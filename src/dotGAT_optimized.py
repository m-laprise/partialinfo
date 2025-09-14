"""
Optimized version of DynamicDotGAT with fused QKV projections and batched matrix multiplications.
Main optimizations:
1. Pre-fused QKV weight storage
2. Batched matrix multiplications instead of einsum
3. Reduced tensor reshaping overhead
"""

import math
from typing import Callable, Optional, Tuple, Union

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from datautils.sensing import SensingMasks

try:
    from datautils.sensing import SensingMasksTemporal
except Exception:
    SensingMasksTemporal = None


def _agent_init(params: torch.Tensor,
                init: Union[str, Callable],
                *args, **kwargs) -> None:
    """Custom in-place initialization of per-agent slices."""
    init_fn = getattr(nn.init, init) if isinstance(init, str) else init
    with torch.no_grad():
        n = params.size(0)
        for i in range(n):
            init_fn(params[i], *args, **kwargs)


class TrainableSmallWorld(nn.Module):
    """Additive attention-bias matrix for DotGAT."""
    @torch.no_grad()
    def __init__(self, A, device, *, k=10, p=0.3,
                 freeze_frac=0.5, symmetric=False):
        super().__init__()
        self.A, self.symmetric = A, symmetric
        G = nx.watts_strogatz_graph(A, k, p)
        adj = torch.tensor(nx.to_numpy_array(G), dtype=torch.bool, device=device)
        zeros = (~adj).nonzero(as_tuple=False)
        n_freeze = int(math.ceil(len(zeros) * freeze_frac))
        frozen_idx = zeros[torch.randperm(len(zeros))[:n_freeze]]
        learn_mask = torch.ones(A, A, dtype=torch.bool, device=device)
        learn_mask[frozen_idx[:, 0], frozen_idx[:, 1]] = False
        if symmetric:
            learn_mask = learn_mask & learn_mask.T
        learnable_idx = learn_mask.nonzero(as_tuple=False)
        self.register_buffer("learn_row",  learnable_idx[:, 0])
        self.register_buffer("learn_col",  learnable_idx[:, 1])
        init = torch.zeros(len(learnable_idx), device=device)
        init[adj[self.learn_row, self.learn_col]] = 0.0
        self.bias_param = nn.Parameter(init)
        self.register_buffer("learn_mask", learn_mask)

    def forward(self) -> torch.Tensor:
        bias = torch.full((self.A, self.A),
                          float('-inf'),
                          device=self.bias_param.device)
        bias[self.learn_row, self.learn_col] = self.bias_param
        bias.fill_diagonal_(1.0)
        if self.symmetric:
            bias = 0.5 * (bias + bias.T)
        return bias.unsqueeze(0)


class OptimizedDynamicDotGAT(nn.Module):
    """
    Optimized time-series variant of DynamicDotGAT with fused QKV projections.
    """
    def __init__(
        self,
        device,
        *,
        m: int,
        num_agents: int,
        hidden_dim: int,
        num_heads: int = 4,
        message_steps: int = 1,
        dropout: float = 0.0,
        adjacency_mode: str = 'socnet',
        sharedV: bool = False,
        k: int = 4,
        p: float = 0.0,
        freeze_zero_frac: float = 1.0,
        sensing_masks_temporal: Optional[object] = None, 
        y_dim: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.m = int(m)
        self.n_agents = int(num_agents)
        self.d_hidden = int(hidden_dim)
        self.heads = int(num_heads)
        self.head_dim = self.d_hidden // self.heads
        self.message_steps = int(message_steps) if self.n_agents > 1 else 1
        self.dropout = float(dropout)
        self.adjacency_mode = adjacency_mode
        self.sharedV = bool(sharedV)
        if y_dim is not None:
            if isinstance(y_dim, tuple):
                _, self.y_dim = y_dim
            else:
                self.y_dim = y_dim
        else:
            self.y_dim = None
        self.device_ref = device

        # Optional sensing masks
        self.smt_available = sensing_masks_temporal is not None and hasattr(
            sensing_masks_temporal, 'col_masks'
        )
        if self.smt_available:
            col_masks = sensing_masks_temporal.col_masks
            self.register_buffer("agent_col_masks", col_masks.bool())
        else:
            self.register_buffer("agent_col_masks", torch.empty(0, dtype=torch.bool))

        # Per-agent row embedding
        self.W_embed = nn.Parameter(torch.empty(self.n_agents, self.m, self.d_hidden))

        # OPTIMIZATION: Fused QKV projections stored as single tensors
        if self.sharedV:
            # Separate Q,K (per-agent) and V (shared)
            self.W_qk_mem = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, 2 * self.d_hidden))
            self.W_qk_soc = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, 2 * self.d_hidden))
            self.W_v_mem_shared = nn.Linear(self.d_hidden, self.d_hidden, bias=False)
            self.W_v_soc_shared = nn.Linear(self.d_hidden, self.d_hidden, bias=False)
        else:
            # Full QKV fusion
            self.W_qkv_mem = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, 3 * self.d_hidden))
            self.W_qkv_soc = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, 3 * self.d_hidden))

        # Optional connectivity bias between agents
        if self.message_steps > 0 and self.n_agents > 1 and self.adjacency_mode == 'socnet':
            self.connect = TrainableSmallWorld(
                self.n_agents, device,
                k=min(k, max(1, self.n_agents - 1)), p=p, freeze_frac=freeze_zero_frac
            )

        # Feed-forward (optimized for batched operations)
        self.W_fwd = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, 2 * self.d_hidden))
        self.b_fwd1 = nn.Parameter(torch.zeros(self.n_agents, self.d_hidden))
        self.b_fwd2 = nn.Parameter(torch.zeros(self.n_agents, self.d_hidden))

        # Norms & activations
        self.prenorm = nn.RMSNorm(self.d_hidden)
        self.memnorm = nn.RMSNorm(self.d_hidden)
        self.socnorm = nn.RMSNorm(self.d_hidden)
        self.mlpnorm = nn.RMSNorm(self.d_hidden)
        self.act = nn.SiLU()

        # Residual dropouts
        self.drop_mem = nn.Dropout(self.dropout)
        self.drop_soc = nn.Dropout(self.dropout)
        self.drop_mlp = nn.Dropout(self.dropout)

        # Optional prediction head
        if self.y_dim is not None and self.y_dim > 0:
            self.W_decode_pred = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.y_dim))
            self.b_decode_pred = nn.Parameter(torch.zeros(self.n_agents, self.y_dim))
            self.prednorm = nn.RMSNorm(self.d_hidden)
        else:
            self.W_decode_pred = None
            self.b_decode_pred = None
            self.prednorm = None

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        a_val = 1.5
        nonlin = "leaky_relu"
        
        # Embedding
        _agent_init(self.W_embed, "kaiming_normal_", a=a_val, mode="fan_out", nonlinearity=nonlin)
        
        # Fused QKV weights
        if self.sharedV:
            _agent_init(self.W_qk_mem, "xavier_uniform_")
            _agent_init(self.W_qk_soc, "xavier_uniform_") 
            self.W_v_mem_shared.reset_parameters()
            self.W_v_soc_shared.reset_parameters()
        else:
            _agent_init(self.W_qkv_mem, "xavier_uniform_")
            _agent_init(self.W_qkv_soc, "xavier_uniform_")
        
        # Feed-forward
        _agent_init(self.W_fwd, "kaiming_normal_", a=a_val, nonlinearity=nonlin)
        
        # Prediction head
        if self.W_decode_pred is not None:
            _agent_init(self.W_decode_pred, "xavier_uniform_")

    def _optimized_mlp(self, h: torch.Tensor) -> torch.Tensor:
        """Optimized MLP with fused weight matrix using einsum."""
        B, T, A, H = h.shape

        # Fused forward pass with einsum
        fwd = torch.einsum('btah,ahd->btad', h, self.W_fwd)  # [B, T, A, 2*H]

        # Split and apply activations
        h1, h2 = fwd.chunk(2, dim=-1)
        h1 = self.act(h1 + self.b_fwd1.unsqueeze(0).unsqueeze(0))
        h2 = h2 + self.b_fwd2.unsqueeze(0).unsqueeze(0)

        return h2

    def _optimized_memory_attention(self, h: torch.Tensor) -> torch.Tensor:
        """Optimized memory attention with fused QKV."""
        B, T, A, H = h.shape

        if self.sharedV:
            # QK fused, V separate
            qk = torch.einsum('btah,ahd->btad', h, self.W_qk_mem)  # [B, T, A, 2*H]
            q, k = qk.chunk(2, dim=-1)
            v = self.W_v_mem_shared(h)
        else:
            # Full QKV fusion
            qkv = torch.einsum('btah,ahd->btad', h, self.W_qkv_mem)  # [B, T, A, 3*H]
            q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention: [B*A, heads, T, head_dim]
        q = q.view(B, T, A, self.heads, self.head_dim)
        k = k.view(B, T, A, self.heads, self.head_dim)
        v = v.view(B, T, A, self.heads, self.head_dim)
        
        q = q.permute(0, 2, 3, 1, 4).contiguous().view(B * A, self.heads, T, self.head_dim)
        k = k.permute(0, 2, 3, 1, 4).contiguous().view(B * A, self.heads, T, self.head_dim)
        v = v.permute(0, 2, 3, 1, 4).contiguous().view(B * A, self.heads, T, self.head_dim)

        dropout_p = self.dropout if self.training else 0.0
        
        if q.is_cuda:
            backends_ok = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            with sdpa_kernel(backends_ok, set_priority=True):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        # Reshape back to [B, T, A, H]
        out = out.view(B, A, self.heads, T, self.head_dim)
        out = out.permute(0, 3, 1, 2, 4).contiguous().view(B, T, A, H)
        return out

    def _optimized_social_attention(self, h: torch.Tensor) -> torch.Tensor:
        """Optimized social attention with fused QKV."""
        B, T, A, H = h.shape

        if self.sharedV:
            # QK fused, V separate
            qk = torch.einsum('btah,ahd->btad', h, self.W_qk_soc)  # [B, T, A, 2*H]
            q, k = qk.chunk(2, dim=-1)
            v = self.W_v_soc_shared(h)
        else:
            # Full QKV fusion
            qkv = torch.einsum('btah,ahd->btad', h, self.W_qkv_soc)  # [B, T, A, 3*H]
            q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention: [B*T, heads, A, head_dim]
        q = q.view(B, T, A, self.heads, self.head_dim)
        k = k.view(B, T, A, self.heads, self.head_dim)
        v = v.view(B, T, A, self.heads, self.head_dim)
        
        q = q.permute(0, 1, 3, 2, 4).contiguous().view(B * T, self.heads, A, self.head_dim)
        k = k.permute(0, 1, 3, 2, 4).contiguous().view(B * T, self.heads, A, self.head_dim)
        v = v.permute(0, 1, 3, 2, 4).contiguous().view(B * T, self.heads, A, self.head_dim)

        attn_bias = None
        if hasattr(self, 'connect') and self.adjacency_mode == 'socnet':
            attn_bias = self.connect()
            attn_bias = attn_bias.to(device=q.device, dtype=q.dtype).unsqueeze(1)

        dropout_p = self.dropout if self.training else 0.0
        
        if q.is_cuda:
            backends_ok = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            with sdpa_kernel(backends_ok, set_priority=True):
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False
                )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False
            )

        # Reshape back to [B, T, A, H]
        out = out.view(B, T, self.heads, A, self.head_dim)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, T, A, H)
        return out

    def _maybe_mask_inputs(self, X: torch.Tensor) -> torch.Tensor:
        """Ensure inputs are [B, A, T, M], applying per-agent column masks if needed."""
        if X.dim() == 3:  # [B, T, M] 
            assert self.smt_available, "Provide SensingMasksTemporal or pass [B, A, T, M] inputs."
            B, T, M = X.shape
            assert M == self.m, f"Expected last dim M={self.m}, got {M}"
            col_masks = self.agent_col_masks
            return X.unsqueeze(1) * col_masks[None, :, None, :]
        elif X.dim() == 2:  # [T, M]
            assert self.smt_available, "Provide SensingMasksTemporal or pass [A, T, M] inputs."
            T, M = X.shape
            assert M == self.m
            col_masks = self.agent_col_masks
            X_A_tm = X.unsqueeze(0)
            X_A_tm = X_A_tm * col_masks[:, None, :]
            return X_A_tm.unsqueeze(0)
        elif X.dim() == 4 and X.shape[1] == self.n_agents:  # [B, A, T, M]
            assert X.shape[-1] == self.m
            return X
        elif X.dim() == 3 and X.shape[0] == self.n_agents:  # [A, T, M]
            return X.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input shape: {X.shape}")

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optimized operations."""
        # Ensure correct input format
        X_batm = self._maybe_mask_inputs(X)  # [B, A, T, M]
        B, A, T, M = X_batm.shape

        # Efficient embedding using einsum
        h = torch.einsum('batm,amh->bath', X_batm, self.W_embed)  # [B, A, T, H]
        h = h.permute(0, 2, 1, 3).contiguous()  # [B, T, A, H]
        h = self.prenorm(h)

        # Message passing with optimized attention
        if self.message_steps > 0 and self.n_agents > 1:
            for _ in range(self.message_steps):
                # Memory attention
                h = h + self.drop_mem(self._optimized_memory_attention(self.memnorm(h)))
                # Social attention
                h = h + self.drop_soc(self._optimized_social_attention(self.socnorm(h)))
                # MLP
                h = h + self.drop_mlp(self._optimized_mlp(self.mlpnorm(h)))

        # Optional prediction head
        y_pred = None
        if self.W_decode_pred is not None:
            h_pred = self.prednorm(h) if self.prednorm is not None else h
            y_pred = torch.einsum('btah,ahy->btay', h_pred, self.W_decode_pred)
            y_pred = y_pred + self.b_decode_pred.unsqueeze(0).unsqueeze(0)

        return h, y_pred