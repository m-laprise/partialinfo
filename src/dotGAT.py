"""
Possible optimization: for einsum operations with batch AND agent dimensions,
a bmm path may be faster: flatten batch/agent dims and use grouped matmuls.
to check with profiling.
"""

import math
from typing import Optional, Tuple, Union

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from datautils.sensing import SensingMasks

try:
    # Optional import; only needed for time-series masking helper
    from datautils.sensing import SensingMasksTemporal  # type: ignore
except Exception:
    SensingMasksTemporal = None  # noqa: N816


class DotGATHead(nn.Module):
    """
    Vectorised distributed attention-message-passing for all agents and all attention heads.
    Shapes:
        x          : [B, A, Din]
        W_q / W_k/ W_v_shared : [A, Din, H*Dh]  (H = #heads, Dh = out_per_head)
        W_v        : [Din, H*Dh]     (shared across agents)
    """
    def __init__(self,
                 num_agents: int, in_features: int, out_features: int, *, 
                 sharedV: bool, dropout: float = 0, heads: int = 4):
        super().__init__()
        assert out_features % heads == 0
        if in_features != out_features:
            raise NotImplementedError("Mismatched message-passing input and output dimensions.")
        self.n_agents = num_agents
        self.d_hidden = in_features
        self.heads = heads
        self.head_dim = self.d_hidden // self.heads
        self.sharedV = sharedV
        self.dropout = dropout
        
        # batched weights: one slice per agent
        self.W_q = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.d_hidden))
        self.W_k = nn.Parameter(torch.empty_like(self.W_q))
        if self.sharedV:
            self.W_v_shared = nn.Linear(self.d_hidden, self.d_hidden, bias=False)
        else:
            self.W_v = nn.Parameter(torch.empty_like(self.W_q))
        
        self.reset_parameters()  # initialize weights
    
    def reset_parameters(self):
        """
        Custom initialization to make sure each agent's slice [Din, Dout] is initialized like a 
        normal 2D linear weight
        """
        with torch.no_grad():
            # Note: gain decreases as the a value increases, which narrows the init bounds
            # gain = √2 / √(1 + a²); bound = gain * (√3 / √fan_mode)
            a_val = 1.5
            nonlin = 'leaky_relu'
            # per-slice, to ignore agent dim
            for i in range(self.W_q.size(0)):
                nn.init.xavier_uniform_(self.W_q[i])
                nn.init.xavier_uniform_(self.W_k[i])
                if not self.sharedV:
                    nn.init.kaiming_normal_(self.W_v[i], a=a_val, nonlinearity=nonlin)
            if self.sharedV:
                self.W_v_shared.reset_parameters()
    
    def _project_qkv(self, x: torch.Tensor):
        # x: [B, A, Din]
        # result: [B, A, H*Dh]
        q = torch.einsum('bad,adh->bah', x, self.W_q)
        k = torch.einsum('bad,adh->bah', x, self.W_k)
        if self.sharedV:
            v = self.W_v_shared(x)   
        else:
            v = torch.einsum('bad,adh->bah', x, self.W_v)
        # reshape for scaled_dot_product_attention
        B, A, _ = q.shape
        q = q.view(B, A, self.heads, self.head_dim).transpose(1, 2)  # [B, H, A, Dh]
        k = k.view(B, A, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, A, self.heads, self.head_dim).transpose(1, 2)
        return q, k, v   
    
    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None):
        # x: [B, A, Din];   attn_bias: [A, A] or [1, A, A]
        #x = self.prenorm(x)
        q, k, v = self._project_qkv(x)
        dropout_p = self.dropout if self.training else 0.0
        if attn_bias is not None:
            # [A, A]  -> [1, 1, A, A] so it broadcasts to (B, H, L, S)
            if attn_bias.ndim == 2:
                attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
            attn_bias = attn_bias.to(device=q.device, dtype=q.dtype)
            
        if q.is_cuda:
            # Enable Flash first, fall back to Mem-Efficient if Flash is unsupported
            # Error rather than fall back on slow math kernel
            backends_ok = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            with sdpa_kernel(backends_ok, set_priority=True):
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False
                )                            # [B, H, A, Dh]
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False
            )                                # [B, H, A, Dh]
        out = out.transpose(1, 2).reshape(x.size(0), -1, self.heads * self.head_dim)
        return out


class TrainableSmallWorld(nn.Module):
    """
    Additive attention-bias matrix for DotGAT.
    Only non-frozen positions become nn.Parameters.
    • Non-frozen entries are a learnt parameter vector, scattered every fwd pass  
    • Frozen entries are hard −∞

    Arguments:
    A               : #agents (nodes)
    k, p            : Watts-Strogatz nearest-neighbours & rewiring prob
    freeze_frac     : fraction of the absent edges to keep permanently −∞
    symmetric       : whether to tie (i,j) and (j,i) parameters
    device          : CPU / CUDA
    """
    @torch.no_grad()
    def __init__(self, A, device, *, k=10, p=0.3,
                 freeze_frac=0.5, symmetric=False):
        super().__init__()
        self.A, self.symmetric = A, symmetric
        # adjacency of a small-world graph
        G = nx.watts_strogatz_graph(A, k, p)
        adj = torch.tensor(nx.to_numpy_array(G), dtype=torch.bool, device=device)  # [A,A]
        # Choose a portion of the *zero* entries to freeze permanently
        zeros = (~adj).nonzero(as_tuple=False)          # list of absent edges
        n_freeze = int(math.ceil(len(zeros) * freeze_frac))
        frozen_idx = zeros[torch.randperm(len(zeros))[:n_freeze]]
        # Boolean mask: True -> learnable, False -> frozen (-∞)
        learn_mask = torch.ones(A, A, dtype=torch.bool, device=device)
        learn_mask[frozen_idx[:, 0], frozen_idx[:, 1]] = False
        if symmetric:
            learn_mask = learn_mask & learn_mask.T
        # list learnable parameters
        learnable_idx = learn_mask.nonzero(as_tuple=False)
        self.register_buffer("learn_row",  learnable_idx[:, 0])
        self.register_buffer("learn_col",  learnable_idx[:, 1])
        # single 1-D parameter for the trainable biases
        init = torch.zeros(len(learnable_idx), device=device)
        init[adj[self.learn_row, self.learn_col]] = 0.0        # type: ignore
        self.bias_param = nn.Parameter(init)
        # keep the mask for inspection
        self.register_buffer("learn_mask", learn_mask)

    def forward(self) -> torch.Tensor:
        """
        Returns an additive bias matrix for `scaled_dot_product_attention`:
            (-∞  on frozen entries, shape [1, A, A] broadcasted over batch & heads).
        """
        bias = torch.full((self.A, self.A),
                          float('-inf'),
                          device=self.bias_param.device)
        bias[self.learn_row, self.learn_col] = self.bias_param # type: ignore
        bias.fill_diagonal_(1.0)
        if self.symmetric: # keep symmetry in differentiable way with an average
            bias = 0.5 * (bias + bias.T)
        return bias.unsqueeze(0)


class DistributedDotGAT(nn.Module):
    def __init__(
        self, device, *,
        input_dim: int, # also n x m if vectorized
        hidden_dim: int, # internal dim, e.g. 128
        n: int, m: int, 
        num_agents: int, 
        num_heads: int,
        sharedV: bool,
        dropout: float,
        adjacency_mode: str,
        message_steps: int,
        sensing_masks: Optional[SensingMasks],
        k: int = 4, p: float = 0.0, 
        freeze_zero_frac: float = 1.0
    ):
        super().__init__()
        self.n, self.m = n, m
        self.d_in = input_dim   # = n * m
        self.n_agents = num_agents
        self.d_hidden = hidden_dim # size of the internal state of each agent
        self.message_steps = message_steps
        self.dropout = dropout
        self.adjacency_mode = adjacency_mode
           
        self.sensing_masks = sensing_masks
        if sensing_masks is not None:
            self.register_buffer("agent_mask", sensing_masks.masks)         # [A, E], bool
            self.register_buffer("global_known_mask", sensing_masks.global_known)  # [E], bool
        
        self.W_embed = nn.Parameter(torch.empty(self.n_agents, self.d_in, self.d_hidden))
        
        if self.message_steps > 0  and num_agents > 1:
            self.DotGAT = DotGATHead(self.n_agents, self.d_hidden, self.d_hidden, 
                                     dropout=dropout, heads=num_heads, sharedV=sharedV)
            
            self.W_fwd1 = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.d_hidden))
            self.b_fwd1 = nn.Parameter(torch.zeros(self.n_agents, self.d_hidden))
            self.W_fwd2 = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.d_hidden))
            self.b_fwd2 = nn.Parameter(torch.zeros(self.n_agents, self.d_hidden))
            
            self.attnorm = nn.RMSNorm(self.d_hidden)
            self.mlpnorm = nn.RMSNorm(self.d_hidden)
            
            if self.message_steps > 0 and self.n_agents > 1 and self.adjacency_mode == 'socnet':
                self.connect = TrainableSmallWorld(self.n_agents, device, k=min(k, self.n_agents - 1), 
                                                   p=p, freeze_frac=freeze_zero_frac)
        
        self.residual_drop1 = nn.Dropout(self.dropout)
        self.residual_drop2 = nn.Dropout(self.dropout)
        
        self.prenorm = nn.RMSNorm(self.d_hidden)
        self.act    = nn.SiLU()            # fused Swish
        
        self.reset_parameters()  # initialize weights
    
    def reset_parameters(self):
        """
        Custom initialization to make sure each agent's slice [Din, Dout] is initialized like a 
        normal 2D linear weight
        """
        with torch.no_grad():
            a_val = 1.5
            nonlin = 'leaky_relu'
            for i in range(self.W_embed.size(0)):
                # dims are reversed so this is actually fan_in
                nn.init.kaiming_normal_(self.W_embed[i], a=a_val, mode='fan_out', nonlinearity=nonlin)
                if self.message_steps > 0 and self.n_agents > 1:
                    nn.init.kaiming_normal_(self.W_fwd1[i], a=a_val, nonlinearity=nonlin)
                    nn.init.kaiming_normal_(self.W_fwd2[i], a=a_val, nonlinearity=nonlin)
        
    def _mlp(self, h: torch.Tensor) -> torch.Tensor:
        # fused forward projection (one GEMM via einsum); broadcast the bias
        h = torch.einsum('bij,ijk->bik', h, self.W_fwd1) + self.b_fwd1.unsqueeze(0)
        h = self.act(h)
        h = torch.einsum('bij,ijk->bik', h, self.W_fwd2) + self.b_fwd2.unsqueeze(0)
        return h
    
    def _message_passing(self, h: torch.Tensor) -> torch.Tensor:
        if self.adjacency_mode == 'socnet':
            attn_bias = self.connect()
        else:
            attn_bias = None
        for _ in range(self.message_steps):
            h = h + self.residual_drop1(self.DotGAT(self.attnorm(h), attn_bias))
            h = h + self.residual_drop2(self._mlp(self.mlpnorm(h)))
        return self.attnorm(h)
    
    def forward(self, x):
        # x: [batch, num_agents, d_in]
        # Apply sparse mask to convey partial information to agents
        if self.sensing_masks is not None:
            # Broadcast: [1, A, E] multiplies into [B, A, E]
            x = x * self.agent_mask.unsqueeze(0)    # type: ignore
        else:
            print("WARNING: No mask applied. Proceeding with full information.")
        # Agent embeddings to internal state
        h = torch.einsum('bij,ijk->bik', x, self.W_embed)
        h = self.prenorm(h)
        # Do attention-based message passing if message_steps > 0; otherwise,
        # network reduces to a simple encoder - decoder
        if self.message_steps > 0 and self.n_agents > 1:
            h = self._message_passing(h)
        return h


class ReconDecoder(nn.Module):
    def __init__(self, hidden_dim, n, m, num_agents):
        super().__init__()
        self.output_dim = n * m
        self.n, self.m = n, m
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.maxrank = min(self.n // 2, self.m // 2)
        self.U_proj = nn.Linear(hidden_dim, self.n * self.maxrank, bias=False)
        self.V_proj = nn.Linear(hidden_dim, self.m * self.maxrank, bias=False)

    def forward(self, h):
        B, A, H = h.shape
        U = self.U_proj(h).view(B, A, self.n, self.maxrank)  # [B, A, n, maxrank]
        V = self.V_proj(h).view(B, A, self.m, self.maxrank)  # [B, A, m, maxrank]

        # Compute H = U @ Vt for each agent
        recon = torch.matmul(U, V.transpose(-1, -2)) / self.maxrank # [B, A, n, m]
        recon = recon.view(B, A, self.n * self.m)  # vectorize: [B, A, n * m]
        return recon  # [batch, num_agents, output_dim]


class Aggregator(nn.Module):
    """
    Input-dependent aggregation module. It reads each agent's output,
    uses a shared MLP to compute per-agent gates based on their output,
    applies a softmax over agent gates, and then performs weighted sum.
    
    We hope this is flexible enough to let the model learn how to downweight
    non-informative agent outputs (e.g. near-zero).
    """
    def __init__(self, num_agents: int, output_dim: int, hidden_dim: int=4):
        super().__init__()
        self.num_agents = num_agents
        self.output_dim = output_dim
        if num_agents > 1:
            self.gate_mlp = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, agent_outputs: torch.Tensor) -> torch.Tensor:
        """
        agent_outputs: [B, A, nm] 
        returns: [B, nm] aggregated output
        """
        B, A, nm = agent_outputs.shape
        assert A == self.num_agents and nm == self.output_dim
        
        if self.num_agents == 1:
            return agent_outputs.squeeze(1)
        else:
            gates = self.gate_mlp(agent_outputs)  # [B, A, 1]
            weights = F.softmax(gates, dim=1)     # [B, A, 1]
            weighted_output = agent_outputs * weights  # [B, A, D]
            return weighted_output.sum(dim=1)     # [B, D]


class CollectiveClassifier(nn.Module):
    """
    Classification module decoding internal states to logits for each agent.
    """
    def __init__(self, num_agents: int, agent_outputs_dim: int, m: int):
        super().__init__()
        self.n_agents = num_agents
        self.agent_d_out = agent_outputs_dim
        self.n_classes = m
        self.W_decode = nn.Parameter(
            torch.empty(self.n_agents, self.agent_d_out, self.n_classes)
        )
        self.b_decode = nn.Parameter(torch.zeros(self.n_agents, self.n_classes))
        self.prenorm = nn.RMSNorm(self.agent_d_out)
        
        self.reset_parameters()  # initialize weights
    
    def reset_parameters(self):
        with torch.no_grad():
            for i in range(self.W_decode.size(0)):
                nn.init.xavier_uniform_(self.W_decode[i])

    def forward(self, agent_outputs: torch.Tensor) -> torch.Tensor:
        """
        agent_outputs: [B, A, H] 
        returns intermediate logits: [B, A, m]
        """
        _, A, H = agent_outputs.shape
        if A != self.n_agents or H != self.agent_d_out:
            raise ValueError(f"Expected a_outputs shape [B,{self.n_agents},{self.agent_d_out}], got {agent_outputs.shape}")

        agent_outputs = self.prenorm(agent_outputs)
        agent_logits = torch.einsum('bij,ijk->bik', agent_outputs, self.W_decode)
        agent_logits = agent_logits + self.b_decode.unsqueeze(0)
        
        return agent_logits


class CollectiveInferPredict(nn.Module):
    """
    Regression module decoding internal states to next row values and outcome prediction
    for each agent.
    """
    def __init__(self, num_agents: int, agent_outputs_dim: int, m: int, y_dim: int = 1):
        super().__init__()
        self.n_agents = num_agents
        self.agent_d_out = agent_outputs_dim
        self.m = m
        self.y_dim = m #y_dim
        
        self.W_fwd_H = nn.Parameter(torch.empty(self.n_agents, self.agent_d_out, self.agent_d_out))
        self.b_fwd_H = nn.Parameter(torch.zeros(self.n_agents, self.agent_d_out))
        
        self.W_decode = nn.Parameter(torch.empty(self.n_agents, self.agent_d_out, self.m))
        self.b_decode = nn.Parameter(torch.zeros(self.n_agents, self.m))
        
        self.W_predict = nn.Parameter(torch.empty(self.n_agents, self.m, self.y_dim))
        self.b_predict = nn.Parameter(torch.zeros(self.n_agents, self.y_dim))
        
        self.prenorm = nn.RMSNorm(self.agent_d_out)
        self.swish = nn.SiLU()
        
        self.reset_parameters()  # initialize weights
    
    def reset_parameters(self):
        with torch.no_grad():
            for i in range(self.W_decode.size(0)):
                nn.init.xavier_uniform_(self.W_fwd_H[i])
                nn.init.xavier_uniform_(self.W_decode[i])
                nn.init.xavier_uniform_(self.W_predict[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, A, H]  (agent_outputs)
        returns intermediate next row prediction and outcome prediction: [B, A, m + y_dim]
        """
        B, A, H = x.shape
        
        if A != self.n_agents or H != self.agent_d_out:
            raise ValueError(f"Expected a_outputs shape [B,{self.n_agents},{self.agent_d_out}], got {x.shape}")

        x = self.prenorm(x) # [B, A, H] 
        x = torch.einsum('bah,ahm->bam', x, self.W_fwd_H) + self.b_fwd_H # [B, A, H] 
        x = self.swish(x)
        
        agent_m = torch.einsum('bah,ahm->bam', x, self.W_decode) + self.b_decode # [B, A, m] 
        agent_m = self.swish(agent_m)
        
        agent_y = torch.einsum('ban,any->bay', agent_m, self.W_predict) + self.b_predict # [B, A, y_dim]
        
        return agent_y


class DynamicDotGAT(nn.Module):
    """
    Time-series variant of DistributedDotGAT for a network of agents that exchange messages.
    Each agent performs two attentions per message step:
      1) Memory (temporal) self-attention over its own past states t' <= t (causal).
      2) Social attention over its neighborhood across agents at the same time t.

    Shapes used inside:
      - Inputs per batch as matrices: [B, T, M] or agent-masked inputs [B, A, T, M].
      - Internal hidden states h: [B, T, A, H].

    If sensing_masks_temporal is provided, it will be used to expand [B, T, M] into [B, A, T, M]
    by masking columns per agent; otherwise inputs are assumed to already be [B, A, T, M].

    Optional per-timestep, per-agent prediction head outputs logits of shape [B, T, A, y_dim].
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
        sensing_masks_temporal: Optional[object] = None,  # SensingMasksTemporal
        y_dim: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.m = int(m)
        self.n_agents = int(num_agents)
        self.d_hidden = int(hidden_dim)
        self.heads = int(num_heads)
        self.head_dim = self.d_hidden // self.heads
        self.message_steps = int(message_steps)
        self.dropout = float(dropout)
        self.adjacency_mode = adjacency_mode
        self.sharedV = bool(sharedV)
        #self.y_dim = int(y_dim) if y_dim is not None else None
        if y_dim is not None:
            if isinstance(y_dim, tuple):
                self.y_dim, self.tm1 = y_dim[1], y_dim[0]
            else:
                self.y_dim = y_dim
        else:
            self.y_dim = None
        self.device_ref = device

        # Optional temporal sensing masks (per-agent column masks) -> [A, M]
        self.smt_available = sensing_masks_temporal is not None and hasattr(sensing_masks_temporal, 'col_masks')
        if self.smt_available:
            col_masks = sensing_masks_temporal.col_masks  # type: ignore[attr-defined]
            self.register_buffer("agent_col_masks", col_masks.bool())  # [A, M]
        else:
            self.register_buffer("agent_col_masks", torch.empty(0, dtype=torch.bool))

        # Per-agent row embedding: [A, M, H]
        self.W_embed = nn.Parameter(torch.empty(self.n_agents, self.m, self.d_hidden))

        # Memory attention (causal across time) per agent
        self.W_q_mem = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.d_hidden))
        self.W_k_mem = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.d_hidden))
        if self.sharedV:
            self.W_v_mem_shared = nn.Linear(self.d_hidden, self.d_hidden, bias=False)
        else:
            self.W_v_mem = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.d_hidden))

        # Social attention (across agents at same time) per agent
        self.W_q_soc = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.d_hidden))
        self.W_k_soc = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.d_hidden))
        if self.sharedV:
            self.W_v_soc_shared = nn.Linear(self.d_hidden, self.d_hidden, bias=False)
        else:
            self.W_v_soc = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.d_hidden))

        # Optional connectivity bias between agents
        if self.message_steps > 0 and self.n_agents > 1 and self.adjacency_mode == 'socnet':
            self.connect = TrainableSmallWorld(
                self.n_agents, device,
                k=min(k, max(1, self.n_agents - 1)), p=p, freeze_frac=freeze_zero_frac
            )

        # Feed-forward (per-agent batched linear via einsum)
        self.W_fwd1 = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.d_hidden))
        self.b_fwd1 = nn.Parameter(torch.zeros(self.n_agents, self.d_hidden))
        self.W_fwd2 = nn.Parameter(torch.empty(self.n_agents, self.d_hidden, self.d_hidden))
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

        # Optional per-timestep prediction head per agent: [A, H, y]
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
        with torch.no_grad():
            a_val = 1.5
            nonlin = 'leaky_relu'
            # Embedding
            for i in range(self.W_embed.size(0)):
                nn.init.kaiming_normal_(self.W_embed[i], a=a_val, mode='fan_out', nonlinearity=nonlin)
            # Memory projections
            for i in range(self.n_agents):
                nn.init.xavier_uniform_(self.W_q_mem[i])
                nn.init.xavier_uniform_(self.W_k_mem[i])
                if not self.sharedV:
                    nn.init.kaiming_normal_(self.W_v_mem[i], a=a_val, nonlinearity=nonlin)
            if self.sharedV:
                self.W_v_mem_shared.reset_parameters()
            # Social projections
            for i in range(self.n_agents):
                nn.init.xavier_uniform_(self.W_q_soc[i])
                nn.init.xavier_uniform_(self.W_k_soc[i])
                if not self.sharedV:
                    nn.init.kaiming_normal_(self.W_v_soc[i], a=a_val, nonlinearity=nonlin)
            if self.sharedV:
                self.W_v_soc_shared.reset_parameters()
            # MLP
            for i in range(self.n_agents):
                nn.init.kaiming_normal_(self.W_fwd1[i], a=a_val, nonlinearity=nonlin)
                nn.init.kaiming_normal_(self.W_fwd2[i], a=a_val, nonlinearity=nonlin)
            # Prediction
            if self.W_decode_pred is not None:
                for i in range(self.n_agents):
                    nn.init.xavier_uniform_(self.W_decode_pred[i])

    def _apply_mlp(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, A, H]
        h = torch.einsum('btah,ahd->btad', h, self.W_fwd1) + self.b_fwd1.unsqueeze(0).unsqueeze(0)
        h = self.act(h)
        h = torch.einsum('btah,ahd->btad', h, self.W_fwd2) + self.b_fwd2.unsqueeze(0).unsqueeze(0)
        return h

    def _memory_attention(self, h: torch.Tensor) -> torch.Tensor:
        """Causal self-attention across time per agent.
        Input/Output shape: [B, T, A, H]
        """
        B, T, A, H = h.shape
        q = torch.einsum('btah,ahd->btad', h, self.W_q_mem)
        k = torch.einsum('btah,ahd->btad', h, self.W_k_mem)
        if self.sharedV:
            v = self.W_v_mem_shared(h)
        else:
            v = torch.einsum('btah,ahd->btad', h, self.W_v_mem)

        # [B, T, A, H] -> [B, A, heads, T, Dh] -> merge BA
        q = q.view(B, T, A, self.heads, self.head_dim).permute(0, 2, 3, 1, 4).contiguous()
        k = k.view(B, T, A, self.heads, self.head_dim).permute(0, 2, 3, 1, 4).contiguous()
        v = v.view(B, T, A, self.heads, self.head_dim).permute(0, 2, 3, 1, 4).contiguous()
        q = q.view(B * A, self.heads, T, self.head_dim)
        k = k.view(B * A, self.heads, T, self.head_dim)
        v = v.view(B * A, self.heads, T, self.head_dim)

        dropout_p = self.dropout if self.training else 0.0
        if q.is_cuda:
            backends_ok = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            with sdpa_kernel(backends_ok, set_priority=True):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        # Back to [B, T, A, H]
        out = out.view(B, A, self.heads, T, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()
        out = out.view(B, T, A, H)
        return out

    def _social_attention(self, h: torch.Tensor) -> torch.Tensor:
        """Attention across agents per time step using optional adjacency bias.
        Input/Output shape: [B, T, A, H]
        """
        B, T, A, H = h.shape
        q = torch.einsum('btah,ahd->btad', h, self.W_q_soc)
        k = torch.einsum('btah,ahd->btad', h, self.W_k_soc)
        if self.sharedV:
            v = self.W_v_soc_shared(h)
        else:
            v = torch.einsum('btah,ahd->btad', h, self.W_v_soc)

        # [B, T, A, H] -> [B, T, heads, A, Dh] -> merge BT
        q = q.view(B, T, A, self.heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        k = k.view(B, T, A, self.heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        v = v.view(B, T, A, self.heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        q = q.view(B * T, self.heads, A, self.head_dim)
        k = k.view(B * T, self.heads, A, self.head_dim)
        v = v.view(B * T, self.heads, A, self.head_dim)

        attn_bias = None
        if hasattr(self, 'connect') and self.adjacency_mode == 'socnet':
            attn_bias = self.connect()  # [1, A, A]
            attn_bias = attn_bias.to(device=q.device, dtype=q.dtype).unsqueeze(1)  # [1, 1, A, A]

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

        out = out.view(B, T, self.heads, A, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        out = out.view(B, T, A, H)
        return out

    def _maybe_mask_inputs(self, X: torch.Tensor) -> torch.Tensor:
        """Ensure inputs are [B, A, T, M], applying per-agent column masks if needed.
        Accepts [B, T, M], [T, M], [B, A, T, M], or [A, T, M].
        """
        if X.dim() == 3:  # [B, T, M]
            assert self.smt_available, "Provide SensingMasksTemporal or pass [B, A, T, M] inputs."
            B, T, M = X.shape
            assert M == self.m, f"Expected last dim M={self.m}, got {M}"
            col_masks = self.agent_col_masks.to(device=X.device)  # [A, M]
            X_BA_tm = X.unsqueeze(1).repeat(1, self.n_agents, 1, 1)  # [B, A, T, M]
            return X_BA_tm * col_masks[None, :, None, :] # type: ignore
        elif X.dim() == 2:  # [T, M]
            assert self.smt_available, "Provide SensingMasksTemporal or pass [A, T, M] inputs."
            T, M = X.shape
            assert M == self.m
            col_masks = self.agent_col_masks.to(device=X.device)
            X_A_tm = X.unsqueeze(0).repeat(self.n_agents, 1, 1)  # [A, T, M]
            return (X_A_tm * col_masks[:, None, :]).unsqueeze(0)  # [1, A, T, M] # type: ignore
        elif X.dim() == 4 and X.shape[1] == self.n_agents:  # [B, A, T, M]
            assert X.shape[-1] == self.m
            return X
        elif X.dim() == 3 and X.shape[0] == self.n_agents:  # [A, T, M]
            assert X.shape[-1] == self.m
            return X.unsqueeze(0)  # [1, A, T, M]
        else:
            raise ValueError(f"Unexpected input shape {tuple(X.shape)}; expected [B,T,M], [T,M], [B,A,T,M], or [A,T,M]")

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward through K message passing steps with memory and social attentions.

        Args:
            X: inputs with shapes [B, T, M] (will be masked using SensingMasksTemporal)
               or pre-masked [B, A, T, M].

        Returns:
            h: final hidden states [B, T, A, H]
            y: optional per-timestep predictions [B, T, A, y_dim] if y_dim was provided
        """
        # Prepare [B, A, T, M]
        X = self._maybe_mask_inputs(X)
        B, A, T, M = X.shape
        assert A == self.n_agents and M == self.m

        # Per-agent embedding of each time row -> [B, T, A, H]
        h = torch.einsum('batm,amh->bath', X, self.W_embed)  # [B, A, T, H]
        h = h.permute(0, 2, 1, 3).contiguous()  # [B, T, A, H]
        h = self.prenorm(h)

        if self.message_steps > 0 and self.n_agents > 0:
            for _ in range(self.message_steps):
                # Memory attention over time (causal) per agent
                h_mem = self._memory_attention(self.memnorm(h))
                h = h + self.drop_mem(h_mem)

                # Social attention across agents at each time step
                if self.n_agents > 1:
                    h_soc = self._social_attention(self.socnorm(h))
                    h = h + self.drop_soc(h_soc)

                # Feed-forward
                h_ff = self._apply_mlp(self.mlpnorm(h))
                h = h + self.drop_mlp(h_ff)

        y = None
        if self.y_dim is not None and self.y_dim > 0:
            # Per-agent, per-time predictions
            h_pred = self.prednorm(h) if self.prednorm is not None else h
            y = torch.einsum('btah,ahy->btay', h_pred, self.W_decode_pred) + self.b_decode_pred.unsqueeze(0).unsqueeze(0)  # type: ignore[arg-type]

        return h, y
    