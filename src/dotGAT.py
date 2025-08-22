import math
from typing import Union

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from datagen_temporal import SensingMasks


class DotGATHead(nn.Module):
    """
    Fused and vectorised distributed attention-message-passing for all agents and all attention head.
    Shapes:
        x          : [B, A, Din]
        W_q / W_k  : [A, Din, H*Dh]  (H = #heads, Dh = out_per_head)
        W_v        : [Din, H*Dh]     (shared across agents)
    """
    def __init__(self,
                 num_agents: int, in_features: int, out_features: int, *, 
                 sharedV: bool, dropout: float = 0, heads: int = 4):
        super().__init__()
        assert out_features % heads == 0
        self.heads = heads
        self.head_dim = out_features // heads
        self.sharedV = sharedV
        
        # batched weights: one slice per agent
        self.W_q = nn.Parameter(torch.empty(num_agents, in_features, self.heads * self.head_dim))
        self.W_k = nn.Parameter(torch.empty_like(self.W_q))
        if self.sharedV:
            self.W_v_shared = nn.Linear(in_features, self.heads * self.head_dim, bias=True)
        else:
            self.W_v = nn.Parameter(torch.empty_like(self.W_q))

        self.W_fwd  = nn.Parameter(torch.empty(num_agents, out_features, out_features))
        self.b_fwd  = nn.Parameter(torch.empty(num_agents, out_features))

        self.norm1   = nn.RMSNorm(out_features)
        self.norm2   = nn.RMSNorm(out_features)
        self.dropout = dropout
        self.act     = nn.SiLU()
        
        self.reset_parameters()  # initialize weights
    
    def reset_parameters(self):
        """
        Custom initialization to make sure each agent's slice [Din, Dout] is initialized like a 
        normal 2D linear weight
        """
        nonlin = 'leaky_relu'
        with torch.no_grad():
            a_val = 0.75
            nn.init.kaiming_uniform_(self.W_fwd, a=a_val, mode='fan_in', nonlinearity=nonlin)
            nn.init.zeros_(self.b_fwd)
            
            # per-slice, to ignore agent dim
            for i in range(self.W_q.size(0)):
                nn.init.kaiming_uniform_(self.W_q[i], a=a_val, mode='fan_in', nonlinearity=nonlin)
                nn.init.kaiming_uniform_(self.W_k[i], a=a_val, mode='fan_in', nonlinearity=nonlin)
                if not self.sharedV:
                    nn.init.kaiming_uniform_(self.W_v[i], a=a_val, mode='fan_in', nonlinearity=nonlin)
                    
            if self.sharedV:
                self.W_v_shared.reset_parameters()
    
    def _project_qkv(self, x: torch.Tensor):
        # x: [B, A, Din]
        # einsum → single GEMM for all agents
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
        # x: [B, A, Din];   connect: [A, A] or [1, A, A]
        x = self.norm1(x)                                        # pre-norm
        q, k, v = self._project_qkv(x)
        dropout_p = self.dropout if self.training else 0.0
        if attn_bias is not None:
            # [A, A]  -> [1, 1, A, A] so it broadcasts to (B, H, L, S)
            if attn_bias.ndim == 2:
                attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
            attn_bias = attn_bias.to(device=q.device, dtype=q.dtype)
            
        if torch.cuda.is_available():
            # Call SDPA in a context that forbids the slow math kernel
            # Enable Flash first, fall back to Mem-Efficient if Flash is unsupported
            backends_ok = [SDPBackend.FLASH_ATTENTION,
                           SDPBackend.EFFICIENT_ATTENTION]
            # (set_priority=True → Flash tried first, Efficient second)
            with sdpa_kernel(backends_ok, set_priority=True):
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False
                )                            # [B, H, A, Dh]
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False
            )                                # [B, H, A, Dh]
            
        out = out.transpose(1, 2).reshape(x.size(0), -1, self.heads * self.head_dim)
        out = self.norm2(out)                                    # post-norm
        # fused forward projection (one GEMM via einsum); broadcast the bias
        out = torch.einsum('bij,ijk->bik', out, self.W_fwd) + self.b_fwd.unsqueeze(0)
        return self.act(out)


class TrainableSmallWorld(nn.Module):
    """
    Additive attention-bias matrix for DotGAT.
    Only non-frozen positions become nn.Parameters.

    • Non-frozen entries are a learnt parameter vector → scattered every fwd pass  
    • Frozen entries are hard −∞ (so the kernel never attends to them)  
    • Initial edges are pre-wired with a positive bias (~3 -> sigmoid ~0.95)  

    Arguments:
    A               : #agents (nodes)
    k, p            : Watts-Strogatz nearest-neighbours & rewiring prob
    freeze_frac     : fraction **of the absent edges** to keep permanently −∞
    symmetric       : tie (i,j) and (j,i) parameters
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
            # out-of-place logical-and avoids aliasing error
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
        bias[self.learn_row, self.learn_col] = self.bias_param       # type: ignore
        bias.fill_diagonal_(1.0)
        if self.symmetric:                                           # keep symmetry
            bias = torch.maximum(bias, bias.T)
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
        sensing_masks: Union[None, SensingMasks] = None,
        k: int = 4, p: float = 0.0, 
        freeze_zero_frac: float = 1.0
    ):
        super().__init__()
        self.output_dim = n * m
        self.n, self.m = n, m
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps
        self.sharedV = sharedV
        self.dropout = dropout
        self.adjacency_mode = adjacency_mode        
        self.sensing_masks = sensing_masks
        
        self.W_embed = nn.Parameter(torch.empty(size=(num_agents, input_dim, hidden_dim)))
        if self.message_steps > 0:
            self.gat_head = DotGATHead(num_agents, hidden_dim, hidden_dim, 
                                       dropout=dropout, heads=num_heads, sharedV = self.sharedV)
            if adjacency_mode == 'learned':
                self.connect = TrainableSmallWorld(num_agents, device, 
                                                   k=k, p=p, freeze_frac=freeze_zero_frac)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.act    = nn.SiLU()            # fused Swish
        
        self.reset_parameters()  # initialize weights
    
    def reset_parameters(self):
        """
        Custom initialization to make sure each agent's slice [Din, Dout] is initialized like a 
        normal 2D linear weight
        """
        with torch.no_grad():
            a_val = 0.75
            for i in range(self.W_embed.size(0)):
                nn.init.kaiming_uniform_(self.W_embed[i], a=a_val, 
                                         mode='fan_in', nonlinearity='leaky_relu')

    def sense(self, x):
        return x if self.sensing_masks is None else self.sensing_masks(x)
        
    def _message_passing(self, h: torch.Tensor) -> torch.Tensor:
        if self.adjacency_mode == 'learned':
            attn_bias = self.connect()
        else:
            attn_bias = None
        # convert to bool
        #if attn_bias is not None:
        #    attn_bias = attn_bias > 0.0
        for _ in range(self.message_steps):
            h = h + self.gat_head(h, attn_bias)
            h = self.norm2(h)
        return self.act(h)
    
    def forward(self, x):
        # x: [batch, num_agents, input_dim]
        x = self.sense(x)
        agents_embeddings = torch.einsum('bij,ijk->bik', x, self.W_embed)
        h = self.norm1(agents_embeddings)
        # Do attention-based message passing if message_steps > 0; otherwise,
        # network reduces to a simple encoder - decoder
        if self.message_steps > 0:
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
        self.num_agents = num_agents
        self.agent_outputs_dim = agent_outputs_dim
        self.output_dim = m
        self.W_decode = nn.Parameter(torch.empty(size=(num_agents, agent_outputs_dim, m)))
        self.norm_in = nn.RMSNorm(agent_outputs_dim)
        self.act = nn.SiLU()
        
        self.reset_parameters()  # initialize weights
    
    def reset_parameters(self):
        with torch.no_grad():
            a_val = 0.75
            for i in range(self.W_decode.size(0)):
                nn.init.kaiming_uniform_(self.W_decode[i], a=a_val, 
                                         mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, agent_outputs: torch.Tensor) -> torch.Tensor:
        """
        agent_outputs: [B, A, H] 
        returns intermediate logits: [B, A, m]
        """
        _, A, H = agent_outputs.shape
        assert A == self.num_agents and H == self.agent_outputs_dim

        agent_outputs = self.norm_in(agent_outputs)
        agent_decoded = torch.einsum('bij,ijk->bik', agent_outputs, self.W_decode)
        agent_preds = self.act(agent_decoded)
        
        if self.num_agents == 1:
            return agent_preds.squeeze(1)
        else:
            return agent_preds
