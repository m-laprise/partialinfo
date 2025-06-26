import math

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class DotGATLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.v_proj = nn.Linear(in_features, out_features, bias=False)
        
        self.forward_proj = nn.Sequential(
            Swish(), nn.LayerNorm(out_features), 
            nn.Linear(out_features, out_features),
            Swish(), nn.LayerNorm(out_features), 
            nn.Linear(out_features, out_features),
        )
        
        self.norm = nn.LayerNorm(out_features)
        self.dropout = dropout
        self.scale = math.sqrt(out_features)
    
    def forward(self, x: torch.Tensor, connectivity: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_agents, hidden_dim]
        B, A, H = x.shape
        
        x = x.view(B * A, H)  # Flatten batch and agents
        # Compute query, key, and value matrices
        Q = self.q_proj(x).view(B, A, -1)
        K = self.k_proj(x).view(B, A, -1)
        V = self.v_proj(x).view(B, A, -1)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # [B, A, A]

        # Add -inf mask where connectivity is 0
        mask = (connectivity == 0).float() * -1e9  # [A, A]
        scores = scores + mask        # [B, A, A]
        
        # Full attention (no top-k)
        alpha = F.softmax(scores, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        H = torch.matmul(alpha, V)  # [B, A, hidden_dim]
        out = self.forward_proj(H)
        return self.norm(out)


class DistributedDotGAT(nn.Module):
    def __init__(
        self, 
        input_dim: int, # also n x m if vectorized
        hidden_dim: int, # internal dim, e.g. 128
        n: int,
        m: int,
        num_agents: int,
        num_heads: int,
        dropout: float,
        adjacency_mode: str = 'none',
        message_steps: int = 3,
    ):
        super().__init__()
        self.output_dim = n * m
        self.n, self.m = n, m
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps
        self.dropout = dropout
        self.adjacency_mode = adjacency_mode
        
        self.agent_input_proj = self._build_input_proj(input_dim, hidden_dim)
        self.connectivity = self._init_connectivity(adjacency_mode, num_agents)
        
        if message_steps > 0:
            self.gat_heads = nn.ModuleList([
                DotGATLayer(hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(num_heads)
            ])

        self.maxrank = min(self.n // 2, self.m // 2)
        self.U_proj = nn.Linear(hidden_dim, self.n * self.maxrank, bias=False)
        self.V_proj = nn.Linear(hidden_dim, self.n * self.maxrank, bias=False)

    def _build_input_proj(self, in_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 2 * hidden_dim, bias=False),
            Swish(), nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=False),
            Swish(), nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
        )
        
    def _init_connectivity(self, mode: str, num_agents: int) -> torch.Tensor:
        if mode == 'none':
            return torch.ones(num_agents, num_agents)
        elif mode == 'learned':
            G = nx.watts_strogatz_graph(n=num_agents, k=10, p=0.3)
            A = torch.zeros(num_agents, num_agents)
            for i, j in G.edges():
                A[i, j] = A[j, i] = 1.0 # Ensure symmetry

            # Learnable parameter initialized with small noise
            init_adj = A + 0.01 * torch.randn(num_agents, num_agents)
            connectivity = nn.Parameter(init_adj)

            # Freeze part of the zero entries
            mask = (A == 0).float()  # Where connections are missing
            flat_mask = mask.view(-1)
            candidates = (flat_mask == 1).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(candidates))
            num_learnable = len(candidates) // 2
            learnable_idx = candidates[perm[:num_learnable]]
            final_mask = torch.ones_like(flat_mask)
            final_mask[flat_mask == 1] = 0.0  # freeze all zero-edges
            final_mask[learnable_idx] = 1.0   # unfreeze selected
            grad_mask = final_mask.view(num_agents, num_agents)
            
            # Register_buffer saves the mask on the model (not trainable)
            self.adj_grad_mask: torch.Tensor
            self.register_buffer('adj_grad_mask', grad_mask)
            def gradient_mask_hook(grad):
                return grad * self.adj_grad_mask
            # Register_hook applies element-wise masking during backprop
            connectivity.register_hook(gradient_mask_hook)
            # Note: Even if the gradient is zeroed, the value might change due to weight decay or momentum.
            # to avoid that, I define a freeze method below to call after each optimizer step.
            self.connectivity = connectivity
            return connectivity
        else:
            raise ValueError(f"Invalid adjacency_mode: {mode}")
        
    def _message_passing(self, h: torch.Tensor) -> torch.Tensor:
        for _ in range(self.message_steps):
            head_outputs = [head(h, self.connectivity.unsqueeze(0)) for head in self.gat_heads]
            stacked = torch.stack(head_outputs)  # [num_heads, B, A, H]
            
            # Max-pool across heads based on absolute values
            abs_vals = torch.abs(stacked)
            max_indices = abs_vals.argmax(dim=0)  # [B, A, H]

            # Gather max across heads
            stacked = stacked.permute(1, 2, 3, 0).contiguous()  # [B, A, H, num_heads]
            h = torch.gather(stacked, dim=-1, index=max_indices.unsqueeze(-1)).squeeze(-1) # [B, A, H]
        return h
    
    def forward(self, x):
        # x: [batch, num_agents, input_dim]
        h = self.agent_input_proj(x)  # [B, A, H]

        # Do attention-based message passing if message_steps > 0; otherwise,
        # network reduces to a simple encoder - decoder
        if self.message_steps > 0:
            h = self._message_passing(h)
        
        B, A, H = h.shape
        # Compute U and V from shared projections
        U = self.U_proj(h).view(B, A, self.n, self.maxrank)  # [B, A, n, maxrank]
        V = self.V_proj(h).view(B, A, self.m, self.maxrank)  # [B, A, m, maxrank]

        # Compute H = U @ Vt for each agent
        recon = torch.matmul(U, V.transpose(-1, -2)) / self.maxrank # [B, A, n, m]
        recon = recon.view(B, A, self.n * self.m)  # vectorize: [B, A, n * m]
        return recon  # [batch, num_agents, output_dim]
        
    @torch.no_grad()
    def freeze_nonlearnable(self):
        if self.adjacency_mode == 'learned':
            if not hasattr(self, 'adj_grad_mask'):
                raise RuntimeError("adj_grad_mask missing.")
            self.connectivity.data = self.connectivity.data * self.adj_grad_mask


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

