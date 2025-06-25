"""
Dot product graph attention network for distributed inductive matrix completion, 
with learned agent-based message passing setup.
Supports both sparse views and low-dimensional projections as agent inputs.

Each agent independently receives their own masked view of the matrix via 
AgentMatrixReconstructionDataset (x: [batch, num_agents, input_dim]). 

During message passing, each DotGATLayer uses dot-product attention with a learned 
connectivity matrix, ensuring localized neighbor communication.
Aggregation only occurs after message passing and is mediated via the GAT network to prevent
information leakage.

self.connectivity is a trainable nn.Parameter representing the agent-to-agent communication graph.
The attention computation adds this weighted adjacency matrix to the attention logits.

Explicit message passing rounds are controlled by message_steps in DistributedDotGAT, with message 
updates applied iteratively through GAT heads.

Design choices:
- *Positional embeddings*: Learned Fourier-features positional embeddings of the sparse entries. Many
alternatives are possible, including integrating structural rather than only coordinate information.
- *Adjacency matrix is not input-dependent*: 
Currently self.connectivity is shared across all heads and batches, and static per model instance.
    An alternative would be to allow connectivity vary by batch or to depend on inputs/agent embeddings.
- *Sparsity of adjacency matrix*: 
The connectivity matrix is initialized as a sparse adjacency matrix of a small world graph, and used
to enforce sparsity throughout training via gradient freezing (structural barriers in connectivity). 
Initial edges (nonzero in A) are always learnable. Half of the 0 entries are allowed to grow via learning. 
The rest are frozen at zero through gradient masking.
    - This could be taken out or modified. 
    - An alternative (which is in a previous version in Github) is top-k sparsification (listening to only some 
    neighbors), which is also static.
    - Other alternatives would be dynamic thresholding or learned gating. ChatGPT suggests thresholded softmax 
    (Sparsemax or Entmax), attention dropout, learned attention masks, or entropic sparsity.
- Agents only weakly specialize through agent-specific embeddings.
    Alternatives include: agent-specific MLPs or gating mechanisms before/after message passing;  
    additional diversity or specialization losses.
- No residual connections.
    Alternative: `h = h + torch.stack(head_outputs).mean(dim=0)` or `h = self.norm(h + ...)`
- Collective aggregation occurs through learned, gated (input-dependent) pooling across agents.
    Alternatives include:
    attention-based aggregation with a learned attention weight for each agent, 
    game-theoretic aggregation of subsets of agents based on utility or diversity contribution

Required improvements:
- Save and plot connectivity matrix over time
- Spectral penalty is unstable. ChatGPT suggests replacing with nuclear norm clipping, 
differentiable low-rank approximations, or penalizing condition number.
- Log agent_diversity_penalty during training to track changes in agent roles
- Per-agent MSE should be correlated with agent input quality/sparsity. Integrate tracking input sparsity 
per agent along with prediction quality.
- Develop view_mode='project' and make projections learnable (e.g., per-agent linear layers).
"""

import argparse
import gc
import math
import os
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from plot_utils import plot_connectivity_matrices, plot_stats


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def unique_filename(base_dir="results", prefix="run"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{prefix}_{timestamp}")


class DotGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.v_proj = nn.Linear(in_features, out_features, bias=False)
        #self.forward_proj = nn.Linear(out_features, out_features)
        self.Swish = Swish()
        self.forward_proj = nn.Sequential(
            Swish(), nn.LayerNorm(out_features), 
            nn.Linear(out_features, out_features),
            Swish(), nn.LayerNorm(out_features), 
            nn.Linear(out_features, out_features),
        )
        self.norm = nn.LayerNorm(out_features)
        self.dropout = dropout
        self.scale = math.sqrt(out_features)
    
    def forward(self, x, connectivity):
        # x: [batch_size, num_agents, hidden_dim]
        B, A, H = x.shape

        x = x.view(B * A, H)  # Flatten batch and agents
        # Compute query, key, and value matrices
        Q = self.q_proj(x).view(B, A, -1)
        K = self.k_proj(x).view(B, A, -1)
        V = self.v_proj(x).view(B, A, -1)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # [B, A, A]
        scores = scores #+ connectivity  # Add learnable connectivity bias

        # Full attention (no top-k)
        alpha = F.softmax(scores, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        H = torch.matmul(alpha, V)  # [B, A, hidden_dim]
        out = self.forward_proj(H)
        return self.norm(out)


class DistributedDotGAT(nn.Module):
    def __init__(self, 
                 input_dim, # also n x m if vectorized
                 hidden_dim, # internal dim, e.g. 128
                 n, m,
                 num_agents, num_heads, dropout, message_steps=3):
        super().__init__()
        self.output_dim = n * m
        self.n = n
        self.m = m
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps
        self.dropout = dropout
        #v2:self.agent_input_projs = nn.ModuleList([
        #    nn.Linear(input_dim, hidden_dim, bias=False) for _ in range(num_agents)
        #])
        self.agent_input_proj = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim, bias=False),
            Swish(), nn.LayerNorm(2 * hidden_dim), 
            nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=False),
            Swish(), nn.LayerNorm(2 * hidden_dim), 
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
        )
        self.connectivity = torch.ones(num_agents, num_agents)
        # Generate small-world graph
        """         G = nx.watts_strogatz_graph(n=num_agents, k=10, p=0.3)
        A = torch.zeros(num_agents, num_agents)
        for i, j in G.edges():
            A[i, j] = 1.0
            A[j, i] = 1.0  # Ensure symmetry

        # Learnable parameter initialized with small noise
        init_adj = A + 0.01 * torch.randn(num_agents, num_agents)
        self.connectivity = nn.Parameter(init_adj)

        # Build a mask: 1 for learnable, 0 for frozen
        mask = (A == 0).float()  # Where connections are missing
        mask_flat = mask.view(-1)
        num_candidates = (mask_flat == 1).nonzero(as_tuple=True)[0]
        perm = torch.randperm(num_candidates.shape[0])
        num_learnable = perm.shape[0] // 2
        learnable_idx = num_candidates[perm[:num_learnable]]
        final_mask = torch.ones_like(mask_flat)
        final_mask[mask_flat == 1] = 0.0  # freeze all zero-edges
        final_mask[learnable_idx] = 1.0   # unfreeze selected

        self.register_buffer('adj_grad_mask', final_mask.view(num_agents, num_agents))

        # Hook to zero gradients for frozen entries
        def gradient_mask_hook(grad):
            return grad * self.adj_grad_mask

        self.connectivity.register_hook(gradient_mask_hook) """
        
        self.gat_heads = nn.ModuleList([
            DotGATLayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_heads)
        ])

        self.swish = Swish()
        self.norm = nn.LayerNorm(self.n * self.m)
        self.maxrank = min(self.n // 2, self.m // 2)
        self.U_proj = nn.Linear(hidden_dim, self.n * self.maxrank, bias=False)
        self.V_proj = nn.Linear(hidden_dim, self.n * self.maxrank, bias=False)

    def forward(self, x):
        # x: [batch, num_agents, input_dim]
        h = self.agent_input_proj(x)  # [B, A, hidden_dim]

        for _ in range(self.message_steps):
            #head_outputs = [head(h, self.connectivity.unsqueeze(0)) for head in self.gat_heads]
            #h = torch.stack(head_outputs).mean(dim=0)
            #h = self.swish(F.dropout(h, p=self.dropout, training=self.training))
            head_outputs = [head(h, self.connectivity.unsqueeze(0)) for head in self.gat_heads]
            # Max-pool across heads based on absolute values
            stacked = torch.stack(head_outputs)  # [num_heads, B, A, hidden_dim]
            abs_vals = torch.abs(stacked)
            max_indices = torch.argmax(abs_vals, dim=0)  # [B, A, hidden_dim]
            # Convert to [B, A, hidden_dim, num_heads] for indexing
            stacked = stacked.permute(1, 2, 3, 0).contiguous()  # [B, A, H, num_heads]
            max_indices = max_indices.unsqueeze(-1)  # [B, A, H, 1]
            h = torch.gather(stacked, dim=-1, index=max_indices).squeeze(-1)  # [B, A, H]
        #out = self.output_proj(h)
        B, A, H = h.shape
        # Compute U and V from shared projections
        U = self.U_proj(h).view(B, A, self.n, self.maxrank)  # [B, A, n, mr]
        V = self.V_proj(h).view(B, A, self.m, self.maxrank)  # [B, A, m, mr]

        # Compute H = U @ Vᵀ for each agent
        H = torch.matmul(U, V.transpose(-1, -2)) / self.maxrank # [B, A, n, m]
        H = H.view(B, A, self.n * self.m)  # vectorize: [B, A, n * m]

        # Aggregate across agents: H.mean(dim=1)  # [B, n*m]
        return H  # [batch, num_agents, output_dim]


class Aggregator(nn.Module):
    """
    Input-dependent aggregation module. It reads each agent's output,
    uses a shared MLP to compute per-agent gates based on their output,
    applies a softmax over agent gates, and then performs weighted sum.
    
    We hope this is flexible enough to let the model learn how to downweight
    non-informative agent outputs (e.g. near-zero).
    """
    def __init__(self, num_agents, output_dim, hidden_dim=4):
        super().__init__()
        self.num_agents = num_agents
        self.output_dim = output_dim
        if num_agents != 1:
            self.gate_mlp = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, agent_outputs):
        """
        agent_outputs: [B, A, nm] 
        returns: [B, nm] aggregated output
        """
        B, A, nm = agent_outputs.shape
        assert self.num_agents == A and self.output_dim == nm
        if self.num_agents == 1:
            return agent_outputs.squeeze(1)
        else:
            # Compute per-agent gates based on their output
            gates = self.gate_mlp(agent_outputs)  # [B, A, 1]
            weights = F.softmax(gates, dim=1)     # [B, A, 1]

            weighted_output = agent_outputs * weights  # [B, A, D]
            return weighted_output.sum(dim=1)     # [B, D]


class AgentMatrixReconstructionDataset(InMemoryDataset):
    def __init__(self, num_graphs=1000, n=20, m=20, r=4, 
                 num_agents=30, density=0.2, sigma=0, verbose=True):
        self.num_graphs = num_graphs
        self.n = n
        self.m = m
        self.r = r
        self.density = density
        self.sigma = sigma
        self.num_agents = num_agents
        self.verbose = verbose
        self.input_dim = n * m
        self.nuclear_norm_mean = 0.0
        self.gap_mean = 0.0
        self.variance_mean = 0.0
        self.agent_overlap_mean = 0.0
        self.agent_endowment_mean = 0.0
        self.actual_known_mean = 0.0
        super().__init__('.')
        self.data, self.slices = self._generate()
    
    def _generate(self):
        data_list = []
        norms = []
        gaps = []
        variances = []
        agent_overlaps = []
        agent_endowments = []
        actual_knowns = []
        total_entries = self.n * self.m
        target_known = int(self.density * total_entries)
        for _ in range(self.num_graphs):
            # Low-rank matrix with Gaussian noise
            U = np.random.randn(self.n, self.r) / np.sqrt(self.r)
            V = np.random.randn(self.m, self.r) / np.sqrt(self.r)
            M = U @ V.T + self.sigma * np.random.randn(self.n, self.m)
            M_tensor = torch.tensor(M, dtype=torch.float32).view(-1)
            norms.append(torch.linalg.norm(M_tensor.view(self.n, self.m), ord='nuc').item())
            S = torch.linalg.svdvals(M_tensor.view(self.n, self.m))
            gaps.append(S[self.r - 1] - S[self.r])
            variances.append(torch.var(M_tensor).item())
            # Controlled global known vs secret mask
            all_indices = torch.randperm(total_entries)
            known_global_idx = all_indices[:target_known]
            global_mask = torch.zeros(total_entries, dtype=torch.bool)
            global_mask[known_global_idx] = True
            # Build observed tensor with zeros at unknowns
            observed_tensor = M_tensor.clone()
            observed_tensor[~global_mask] = 0.0
            # Agent-specific views
            features = []
            for i in range(self.num_agents):
                # Each agent samples (with replacement) a variable-sized subset 
                # of the global known entries
                # control expected agent overlap by tweaking agent_sample_size range
                agent_sample_size = np.random.randint(
                    int(2.0 * (target_known // self.num_agents)),
                    int(4.0 * (target_known // self.num_agents)) 
                )
                if agent_sample_size > target_known:
                    agent_sample_size = target_known
                    print(f"Warning: agent {i} sampled all known entries.")
                agent_endowments.append(agent_sample_size)
                sample_idx = known_global_idx[
                    torch.randint(len(known_global_idx), (agent_sample_size,))
                ]
                agent_view = torch.zeros(total_entries, dtype=torch.float32)
                agent_view[sample_idx] = observed_tensor[sample_idx]
                features.append(agent_view)
            x = torch.stack(features)
            # Create mask tensor from agent views reflecting entries actually seen by any agent
            mask_tensor = (x != 0).sum(dim=0) > 0
            
            overlap_matrix = (torch.stack(features) > 0).float() @ (torch.stack(features) > 0).float().T
            overlap_matrix /= overlap_matrix.diagonal().view(-1, 1)  # normalize
            avg_overlap = (overlap_matrix.sum() - overlap_matrix.trace()) / (self.num_agents * (self.num_agents - 1))
            agent_overlaps.append(avg_overlap)
            # Count how many entries known by any agent, avoid double counting
            actual_known = mask_tensor.sum().item()
            actual_knowns.append(actual_known)
            # Create Data object
            data = Data(x=x, y=M_tensor, mask=mask_tensor, nb_known_entries=actual_known)
            data_list.append(data)
        self.nuclear_norm_mean = np.mean(norms)
        self.gap_mean = np.mean(gaps)
        self.variance_mean = np.mean(variances)
        self.agent_overlap_mean = np.mean(agent_overlaps)
        self.agent_endowment_mean = np.mean(agent_endowments)
        self.actual_known_mean = np.mean(actual_knowns)
        output = self.collate(data_list)
        if self.verbose:
            print(f"Generated {self.num_graphs} rank-{self.r} matrices of size {self.n}x{self.m} " +
                f"with mean nuclear norm {self.nuclear_norm_mean:.4f} and mean gap {self.gap_mean:.4f}.")
            print(f"Global observed density {self.density:.4f} and noise level {self.sigma:.4f}.")
            print(f"Total entries {total_entries}; target known entries {target_known}; mean actual known entries {self.actual_known_mean}.")
            print(f"Entries distributed among {self.num_agents} agents with mean overlap {self.agent_overlap_mean:.4f}.")
            print(f"Average number of entries per agent: {self.agent_endowment_mean:.4f}.")
        return output


def spectral_penalty(output, rank, evalmode=False):
    """
    Compute spectral penalty for a low rank matrix output.
    
    The spectral penalty is defined as the sum of the singular values
    of the output matrix, excluding the r largest singular values.
    
    Additionally, if the largest singular value is greater than 2
    times the smaller dimension of the output matrix, or if it is
    less than half the smaller dimension, add a penalty term of
    (s0 - 2 * N) ** 2 or (N / 2 - s0) ** 2, respectively.
    
    Return the spectral penalty, the largest singular value, and the
    gap between the rth and the next largest singular values.
    """
    try:
        U, S, Vt = torch.linalg.svd(output, full_matrices=False)
    except RuntimeError:
        S = torch.linalg.svdvals(output)
    nuc = S.sum()
    sum_rest = S[rank:].sum()
    s_last = S[rank - 1].item()
    s_next = S[rank].item()
    svdcount = len(S) # or min(output.shape)
    if evalmode:
        gap = (s_last - s_next)
        return gap
    else:
        #ratio = sum_rest / (s_last + 1e-6)
        penalty = sum_rest/(svdcount-rank) #+ ratio
        #penalty = 0
        if nuc > (2 * svdcount):
            penalty += nuc/(2 * svdcount) - 1
        elif nuc < (svdcount // 2):
            penalty -= (nuc/svdcount) + 1
        return penalty

def train(model, aggregator, loader, optimizer, theta, criterion, n, m, rank, 
          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        batch_size = batch.num_graphs
        x = batch.x 
        x = x.view(batch_size, model.num_agents, -1)
        out = model(x) 
        prediction = aggregator(out)
        target = batch.y.view(batch_size, -1)
        reconstructionloss = criterion(prediction, target)
        penalty = sum(
            spectral_penalty(prediction[i].view(n,m), rank) for i in range(batch_size)
        )
        loss = theta * reconstructionloss + (1 - theta) * penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def train_additional_stats(
    model, aggregator, loader, criterion, n, m, rank, 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    model.eval()
    t_known_mse, t_unknown_mse, t_nuclear_norms, t_variances, t_gaps = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.num_graphs
        # Forward pass
        x = batch.x.view(batch_size, model.num_agents, -1)
        out = model(x)
        prediction = aggregator(out)  # [batch_size, n*m]
        target = batch.y.view(batch_size, -1)  # [batch_size, n*m]
        mask = batch.mask.view(batch_size, -1)  # [batch_size, n*m]
        for i in range(batch_size):
            pred_i = prediction[i]  # [n*m]
            target_i = target[i]    # [n*m]
            mask_i = mask[i]        # [n*m]
            known_i = mask_i
            unknown_i = ~mask_i
            if known_i.sum() > 0:
                t_known_mse.append(criterion(pred_i[known_i], target_i[known_i]).item())
            if unknown_i.sum() > 0:
                t_unknown_mse.append(criterion(pred_i[unknown_i], target_i[unknown_i]).item())
            assert known_i.sum() + unknown_i.sum() == n * m
            # Reshape for nuclear norm
            matrix_2d = pred_i.view(n, m)
            t_nuclear_norms.append(torch.linalg.norm(matrix_2d, ord='nuc').item())
            # Spectral penalty metrics
            gap = spectral_penalty(matrix_2d, rank, evalmode=True) 
            t_gaps.append(gap)
            # Variance of the agent outputs
            t_variances.append(matrix_2d.var().item())
    return (
        np.mean(t_known_mse) if t_known_mse else float('nan'),
        np.mean(t_unknown_mse) if t_unknown_mse else float('nan'),
        np.mean(t_nuclear_norms),
        np.mean(t_variances),
        np.mean(t_gaps),
    )

@torch.no_grad()
def evaluate(model, aggregator, loader, criterion, n, m, rank, 
             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    known_mse, unknown_mse, nuclear_norms, variances, gaps = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.num_graphs
        # Forward pass
        x = batch.x.view(batch_size, model.num_agents, -1)
        out = model(x)
        prediction = aggregator(out)  # [batch_size, n*m]
        target = batch.y.view(batch_size, -1)  # [batch_size, n*m]
        mask = batch.mask.view(batch_size, -1)  # [batch_size, n*m]
        for i in range(batch_size):
            pred_i = prediction[i]  # [n*m]
            target_i = target[i]    # [n*m]
            mask_i = mask[i]        # [n*m]
            known_i = mask_i
            unknown_i = ~mask_i
            if known_i.sum() > 0:
                known_mse.append(criterion(pred_i[known_i], target_i[known_i]).item())
            if unknown_i.sum() > 0:
                unknown_mse.append(criterion(pred_i[unknown_i], target_i[unknown_i]).item())
            assert known_i.sum() + unknown_i.sum() == n * m
            # Reshape for nuclear norm
            matrix_2d = pred_i.view(n, m)
            nuclear_norms.append(torch.linalg.norm(matrix_2d, ord='nuc').item())
            # Spectral penalty metrics
            gap = spectral_penalty(matrix_2d, rank, evalmode=True) 
            gaps.append(gap)
            # Variance of the agent outputs
            variances.append(matrix_2d.var().item())
    return (
        np.mean(known_mse) if known_mse else float('nan'),
        np.mean(unknown_mse) if unknown_mse else float('nan'),
        np.mean(nuclear_norms),
        np.mean(variances),
        np.mean(gaps),
    )


@torch.no_grad()
def evaluate_agent_contributions(model, loader, criterion, n, m, 
                                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    print("\nEvaluating individual agent contributions...")
    all_agent_errors = []
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.num_graphs
        num_agents = batch.x.shape[0] // batch_size
        nm = n * m
        x = batch.x.view(batch_size, model.num_agents, -1)
        out = model(x)
        targets = batch.y.view(batch_size, nm)      # [B, n*m]
        for i in range(batch_size):
            individual_errors = []
            for a in range(num_agents):
                agent_pred = out[i, a]     # [n*m]
                target = targets[i]        # [n*m]
                error = criterion(agent_pred, target).item()
                individual_errors.append((a, error))
            all_agent_errors.append(individual_errors)
    # Aggregate and report
    flat_errors = [err for sample in all_agent_errors for _, err in sample]
    agent_errors = torch.tensor(flat_errors)
    agent_mean = agent_errors.mean().item()
    agent_std = agent_errors.std().item()
    agent_min = agent_errors.min().item()
    agent_max = agent_errors.max().item()
    print(f"Mean agent MSE: {agent_mean:.4f}, Std: {agent_std:.4f}")
    print(f"Min agent MSE: {agent_min:.4f}, Max: {agent_max:.4f}")


def init_weights(m):
    if isinstance(m, nn.Linear):
        #nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
            
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    print("Breakdown by layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:60} {param.numel():>10}")
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=30)
    parser.add_argument('--m', type=int, default=30)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--density', type=float, default=0.3)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--num_agents', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--theta', type=float, default=0.95, help='Weight for the known entry loss vs penalty')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--eval_agents', action='store_true', help='Always evaluate agent contributions')
    parser.add_argument('--steps', type=int, default=5, help='Number of message passing steps')
    parser.add_argument('--train_n', type=int, default=1000, help='Number of training matrices')
    parser.add_argument('--val_n', type=int, default=64, help='Number of validation matrices')
    parser.add_argument('--test_n', type=int, default=64, help='Number of test matrices')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = AgentMatrixReconstructionDataset(
        num_graphs=args.train_n, n=args.n, m=args.m, r=args.r, num_agents=args.num_agents,
        density=args.density, sigma=args.sigma
    )
    val_set = AgentMatrixReconstructionDataset(
        num_graphs=args.val_n, n=args.n, m=args.m, r=args.r, num_agents=args.num_agents,
        density=args.density, sigma=args.sigma, verbose=False
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = DistributedDotGAT(
        input_dim=train_set.input_dim,
        hidden_dim=args.hidden_dim,
        n=args.n, m=args.m,
        num_agents=args.num_agents,
        num_heads=args.num_heads,
        dropout=args.dropout,
        message_steps=args.steps
    ).to(device)
    model.apply(init_weights)
    count_parameters(model)
    
    aggregator = Aggregator(
        num_agents=args.num_agents, output_dim=args.n * args.m
    ).to(device)
    aggregator.apply(init_weights)
    count_parameters(aggregator)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(aggregator.parameters()), lr=args.lr
    )
    criterion = nn.MSELoss()
    
    stats = {
        "train_loss": [],
        "t_known_mse": [],
        "t_unknown_mse": [],
        "t_nuclear_norm": [],
        "t_variance": [],
        "t_spectral_gap": [],
        "val_known_mse": [],
        "val_unknown_mse": [],
        "val_nuclear_norm": [],
        "val_variance": [],
        "val_spectral_gap": []
    }
    file_base = unique_filename()
    checkpoint_path = f"{file_base}_checkpoint.pt"
    
    best_loss = float('inf')
    patience_counter = 0
    
    with torch.no_grad():
        batch = next(iter(train_loader)).to(device)
        x = batch.x.view(batch.num_graphs, model.num_agents, -1)
        out = model(x)
        recon = aggregator(out)
        print("Initial output variance:", recon.var().item())

    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model, aggregator, train_loader, optimizer, args.theta, criterion, args.n, args.m, args.r
        )
        t_known, t_unknown, t_nuc, t_var, t_gap = train_additional_stats(
            model, aggregator, train_loader, criterion, args.n, args.m, args.r
        )
        val_known, val_unknown, nuc, var, gap = evaluate(
            model, aggregator, val_loader, criterion, args.n, args.m, args.r
        )
        
        stats["train_loss"].append(train_loss)
        stats["t_known_mse"].append(t_known)
        stats["t_unknown_mse"].append(t_unknown)
        stats["t_nuclear_norm"].append(t_nuc)
        stats["t_variance"].append(t_var)
        stats["t_spectral_gap"].append(t_gap)
        stats["val_known_mse"].append(val_known)
        stats["val_unknown_mse"].append(val_unknown)
        stats["val_nuclear_norm"].append(nuc)
        stats["val_variance"].append(var)
        stats["val_spectral_gap"].append(gap)
        
        print(f"Ep {epoch:03d}. L: {train_loss:.4f} | Kn: {t_known:.4f} | "+
            f"Unkn: {t_unknown:.4f} | NN: {t_nuc:.2f} | Gap: {t_gap:.2f} | Var: {t_var:.4f}")
        
        print(f"--------------------------------------VAL--: K/U: {val_known:.2f} / "+
            f"{val_unknown:.2f}. NN/G: {nuc:.1f} / {gap:.1f}. Var: {var:.2f}.")
        
        # Save connectivity matrix for visualization
        #adj_matrix = model.connectivity.detach().cpu().numpy()
        #np.save(f"{file_base}_adj_epoch{epoch}.npy", adj_matrix)
        
        if t_unknown < best_loss - 1e-5:
            best_loss = t_unknown
            torch.save(model.state_dict(), checkpoint_path)
            
        if t_unknown < 0.001:
            print(f"Early stopping at epoch {epoch}.")
            break
        
        #if val_unknown < best_loss - 1e-5:
        #    best_loss = val_unknown
        #    patience_counter = 0
        #    torch.save(model.state_dict(), checkpoint_path)
        #    
        #else:
        #    patience_counter += 1
        #    if patience_counter >= args.patience:
        #        print(f"Early stopping at epoch {epoch}.")
        #        break

    # Clear memory (avoid OOM) and load best model
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    print("Loaded best model from checkpoint.")

    plot_stats(stats, file_base, 
               train_set.nuclear_norm_mean, train_set.gap_mean, train_set.variance_mean)
    
    # Final test evaluation on fresh data
    test_dataset = AgentMatrixReconstructionDataset(
        num_graphs=args.test_n, n=args.n, m=args.m, r=args.r, num_agents=args.num_agents, 
        density=args.density, sigma=args.sigma, verbose=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    test_known, test_unknown, test_nuc, test_var, test_gap = evaluate(
        model, aggregator, test_loader, criterion, args.n, args.m, args.r)
    print(f"Test Set Performance | Known MSE: {test_known:.4f}, Unknown MSE: {test_unknown:.4f}"+
        f" Nuclear Norm: {test_nuc:.2f}, Spectral Gap: {test_gap:.2f}, Variance: {test_var:.4f}")

    #file_prefix = Path(file_base).name  # Extracts just 'run_YYYYMMDD_HHMMSS'
    #plot_connectivity_matrices("results", prefix=file_prefix, cmap="coolwarm")

    # Agent contribution eval (optional)
    if test_unknown < 0.1 or args.eval_agents:
        evaluate_agent_contributions(model, test_loader, criterion, args.n, args.m)
    