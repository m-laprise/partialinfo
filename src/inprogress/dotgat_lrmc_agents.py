"""
Dot product graph attention network for distributed inductive matrix completion, 
with learned agent-based message passing setup.
Supports both sparse views and low-dimensional projections as agent inputs.

(Currently, view-mode 'sparse' runs, but view-mode 'project' needs to be debugged.)

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
- Collective aggregation occurs through mean pooling across agents.
    Alternatives include:
    attention-based aggregation with a learned attention weight for each agent, 
    aggregate using learned gating, 
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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
from plot_utils import plot_connectivity_matrices
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def unique_filename(base_dir="results", prefix="run"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{prefix}_{timestamp}")


def plot_stats(stats, filename_base, true_nuclear_mean):
    epochs = np.arange(1, len(stats["train_loss"]) + 1)
    # Plot loss-related metrics in two panels
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    axs[0].plot(epochs, stats["train_loss"], label="Train Loss", color='tab:blue')
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].plot(epochs, stats["val_known_mse"], 
                label="Val MSE Known Entries", color='tab:green')
    axs[1].plot(epochs, stats["val_unknown_mse"], 
                label="Val MSE Unknown Entries", color='tab:orange')
    axs[1].plot(epochs, stats["variance"], 
                label="Variance of Reconstructed Entries", color='tab:red')
    axs[1].set_title("Validation Loss & Variance")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Metric")
    axs[1].grid(True)
    axs[1].legend()
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(f"{filename_base}_loss_metrics.png")
    plt.close(fig)
    # Plot spectral diagnostics
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=120)
    ax.plot(epochs, stats["nuclear_norm"], label="Nuclear Norm", color='tab:purple')
    ax.plot(epochs, stats["spectral_gap"], label="Spectral Gap", color='tab:orange')
    ax.axhline(y=true_nuclear_mean, color='gray', linestyle='--', 
               label="Ground Truth Mean Nuclear Norm")
    ax.set_title("Spectral Properties Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Singular Value Scale")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(f"{filename_base}_spectral_diagnostics.png")
    plt.close(fig)


class FourierPositionalEncoder(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.B = nn.Parameter(torch.randn(num_frequencies, 2))

    def forward(self, coords):
        proj = 2 * math.pi * coords @ self.B.T  # (N, F)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2F)


class EntryEncoder(nn.Module):
    def __init__(self, fourier_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * fourier_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, values, pos_features):
        x = torch.cat([values, pos_features], dim=-1)
        return self.mlp(x)


class DotGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.v_proj = nn.Linear(in_features, out_features, bias=False)
        #self.forward_proj = nn.Linear(out_features, out_features)
        self.forward_proj = nn.Sequential(
            Swish(),
            nn.Linear(out_features, out_features),
            Swish(),
            nn.Linear(out_features, out_features),
        )
        self.Swish = Swish()
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
        scores = scores + connectivity  # Add learnable connectivity bias

        # Full attention (no top-k)
        alpha = F.softmax(scores, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        H = torch.matmul(alpha, V)  # [B, A, hidden_dim]
        out = self.forward_proj(H)
        return self.norm(out)


class MatrixDecoder(nn.Module):
    def __init__(self, hidden_dim, num_frequencies=16):
        super().__init__()
        self.fourier_dim = num_frequencies
        self.pos_encoder = FourierPositionalEncoder(num_frequencies)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 2 * num_frequencies, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, coords: torch.Tensor, agent_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, T, 2] - target positions (i,j)
            agent_embedding: [B, D] - agent-level embedding

        Returns:
            predictions: [B, T] - predicted values at positions
        """
        B, T, _ = coords.shape
        pos_feats = self.pos_encoder(coords.view(-1, 2)).view(B, T, -1)  # [B, T, 2F]

        agent_rep = agent_embedding.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        joint = torch.cat([agent_rep, pos_feats], dim=-1)  # [B, T, D + 2F]

        out = self.decoder(joint)  # [B, T, 1]
        return out.squeeze(-1)  # [B, T]
    

class DistributedDotGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_agents, 
                 num_heads, dropout, message_steps=3):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps
        #self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = dropout
        #v1:self.agent_embeddings = nn.Parameter(torch.randn(num_agents, hidden_dim))
        #v2:self.agent_input_projs = nn.ModuleList([
        #    nn.Linear(input_dim, hidden_dim, bias=False) for _ in range(num_agents)
        #])
        pos_emb_dim = 16
        self.pos_encoder = FourierPositionalEncoder(num_frequencies=pos_emb_dim)
        self.entry_encoder = EntryEncoder(
            fourier_dim=pos_emb_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        self.max_entries = round((output_dim * 0.5) / (num_agents / 2))
        self.pad_token = nn.Parameter(torch.zeros(hidden_dim))  # learnable pad
        self.entry_embed_compression = nn.Sequential(
            nn.Linear(self.max_entries * hidden_dim, 2 * hidden_dim),
            Swish(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        # Generate small-world graph
        G = nx.watts_strogatz_graph(n=num_agents, k=10, p=0.3)
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

        self.connectivity.register_hook(gradient_mask_hook)
        
        self.gat_heads = nn.ModuleList([
            DotGATLayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_heads)
        ])

        self.swish = Swish()
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        #self.output_proj = nn.Sequential(
        #    nn.Linear(hidden_dim, hidden_dim),
        #    Swish(),
        #    nn.Linear(hidden_dim, output_dim),
        #)

    def forward(self, x):
        # x: [batch, num_agents, input_dim]
        #v1:h = self.input_proj(x) + self.agent_embeddings.unsqueeze(0)  # Inject identity
        #v2:h = torch.stack([
        #    self.agent_input_projs[i](x[:, i, :])  # [batch, hidden_dim] for agent i
        #    for i in range(self.num_agents)
        #], dim=1)  # -> [batch, num_agents, hidden_dim]
        B, A, D = x.shape  # [batch, num_agents, n*m]
        device = x.device
        states = []
        # For each agent’s input vector x \in \mathbb{R}^{n \times m}:
        for agent_id in range(self.num_agents):
            # 1. Extract non-zero values and indices
            agent_inputs = x[:, agent_id, :]  # [B, n*m]
            nonzero_mask = agent_inputs != 0  # [B, n*m]
            coords = nonzero_mask.nonzero(as_tuple=False)
            batch_ids = coords[:, 0]
            flat_indices = coords[:, 1]
            values = agent_inputs[batch_ids, flat_indices].unsqueeze(-1)
            
            row = flat_indices // int(math.sqrt(D))
            col = flat_indices % int(math.sqrt(D))
            ij_coords = torch.stack([row, col], dim=1).float().to(device)
            
            # 2. Apply positional encoding and value embedding
            # & 3. Combine position and value into token embeddings: e_i = f(value_i, position_i)
            pos_feats = self.pos_encoder(ij_coords)
            entry_embeds = self.entry_encoder(values, pos_feats)
            
            # 4. Aggregate into a fixed-size state vector per agent
            # Group per batch
            grouped = [[] for _ in range(B)]
            for idx, b in enumerate(batch_ids):
                grouped[b.item()].append(entry_embeds[idx])
                
            # Truncate or pad each to max_entries
            agent_embed = []
            for entry_list in grouped:
                if len(entry_list) > self.max_entries:
                    entry_list = entry_list[:self.max_entries]
                while len(entry_list) < self.max_entries:
                    entry_list.append(self.pad_token)
                agent_embed.append(torch.stack(entry_list))  # shape: [max_entries, hidden_dim]
            
            agent_embed = torch.stack(agent_embed)  # [B, max_entries, hidden_dim]
            # Flatten or pool
            agent_embed = agent_embed.view(B, -1)  # [B, max_entries * hidden_dim]
            agent_embed = self.entry_embed_compression(agent_embed)  # [B, hidden_dim]
            #counts = nonzero_mask.sum(dim=1)  # shape: [B], type: torch.Tensor
            #batch_idx = torch.arange(B, device=device).repeat_interleave(counts)
            #agent_embed = torch.zeros(B, self.hidden_dim, device=device).index_add_(
            #    0, batch_idx, entry_embeds
            #)
            #norm_factors = nonzero_mask.sum(dim=1).clamp(min=1).unsqueeze(1)
            #agent_embed /= norm_factors  # [B, hidden_dim]

            states.append(agent_embed)

        h = torch.stack(states, dim=1)  # [B, A, hidden_dim]

        for _ in range(self.message_steps):
            head_outputs = [head(h, self.connectivity.unsqueeze(0)) for head in self.gat_heads]
            h = torch.stack(head_outputs).mean(dim=0)
            #h = self.swish(F.dropout(h, p=self.dropout, training=self.training))

        out = self.output_proj(h)
        return out  # [batch, num_agents, output_dim]


class AgentMatrixReconstructionDataset(InMemoryDataset):
    def __init__(self, num_graphs=1000, n=20, m=20, r=4, 
                 num_agents=30, view_mode='sparse', density=0.2, sigma=0.01):
        self.num_graphs = num_graphs
        self.n = n
        self.m = m
        self.r = r
        self.density = density
        self.sigma = sigma
        self.num_agents = num_agents
        self.view_mode = view_mode
        self.input_dim = n * m if view_mode == 'sparse' else min(n * m, 128)
        self.nuclear_norm_mean = 0.0
        super().__init__('.')
        self.data, self.slices = self._generate()
    
    def _generate(self):
        data_list = []
        norms = []
        total_entries = self.n * self.m
        target_known = int(self.density * total_entries)
        for _ in range(self.num_graphs):
            # Low-rank matrix with Gaussian noise
            U = np.random.randn(self.n, self.r) / np.sqrt(self.r)
            V = np.random.randn(self.m, self.r) / np.sqrt(self.r)
            M = U @ V.T + self.sigma * np.random.randn(self.n, self.m)
            M_tensor = torch.tensor(M, dtype=torch.float32).view(-1)
            norms.append(torch.linalg.norm(M_tensor.view(self.n, self.m), ord='nuc').item())
            # Controlled global known vs secret mask
            all_indices = torch.randperm(total_entries)
            known_global_idx = all_indices[:target_known]
            global_mask = torch.zeros(total_entries, dtype=torch.bool)
            global_mask[known_global_idx] = True
            mask_tensor = global_mask.clone()
            # Build observed tensor with zeros at unknowns
            observed_tensor = M_tensor.clone()
            observed_tensor[~global_mask] = 0.0
            # Agent-specific views
            if self.view_mode == 'sparse':
                features = []
                for _ in range(self.num_agents):
                    # Each agent samples (with replacement) a variable-sized subset 
                    # of the global known entries
                    # control expected agent overlap by tweaking agent_sample_size range
                    agent_sample_size = np.random.randint(
                        int(0.6 * target_known // self.num_agents),
                        int(1.2 * target_known // self.num_agents) + 1
                    )
                    sample_idx = known_global_idx[
                        torch.randint(len(known_global_idx), (agent_sample_size,))
                    ]
                    agent_view = torch.zeros(total_entries, dtype=torch.float32)
                    agent_view[sample_idx] = observed_tensor[sample_idx]
                    features.append(agent_view)
                x = torch.stack(features)
            else:  # projection view
                x = torch.stack([
                    torch.matmul(torch.randn(self.input_dim, total_entries), observed_tensor)
                    for _ in range(self.num_agents)
                ])
            # Create Data object
            data = Data(x=x, y=M_tensor, mask=mask_tensor)
            data_list.append(data)
        self.nuclear_norm_mean = np.mean(norms)
        return self.collate(data_list)


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
    sum_rest = S[rank:].sum()
    s_last = S[rank - 1].item()
    s_next = S[rank].item()
    gap = (s_last - s_next)
    svdcount = len(S) # or min(output.shape)
    ratio = sum_rest / (s_last + 1e-6)
    penalty = sum_rest/(svdcount-rank) + ratio #
    if s_last > 2 * svdcount:
        penalty += (s_last - svdcount)/svdcount ** 2
    elif s_last < svdcount / 2:
        penalty += (svdcount - s_last)/svdcount ** 2
    if evalmode:
        return gap
    else:
        return penalty

def train(model, loader, optimizer, theta, criterion, rank, 
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
        prediction = out.mean(dim=1)
        target = batch.y.view(batch_size, -1)
        reconstructionloss = criterion(prediction, target)
        penalty = sum(
            spectral_penalty(out[i], rank) for i in range(batch_size)
        ) / batch_size
        #diversity = agent_diversity_penalty(out)  # out: [B, A, D]
        loss = theta * reconstructionloss + (1 - theta) * penalty
        # Uncomment following lines to try variance penalty.
        # This seems to prevent learning any useful solutions for the other components of the loss
        #var_penalty = F.mse_loss(out.var(), torch.tensor(1.0).to(out.device))
        #loss += 0.5 * var_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, n, m, rank, 
             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    known_mse, unknown_mse, nuclear_norms, variances, gaps = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.num_graphs
        # Forward pass
        x = batch.x.view(batch_size, model.num_agents, -1)
        out = model(x)
        prediction = out.mean(dim=1)  # [batch_size, n*m]
        target = batch.y.view(batch_size, -1)  # [batch_size, n*m]
        mask = batch.mask.view(batch_size, -1)  # [batch_size, n*m]
        for i in range(batch_size):
            pred_i = prediction[i]  # [n*m]
            target_i = target[i]    # [n*m]
            mask_i = mask[i]        # [n*m]
            known_i = mask_i
            unknown_i = ~mask_i
            if known_i.any():
                known_mse.append(criterion(pred_i[known_i], target_i[known_i]).item())
            if unknown_i.any():
                unknown_mse.append(criterion(pred_i[unknown_i], target_i[unknown_i]).item())
            # Reshape for nuclear norm
            matrix_2d = pred_i.view(n, m)
            nuclear_norms.append(torch.linalg.norm(matrix_2d, ord='nuc').item())
            # Spectral penalty metrics
            gap = spectral_penalty(out[i], rank, evalmode=True)  # [num_agents, n*m]
            gaps.append(gap)
            # Variance of the agent outputs
            variances.append(out[i].var().item())
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
        nn.init.xavier_uniform_(m.weight, gain=5.0)
        #nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
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
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--r', type=int, default=4)
    parser.add_argument('--density', type=float, default=0.2)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--num_agents', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--theta', type=float, default=0.95, help='Weight for the known entry loss vs penalty')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--view_mode', type=str, choices=['sparse', 'project'], default='sparse')
    parser.add_argument('--eval_agents', action='store_true', help='Always evaluate agent contributions')
    parser.add_argument('--steps', type=int, default=3, help='Number of message passing steps')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = AgentMatrixReconstructionDataset(
        num_graphs=1000, n=args.n, m=args.m, r=args.r, num_agents=args.num_agents,
        view_mode=args.view_mode, density=args.density, sigma=args.sigma
    )
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = DistributedDotGAT(
        input_dim=dataset.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=dataset.n * dataset.m,
        num_agents=args.num_agents,
        num_heads=args.num_heads,
        dropout=args.dropout,
        message_steps=args.steps
    ).to(device)
    model.apply(init_weights)
    count_parameters(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    stats = {
        "train_loss": [],
        "val_known_mse": [],
        "val_unknown_mse": [],
        "nuclear_norm": [],
        "variance": [],
        "spectral_gap": []
    }
    file_base = unique_filename()
    checkpoint_path = f"{file_base}_checkpoint.pt"
    
    best_loss = float('inf')
    patience_counter = 0
    
    with torch.no_grad():
        batch = next(iter(train_loader)).to(device)
        x = batch.x.view(batch.num_graphs, model.num_agents, -1)
        out = model(x)
        recon = out.mean(dim=1)
        print("Initial output variance:", recon.var().item())

    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model, train_loader, optimizer, args.theta, criterion, args.r)
        val_known, val_unknown, nuc, var, gap = evaluate(
            model, val_loader, criterion, args.n, args.m, args.r)
        
        stats["train_loss"].append(train_loss)
        stats["val_known_mse"].append(val_known)
        stats["val_unknown_mse"].append(val_unknown)
        stats["nuclear_norm"].append(nuc)
        stats["variance"].append(var)
        stats["spectral_gap"].append(gap)
        
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Known: {val_known:.4f} | "+
            f"Unknown: {val_unknown:.4f} | Nucl: {nuc:.2f} | Gap: {gap:.2f} | Var: {var:.4f}")
        
        # Save connectivity matrix for visualization
        #adj_matrix = model.connectivity.detach().cpu().numpy()
        #np.save(f"{file_base}_adj_epoch{epoch}.npy", adj_matrix)
        
        if val_unknown < best_loss - 1e-5:
            best_loss = val_unknown
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Clear memory (avoid OOM) and load best model
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    print("Loaded best model from checkpoint.")

    plot_stats(stats, file_base, dataset.nuclear_norm_mean)
    
    # Final test evaluation on fresh data
    test_dataset = AgentMatrixReconstructionDataset(
        num_graphs=64, n=args.n, m=args.m, r=args.r,
        num_agents=args.num_agents, view_mode=args.view_mode,
        density=args.density, sigma=args.sigma
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    test_known, test_unknown, test_nuc, test_var, test_gap = evaluate(
        model, test_loader, criterion, args.n, args.m, args.r)
    print(f"Test Set Performance | Known MSE: {test_known:.4f}, Unknown MSE: {test_unknown:.4f}"+
        f" Nuclear Norm: {test_nuc:.2f}, Spectral Gap: {test_gap:.2f}, Variance: {test_var:.4f}")

    #file_prefix = Path(file_base).name  # Extracts just 'run_YYYYMMDD_HHMMSS'
    #plot_connectivity_matrices("results", prefix=file_prefix, cmap="coolwarm")

    # Agent contribution eval (optional)
    if test_unknown < 0.1 or args.eval_agents:
        evaluate_agent_contributions(model, test_loader, criterion, args.n, args.m)
    