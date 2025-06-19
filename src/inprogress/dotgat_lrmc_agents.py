"""
Dot product graph attention network for distributed matrix completion, 
with learned agent-based message passing setup.
Supports both sparse views and low-dimensional projections as agent inputs.

(Currently, view-mode 'sparse' runs, but view-mode 'project' needs to be debugged.)

Each agent independently receives their own masked view of the matrix via 
AgentMatrixReconstructionDataset (x: [batch, num_agents, input_dim]). 

During message passing, each DotGATLayer uses dot-product attention with a learned 
connectivity matrix and applies top-k sparsification, ensuring localized neighbor communication.
Aggregation only occurs after message passing and is mediated via the GAT network to prevent
information leakage.

self.connectivity is a trainable nn.Parameter representing the agent-to-agent communication graph.
The attention computation adds this weighted adjacency matrix to the attention logits, making the 
interactions learnable.

self.agent_embeddings adds agent-specific vectors into the processing.
During training and inference, each agent maintains separate outputs that are optionally aggregated.

Explicit message passing rounds are controlled by message_steps in DistributedDotGAT, with message 
updates applied iteratively through GAT heads.

Design choices:
- Currently self.connectivity is shared across all batches, and static per model instance.
    An alternative would be to allow connectivity vary by batch or to depend on inputs/agent embeddings.
- topk sparsification is static.
    An alternative would be dynamic thresholding or learned gating. ChatGPT suggests thresholded softmax 
    (Sparsemax or Entmax), attention dropout, learned attention masks, or entropic sparsity.
- Agents only weakly specialize through agent-specific embeddings.
    Alternatives include: agent-specific MLPs or gating mechanisms before/after message passing;  
    additional diversity or specialization losses
- No residual connections.
    Alternative: `h = h + torch.stack(head_outputs).mean(dim=0)` or `h = self.norm(h + ...)`
- collective aggregation occurs through mean pooling across agents.
    Alternatives include:
    attention-based aggregation with a learned attention weight for each agent, 
    aggregate using learned gating, 
    game-theoretic aggregation of subsets of agents based on utility or diversity contribution

Required improvements:
- Save and plot connectivity matrix over time
- Spectral penalty is unstable. ChatGPT suggests eplacing gap-based penalty with nuclear norm clipping, 
differentiable low-rank approximations, or penalizing condition number.
- Log agent_diversity_penalty during training to track changes in agent roles
- Per-agent MSE should be correlated with agent input quality/sparsity. Integrate tracking input sparsity 
per agent along with prediction quality.
- Develop view_mode='project' and make projections learnable (e.g., per-agent linear layers).
"""

import argparse
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
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
    axs[1].plot(epochs, stats["val_known_mse"], label="Val MSE Known Entries", color='tab:green')
    axs[1].plot(epochs, stats["val_unknown_mse"], label="Val MSE Unknown Entries", color='tab:orange')
    axs[1].plot(epochs, stats["variance"], label="Variance of Reconstructed Entries", color='tab:red')
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
    ax.axhline(y=true_nuclear_mean, color='gray', linestyle='--', label="Ground Truth Mean Nuclear Norm")
    ax.set_title("Spectral Properties Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Singular Value Scale")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(f"{filename_base}_spectral_diagnostics.png")
    plt.close(fig)


class DotGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, topk=5):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.v_proj = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = dropout
        self.scale = math.sqrt(out_features)
        self.topk = topk

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
        scores = scores + connectivity  # [B, A, A]
        # Apply attention with top-k sparsification
        topk_vals, topk_idx = torch.topk(scores, self.topk, dim=-1)
        mask = scores.new_full(scores.shape, float('-inf'))
        mask.scatter_(dim=-1, index=topk_idx, src=topk_vals)

        alpha = F.softmax(mask, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.matmul(alpha, V)  # [B, A, hidden_dim]
        return self.norm(out)


class DistributedDotGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_agents, 
                 num_heads, dropout, message_steps=3):
        super().__init__()
        self.num_agents = num_agents
        self.message_steps = message_steps
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.agent_embeddings = nn.Parameter(torch.randn(num_agents, hidden_dim))
        
        # Generate small-world graph with networkx
        G = nx.watts_strogatz_graph(n=num_agents, k=4, p=0.3)
        A = torch.zeros(num_agents, num_agents)
        for i, j in G.edges():
            A[i, j] = 1.0
            A[j, i] = 1.0  # Ensure symmetry

        # Make it learnable, but initialize with sparse structure
        self.connectivity = nn.Parameter(A + 0.01 * torch.randn(num_agents, num_agents))
        
        #self.connectivity = nn.Parameter(torch.randn(num_agents, num_agents))
        self.gat_heads = nn.ModuleList([
            DotGATLayer(hidden_dim, hidden_dim, dropout=dropout, topk=5)
            for _ in range(num_heads)
        ])
        final_dim = hidden_dim
        self.swish = Swish()
        self.output_proj = nn.Linear(final_dim, output_dim)

    def forward(self, x):
        # x: [batch, num_agents, input_dim]
        h = self.input_proj(x) + self.agent_embeddings.unsqueeze(0)  # Inject identity

        for _ in range(self.message_steps):
            head_outputs = [head(h, self.connectivity.unsqueeze(0)) for head in self.gat_heads]
            h = torch.stack(head_outputs).mean(dim=0)
            h = self.swish(F.dropout(h, p=0.3, training=self.training))

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
            # === Agent-specific views ===
            if self.view_mode == 'sparse':
                features = []
                for _ in range(self.num_agents):
                    # Each agent samples (with replacement) a variable-sized subset 
                    # of the global known entries
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
    gap = (s_last - s_next) if len(S) > 1 else 0.0
    #ratio = sum_rest / (s_last + 1e-6)
    penalty = sum_rest - 2*gap #+ ratio
    if evalmode:
        return gap
    else:
        return penalty

def agent_diversity_penalty(agent_outputs):
    # agent_outputs: [num_agents, n*m] or [batch_size, num_agents, n*m]
    if agent_outputs.dim() == 3:
        B, A, D = agent_outputs.shape
        agent_outputs = F.normalize(agent_outputs, dim=-1)  # cosine space
        sim = torch.matmul(agent_outputs, agent_outputs.transpose(1, 2))  # [B, A, A]
        mask = ~torch.eye(A, dtype=torch.bool, device=sim.device)
        pairwise_sim = sim[:, mask].view(B, -1)  # remove diagonal
        return pairwise_sim.mean()
    else:
        A, D = agent_outputs.shape
        agent_outputs = F.normalize(agent_outputs, dim=-1)
        sim = torch.matmul(agent_outputs, agent_outputs.T)  # [A, A]
        mask = ~torch.eye(A, dtype=torch.bool, device=sim.device)
        pairwise_sim = sim[mask].view(A, -1)
        return pairwise_sim.mean()

def train(model, loader, optimizer, theta, criterion, rank, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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
        diversity = agent_diversity_penalty(out)  # out: [B, A, D]
        loss = theta * reconstructionloss + (1 - theta) * penalty + 0.2 * diversity
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, n, m, rank, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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
def evaluate_agent_contributions(model, loader, criterion, n, m, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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
        nn.init.xavier_uniform_(m.weight)
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
        
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Known: {val_known:.4f} | Unknown: {val_unknown:.4f} | Nucl: {nuc:.2f} | Gap: {gap:.2f} | Var: {var:.4f}")
        
        if val_known < best_loss - 1e-5:
            best_loss = val_known
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(checkpoint_path))
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
    print(f"Test Set Performance | Known MSE: {test_known:.4f}, Unknown MSE: {test_unknown:.4f}, Nuclear Norm: {test_nuc:.2f}, Spectral Gap: {test_gap:.2f}, Variance: {test_var:.4f}")

    # Agent contribution eval (optional)
    if test_unknown < 0.1 or args.eval_agents:
        evaluate_agent_contributions(model, test_loader, criterion, args.n, args.m)
    