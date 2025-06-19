"""
Dot product graph attention network for dsitributed matrix completion, 
with learned agent-based message passing setup.
Supports both sparse views and low-dimensional projections as agent inputs.
"""

import argparse
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
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
    ax.plot(epochs, stats["spectral_gap"], label="Spectral Gap (s1 - s2)", color='tab:orange')
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
    def __init__(self, in_features, out_features, dropout=0.6):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.dropout = dropout
        self.scale = math.sqrt(out_features)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        scores = torch.matmul(Q, K.T) / self.scale
        alpha = F.softmax(scores, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.matmul(alpha, Q)
        return out + self.residual(x)


class MultiHeadDotGAT(nn.Module):
    def __init__(self, in_features, hidden_features, output_dim, num_heads, dropout, agg_mode='concat'):
        super().__init__()
        assert agg_mode in ['concat', 'mean']
        self.heads = nn.ModuleList([
            DotGATLayer(in_features, hidden_features, dropout=dropout)
            for _ in range(num_heads)
        ])
        self.agg_mode = agg_mode
        final_in_dim = hidden_features * num_heads if agg_mode == 'concat' else hidden_features
        self.norm = nn.LayerNorm(final_in_dim)
        self.swish = Swish()
        self.output = nn.Linear(final_in_dim, output_dim)
        self.residual = nn.Linear(in_features, final_in_dim) if in_features != final_in_dim else nn.Identity()
        self.dropout = dropout

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        x_out = torch.cat(head_outputs, dim=1) if self.agg_mode == 'concat' else torch.stack(head_outputs).mean(dim=0)
        x_out = self.norm(x_out)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        x_out = self.swish(x_out)
        x_out = x_out + self.residual(x)
        return self.output(x_out)


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
        for _ in range(self.num_graphs):
            U = np.random.randn(self.n, self.r) / np.sqrt(self.r)
            V = np.random.randn(self.m, self.r) / np.sqrt(self.r)
            M = U @ V.T + self.sigma * np.random.randn(self.n, self.m)
            mask = np.random.rand(self.n, self.m) < self.density
            observed = M * mask
            M_tensor = torch.tensor(M, dtype=torch.float).view(-1)
            mask_tensor = torch.tensor(mask, dtype=torch.bool).view(-1)
            norms.append(torch.linalg.norm(torch.tensor(M, dtype=torch.float), ord='nuc').item())
            
            if self.view_mode == 'sparse':
                observed_tensor = torch.tensor(observed, dtype=torch.float).view(-1)
                features = []
                for _ in range(self.num_agents):
                    agent_mask = torch.rand(self.n * self.m) < self.density
                    agent_view = observed_tensor.clone()
                    agent_view[~agent_mask] = 0.0
                    features.append(agent_view)
                x = torch.stack(features)
            else:  # projection
                observed_tensor = torch.tensor(observed, dtype=torch.float).view(-1)
                x = torch.stack([
                    torch.matmul(torch.randn(self.input_dim, self.n * self.m), observed_tensor)
                    for _ in range(self.num_agents)
                ])

            data = Data(x=x, y=M_tensor, mask=mask_tensor)
            data_list.append(data)
        self.nuclear_norm_mean = np.mean(norms)
        return self.collate(data_list)

def spectral_penalty(output):
    """
    Compute spectral penalty for a low rank matrix output.
    
    The spectral penalty is defined as the sum of the singular values
    of the output matrix, excluding the largest singular value.
    
    Additionally, if the largest singular value is greater than 2
    times the smaller dimension of the output matrix, or if it is
    less than half the smaller dimension, add a penalty term of
    (s0 - 2 * N) ** 2 or (N / 2 - s0) ** 2, respectively.
    
    Return the spectral penalty, the largest singular value, and the
    gap between the largest and second largest singular values.
    """
    try:
        U, S, Vt = torch.linalg.svd(output, full_matrices=False)
    except RuntimeError:
        S = torch.linalg.svdvals(output)
    sum_rest = S[1:].sum()
    N = min(output.shape)
    penalty = sum_rest
    s0 = S[0].item()
    if s0 > 2 * N:
        penalty += (s0 - 2 * N) ** 2
    elif s0 < N / 2:
        penalty += (N / 2 - s0) ** 2
    gap = (s0 - S[1].item()) if len(S) > 1 else 0.0
    return penalty, s0, gap

def train(model, loader, optimizer, theta, criterion, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        batch_size = batch.num_graphs
        out = model(batch.x)  # [batch_size * num_agents, n*m]
        out = out.view(batch_size, -1, out.shape[-1])  # [batch_size, num_agents, n*m]
        prediction = out.mean(dim=1)  # [batch_size, n*m] # Collective prediction: mean
        target = batch.y.view(batch_size, -1)
        reconstructionloss = criterion(prediction, target)
        penalty = 0.0
        for i in range(out.shape[0]):  # batch size
            p, _, _ = spectral_penalty(out[i])  # [num_agents, n*m] per sample
            penalty += p
        penalty /= out.shape[0]
        loss = theta * reconstructionloss + (1 - theta) * penalty
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, n, m, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    known_mse, unknown_mse, nuclear_norms, variances, gaps = [], [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.num_graphs

        # Forward pass
        out = model(batch.x)  # [batch_size * num_agents, n*m]
        out = out.view(batch_size, -1, out.shape[-1])  # [batch_size, num_agents, n*m]
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
            _, _, gap = spectral_penalty(out[i])  # [num_agents, n*m]
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

        out = model(batch.x)  # [batch_size * num_agents, n*m]
        out = out.view(batch_size, num_agents, nm)  # [B, A, n*m]
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
    #parser.add_argument('--agg_mode', type=str, choices=['concat', 'mean'], default='concat', help='Aggregation mode for multi-head attention')
    parser.add_argument('--theta', type=float, default=0.95, help='Weight for the known entry loss vs penalty')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--view_mode', type=str, choices=['sparse', 'project'], default='sparse')
    parser.add_argument('--eval_agents', action='store_true', help='Always evaluate agent contributions')
    
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

    model = MultiHeadDotGAT(
        in_features=dataset.input_dim,
        hidden_features=args.hidden_dim,
        output_dim=dataset.n * dataset.m,
        num_heads=args.num_heads,
        dropout=args.dropout,
        agg_mode='concat'
    ).to(device)

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
            model, train_loader, optimizer, args.theta, criterion)
        val_known, val_unknown, nuc, var, gap = evaluate(
            model, val_loader, criterion, args.n, args.m)
        
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
        model, test_loader, criterion, args.n, args.m)
    print(f"Test Set Performance | Known MSE: {test_known:.4f}, Unknown MSE: {test_unknown:.4f}, Nuclear Norm: {test_nuc:.2f}, Spectral Gap: {test_gap:.2f}, Variance: {test_var:.4f}")

    # Agent contribution eval (optional)
    if test_unknown < 0.1 or args.eval_agents:
        evaluate_agent_contributions(model, test_loader, criterion, args.n, args.m)
    