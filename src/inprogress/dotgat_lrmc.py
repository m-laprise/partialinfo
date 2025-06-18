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
from torch_geometric.utils import dense_to_sparse


def generate_low_rank(n, m, r, density=0.2, sigma=0.01):
    U = np.random.randn(n, r) / np.sqrt(r)
    V = np.random.randn(m, r) / np.sqrt(r)
    M = U @ V.T + sigma * np.random.randn(n, m)
    mask = np.random.rand(n, m) < density
    return M, mask

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

    axs[1].plot(epochs, stats["val_known_loss"], label="Val MSE Known Entries", color='tab:green')
    axs[1].plot(epochs, stats["val_unknown_loss"], label="Val MSE Unknown Entries", color='tab:orange')
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

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class DotGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.dropout = dropout
        self.scale = math.sqrt(out_features)

    def forward(self, x, edge_index):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        row, col = edge_index
        alpha = (Q[row] * K[col]).sum(dim=-1) / self.scale
        alpha = F.softmax(alpha, dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = Q[row] * alpha.unsqueeze(-1)
        out = torch.zeros_like(Q).index_add_(0, col, out)
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

    def forward(self, x, edge_index):
        head_outputs = [head(x, edge_index) for head in self.heads]
        x_out = torch.cat(head_outputs, dim=1) if self.agg_mode == 'concat' else torch.stack(head_outputs).mean(dim=0)
        x_out = self.norm(x_out)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        x_out = self.swish(x_out)
        x_out = x_out + self.residual(x)
        return self.output(x_out)

class LowRankMatrixGraphDataset(InMemoryDataset):
    def __init__(self, num_graphs=1000, n=20, m=20, r=4, density=0.2, sigma=0.01):
        self.num_graphs = num_graphs
        self.n = n
        self.m = m
        self.r = r
        self.density = density
        self.sigma = sigma
        self.nuclear_norm_mean = 0.0
        super().__init__('.', transform=None)
        self.data, self.slices = self._generate()

    def _generate(self):
        data_list = []
        norms = []
        for _ in range(self.num_graphs):
            M, mask = generate_low_rank(self.n, self.m, self.r, self.density, self.sigma)
            M_tensor = torch.tensor(M, dtype=torch.float)
            norms.append(torch.norm(M_tensor, p='nuc').item())
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            observed = M_tensor * mask_tensor
            edge_index = dense_to_sparse(mask_tensor.float())[0]
            data = Data(x=observed.T, edge_index=edge_index, y=M_tensor.T, mask=mask_tensor.T)
            data_list.append(data)
        self.nuclear_norm_mean = np.mean(norms)
        return self.collate(data_list)

def spectral_penalty(output):
    U, S, V = torch.svd(output)
    sum_rest = S[1:].sum()
    #ratio = sum_rest / (S[0] + 1e-6)
    N = min(output.shape)
    penalty = sum_rest #+ ratio
    if S[0] > 2 * N:
        penalty += (S[0] - 2 * N) ** 2
    elif S[0] < N / 2:
        penalty += (N / 2 - S[0]) ** 2
    gap = (S[0] - S[1]).item() if len(S) > 1 else 0.0
    return penalty, S[0].item(), gap

def train(model, loader, optimizer, theta):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        l2_loss = F.mse_loss(out[batch.mask], batch.y[batch.mask])
        penalty, _, _ = spectral_penalty(out)
        loss = theta * l2_loss + (1 - theta) * penalty
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    known_loss, unknown_loss, nuclear_norms, variances, gaps = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        known = batch.mask
        unknown = ~batch.mask
        known_loss.append(F.mse_loss(out[known], batch.y[known]).item())
        unknown_loss.append(F.mse_loss(out[unknown], batch.y[unknown]).item())
        _, top_singular, gap = spectral_penalty(out)
        nuclear_norms.append(torch.norm(out, p='nuc').item())
        gaps.append(gap)
        variances.append(out.var().item())
    return (
        np.mean(known_loss),
        np.mean(unknown_loss),
        np.mean(nuclear_norms),
        np.mean(variances),
        np.mean(gaps),
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--r', type=int, default=4)
    parser.add_argument('--density', type=float, default=0.2)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--agg_mode', type=str, choices=['concat', 'mean'], default='concat', help='Aggregation mode for multi-head attention')
    parser.add_argument('--theta', type=float, default=0.9, help='Weight for the known entry loss vs penalty')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LowRankMatrixGraphDataset(1000, args.n, args.m, args.r, args.density, args.sigma)
    train_set, val_set = torch.utils.data.random_split(dataset, [800, 200])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = MultiHeadDotGAT(
        in_features=args.n,
        hidden_features=args.hidden_dim,
        output_dim=args.n,
        num_heads=args.num_heads,
        dropout=args.dropout,
        agg_mode=args.agg_mode
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    stats = {"train_loss": [], 
             "val_known_loss": [], "val_unknown_loss": [], 
             "nuclear_norm": [], "variance": [], "spectral_gap": []}
    file_base = unique_filename()
    checkpoint_path = f"{file_base}_best_model.pt"
    
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, args.theta)
        val_known, val_unknown, nuc, var, gap = evaluate(model, val_loader)

        stats["train_loss"].append(train_loss)
        stats["val_known_loss"].append(val_known)
        stats["val_unknown_loss"].append(val_unknown)
        stats["nuclear_norm"].append(nuc)
        stats["variance"].append(var)
        stats["spectral_gap"].append(gap)

        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Known: {val_known:.4f} | Unknown: {val_unknown:.4f} | Nucl: {nuc:.2f} | Gap: {gap:.2f} | Var: {var:.4f}")
        
        if val_unknown < best_loss - 1e-5:
            best_loss = val_unknown
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
            
    model.load_state_dict(torch.load(checkpoint_path))
    print("Loaded best model from checkpoint.")

    plot_stats(stats, file_base, dataset.nuclear_norm_mean)
    
    # Test on fresh data
    test_dataset = LowRankMatrixGraphDataset(num_graphs=64, n=args.n, m=args.m, r=args.r, density=args.density, sigma=args.sigma)
    test_loader = DataLoader(test_dataset, batch_size=32)
    test_known, test_unknown, test_nuc, test_var, test_gap = evaluate(model, test_loader)
    print(f"Test Set Performance | Known MSE: {test_known:.4f}, Unknown MSE: {test_unknown:.4f}, Nuclear Norm: {test_nuc:.2f}, Spectral Gap: {test_gap:.2f}, Variance: {test_var:.4f}")
