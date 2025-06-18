import argparse
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def plot_stats(stats, filename_base):
    plt.figure(figsize=(10, 6))
    for key, values in stats.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Metrics Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename_base}_metrics.png")
    plt.close()


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

    def forward(self, x, edge_index):
        head_outputs = [head(x, edge_index) for head in self.heads]
        x_out = torch.cat(head_outputs, dim=1) if self.agg_mode == 'concat' else torch.stack(head_outputs).mean(dim=0)
        x_out = self.norm(x_out)
        x_out = F.dropout(x_out, p=0.5, training=self.training)
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
        super().__init__('.', transform=None)
        self.data, self.slices = self._generate()

    def _generate(self):
        data_list = []
        for _ in range(self.num_graphs):
            M, mask = generate_low_rank(self.n, self.m, self.r, self.density, self.sigma)
            M = torch.tensor(M, dtype=torch.float)
            mask = torch.tensor(mask, dtype=torch.bool)
            observed = M * mask
            edge_index = dense_to_sparse(mask.float())[0]
            data = Data(x=observed.T, edge_index=edge_index, y=M.T, mask=mask.T)
            data_list.append(data)
        return self.collate(data_list)


def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.mse_loss(out[batch.mask], batch.y[batch.mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    known_loss, unknown_loss, nuclear_norms, variances = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        known = batch.mask
        unknown = ~batch.mask
        known_loss.append(F.mse_loss(out[known], batch.y[known]).item())
        unknown_loss.append(F.mse_loss(out[unknown], batch.y[unknown]).item())
        nuclear_norms.append(torch.norm(out, p='nuc').item())
        variances.append(out.var().item())
    return (
        np.mean(known_loss),
        np.mean(unknown_loss),
        np.mean(nuclear_norms),
        np.mean(variances),
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

    stats = {"train_loss": [], "val_known_loss": [], "val_unknown_loss": [], "nuclear_norm": [], "variance": []}
    file_base = unique_filename()

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        val_known, val_unknown, nuc, var = evaluate(model, val_loader)

        stats["train_loss"].append(train_loss)
        stats["val_known_loss"].append(val_known)
        stats["val_unknown_loss"].append(val_unknown)
        stats["nuclear_norm"].append(nuc)
        stats["variance"].append(var)

        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Known: {val_known:.4f} | Unknown: {val_unknown:.4f} | Nuclear: {nuc:.2f} | Var: {var:.4f}")

    plot_stats(stats, file_base)