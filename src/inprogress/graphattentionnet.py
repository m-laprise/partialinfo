import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Dot-product attention GAT layer (like Julia version)
class DotGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.dropout = dropout
        self.scale = math.sqrt(out_features)

    def forward(self, x, edge_index):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        row, col = edge_index

        alpha = (Q[row] * K[col]).sum(dim=-1) / self.scale
        alpha = softmax(alpha, col)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = Q[row] * alpha.unsqueeze(-1)
        out = torch.zeros_like(Q).index_add_(0, col, out)
        return out


# Multi-head dot-product GAT with concat or mean
class MultiHeadDotGAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout, agg_mode='concat'):
        super().__init__()
        assert agg_mode in ['concat', 'mean'], "agg_mode must be 'concat' or 'mean'"
        self.heads = nn.ModuleList([
            DotGATLayer(in_features, hidden_features, dropout=dropout)
            for _ in range(num_heads)
        ])
        self.agg_mode = agg_mode
        self.norm = nn.LayerNorm(hidden_features * num_heads if agg_mode == 'concat' else hidden_features)
        self.swish = Swish()
        final_in_dim = hidden_features * num_heads if agg_mode == 'concat' else hidden_features
        self.output = nn.Linear(final_in_dim, out_features)
        self.dropout = dropout

    def forward(self, x, edge_index):
        head_outputs = [head(x, edge_index) for head in self.heads]
        if self.agg_mode == 'concat':
            x = torch.cat(head_outputs, dim=1)
        else:  # mean
            x = torch.stack(head_outputs, dim=0).mean(dim=0)

        x = self.norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.swish(x)
        x = self.output(x)
        return x


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, criterion):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    losses = {}
    accs = {}
    for split in ['train', 'val', 'test']:
        mask = data[f'{split}_mask']
        losses[split] = criterion(out[mask], data.y[mask]).item()
        accs[split] = (pred[mask] == data.y[mask]).float().mean().item() * 100
    return losses, accs


def main(args):
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    model = MultiHeadDotGAT(
        in_features=dataset.num_node_features,
        hidden_features=args.hidden_dim,
        out_features=dataset.num_classes,
        num_heads=args.num_heads,
        dropout=args.dropout,
        agg_mode=args.agg
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, data, optimizer, criterion)
        losses, accs = evaluate(model, data, criterion)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {losses['val']:.4f} | "
                  f"Test Acc: {accs['test']:.2f}%")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model_state)
    _, accs = evaluate(model, data, criterion)
    print(f"\nFinal Test Accuracy: {accs['test']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dot-Product GAT on Cora with CLI options')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=8, help='Hidden dimension per head')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--agg', type=str, choices=['concat', 'mean'], default='concat',
                        help='Aggregation mode for multi-head (concat or mean)')
    args = parser.parse_args()

    main(args)