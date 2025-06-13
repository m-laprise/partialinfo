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

# Dot-product attention GAT layer
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

# EarlyStopping utility class
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score, model):
        score = -score if self.mode == 'min' else score

        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.best_state = model.state_dict()
            self.counter = 0

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, criterion, split):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    mask = data[f'{split}_mask']
    loss = criterion(out[mask], data.y[mask]).item()
    acc = (pred[mask] == data.y[mask]).float().mean().item() * 100
    return loss, acc


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

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()

    early_stopper = EarlyStopping(patience=args.patience, delta=1e-4, mode='min')
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, data, optimizer, criterion)
        val_loss, val_acc = evaluate(model, data, criterion, split='val')

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        early_stopper(val_loss, model)
        if early_stopper.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(early_stopper.best_state)
    
    test_loss, test_acc = evaluate(model, data, criterion, split='test')
    print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")
    print(f"Final Test Accuracy (best model): {test_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dot-Product GAT on Cora with CLI options')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=8, help='Hidden dimension per head')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 regularization) for AdamW optimizer')
    parser.add_argument('--agg', type=str, choices=['concat', 'mean'], default='concat',
                        help='Aggregation mode for multi-head (concat or mean)')
    args = parser.parse_args()

    main(args)