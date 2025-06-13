import argparse
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class DotGATLayer(nn.Module):
    """
    Single-head dot-product Graph Attention Layer.

    Projects input features into query and key spaces,
    computes attention scores using scaled dot-product attention,
    and aggregates information from neighboring nodes.
    """
    def __init__(self, in_features, out_features, dropout=0.6):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.dropout = dropout
        self.scale = math.sqrt(out_features)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node feature matrix of shape [num_nodes, in_features]
            edge_index: Graph connectivity matrix [2, num_edges]

        Returns:
            Aggregated node features of shape [num_nodes, out_features]
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        row, col = edge_index   # Source and target node indices

        # Compute scaled dot-product attention
        alpha = (Q[row] * K[col]).sum(dim=-1) / self.scale
        alpha = softmax(alpha, col)     # Normalize over neighbors
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted aggregation of neighbor features
        out = Q[row] * alpha.unsqueeze(-1)
        out = torch.zeros_like(Q).index_add_(0, col, out)
        return out + self.residual(x)


# Multi-head dot-product GAT with concat or mean
class MultiHeadDotGAT(nn.Module):
    """
    Multi-head dot-product Graph Attention Network.

    Applies several attention heads in parallel and combines
    the results using concatenation or mean pooling.
    """
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout, agg_mode='concat'):
        """
        Args:
            in_features: Dimensionality of input node features
            hidden_features: Output dimensionality per attention head
            out_features: Number of classes (output dimensionality)
            num_heads: Number of parallel attention heads
            dropout: Dropout rate applied after normalization
            agg_mode: Aggregation mode for multiple heads: 'concat' or 'mean'
        """
        super().__init__()
        assert agg_mode in ['concat', 'mean'], "agg_mode must be 'concat' or 'mean'"
        
        # Create multiple attention heads
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
        self.residual = nn.Linear(in_features, final_in_dim) if in_features != final_in_dim else nn.Identity()

    def forward(self, x, edge_index):
        head_outputs = [head(x, edge_index) for head in self.heads]
        if self.agg_mode == 'concat':
            x_out = torch.cat(head_outputs, dim=1)
        else:  # mean
            x_out = torch.stack(head_outputs, dim=0).mean(dim=0)

        x_out = self.norm(x_out)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        x_out = self.swish(x_out)
        x_out = self.output(x_out + self.residual(x))
        return x_out

# EarlyStopping utility class
class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.

    Monitors a validation metric and stops training if it doesn't improve
    after a certain number of epochs (`patience`).
    """
    def __init__(self, patience=10, delta=1e-4, mode='min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            delta: Minimum change to be considered as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score, model):
        # Flip sign if we're minimizing (e.g. loss)
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
    """
    One training step over the dataset.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, criterion, split):
    """
    Evaluates model on a given dataset split.
    
    Args:
        split: One of 'train', 'val', or 'test'
    
    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    mask = data[f'{split}_mask']
    loss = criterion(out[mask], data.y[mask]).item()
    acc = (pred[mask] == data.y[mask]).float().mean().item() * 100
    return loss, acc

def plot_metrics(train_losses, val_losses, val_accs, save_path):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Over Epochs")
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))

    plt.figure()
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Over Epochs")
    plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))


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
    
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, data, optimizer, criterion)
        val_loss, val_acc = evaluate(model, data, criterion, split='val')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")

        early_stopper(val_loss, model)
        if early_stopper.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(early_stopper.best_state)
    
    test_loss, test_acc = evaluate(model, data, criterion, split='test')
    print(f"Final Test Accuracy (best model): {test_acc:.2f}%")
    plot_metrics(train_losses, val_losses, val_accs, save_path='.')

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