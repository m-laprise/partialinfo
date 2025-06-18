"""
Dot-product graph attention network tested on Cora dataset and random data, for node classification.
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
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

    def forward(self, x, edge_index, batch=None):
        head_outputs = [head(x, edge_index) for head in self.heads]
        if self.agg_mode == 'concat':
            x_out = torch.cat(head_outputs, dim=1)
        else:
            x_out = torch.stack(head_outputs, dim=0).mean(dim=0)

        x_out = self.norm(x_out)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        x_out = self.swish(x_out)
        x_out = x_out + self.residual(x)

        if batch is None:
            return self.output(x_out)
        else:
            pooled = global_mean_pool(x_out, batch)
            return self.output(pooled)

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

def train(model, loader, optimizer, criterion):
    """
    One training step over the dataset.
    """
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total_loss += loss.item() * batch.num_graphs
    acc = 100 * correct / len(loader.dataset)
    return total_loss / len(loader.dataset), acc

def train_transductive(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_transductive(model, data, criterion, split):
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

class RandomStructuredGraphDataset(InMemoryDataset):
    """
    Generates a set of random graphs with structured node features.
    Each class has a different feature centroid.
    """
    def __init__(self, num_graphs=100, num_nodes=30, num_edges=100,
                 num_features=16, num_classes=3, feature_std=1.0, 
                 center_scale=0.3, structural_variation=True, transform=None):
        self.num_graphs = num_graphs
        self._num_nodes = num_nodes
        self._num_edges = num_edges
        self._num_features = num_features
        self._num_classes = num_classes
        self._feature_std = feature_std
        self._center_scale = center_scale
        self._structural_variation = structural_variation
        super().__init__('.', transform)
        self.data, self.slices = self._generate_data()

    def _generate_data(self):
        data_list = []
        
        # Class-specific feature centroids (closer together if center_scale < 1)
        class_centers = torch.randn(self._num_classes, self._num_features) * self._center_scale

        for _ in range(self.num_graphs):
            label = torch.randint(0, self._num_classes, (1,))
            center = class_centers[label]

            # Add graph-specific noise
            x = torch.randn((self._num_nodes, self._num_features)) * self._feature_std + center

            # Generate class-conditional graph structure
            if self._structural_variation:
                if label == 0:
                    G = nx.erdos_renyi_graph(self._num_nodes, p=0.2)
                elif label == 1:
                    G = nx.barabasi_albert_graph(self._num_nodes, m=2)
                else:
                    G = nx.minimum_spanning_tree(nx.complete_graph(self._num_nodes), weight=None)
            else:
                G = nx.gnm_random_graph(self._num_nodes, self._num_edges)

            # Ensure edges exist
            if G.number_of_edges() == 0:
                continue

            edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)  # make undirected

            data = Data(x=x, edge_index=edge_index, y=label)
            data_list.append(data)

        return self.collate(data_list)

def main(args):
    if args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
        data = dataset[0].to(device)
        in_features = dataset.num_node_features
        out_features = dataset.num_classes
        use_loader = False
    else:  # inductive/random
        dataset = RandomStructuredGraphDataset(
            num_graphs=300,
            num_nodes=30,
            num_edges=100,
            num_features=args.random_features,
            num_classes=args.random_classes,
            feature_std=args.feature_std,
            center_scale=args.center_scale,
            structural_variation=not args.no_structure
        ).to(device)
        
        train_len = int(0.6 * len(dataset))
        val_len = int(0.2 * len(dataset))
        test_len = len(dataset) - train_len - val_len
        train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)
        test_loader = DataLoader(test_set, batch_size=32)

        in_features = args.random_features
        out_features = args.random_classes
        use_loader = True
        
    model = MultiHeadDotGAT(
        in_features=in_features,
        hidden_features=args.hidden_dim,
        out_features=out_features,
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
        if use_loader:
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
        else:
            train_loss = train_transductive(model, data, optimizer, criterion)
            val_loss, val_acc = evaluate_transductive(model, data, criterion, split='val')

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
    
    if use_loader:
        test_loss, test_acc = evaluate(model, test_loader, criterion)
    else:
        test_loss, test_acc = evaluate_transductive(model, data, criterion, split='test')

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
    parser.add_argument('--dataset', type=str, choices=['cora', 'random'], default='cora',
                    help='Dataset to use: cora (transductive) or random (inductive)')
    parser.add_argument('--random_features', type=int, default=16,
                        help='Number of features for random graph nodes')
    parser.add_argument('--random_classes', type=int, default=3,
                        help='Number of classes for random graph labels')
    parser.add_argument('--feature_std', type=float, default=1.0,
                    help='Standard deviation for node features')
    parser.add_argument('--center_scale', type=float, default=0.3,
                        help='Spread of class-specific feature centers')
    parser.add_argument('--no_structure', action='store_true',
                        help='Disable structural variation between classes')
    args = parser.parse_args()

    main(args)