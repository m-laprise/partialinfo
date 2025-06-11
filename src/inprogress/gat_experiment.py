import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.transforms import RandomNodeSplit

# ─── Argument Parsing ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GAT on Cora with CLI hyperparameters")
parser.add_argument("--hidden", type=int, default=8, help="Hidden channels")
parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate")
parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
args = parser.parse_args()

# ─── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ─── Device setup ───────────────────────────────────────────────────────────────
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─── Data Loading and Splitting ────────────────────────────────────────────────
dataset = Planetoid(
    root='data/Cora', 
    name='Cora'
    )

data = dataset[0].to(device)

# ─── Model Definition ─────────────────────────────────────────────────────────
class GAT(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, heads, dropout):
        super().__init__()
        self.conv1 = GATConv(in_ch, hid_ch, 
                             heads=heads, dropout=dropout)
        self.conv2 = GATConv(hid_ch * heads, out_ch,
                             heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

model = GAT(
    dataset.num_features, 
    args.hidden, 
    dataset.num_classes, 
    args.heads, 
    args.dropout
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)

# ─── TensorBoard Logging ───────────────────────────────────────────────────────
writer = SummaryWriter(log_dir='runs/GAT_Cora_experiment')

@torch.no_grad()
def evaluate(mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits[mask].argmax(dim=1)
    return float((pred == data.y[mask]).sum()) / int(mask.sum()) #mask.sum().item()

# ─── Training Loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, args.epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    train_acc = evaluate(data.train_mask)
    val_acc = evaluate(data.val_mask)

    writer.add_scalars(
        'Accuracy', {
            'Train': train_acc,
            'Validation': val_acc,
        }, epoch)
    writer.add_scalar('Loss/Train', loss.item(), epoch)

    if val_acc > best_val_acc:
        best_val_acc = val_acc

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} "
              f"| Train: {train_acc:.4f} | Val: {val_acc:.4f}")

# ─── Final Report ────────────────────────────────────────────────────────────
test_acc = evaluate(data.test_mask)

print(f"\n✅ Best validation accuracy: {best_val_acc:.4f}")
print(f"Final test accuracy: {test_acc:.4f}")
writer.close()