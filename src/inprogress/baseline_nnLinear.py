#!/usr/bin/env python3
"""
Train a simple linear model (centralized case) to predict the index of the column
with the maximal value in row t+1 (the last row) of M = U V^T, using ONLY the
masked information up to time t (global sensing mask).

Usage (defaults shown):
    python train_centralized.py \
        --N 2000 --t 20 --m 15 --r 5 \
        --density 0.2 --num_agents 5 \
        --epochs 50 --batch_size 128 --seed 42

Notes
-----
- Centralized setting: we use ONLY the global mask (union of agents) for inputs,
  and set future_only=True to hide the final row (t+1) during training.
- Label is argmax over the final row (time t+1) of each M, as defined by
  datagen_temporal.TemporalData.
"""
import argparse
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import datagen_temporal as dt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    N: int = 2000
    t: int = 20
    m: int = 15
    r: int = 5
    density: float = 0.2
    num_agents: int = 5
    realizations: int = 10
    structured: bool = True
    seed: int = 42
    epochs: int = 50
    batch_size: int = 128
    lr: float = 5e-3
    weight_decay: float = 0.0
    train_frac: float = 0.7
    val_frac: float = 0.15  # test_frac implied as 1 - train - val


class LinearHead(nn.Module):
    """Simple linear classifier: flatten(masked M up to t) -> m classes."""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def build_dataset(cfg: Config):
    matrices = dt.GTMatrices(
        N=cfg.N, t=cfg.t, m=cfg.m, r=cfg.r,
        structured=cfg.structured, realizations=cfg.realizations, seed=cfg.seed
    )
    temporal_data = dt.TemporalData(matrices, verbose=False)
    sensing = dt.SensingMasks(
        temporal_data, rank=cfg.r, num_agents=cfg.num_agents,
        density=cfg.density, future_only=True
    )
    # Materialize X (masked) and y.
    X_list, y_list = [], []
    for i in range(len(temporal_data)):
        sample = temporal_data[i]  # expects dict with keys 'matrix' (1 x t*m) and 'label' (int)
        x = sample['matrix']                   # shape: (1, t*m)
        x_masked = sensing(x, global_mask=True)  # only global sensing mask
        X_list.append(x_masked.view(-1))      # flatten to (t*m,)
        y_list.append(int(sample['label']))

    X = torch.stack(X_list)                   # (N, t*m)
    y = torch.tensor(y_list, dtype=torch.long)  # (N,)

    return X, y, temporal_data, sensing


def make_splits(X, y, cfg: Config):
    N = X.shape[0]
    n_train = int(round(cfg.train_frac * N))
    n_val = int(round(cfg.val_frac * N))
    n_test = N - n_train - n_val
    assert n_train > 0 and n_val > 0 and n_test > 0, "Bad splits; adjust fractions or N."

    dataset = TensorDataset(X, y)
    train_set, val_set, test_set = random_split(
        dataset, lengths=[n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    return train_set, val_set, test_set


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean().item()


def train_eval_loop(model, train_loader, val_loader, test_loader, cfg: Config, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        # Train
        model.train()
        total_loss, total_acc, total_samples = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            bs = xb.size(0)
            total_samples += bs
            total_loss += loss.item() * bs
            total_acc += accuracy(logits.detach(), yb) * bs

        train_loss = total_loss / total_samples
        train_acc = total_acc / total_samples

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_samples = 0.0, 0.0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                bs = xb.size(0)
                val_samples += bs
                val_loss += loss.item() * bs
                val_acc += accuracy(logits, yb) * bs

        val_loss /= max(1, val_samples)
        val_acc /= max(1, val_samples)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f}")

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_samples = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = xb.size(0)
            test_samples += bs
            test_loss += loss.item() * bs
            test_acc += accuracy(logits, yb) * bs
    test_loss /= max(1, test_samples)
    test_acc /= max(1, test_samples)

    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    print(f"Test Loss:    {test_loss:.4f}")
    print(f"Test Acc:     {test_acc:.4f}")
    return best_val_acc, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--t", type=int, default=20)
    parser.add_argument("--m", type=int, default=15)
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--density", type=float, default=0.2, help="Target global known density")
    parser.add_argument("--num_agents", type=int, default=5, help="Only used to build the global mask")
    parser.add_argument("--realizations", type=int, default=10)
    parser.add_argument("--structured", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    cfg = Config(
        N=args.N, t=args.t, m=args.m, r=args.r, density=args.density, num_agents=args.num_agents,
        realizations=args.realizations, structured=args.structured, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        train_frac=args.train_frac, val_frac=args.val_frac
    )
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build data
    X, y, temporal_data, sensing = build_dataset(cfg)
    in_dim = X.shape[1]
    num_classes = cfg.m

    # Splits
    train_set, val_set, test_set = make_splits(X, y, cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    # Model
    model = LinearHead(in_dim, num_classes)

    # Train & evaluate
    train_eval_loop(model, train_loader, val_loader, test_loader, cfg, device=device)


if __name__ == "__main__":
    main()
