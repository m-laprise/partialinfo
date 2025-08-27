"""
Notes about the CNN baseline:
The inductive bias of a regular CNN discards the signal that matters.
This is a column-wise CNN to retain within column dependency information.
"""

import argparse
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# --- Assume datagen_temporal is in the parent directory ---
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

try:
    import datagen_temporal as dt
except ImportError:
    print("Error: Could not import 'datagen_temporal'.")
    sys.exit(1)

# ---------- Configuration ----------

@dataclass
class Config:
    """Configuration for the CNN classification script."""
    # Data generation
    N: int = 2000
    t: int = 50
    m: int = 25
    r: int = 25
    realizations: int = 1
    seed: int = 42
    
    # Sensing mask
    num_agents: int = 5
    density: float = 0.5
    
    # CNN Training Hyperparameters
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 20
    train_frac: float = 0.8
    val_frac: float = 0.1

# ---------- Helper Functions & Classes ----------

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ColwiseCNN(nn.Module):
    def __init__(self, t, m, k=5, hidden=32):
        super().__init__()
        assert k % 2 == 1
        # 2 channels: values + mask (so the model knows what's hidden)
        # Temporal-only convs: kernel spans time, not columns
        self.block = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=(k, 1), padding=(k // 2, 0)),  # no column mixing
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=(k, 1), padding=(k // 2, 0)),
            nn.ReLU(inplace=True),
        )
        
        self.time_pool = nn.AdaptiveAvgPool2d((1, None))  # (B, hidden, 1, m)
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)   # (B, 1, 1, m)

    def forward(self, x):            # x: (B, 2, t, m)
        h = self.block(x)            # (B, hidden, t, m)   (t unchanged by design)
        h = self.time_pool(h)        # (B, hidden, 1, m)
        logits = self.head(h)        # (B, 1, 1, m)
        return logits.squeeze(1).squeeze(1)  # (B, m)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)                    # (B, m)
        loss = criterion(outputs, labels)          # CE expects [B,m] vs [B]
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs            # weight by batch size
        total += bs

        preds = outputs.detach().argmax(dim=1)    # no .data
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Build data with the provided utilities (no manual labels)
    print("Generating data...")
    matrices = dt.GTMatrices(
        N=cfg.N, t=cfg.t, m=cfg.m, r=cfg.r,
        structured=True, realizations=cfg.realizations, seed=cfg.seed
    )
    temporal_data = dt.TemporalData(matrices, verbose=False)

    # Global sensing masks (use future_only=True to hide last row)
    sensing = dt.SensingMasks(
        temporal_data, rank=cfg.r, num_agents=cfg.num_agents,
        density=cfg.density, future_only=True
    )
    
    # Build global mask once 
    gm_flat = sensing.global_known.to(torch.float32)     # shape: (t*m,)
    gm_img1 = gm_flat.view(cfg.t, cfg.m).unsqueeze(0).unsqueeze(0)  # (1,1,t,m)

    # 2) Split indices once (no reprocessing of labels/masks)
    N_total = len(temporal_data)
    train_N = int(cfg.train_frac * N_total)
    val_N = int(cfg.val_frac * N_total)

    all_idx = torch.randperm(N_total)
    train_idx = all_idx[:train_N]
    val_idx = all_idx[train_N:train_N + val_N]
    test_idx = all_idx[train_N + val_N:]

    # 3) Collate: apply global mask and reshape for CNN; use labels from TemporalData
    def collate_mask_reshape(batch):
        X_flat = torch.stack([b['matrix'].squeeze(0) for b in batch], dim=0)  # (B, t*m), float later
        y = torch.tensor([int(b['label']) for b in batch], dtype=torch.long)   # (B,)

        gm_flat_b = gm_flat.unsqueeze(0).expand(X_flat.size(0), -1)            # (B, t*m)
        X_masked_flat = X_flat * gm_flat_b

        B = X_masked_flat.size(0)
        X_val = X_masked_flat.view(B, 1, cfg.t, cfg.m).to(torch.float32)       # (B,1,t,m)
        X_msk = gm_img1.expand(B, -1, -1, -1).to(torch.float32)                # (B,1,t,m)
        X = torch.cat([X_val, X_msk], dim=1)                                   # (B,2,t,m)
        return X, y

    # 4) DataLoaders built directly from TemporalData (no custom dataset)
    train_loader = DataLoader(Subset(temporal_data, train_idx),
                              batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_mask_reshape)
    val_loader = DataLoader(Subset(temporal_data, val_idx),
                            batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate_mask_reshape)
    test_loader = DataLoader(Subset(temporal_data, test_idx),
                             batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_mask_reshape)
    print(f"Data splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # 5) Model / loss / optim
    model = ColwiseCNN(t=cfg.t, m=cfg.m).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # 6) Train / validate
    best_val_acc = -1.0
    print("\nStarting training...")
    for epoch in range(cfg.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{cfg.epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> New best model saved with accuracy: {best_val_acc:.4f}")

    # 7) Final eval
    print("\nTraining finished. Loading best model for final evaluation...")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    _, train_acc = evaluate(model, train_loader, criterion, device)
    _, val_acc = evaluate(model, val_loader, criterion, device)
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a CNN baseline for temporal prediction.")
    # Dynamically add arguments from the Config dataclass
    for field_name, field_type in Config.__annotations__.items():
        default_val = Config.__dataclass_fields__[field_name].default
        if field_type is bool:
            parser.add_argument(f'--{field_name}', action='store_true', default=default_val)
        else:
            parser.add_argument(f'--{field_name}', type=field_type, default=default_val)
    
    args = parser.parse_args()
    cfg = Config(**vars(args))
    main(cfg)