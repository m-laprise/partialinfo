"""
Notes about the CNN baseline:
This performs at chance level. This is expected.
This is because the inductive bias of the CNN discards the signal that matters: 
per-column identity over time. The traditional CNN generally does the following:
	1.	Mixes columns early (kernels wider than 1 along the column axis or pooling across columns),
	2.	Collapses column identity (global avg/max pooling over the column axis, or flattening that no longer maps “this position = this class”), and
	3.	Confuses zeros from masking with true values (no mask channel), so it learns invariances that make the target unrecoverable.
permutation invariance across columns = cannot say which column wins = no usable signal
"""

import argparse
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

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
    learning_rate: float = 1e-3
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

class MatrixDataset(Dataset):
    """Custom PyTorch Dataset for matrix data."""
    def __init__(self, data_matrices, masks, labels, device):
        self.device = device
        # Apply the mask to the data
        self.masked_data = (data_matrices * masks).to(torch.float32)
        self.labels = labels.to(torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Add a channel dimension for the CNN
        matrix = self.masked_data[idx].unsqueeze(0)
        label = self.labels[idx]
        return matrix.to(self.device), label.to(self.device)

# ---------- The CNN Model ----------

class SimpleCNN(nn.Module):
    """A simple CNN to classify the argmax of the final row."""
    def __init__(self, t, m):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the size after convolutions and pooling
        conv_output_t = t // 2
        conv_output_m = m // 2
        flattened_size = 32 * conv_output_t * conv_output_m
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, m)  # Output layer with 'm' neurons for 'm' classes
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        logits = self.classifier(x)
        return logits

# ---------- Training & Evaluation Logic ----------

def train_one_epoch(model, dataloader, criterion, optimizer):
    """Runs a single training epoch."""
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, criterion):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


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
    sensing.to(device)

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
        # batch[i]['matrix'] is (1, t*m); remove channel -> (t*m)
        X_flat = torch.stack([b['matrix'].squeeze(0) for b in batch], dim=0)   # (B, t*m)
        y = torch.tensor([int(b['label']) for b in batch], dtype=torch.long)

        gm = sensing.global_known.to(X_flat.device).view(1, -1)                # (1, t*m)
        X_masked_flat = X_flat * gm.expand(X_flat.size(0), -1)                 # (B, t*m)

        # reshape for the CNN
        B = X_masked_flat.size(0)
        X_img = X_masked_flat.view(B, cfg.t, cfg.m).unsqueeze(1).to(torch.float32)  # (B, 1, t, m)

        return X_img.to(device), y.to(device)

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
    model = SimpleCNN(t=cfg.t, m=cfg.m).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # 6) Train / validate
    best_val_acc = -1.0
    print("\nStarting training...")
    for epoch in range(cfg.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{cfg.epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> New best model saved with accuracy: {best_val_acc:.4f}")

    # 7) Final eval
    print("\nTraining finished. Loading best model for final evaluation...")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    _, train_acc = evaluate(model, train_loader, criterion)
    _, val_acc = evaluate(model, val_loader, criterion)
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a CNN baseline for temporal prediction.")
    # Dynamically add arguments from the Config dataclass
    for field_name, field_type in Config.__annotations__.items():
        default_val = Config.__dataclass_fields__[field_name].default
        if field_type == bool:
            parser.add_argument(f'--{field_name}', action='store_true', default=default_val)
        else:
            parser.add_argument(f'--{field_name}', type=field_type, default=default_val)
    
    args = parser.parse_args()
    cfg = Config(**vars(args))
    main(cfg)