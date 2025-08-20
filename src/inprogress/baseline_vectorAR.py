#!/usr/bin/env python3
"""
Baseline VAR(1) model for predicting the index of the column with the maximal
value in row t (the last row) of M = U V^T.

This script implements a simple, non-neural baseline. For each data matrix,
it fits an independent Vector Autoregression (VAR) model of order 1. The model
is trained on the historical, partially-observed entries of the matrix to learn
a transition matrix A. It then uses the last observed row to predict the
final row and evaluates the classification accuracy of the argmax.

The VAR(1) model is of the form:
    M[t, :] ≈ M[t-1, :] @ A

where A is an (m x m) coefficient matrix fit via ridge regression on the
observed entries.

Usage (defaults shown):
    python src/baseline_var.py --N 1000 --t 20 --m 15 --r 5 --density 0.2
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List

import torch

# Ensure the parent directory is in the system path to find datagen_temporal
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

try:
    import datagen_temporal as dt
except ImportError:
    print("Error: Could not import 'datagen_temporal'.")
    print("Please ensure the parent directory containing 'datagen_temporal.py' is accessible.")
    sys.exit(1)

@dataclass
class Config:
    """Configuration for the VAR baseline script."""
    # Data generation
    N: int = 1000
    t: int = 50
    m: int = 25
    r: int = 25
    realizations: int = 10
    seed: int = 42
    
    # Sensing mask
    num_agents: int = 5
    density: float = 0.5
    
    # Model and Evaluation
    train_frac: float = 0.8
    lambdas: List[float] = field(default_factory=lambda: [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 5e2, 1e3, 5e3, 1e4])


def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_splits(dataset_size, train_frac):
    """Splits a dataset size into train, validation, and test indices."""
    assert train_frac + (1.0 / dataset_size) < 1.0
    train_size = int(train_frac * dataset_size)
    
    indices = torch.randperm(dataset_size).tolist()
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size :]
    
    return train_idx, val_idx

def _ridge_solve(X, y, lam=1e-2):
    """
    Solve (X^T X + lam I) w = X^T y for w [ridge/closed-form]
    X: (n, d), y: (n,)
    Returns w: (d,)
    """
    d = X.shape[1]
    XT_X = X.T @ X
    XT_y = X.T @ y
    # Use torch.linalg.solve for numerical stability
    w = torch.linalg.solve(XT_X + lam * torch.eye(d, device=X.device, dtype=X.dtype), XT_y)
    return w

def argmax_column(tensor_1d):
    """Returns the integer index of the maximum value in a 1D tensor."""
    return int(torch.argmax(tensor_1d).item())

def maxcol_label(M):
    """Ground-truth label is argmax of the final row (time t)."""
    return int(torch.argmax(M[-1, :]).item())


def fit_var1_coeffs_for_matrix(M, G, lam=1e-2):
    """
    Fit a VAR(1) on a single matrix representing the history of a data point.
    M:     (t-1, m) ground-truth matrix history
    G:     (t-1, m) boolean global-known mask for the history
    lam:   ridge strength

    Strategy:
      - Build X = rows 0..t-2 (predictors), Y = rows 1..t-1 (targets).
      - For each target dimension j, fit a ridge regression using only rows where Y[:, j] is observed.
      - Missing predictors in X are set to 0 (masked out).
    Returns:
      A: (m, m) coefficient matrix for VAR(1) [y_t ≈ x_{t-1} @ A]
    """
    t, m = M.shape
    assert t >= 2, "Need at least 2 time steps for VAR(1) fitting."

    X_raw = M[:-1, :]                      # (t-2, m)
    Y_raw = M[1:, :]                       # (t-2, m)
    X_mask = G[:-1, :].to(M.dtype)         # (t-2, m)
    Y_mask = G[1:, :].to(M.dtype)          # (t-2, m)

    # Zero-out missing predictors; we'll select rows per target for Y.
    X = X_raw * X_mask
    A = torch.zeros((m, m), dtype=M.dtype, device=M.device)

    for j in range(m):
        keep = Y_mask[:, j] > 0.5
        if not torch.any(keep):
            A[j, j] = 1e-3  # Fallback: tiny self-weight if no observations
            continue

        Xj = X[keep, :]
        yj = Y_raw[keep, j]
        
        # Ridge regression keeps this stable even if Xj is rank-deficient
        wj = _ridge_solve(Xj, yj, lam=lam)
        A[:, j] = wj

    return A

# NEW generalized function
def fit_varp_coeffs(M, G, p, lam=1e-2):
    """
    Fit a VAR(p) on a single matrix representing the history of a data point.
    """
    t, m = M.shape
    assert t >= p + 1, f"Need at least {p+1} time steps for VAR({p}) fitting."

    # Create lagged predictors X and targets Y
    # Y will be rows p, p+1, ..., t-1
    # X will be the corresponding concatenated p previous rows
    Y_raw = M[p:, :]
    Y_mask = G[p:, :].to(M.dtype)

    # Build the predictor matrix X by stacking p lagged vectors for each target
    X_list = []
    X_mask_list = []
    for i in range(p, t):
        # For target at time i, predictor is concatenation of i-1, i-2, ..., i-p
        lagged_vectors = M[i-p:i, :].flip(dims=[0]).reshape(1, -1) # [x_{i-1}, x_{i-2}, ...]
        lagged_masks = G[i-p:i, :].flip(dims=[0]).reshape(1, -1)
        X_list.append(lagged_vectors)
        X_mask_list.append(lagged_masks)
    
    X_raw = torch.cat(X_list, dim=0)      # (t-p, p*m)
    X_mask = torch.cat(X_mask_list, dim=0).to(M.dtype) # (t-p, p*m)

    # Zero-out missing predictors
    X = X_raw * X_mask
    A = torch.zeros((p * m, m), dtype=M.dtype, device=M.device) # Note the new shape

    for j in range(m):
        keep = Y_mask[:, j] > 0.5
        if not torch.any(keep):
            continue # Leave column as zeros if no targets observed

        Xj = X[keep, :]
        yj = Y_raw[keep, j]
        
        wj = _ridge_solve(Xj, yj, lam=lam) # wj has shape (p*m,)
        A[:, j] = wj

    return A


# MODIFIED function
def predict_next_row(M_hist, G_hist, A, p):
    """
    Predict the next row using the last 'p' observed rows of the history.
    """
    # Create the predictor by concatenating the last p masked rows of history
    # [x_{t-2}, x_{t-3}, ..., x_{t-p-1}]
    predictor_vecs = []
    for i in range(1, p + 1):
        vec = M_hist[-i, :] * G_hist[-i, :].to(M_hist.dtype)
        predictor_vecs.append(vec)
    
    x_last_p_observed = torch.cat(predictor_vecs) # Shape (p*m,)
    y_pred = x_last_p_observed @ A                # (p*m,) @ (p*m, m) -> (m,)
    return y_pred


def eval_split(indices, temporal_data, sensing, lam, p):
    """
    Fit a per-matrix VAR(1), predict the final row, and compute accuracy.
    """
    correct, total = 0, 0
    t, m = temporal_data.t, temporal_data.m
    G = sensing.global_known.view(t, m)
    
    for idx in indices:
        M = temporal_data.data[idx]

        # Define the history (all data except the final row to be predicted)
        M_hist = M[:-1, :]
        G_hist = G[:-1, :]

        # Fit A on the history
        A = fit_varp_coeffs(M_hist, G_hist, p=p, lam=lam) # Call the new function
        
        # Predict the final row using the history
        y_pred = predict_next_row(M_hist, G_hist, A, p=p) # Pass p here
        
        # Get prediction and ground truth labels
        y_hat = argmax_column(y_pred)
        y_true = maxcol_label(M)

        correct += int(y_hat == y_true)
        total += 1
        
    return correct / max(1, total)

# ---------- Main Execution ----------

def main(cfg):
    """Main function to run the VAR baseline experiment."""
    set_seed(cfg.seed)
    print("Configuration:\n", cfg)

    # 1. Generate Data
    print("\nGenerating data...")
    matrices = dt.GTMatrices(
        N=cfg.N, t=cfg.t, m=cfg.m, r=cfg.r,
        structured=True, realizations=cfg.realizations, seed=cfg.seed
    )
    temporal_data = dt.TemporalData(matrices, verbose=False)
    sensing = dt.SensingMasks(
        temporal_data, rank=cfg.r, num_agents=cfg.num_agents,
        density=cfg.density, future_only=True
    )
    print(f"Data generated: {cfg.N} samples of shape ({cfg.t}, {cfg.m})")

    # 2. Create data splits
    train_idx, val_idx = make_splits(cfg.N, cfg.train_frac)
    print(f"Data splits: Train={len(train_idx)}, Val={len(val_idx)}")

    # 3. Experimental Loop
    orders_to_test = [1, 2, 3, 4, 5, 10, 15, 20]
    print(f"\nTesting VAR orders: {orders_to_test}")

    for p in orders_to_test:
        print(f"\n----- Evaluating VAR({p}) -----")
        
        # 1. Tune lambda on the training set for this p
        print(f"Tuning lambda for VAR({p}) on training set...")
        best_lam = None
        best_train_acc = -1.0

        for lam in cfg.lambdas:
            # Pass the current order 'p' to the evaluation function
            train_acc = eval_split(train_idx, temporal_data, sensing, p=p, lam=lam)
            print(f"  Train — p={p}, lam={lam:g}: acc={train_acc:.4f}")
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_lam = lam

        print(f"Best lambda for VAR({p}): {best_lam:g} with train_acc: {best_train_acc:.4f}")

        # 2. Evaluate on the test sets with the best lambda for this p
        val_acc = eval_split(val_idx, temporal_data, sensing, p=p, lam=best_lam)
        
        print(f"\n--- Results for VAR({p}) ---")
        print(f"  Train Accuracy:   {best_train_acc:.4f}")
        print(f"  Val Accuracy:     {val_acc:.4f}")
        print("------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a VAR(1) baseline for temporal prediction.")
    # Add arguments from the Config dataclass
    # This allows for easy command-line overrides of default parameters
    for field_name, field_type in Config.__annotations__.items():
        if field_type == bool:
            parser.add_argument(f'--{field_name}', action='store_true', help=f'Enable {field_name}')
        else:
            parser.add_argument(f'--{field_name}', type=field_type, help=f'Set {field_name}')
    
    args = parser.parse_args()
    
    # Create a config instance, updating with any command-line arguments
    cfg = Config()
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            setattr(cfg, arg_name, arg_value)
            
    main(cfg)
    