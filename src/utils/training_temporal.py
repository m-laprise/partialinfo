
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from dotGAT import DynamicDotGAT


def _btm_from_batch(batch, t: int, m: int, device: torch.device) -> torch.Tensor:
    x = batch['matrix'].to(device, non_blocking=True)
    if x.dim() == 2 and x.shape[1] == t * m:
        return x.view(-1, t, m)
    if x.dim() == 3 and x.shape[-1] == t * m:
        B = x.shape[0]
        return x.view(B, t, m)
    if x.dim() == 3 and x.shape[1:] == (t, m):
        return x
    raise ValueError(f"Unexpected matrix shape {tuple(x.shape)}; expected [B,t*m], [B,1,t*m], or [B,t,m]")


def _bty_labels_from_batch(batch, t: int, device: torch.device) -> torch.Tensor:
    y = batch['label'].to(device, non_blocking=True)  # [B,T,y_dim]
    if y.dim() == 3 and y.shape[1] == t:
        return y
    raise ValueError(f"Unexpected label shape {tuple(y.shape)}; expected [B,T,y_dim]")


def stacked_cross_entropy_loss(logits: torch.Tensor,
                               targets: torch.Tensor,
                               reduction: str = 'mean',
                               label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Computes cross-entropy loss over stacked agent logits.
    logits:  [B, A, m]
    targets: [B] (class indices)

    Returns torch.Tensor: Scalar loss (if reduced) or tensor of shape [B, A] (if 'none').
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits with 3 dims [B, A, m], got {logits.shape}")
    if targets.dim() != 1:
        raise ValueError(f"Expected targets with 1 dim [B], got {targets.shape}")
    
    B, A, n_classes = logits.shape
    
    if targets.dtype != torch.long:
        targets = targets.long()
    if torch.any((targets < 0) | (targets >= n_classes)):
        raise ValueError("targets contain indices outside [0, n_classes-1].")
    
    # Expand and reshape targets and logits
    expanded_targets = targets.unsqueeze(1).expand(-1, A).reshape(-1)  # [B * A]
    logits_flat = logits.reshape(-1, n_classes)                        # [B * A, m]
    # Compute per-prediction cross-entropy loss
    losses = nn.CrossEntropyLoss(
        reduction='none',
        label_smoothing=label_smoothing
    )(logits_flat, expanded_targets)                                   # [B * A]
    losses = losses.view(B, A)                         # [B, A]
    # Apply reduction
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")
    

import torch


def stacked_MSE(predictions: torch.Tensor,
                targets: torch.Tensor,
                reduction: str = 'mean') -> torch.Tensor:
    """
    Computes MSE loss over stacked agent predictions.

    Returns:
        Scalar loss (if reduced) or tensor of shape [B, A] (if 'none').
    """
    if predictions.shape[-1] != targets.shape[-1]:
        raise ValueError(f"Mismatched predictions ({predictions.shape}) and targets ({targets.shape}), loss cannot be computed.")

    if predictions.dim() == 4:
        # predictions: [B, T, A, y_dim]
        # targets:     [B, T, y_dim]
        B, T, A, y_dim = predictions.shape
        if targets.shape != (B, T, y_dim):
            raise ValueError(f"For 4D predictions expected targets shape [B, T, y_dim], got {targets.shape}")

        # elementwise squared error: [B, T, A, y_dim]
        errors = (predictions - targets.unsqueeze(2)) ** 2

        if reduction == 'mean':
            return errors.mean()
        elif reduction == 'sum':
            return errors.sum()
        elif reduction == 'none':
            # average over y_dim then over time -> [B, A]
            return errors.mean(dim=-1).mean(dim=1)
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")

    elif predictions.dim() == 3:
        # predictions: [B, A, y_dim]
        # targets:     [B, y_dim]
        B, A, y_dim = predictions.shape
        if targets.shape != (B, y_dim):
            raise ValueError(f"For 3D predictions expected targets shape [B, y_dim], got {targets.shape}")

        errors = (predictions - targets.unsqueeze(1)) ** 2  # [B, A, y_dim]

        if reduction == 'mean':
            return errors.mean()
        elif reduction == 'sum':
            return errors.sum()
        elif reduction == 'none':
            # average over y_dim -> [B, A]
            return errors.mean(dim=-1)
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")

    else:
        raise ValueError(f"Expected predictions with 3 dims [B, n_agents, y_dim] or 4 dims [B, time, n_agents, y_dim], got {predictions.shape}")


def train(model, aggregator, loader, optimizer, criterion, device, scaler: Optional[GradScaler], *, t, m):
    model.train()
    if aggregator is not None:
        aggregator.train()
    use_amp = (device.type == 'cuda') and (scaler is not None)
    autocast_ctx = autocast(device_type='cuda') if use_amp else nullcontext()
    
    total_loss = 0.0
    total_examples = 0
    
    for batch in loader:
        if device.type == 'cuda':
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except Exception:
                pass

        optimizer.zero_grad(set_to_none=True)

        if hasattr(model, "W_q_mem"):
            x_btm = _btm_from_batch(batch, t, m, device)           # [B,T,M]
            target = _bty_labels_from_batch(batch, t, device)       # [B,T,y_dim]
            B = target.size(0)
            with autocast_ctx:
                h, y_btay = model(x_btm)        # y: [B,T,A,y_dim]
                loss = criterion(y_btay, target, reduction ='mean')
        else:
            x = batch['matrix'].to(device, non_blocking=True)       # [batch_size, t * m]
            target = batch['label'].to(device, non_blocking=True)   # [batch_size]
            B = target.size(0)
            with autocast_ctx:
                logits = aggregator(model(x))
                loss = criterion(logits, target, reduction='mean')
            
        if use_amp:
            scaler.scale(loss).backward()   # type: ignore
            scaler.unscale_(optimizer)      # type: ignore
            # clip both model + aggregator
            if aggregator is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(aggregator.parameters()),
                    max_norm=1.0
                )
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)          # type: ignore
            scaler.update()                 # type: ignore
        else:
            loss.backward()
            if aggregator is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(aggregator.parameters()),
                    max_norm=1.0
                )
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * B
        total_examples += B
        
    return total_loss / max(total_examples, 1)
    

def classif_vote_acc_agree(logits, target):
    """
    For each example, take agent level logits, infer answers for classification task and derive majority vote.
    Compute and return sum of accuracy and agreement across examples.
    Args: 
        logits: [B, A, C]
        target: [B]
    """
    B, A, C = logits.shape
    agent_preds = logits.argmax(dim=-1)        # [B, A]
    
    counts = torch.zeros(B, C, device=logits.device, dtype=torch.int32)
    ones = torch.ones_like(agent_preds, dtype=torch.int32)
    counts.scatter_add_(1, agent_preds, ones)  # [B, C]

    majority_class = counts.argmax(dim=1)      # [B]
    vote_accuracy = (majority_class == target).float().mean().item()
    batch_acc = vote_accuracy * B

    avg_majority_fraction = (counts.max(dim=1).values.float() / A).mean().item()
    batch_agree = avg_majority_fraction * B
    return batch_acc, batch_agree


def regression_acc_agree(logits: torch.Tensor, target: torch.Tensor):
    """
    For each example, take average prediction and compute per-entry MSE of the average with the target.
    For each example, compute inter-agent variance of predictions.
    Return batch sum for each quantity.
    Args:
        logits: [B, ..., A, D]   (A = n_agents, D = m or y_dim)
        target: [B, ..., D]
    Returns:
        (batch_mse_sum: float, batch_variance_sum: float)
    """
    # Accept 2D inputs for scalar target by promoting as needed
    if logits.dim() == 2:          # [B, A] -> [B, A, 1]
        logits = logits.unsqueeze(-1)
    if target.dim() == 1:         # [B] -> [B, 1]
        target = target.unsqueeze(-1)

    if logits.dim() < 3:
        raise ValueError(f"Expected logits with at least 3 dims [B, ..., A, D], got {tuple(logits.shape)}")

    # Shape-check: target must match logits except for the agent axis
    expected_target_shape = tuple(logits.shape[:-2]) + (logits.shape[-1],)
    if tuple(target.shape) != expected_target_shape:
        raise ValueError(f"Shape mismatch: expected target shape {expected_target_shape}, got {tuple(target.shape)}")

    # Average across agents -> [B, *extra, D]
    avg_preds = logits.mean(dim=-2)

    # MSE per example: flatten all non-batch dims and average per-row -> [B]
    mse_per_example = ((avg_preds - target).pow(2)).flatten(start_dim=1).mean(dim=1)
    batch_mse_sum = float(mse_per_example.sum().item())

    # Inter-agent variance: var across agents -> [B, *extra, D], then flatten & mean as above
    A = logits.shape[-2]
    if A == 1:
        batch_variance_sum = 0.0
    else:
        var_across_agents = logits.var(dim=-2, unbiased=False)
        variance_per_example = var_across_agents.flatten(start_dim=1).mean(dim=1)
        batch_variance_sum = float(variance_per_example.sum().item())

    return batch_mse_sum, batch_variance_sum


@torch.inference_mode()
def evaluate(model, aggregator, loader, criterion, device, *, task, t, m, max_batches=None):
    """
    Evaluation for stacked and collective predictions. 
    For classification, returns collective accuracy and inter-agent agreement (% in plurality).
    For regression, returns averaged across agents per-entry MSE and inter-agent prediction diversity
    (entropy), for both next row (m-dimensional) and outcome (1-dimensional).
    """
    model.eval()
    if aggregator is not None:
        aggregator.eval()
    autocast_ctx = autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()
    
    total_loss = 0.0
    total_examples = 0
    if task == 'classif':
        total_acc = 0.0
        total_agree = 0.0
    elif task == 'regression':
        total_mse = 0.0
        total_diversity = 0.0
    else:
        raise NotImplementedError(f"Task {task} not implemented")

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        if isinstance(model, DynamicDotGAT):
            x_btm = _btm_from_batch(batch, t, m, device)
            target = _bty_labels_from_batch(batch, t, device)
            B = target.size(0)
            with autocast_ctx:
                h, logits = model(x_btm)
                loss = criterion(logits, target, reduction ='mean')
        else:
            x = batch['matrix'].to(device, non_blocking=True)  # shape: [B, t * m]
            target = batch['label'].to(device, non_blocking=True)  # shape: [B] or [B, y_dim]

            with autocast_ctx:
                logits = aggregator(model(x)) # [B, A, C] or [B, A, y_dim]
                loss = criterion(logits, target, reduction='mean')
            
        B = target.size(0)
        total_loss += loss.item() * B
        total_examples += B

        # Compute % accuracy
        # Take majority vote
        if task == 'classif':
            batch_acc, batch_agree = classif_vote_acc_agree(logits, target)
            total_acc += batch_acc
            total_agree += batch_agree
        else:
            #B,y or B,t,y target
            batch_mse, batch_diversity = regression_acc_agree(logits, target)
            total_mse += batch_mse
            total_diversity += batch_diversity

    mean_loss = total_loss / max(total_examples, 1)
    if task == 'classif':
        mean_acc = total_acc / max(total_examples, 1)
        mean_agree = total_agree / max(total_examples, 1)
        
        return (
            float(mean_loss), 
            float(mean_acc), 
            float(mean_agree)
        )
    else:
        mean_mse = total_mse / max(total_examples, 1)
        mean_diversity = total_diversity / max(total_examples, 1)
    
        return (
            float(mean_loss), 
            float(mean_mse), 
            float(mean_diversity),
        )


def _last_nonzero_per_col(A: torch.Tensor,
                          mask: torch.Tensor,
                          default: float = 0.0) -> torch.Tensor:
    """
    Takes a batch of masked t x m matrices and returns of batch of vectors of length m
    containing the last seen entry for each column.
    A : (B, t, m) tensor
    mask : bool tensor of same shape (t, m); True where entry counts as "non-zero"
    default : scalar to place for columns with no True in mask.
    Returns: 1D tensor of length m with dtype A.dtype on the same device as A.
    """
    B, t, m = A.shape
    device = A.device
    dtype = A.dtype

    if mask.dtype != torch.bool:
        mask = mask.bool()
        
    mask = mask.unsqueeze(0).expand(B, t, m)
    # row indices broadcasted to (B, t, m)
    rows = torch.arange(t, device=device, dtype=torch.long).view(1, t, 1).expand(B, t, m)
    # index matrix: row index where mask True, else -1
    idx = torch.where(mask, rows, torch.tensor(-1, device=device, dtype=torch.long))

    # last index per (B,m) (-1 means none)
    last_idx = idx.max(dim=1).values          # shape (B, m), long
    # prepare output filled with default
    values = torch.full((B, m), default, dtype=dtype, device=device)
    # fill valid positions
    valid = last_idx >= 0                     # (B, m) bool
    if valid.any():
        batch_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, m)
        col_idx = torch.arange(m, device=device, dtype=torch.long).unsqueeze(0).expand(B, m)
        values[valid] = A[batch_idx[valid], last_idx[valid], col_idx[valid]]

    return values


def baseline_mse_m(dataloader, globalmask, t, m, reduction = 'mean'):
    """
    Returns the naive baseline with partial and full information.
    Naive means predicting x_{t + 1} = x_{t}.
    
    For full information, this is the MSE resulting from predicting the last row of each matrix 
    to be the same as the row before last (including true values for masked entries).
    
    For partial information, masked entries are replaced by the last known element of each column.
    """
    total_mse_full = 0.0
    total_mse_partial = 0.0
    total_examples = 0
    for batch in dataloader:
        x = batch['matrix']
        B, _, _ = x.shape
        x = x.view(B, t, m)
        targets = x[:, -1, :]
        
        predictions_full = x[:, -2, :]
        mse_full = nn.MSELoss(reduction=reduction)(predictions_full, targets)
        total_mse_full += mse_full.item() * B
        
        globalmask = globalmask.view(t, m)
        masked_x = torch.zeros_like(x)
        masked_x[:, globalmask] = x[:, globalmask]
        predictions_partial = _last_nonzero_per_col(masked_x, globalmask, default=0.0)
        mse_partial = nn.MSELoss(reduction=reduction)(predictions_partial, targets)
        total_mse_partial += mse_partial.item() * B
        
        total_examples += B
        
    return (
        total_mse_partial / max(total_examples, 1), 
        total_mse_full / max(total_examples, 1)
    )


def baseline_classif(dataloader, globalmask, t: int, m: int):
    """
    Returns the naive baseline with partial and full information.
    Naive means predicting x_{t + 1} = x_{t}.
    
    For full information, this is the % accuracy resulting from taking the last element of each 
    column (including true values for masked entries), taking the argmax, and predicting that to be 
    the next argmax.
    
    For partial information, this is the % accuracy resulting from taking the last known element of each column, taking the argmax, and predicting that to be the next argmax.
    """
    total_acc_partial = 0.0
    total_acc_full = 0.0
    total_examples = 0
    for batch in dataloader:
        target = batch['label']
        x = batch['matrix'].squeeze()
        B, _ = x.shape
        x = x.view(B, t, m)
        
        predictions_full = x[:, -2, :].argmax(dim=1)
        acc_full = (predictions_full == target).float().mean().item()
        total_acc_full += acc_full * B
        
        globalmask = globalmask.view(t, m)
        masked_x = torch.zeros_like(x)
        masked_x[:, globalmask] = x[:, globalmask]
        last_seen_row = _last_nonzero_per_col(masked_x, globalmask, default=0.0)
        predictions_partial = last_seen_row.argmax(dim=1)
        acc_partial = (predictions_partial == target).float().mean().item()
        total_acc_partial += acc_partial * B
        
        total_examples += B
    
    return (
        total_acc_partial / max(total_examples, 1), 
        total_acc_full / max(total_examples, 1)
    )
    

def final_test(model, aggregator, test_loader, globalmask, criterion, device, task_cat, cfg):
    if task_cat == 'classif':
        test_loss, test_acc, test_agree = evaluate(                             # type: ignore
            model, aggregator, test_loader, criterion, device, task=task_cat,
            t=cfg.t, m=cfg.m
        )
        test_stats = (test_loss, test_acc, test_agree)
        print("Test Set Performance | ",
              f"Loss: {test_loss:.2e}, Accuracy: {test_acc:.2f}, % maj: {test_agree:.2f}")
        naive_partial, naive_full = baseline_classif(
            test_loader, globalmask,
            cfg.t, cfg.m
        )
        print(f"Accuracy for naive prediction on test set, full info: {naive_full:.2f}")
        if cfg.memory is False:
            print(f"Accuracy for naive prediction on test set, partial info: {naive_partial:.2f}")
    else:
        test_loss, test_mse, test_diversity = evaluate(   # type: ignore
            model, aggregator, test_loader, criterion, device, task=task_cat,
            t=cfg.t, m=cfg.m
        )
        test_stats = (test_loss, test_mse, test_diversity)
        print("Test Set Performance | ",
              f"Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, Diversity: {test_diversity:.2f}")
        
        naive_partial, naive_full = baseline_mse_m(
            test_loader, globalmask,
            cfg.t, cfg.m
        )
        print(f"MSE for naive prediction on test set, full info: {naive_full:.4f}")
        if cfg.memory is False:
            print(f"MSE for naive prediction on test set, partial info: {naive_partial:.4f}")
    return test_stats
