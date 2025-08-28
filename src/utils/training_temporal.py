import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


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
    
    
def stacked_MSE(predictions: torch.Tensor,
                targets: torch.Tensor,
                reduction: str = 'mean') -> torch.Tensor:
    """
    Computes MSE loss over stacked agent predictions of last row and outcome.
    predictions: [B, A, m + y_dim]
    targets: [B, m + y_dim]
    
    Returns torch.Tensor: Scalar loss (if reduced) or tensor of shape [batch_size, A] (if 'none').
    """
    if predictions.dim() != 3:
        raise ValueError(f"Expected logits with 3 dims [batch_size, n_agents, m + y_dim], got {predictions.shape}")
    if predictions.shape[-1] != targets.shape[-1]:
        raise ValueError("Mismatched predictions and targets, loss cannot be computed.")
    if targets.dim() != 2:
        raise ValueError(f"Expected targets with 2 dims [batch_size, m + y_dim], got {targets.shape}")
    
    B, A, m_y = predictions.shape # m_y = m + 1
    
    # Expand and reshape 
    expanded_targets = targets.unsqueeze(1).expand(-1, A, -1).reshape(-1, m_y)  # [B * A, m + y_dim]
    predictions_flat = predictions.reshape(-1, m_y)                             # [B * A, m + y_dim]
    # Compute per-prediction MSE loss
    losses = nn.MSELoss(reduction='none')(predictions_flat, expanded_targets)   # [B * A, m + y_dim]

    # Apply reduction
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.mean(dim=1).view(B, A)                                    # [B, A]
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")


def train(model, aggregator, loader, optimizer, criterion, device, scaler: Optional[GradScaler]):
    model.train()
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
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(aggregator.parameters()),
                max_norm=1.0
            )
            scaler.step(optimizer)          # type: ignore
            scaler.update()                 # type: ignore
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(aggregator.parameters()),
                max_norm=1.0
            )
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


def regression_acc_agree(logits, target):    
    """
    For each example, take average prediction and compute per-entry MSE of the average with the targer.
    For each example, compute entropy of distribution of agent-level values.
    Return batch sum for each quantity.
    Args: 
        logits: [B, A, m or y_dim]
        target: [B, m or y_dim]
    """
    # Accept 2D inputs for scalar target case by promoting to 3D/2D as needed.
    if logits.dim() == 2:         # [B, A] -> [B, A, 1]
        logits = logits.unsqueeze(-1)
    if target.dim() == 1:         # [B]    -> [B, 1]
        target = target.unsqueeze(-1)

    B, A, D = logits.shape

    # Average prediction across agents: [B, D]
    avg_preds = logits.mean(dim=1)

    # Per-example, per-entry MSE -> [B]; then sum across batch
    mse_per_example = ((avg_preds - target) ** 2).mean(dim=1)  # mean over D
    batch_mse_sum = mse_per_example.sum().item()

    # Inter-agent diversity: Gaussian differential entropy from across-agent variance
    # var shape: [B, D]
    var = logits.var(dim=1, unbiased=False)
    eps = 1e-12
    entropy_per_entry = 0.5 * torch.log(torch.clamp(var, min=eps) * (2.0 * math.pi * math.e))
    entropy_per_example = entropy_per_entry.mean(dim=1)        # mean over D
    batch_entropy_sum = entropy_per_example.sum().item()

    return batch_mse_sum, batch_entropy_sum


@torch.inference_mode()
def evaluate(model, aggregator, loader, criterion, device, *, task, max_batches=None):
    """
    Evaluation for stacked and collective predictions. 
    For classification, returns collective accuracy and inter-agent agreement (% in plurality).
    For regression, returns averaged across agents per-entry MSE and inter-agent prediction diversity
    (entropy), for both next row (m-dimensional) and outcome (1-dimensional).
    """
    model.eval()
    aggregator.eval()
    autocast_ctx = autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()
    
    total_loss = 0.0
    total_examples = 0
    if task == 'classif':
        total_acc = 0.0
        total_agree = 0.0
    elif task == 'regression':
        total_mse_m = 0.0
        total_entropy_m = 0.0
        total_mse_y = 0.0
        total_entropy_y = 0.0
    else:
        raise NotImplementedError(f"Task {task} not implemented")

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        x = batch['matrix'].to(device, non_blocking=True)  # shape: [B, t * m]
        target = batch['label'].to(device, non_blocking=True)  # shape: [B] or [B, m + y_dim]

        with autocast_ctx:
            logits = aggregator(model(x))          # [B, A, C] or [B, A, m + y_dim]
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
            B, m_ydim = target.shape
            m = m_ydim - 1
            batch_mse_m, batch_entropy_m = regression_acc_agree(logits[:, :, :m], target[:, :m])
            batch_mse_y, batch_entropy_y = regression_acc_agree(logits[:, :, -1], target[:, -1])
            total_mse_m += batch_mse_m
            total_entropy_m += batch_entropy_m
            total_mse_y += batch_mse_y
            total_entropy_y += batch_entropy_y

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
        mean_mse_m = total_mse_m / max(total_examples, 1)
        mean_entropy_m = total_entropy_m / max(total_examples, 1)
        mean_mse_y = total_mse_y / max(total_examples, 1)
        mean_entropy_y = total_entropy_y / max(total_examples, 1)
    
        return (
            float(mean_loss), 
            float(mean_mse_m), 
            float(mean_entropy_m),
            float(mean_mse_y),
            float(mean_entropy_y)
        )


def init_stats(task_cat):
    if task_cat == 'classif':
        return {
            "train_loss": [],
            "t_accuracy": [],
            "t_agreement": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_agreement": []
        }
    else:
        return {
            "train_loss": [],
            "t_mse_m": [],
            "t_entropy_m": [],
            "t_mse_y": [],
            "t_entropy_y": [],
            "val_loss": [],
            "val_mse_m": [],
            "val_entropy_m": [],
            "val_mse_y": [],
            "val_entropy_y": []
        }
    
def printlog(task, epoch, stats):
    if task == 'classif':
        print(f"Ep {epoch:03d}. ",
              f"T loss: {stats['train_loss'][-1]:.2e} | T acc: {stats['t_accuracy'][-1]:.2f} | T % maj: {stats['t_agreement'][-1]:.2f} | ",
              f"V loss: {stats['val_loss'][-1]:.2e} | V acc: {stats['val_accuracy'][-1]:.2f} | V % maj: {stats['val_agreement'][-1]:.2f}")
    else:
        print(f"Ep {epoch:03d}. ",
              f"T loss: {stats['train_loss'][-1]:.2e} | T mse_m: {stats['t_mse_m'][-1]:.2e} | T entropy_m: {stats['t_entropy_m'][-1]:.2e} | ",
              f"T mse_y: {stats['t_mse_y'][-1]:.2e} | T entropy_y: {stats['t_entropy_y'][-1]:.2e} | ",
              f"V loss: {stats['val_loss'][-1]:.2e} | V mse_m: {stats['val_mse_m'][-1]:.2e} | V entropy_m: {stats['val_entropy_m'][-1]:.2e} | ",
              f"V mse_y: {stats['val_mse_y'][-1]:.2e} | V entropy_y: {stats['val_entropy_y'][-1]:.2e}")
    pass
