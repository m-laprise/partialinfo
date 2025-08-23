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

    Returns torch.Tensor: Scalar loss (if reduced) or tensor of shape [batch_size, k] (if 'none').
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits with 3 dims [batch_size, n_agents, m], got {logits.shape}")
    if targets.dim() != 1:
        raise ValueError(f"Expected targets with 1 dim [batch_size], got {targets.shape}")
    
    batch_size, n_agents, n_classes = logits.shape
    
    if targets.dtype != torch.long:
        targets = targets.long()
    if torch.any((targets < 0) | (targets >= n_classes)):
        raise ValueError("targets contain indices outside [0, n_classes-1].")
    
    # Expand and reshape targets
    expanded_targets = targets.unsqueeze(1).expand(-1, n_agents).reshape(-1)  # [batch_size * A]
    logits_flat = logits.reshape(-1, n_classes)                        # [batch_size * A, m]
    # Compute per-prediction cross-entropy loss
    losses = nn.CrossEntropyLoss(
        reduction='none',
        label_smoothing=label_smoothing
    )(logits_flat, expanded_targets)                                   # [batch_size * A]
    losses = losses.view(batch_size, n_agents)                         # [batch_size, A]
    # Apply reduction
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")


def train(model, aggregator, loader, optimizer, criterion, device, scaler: Optional[GradScaler]):
    model.train()
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

        total_loss += loss.item()
        total_examples += B
        
    return total_loss / max(total_examples, 1)
    

@torch.inference_mode()
def evaluate(model, aggregator, loader, criterion, device, max_batches=None):
    model.eval()
    autocast_ctx = autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()
    
    #loss, accuracy, agreement = [], [], []
    total_loss = 0.0
    total_acc = 0.0
    total_agree = 0.0
    total_examples = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        x = batch['matrix'].to(device, non_blocking=True)  # shape: [batch_size, t * m]
        target = batch['label'].to(device, non_blocking=True)  # shape: [batch_size]

        with autocast_ctx:
            logits = aggregator(model(x))          # [B, A, C] or [B, C]
            loss = criterion(logits, target, reduction='mean')
            
        B = target.size(0)
        total_loss += loss.item() * B
        total_examples += B
        _, A, C = logits.shape

        # Compute % accuracy
        # Take majority vote
        agent_preds = logits.argmax(dim=-1)         # [B, A]
        
        counts = torch.zeros(B, C, device=logits.device, dtype=torch.int32)
        ones = torch.ones_like(agent_preds, dtype=torch.int32)
        counts.scatter_add_(1, agent_preds, ones)  # [B, C]

        majority_class = counts.argmax(dim=1)      # [B]
        vote_accuracy = (majority_class == target).float().mean().item()
        total_acc += vote_accuracy * B

        avg_majority_fraction = (counts.max(dim=1).values.float() / A).mean().item()
        total_agree += avg_majority_fraction * B

    mean_loss = total_loss / max(total_examples, 1)
    mean_acc = total_acc / max(total_examples, 1)
    mean_agree = total_agree / max(total_examples, 1)
    
    return (
        float(mean_loss), 
        float(mean_acc), 
        float(mean_agree)
    )


def init_stats():
    return {
        "train_loss": [],
        "t_accuracy": [],
        "t_agreement": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_agreement": []
    }
    