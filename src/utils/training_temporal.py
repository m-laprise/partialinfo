import numpy as np
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
            

def init_stats():
    return {
        "train_loss": [],
        "t_accuracy": [],
        "t_agreement": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_agreement": []
    }


def stacked_cross_entropy_loss(logits: torch.Tensor,
                               targets: torch.Tensor,
                               reduction: str = 'mean',
                               label_smoothing: float = 0.0,
                               ignore_index: int = -100) -> torch.Tensor:
    """
    Computes cross-entropy loss over stacked agent logits.
    logits:  [B, A, m] or [B, m] (A defaults to 1 if omitted)
    targets: [B] (class indices)

    Returns torch.Tensor: Scalar loss (if reduced) or tensor of shape [batch_size, k] (if 'none').
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits with 3 dimensions [batch_size, k, m], got {logits.shape}")
    if targets.dim() != 1:
        raise ValueError(f"Expected targets with 1 dimension [batch_size], got {targets.shape}")
    
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
        label_smoothing=label_smoothing,
        ignore_index=ignore_index
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


def train(model, aggregator, loader, optimizer, criterion, 
          t, m, rank, 
          device, scaler):
    model.train()
    total_loss = 0
    for batch in loader:
        torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)

        x = batch['matrix'].to(device, 
                               non_blocking=torch.cuda.is_available())  # shape: [batch_size, t * m]
        target = batch['label'].to(device, 
                                   non_blocking=torch.cuda.is_available())  # shape: [batch_size]
        
        # Conditionally autocast if on GPU
        if torch.cuda.is_available() and scaler is not None:
            with autocast(device_type="cuda"):
                out = model(x)
                logits = aggregator(out) 
                loss = criterion(logits, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            logits = aggregator(out) 
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model, aggregator, loader, criterion, n, m, rank, device, 
    tag: str = "eval", max_batches=None
):
    model.eval()
    loss, accuracy, agreement = [], [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        x = batch['matrix'].to(
            device, non_blocking=torch.cuda.is_available())  # shape: [batch_size, t * m]
        target = batch['label'].to(
            device, non_blocking=torch.cuda.is_available())  # shape: [batch_size]

        out = model(x)
        logits = aggregator(out)
        loss.append(criterion(logits, target).item())

        # Compute % accuracy
        # Take majority vote
        agent_preds = torch.argmax(logits, dim=2)
        B, A = agent_preds.shape
        # One-hot encode predictions: [batch_size, k, m]
        one_hot = torch.nn.functional.one_hot(agent_preds, num_classes=m).sum(dim=1)  # [batch_size, m]
        # Majority vote is class with highest count
        majority_class = torch.argmax(one_hot, dim=1)  # [batch_size]
        vote_accuracy = (majority_class == target).float().mean().item()
        accuracy.append(vote_accuracy)
        
        agent_agreement = (agent_preds == majority_class.unsqueeze(1)).float().sum(dim=1) / A
        avg_majority_fraction = agent_agreement.mean().item()
        agreement.append(avg_majority_fraction)

    return (
        float(np.mean(loss)),
        float(np.mean(accuracy)),
        float(np.mean(agreement))
    )
