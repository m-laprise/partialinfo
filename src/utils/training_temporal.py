import numpy as np
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast


def init_weights(m):
    if isinstance(m, nn.Linear):
        #nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            

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
                               reduction: str = 'mean') -> torch.Tensor:
    """
    Computes cross-entropy loss for stacked logits.

    Args:
        logits (torch.Tensor): Tensor of shape [batch_size, num_agents, m],
                               where num_agents is the number of stacked predictions,
                               and m is the number of classes.
        targets (torch.Tensor): Tensor of shape [batch_size], containing class indices.
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'. Default: 'mean'.

    Returns:
        torch.Tensor: Scalar loss (if reduced) or tensor of shape [batch_size, k] (if 'none').
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits with 3 dimensions [batch_size, k, m], got {logits.shape}")
    if targets.dim() != 1:
        raise ValueError(f"Expected targets with 1 dimension [batch_size], got {targets.shape}")

    batch_size, k, num_classes = logits.shape

    # Expand and reshape targets
    expanded_targets = targets.unsqueeze(1).expand(-1, k).reshape(-1)  # [batch_size * k]
    logits_flat = logits.reshape(-1, num_classes)                     # [batch_size * k, m]

    # Compute per-prediction cross-entropy loss
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    losses = loss_fn(logits_flat, expanded_targets)                   # [batch_size * k]
    losses = losses.view(batch_size, k)                               # [batch_size, k]

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

        x = batch['matrix'].to(device, non_blocking=torch.cuda.is_available())  # shape: [batch_size, t * m]
        target = batch['label'].to(device, non_blocking=torch.cuda.is_available())  # shape: [batch_size]
        
        # Conditionally use autocast if on GPU
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

        x = batch['matrix'].to(device, non_blocking=torch.cuda.is_available())  # shape: [batch_size, t * m]
        target = batch['label'].to(device, non_blocking=torch.cuda.is_available())  # shape: [batch_size]

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
        #preds = torch.argmax(logits, dim=1)
        #accuracy.append(float((preds == target).sum()) / int(target.size(0)))
        agent_agreement = (agent_preds == majority_class.unsqueeze(1)).float().sum(dim=1) / A
        avg_majority_fraction = agent_agreement.mean().item()
        agreement.append(avg_majority_fraction)

    return (
        float(np.mean(loss)),
        float(np.mean(accuracy)),
        float(np.mean(agreement))
    )
