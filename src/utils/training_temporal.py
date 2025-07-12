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
        "val_loss": [],
        "val_accuracy": []
    }


def train(model, aggregator, loader, optimizer, criterion, 
          t, m, rank, 
          device, scaler):
    model.train()
    total_loss = 0
    for batch in loader:
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

        # Apply structural constraints (e.g., freeze connectivity)
        model.freeze_nonlearnable()

        total_loss += loss.item()
        
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model, aggregator, loader, criterion, n, m, rank, device, 
    tag: str = "eval", max_batches=None
):
    model.eval()
    loss, accuracy = [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        x = batch['matrix'].to(device, non_blocking=torch.cuda.is_available())  # shape: [batch_size, t * m]
        target = batch['label'].to(device, non_blocking=torch.cuda.is_available())  # shape: [batch_size]

        out = model(x)
        logits = aggregator(out)
        loss.append(criterion(logits, target).item())

        # Compute % accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy.append(float((preds == target).sum()) / int(target.size(0)))

    return (
        float(np.mean(loss)),
        float(np.mean(accuracy)),
    )
