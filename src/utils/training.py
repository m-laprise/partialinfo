import numpy as np
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        #nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            

def spectral_penalty_batched(prediction, rank, evalmode=False, eps=1e-6, gamma=1.0):
    """
    prediction: tensor of shape [batch_size, n, m]
    Returns: scalar tensor loss or list of eval gaps
    """
    try:
        S = torch.linalg.svdvals(prediction)  # shape: [batch_size, min(n, m)]
    except RuntimeError:
        return torch.tensor(0.0, device=prediction.device)

    if evalmode:
        # Return spectral gap between rank-th and (rank+1)-th singular value for each sample
        return (S[:, rank - 1] - S[:, rank]).tolist()

    # sum of singular values from index 'rank' onward
    sum_rest = S[:, rank:].sum(dim=1)  # shape: [batch_size]

    s_max = S[:, 0]  # shape: [batch_size]
    N = torch.tensor(min(prediction.shape[1], prediction.shape[2]), 
                     device=prediction.device, dtype=prediction.dtype)

    soft_upper = torch.relu(s_max - 2 * N)
    soft_lower = torch.relu((N // 2) - s_max)
    range_penalty = soft_upper**2 + soft_lower**2  # shape: [batch_size]

    penalty = sum_rest / (N - rank) + gamma * range_penalty  # shape: [batch_size]
    return penalty.mean()  # return mean over batch


def train(model, aggregator, loader, optimizer, theta, criterion, n, m, rank, 
          device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        batch_size = batch.num_graphs

        # Inputs to the model (agent views)
        x = batch.x.view(batch_size, model.num_agents, -1)
        x = x.to(device)
        out = model(x)

        # Aggregated matrix prediction
        prediction = aggregator(out)  # shape: [batch_size, n * m]
        target = batch.y.view(batch_size, -1).to(device)  # shape: [batch_size, n * m]
        mask = batch.mask.view(batch_size, -1).to(device)  # shape: [batch_size, n * m]

        # Compute masked reconstruction loss (only known entries)
        reconstructionloss = criterion(prediction[mask], target[mask])

        # Spectral penalty applied to full matrix prediction
        penalty = spectral_penalty_batched(prediction.view(batch_size, n, m), rank)

        # Combined loss
        loss = theta * reconstructionloss + (1 - theta) * penalty
        loss.backward()

        # Gradient clipping and update
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Apply structural constraints (e.g., freeze connectivity)
        model.freeze_nonlearnable()

        total_loss += loss.item()
        
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model, aggregator, loader, criterion, n, m, rank, device, 
    tag: str = "eval"
):
    model.eval()
    known_mse, unknown_mse, nuclear_norms, variances, gaps = [], [], [], [], []

    for batch in loader:
        batch_size = batch.num_graphs

        x = batch.x.view(batch_size, model.num_agents, -1).to(device)
        out = model(x)
        prediction = aggregator(out)  # [B, nm]
        target = batch.y.view(batch_size, -1).to(device)
        mask = batch.mask.view(batch_size, -1).to(device)

        known_mask = mask
        unknown_mask = ~mask

        # Compute known MSE
        known_pred = prediction[known_mask]
        known_target = target[known_mask]
        if known_pred.numel() > 0:
            known_mse.append(criterion(known_pred, known_target).item())

        # Compute unknown MSE
        unknown_pred = prediction[unknown_mask]
        unknown_target = target[unknown_mask]
        if unknown_pred.numel() > 0:
            unknown_mse.append(criterion(unknown_pred, unknown_target).item())

        # Reshape for spectral and matrix statistics
        prediction_2d = prediction.view(batch_size, n, m)

        # Spectral gaps (batched)
        gaps.extend(spectral_penalty_batched(prediction_2d, rank, evalmode=True))

        # Nuclear norms and variances
        variances.append(prediction_2d.var(dim=(-2, -1)).mean().item())
        # MPS compatibility for SVD
        if prediction_2d.device.type == "mps":
            prediction_2d = prediction_2d.cpu()
        nuclear_norms.append(torch.linalg.matrix_norm(prediction_2d, ord='nuc', dim=(-2, -1)).mean().item())

    return (
        float(np.mean(known_mse)),
        float(np.mean(unknown_mse)),
        float(np.mean(nuclear_norms)),
        float(np.mean(variances)),
        float(np.mean(gaps)),
    )