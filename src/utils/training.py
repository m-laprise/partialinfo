import numpy as np
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        #nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
            
def spectral_penalty(output, rank, evalmode=False, eps=1e-6, gamma=1.0):
    try:
        S = torch.linalg.svdvals(output)
    except RuntimeError:
        return torch.tensor(0.0, device=output.device)
    #nuc = S.sum()
    sum_rest = S[rank:].sum()
    #svdcount = len(S) 
    if evalmode:
        return (S[rank - 1] - S[rank]).item()

    # Add soft range constraint on s_max to prevent exploding/shrinking of spectrum
    s_max = S[0]
    N = torch.tensor(min(output.shape), device=output.device, dtype=s_max.dtype)
    soft_upper = torch.relu(s_max - 2 * N)
    soft_lower = torch.relu((N // 2) - s_max)
    range_penalty = soft_upper**2 + soft_lower**2

    # Final combined penalty
    penalty = sum_rest/(N - rank) + gamma * range_penalty

    return penalty
    """
    #ratio = sum_rest / (s_last + 1e-6)
    #penalty = sum_rest/(svdcount-rank) #+ ratio
    penalty = 0
    if nuc > (2 * svdcount):
        penalty += nuc/(2 * svdcount) - 1
    elif nuc < (svdcount // 2):
        penalty -= (nuc/svdcount) + 1
    return penalty
    """


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
        penalty = sum(
            spectral_penalty(prediction[i].view(n, m), rank)
            for i in range(batch_size)
        )

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

        for i in range(batch_size):
            pred_i = prediction[i]  # [n*m]
            target_i = target[i]    # [n*m]
            mask_i = mask[i]        # [n*m]

            known_i = mask_i
            unknown_i = ~mask_i

            if known_i.sum() > 0:
                known_mse.append(
                    criterion(pred_i[known_i], target_i[known_i]).item()
                )
            if unknown_i.sum() > 0:
                unknown_mse.append(
                    criterion(pred_i[unknown_i], target_i[unknown_i]).item()
                )
            assert known_i.sum() + unknown_i.sum() == n * m
            
            # Spectral penalty metrics
            matrix_2d = pred_i.view(n, m)

            gaps.append(float(spectral_penalty(matrix_2d, rank, evalmode=True)))

            # SVDVALS not implemented for MPS
            if matrix_2d.device.type == 'mps':
                matrix_2d = matrix_2d.cpu()

            nuclear_norms.append(torch.linalg.norm(matrix_2d, ord='nuc').item())
            variances.append(matrix_2d.var().item())

    return (
        float(np.mean(known_mse)) if known_mse else float('nan'),
        float(np.mean(unknown_mse)) if unknown_mse else float('nan'),
        float(np.mean(nuclear_norms)),
        float(np.mean(variances)),
        float(np.mean(gaps)),
    )