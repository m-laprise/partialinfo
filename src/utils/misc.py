import os
from datetime import datetime
from itertools import accumulate

import torch
from torch.utils.data import Subset


def sequential_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("lengths must sum to len(dataset)")
    cuts = [0, *accumulate(lengths)]
    return [Subset(dataset, range(a, b)) for a, b in zip(cuts, cuts[1:])]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    print("Breakdown by layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:60} {param.numel():>10}")
            
            
def unique_filename(base_dir="results", prefix="run"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{prefix}_{timestamp}")


@torch.no_grad()
def evaluate_agent_contributions(model, loader, criterion, n, m, 
                                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    print("\nEvaluating individual agent contributions...")
    all_agent_errors = []
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.num_graphs
        num_agents = batch.x.shape[0] // batch_size
        nm = n * m
        x = batch.x.view(batch_size, model.num_agents, -1)
        out = model(x)
        targets = batch.y.view(batch_size, nm)      # [B, n*m]
        for i in range(batch_size):
            individual_errors = []
            for a in range(num_agents):
                agent_pred = out[i, a]     # [n*m]
                target = targets[i]        # [n*m]
                error = criterion(agent_pred, target).item()
                individual_errors.append((a, error))
            all_agent_errors.append(individual_errors)
    # Aggregate and report
    flat_errors = [err for sample in all_agent_errors for _, err in sample]
    agent_errors = torch.tensor(flat_errors)
    agent_mean = agent_errors.mean().item()
    agent_std = agent_errors.std().item()
    agent_min = agent_errors.min().item()
    agent_max = agent_errors.max().item()
    print(f"Mean agent MSE: {agent_mean:.4f}, Std: {agent_std:.4f}")
    print(f"Min agent MSE: {agent_min:.4f}, Max: {agent_max:.4f}")