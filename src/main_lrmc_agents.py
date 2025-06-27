"""
Dot product graph attention network for distributed inductive matrix completion, 
with learned agent-based message passing setup.
Supports sparse views as agent inputs.

Each agent independently receives their own masked view of the matrix via 
AgentMatrixReconstructionDataset (x: [batch, num_agents, input_dim]). 

During message passing, each DotGATLayer uses dot-product attention with a learned 
connectivity matrix, ensuring localized neighbor communication.
Aggregation only occurs after message passing and is mediated via the GAT network to prevent
information leakage.

self.connectivity is a trainable nn.Parameter representing the agent-to-agent communication graph.
The attention computation adds this weighted adjacency matrix to the attention logits.

Explicit message passing rounds are controlled by message_steps in DistributedDotGAT, with message 
updates applied iteratively through GAT heads.

Design choices:
- *Positional embeddings*: None in this version. Many
alternatives are possible, including integrating structural rather than only coordinate information.
- *Adjacency matrix is not input-dependent*: 
Currently self.connectivity is shared across all heads and batches, and static per model instance.
    An alternative would be to allow connectivity vary by batch or to depend on inputs/agent embeddings.
- *Sparsity of adjacency matrix*: 
The connectivity matrix is initialized as a sparse adjacency matrix of a small world graph, and used
to enforce sparsity throughout training via gradient freezing (structural barriers in connectivity). 
Initial edges (nonzero in A) are always learnable. Half of the 0 entries are allowed to grow via learning. 
The rest are frozen at zero through gradient masking.
    - This could be taken out or modified. 
    - An alternative (which is in a previous version in Github) is top-k sparsification (listening to only some 
    neighbors), which is also static.
    - Other alternatives would be dynamic thresholding or learned gating. ChatGPT suggests thresholded softmax 
    (Sparsemax or Entmax), attention dropout, learned attention masks, or entropic sparsity.
- Agents do not specialize. There are no agent-specific embeddings or parameters.
    Alternatives include: agent-specific MLPs or gating mechanisms before/after message passing;  
    additional diversity or specialization losses.
- No residual connections.
    Alternative: `h = h + torch.stack(head_outputs).mean(dim=0)` or `h = self.norm(h + ...)`
- Collective aggregation occurs through learned, gated (input-dependent) pooling across agents.
    Alternatives include:
    attention-based aggregation with a learned attention weight for each agent, 
    game-theoretic aggregation of subsets of agents based on utility or diversity contribution

Required improvements:
- Save and plot connectivity matrix over time
- Spectral penalty is unstable. ChatGPT suggests replacing with nuclear norm clipping, 
differentiable low-rank approximations, or penalizing condition number.
- Log agent_diversity_penalty during training to track changes in agent roles
- Per-agent MSE should be correlated with agent input quality/sparsity. Integrate tracking input sparsity 
per agent along with prediction quality.
- Develop view_mode='project' and make projections learnable (e.g., per-agent linear layers).
"""

import argparse
import gc

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from datagen import AgentMatrixReconstructionDataset
from dotGAT import Aggregator, DistributedDotGAT
from utils.misc import count_parameters, unique_filename
from utils.plotting import plot_stats


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


def init_weights(m):
    if isinstance(m, nn.Linear):
        #nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=30)
    parser.add_argument('--m', type=int, default=30)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--density', type=float, default=0.3)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--num_agents', type=int, default=30)
    parser.add_argument('--agentdistrib', type=str, default='all-see-all')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--adjacency_mode', type=str, default='none', choices=['none', 'learned'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--theta', type=float, default=0.95, help='Weight for the known entry loss vs penalty')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    #parser.add_argument('--eval_agents', action='store_true', help='Always evaluate agent contributions')
    parser.add_argument('--steps', type=int, default=5, help='Number of message passing steps. If 0, the model reduces to an encoder-decoder.')
    parser.add_argument('--train_n', type=int, default=1000, help='Number of training matrices')
    parser.add_argument('--val_n', type=int, default=64, help='Number of validation matrices')
    parser.add_argument('--test_n', type=int, default=64, help='Number of test matrices')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() 
                          #else 'mps' if torch.backends.mps.is_available() 
                          else 'cpu')
    train_set = AgentMatrixReconstructionDataset(
        num_matrices=args.train_n, n=args.n, m=args.m, r=args.r, 
        num_agents=args.num_agents, agentdistrib=args.agentdistrib,
        density=args.density, sigma=args.sigma
    )
    val_set = AgentMatrixReconstructionDataset(
        num_matrices=args.val_n, n=args.n, m=args.m, r=args.r, 
        num_agents=args.num_agents, agentdistrib=args.agentdistrib,
        density=args.density, sigma=args.sigma, verbose=False
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = DistributedDotGAT(
        input_dim=train_set.input_dim,
        hidden_dim=args.hidden_dim,
        n=args.n, m=args.m,
        num_agents=args.num_agents,
        num_heads=args.num_heads,
        dropout=args.dropout,
        message_steps=args.steps,
        adjacency_mode=args.adjacency_mode
    ).to(device)
    model.apply(init_weights)
    count_parameters(model)
    
    aggregator = Aggregator(
        num_agents=args.num_agents, output_dim=args.n * args.m
    ).to(device)
    aggregator.apply(init_weights)
    count_parameters(aggregator)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(aggregator.parameters()), lr=args.lr
    )
    criterion = nn.MSELoss()
    
    stats = {
        "train_loss": [],
        "t_known_mse": [],
        "t_unknown_mse": [],
        "t_nuclear_norm": [],
        "t_variance": [],
        "t_spectral_gap": [],
        "val_known_mse": [],
        "val_unknown_mse": [],
        "val_nuclear_norm": [],
        "val_variance": [],
        "val_spectral_gap": []
    }
    file_base = unique_filename()
    checkpoint_path = f"{file_base}_checkpoint.pt"
    
    best_loss = float('inf')
    patience_counter = 0
    
    with torch.no_grad():
        batch = next(iter(train_loader))
        x = batch.x.view(batch.num_graphs, model.num_agents, -1).to(device)
        out = model(x)
        recon = aggregator(out)
        print("Initial output variance:", recon.var().item())

    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model, aggregator, train_loader, optimizer, args.theta, criterion, 
            args.n, args.m, args.r, device
        )
        t_known, t_unknown, t_nuc, t_var, t_gap = evaluate(
            model, aggregator, train_loader, criterion, 
            args.n, args.m, args.r, device, tag="train"
        )
        val_known, val_unknown, nuc, var, gap = evaluate(
            model, aggregator, val_loader, criterion, 
            args.n, args.m, args.r, device, tag="val"
        )
        
        stats["train_loss"].append(train_loss)
        stats["t_known_mse"].append(t_known)
        stats["t_unknown_mse"].append(t_unknown)
        stats["t_nuclear_norm"].append(t_nuc)
        stats["t_variance"].append(t_var)
        stats["t_spectral_gap"].append(t_gap)
        stats["val_known_mse"].append(val_known)
        stats["val_unknown_mse"].append(val_unknown)
        stats["val_nuclear_norm"].append(nuc)
        stats["val_variance"].append(var)
        stats["val_spectral_gap"].append(gap)
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Ep {epoch:03d}. L: {train_loss:.4f} | Kn: {t_known:.4f} | "+
                f"Unkn: {t_unknown:.4f} | NN: {t_nuc:.2f} | Gap: {t_gap:.2f} | Var: {t_var:.4f}")
            
            print(f"--------------------------------------VAL--: K/U: {val_known:.2f} / "+
                f"{val_unknown:.2f}. NN/G: {nuc:.1f} / {gap:.1f}. Var: {var:.2f}.")
        
        # Save connectivity matrix for visualization
        #adj_matrix = model.connectivity.detach().cpu().numpy()
        #np.save(f"{file_base}_adj_epoch{epoch}.npy", adj_matrix)
        
        if t_unknown < best_loss - 1e-5:
            best_loss = t_unknown
            torch.save(model.state_dict(), checkpoint_path)
            
        if t_unknown < 0.001:
            print(f"Early stopping at epoch {epoch}.")
            break
        
        #if val_unknown < best_loss - 1e-5:
        #    best_loss = val_unknown
        #    patience_counter = 0
        #    torch.save(model.state_dict(), checkpoint_path)
        #    
        #else:
        #    patience_counter += 1
        #    if patience_counter >= args.patience:
        #        print(f"Early stopping at epoch {epoch}.")
        #        break

    # Clear memory (avoid OOM) and load best model
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    print("Loaded best model from checkpoint.")

    plot_stats(stats, file_base, 
               train_set.nuclear_norm_mean, train_set.gap_mean, train_set.variance_mean)
    
    # Final test evaluation on fresh data
    test_dataset = AgentMatrixReconstructionDataset(
        num_matrices=args.test_n, n=args.n, m=args.m, r=args.r, 
        num_agents=args.num_agents, agentdistrib=args.agentdistrib,
        density=args.density, sigma=args.sigma, verbose=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    test_known, test_unknown, test_nuc, test_var, test_gap = evaluate(
        model, aggregator, test_loader, criterion, 
        args.n, args.m, args.r, device, tag="test"
    )
    print(f"Test Set Performance | Known MSE: {test_known:.4f}, Unknown MSE: {test_unknown:.4f}"+
        f" Nuclear Norm: {test_nuc:.2f}, Spectral Gap: {test_gap:.2f}, Variance: {test_var:.4f}")

    #file_prefix = Path(file_base).name  # Extracts just 'run_YYYYMMDD_HHMMSS'
    #plot_connectivity_matrices("results", prefix=file_prefix, cmap="coolwarm")

    # Agent contribution eval (optional)
    #if test_unknown < 0.1 or args.eval_agents:
    #    evaluate_agent_contributions(model, test_loader, criterion, args.n, args.m)
    