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
- Agents do not specialize. There are no agent-specific embeddings or parameters.
    Alternatives include: agent-specific MLPs or gating mechanisms before/after message passing;  
    additional diversity or specialization losses.
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
- No residual connections.
    Alternative: `h = h + torch.stack(head_outputs).mean(dim=0)` or `h = self.norm(h + ...)`
- Collective aggregation occurs through learned, gated (input-dependent) pooling across agents.
    Alternatives include:
    attention-based aggregation with a learned attention weight for each agent, 
    game-theoretic aggregation of subsets of agents based on utility or diversity contribution

Required improvements:
- Save and plot connectivity matrix over time
- Spectral penalty is unstable. Explore: penalizing condition number.
- Log agent_diversity_penalty during training to track changes in agent roles
- Per-agent MSE should be correlated with agent input quality/sparsity. Integrate tracking input sparsity 
per agent along with prediction quality.
- Develop view_mode='project' and make projections learnable (e.g., per-agent linear layers).
"""

import argparse
import gc
from datetime import datetime

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch_geometric.loader import DataLoader

from datautils.datagen_lrmc import AgentMatrixReconstructionDataset
from dotGAT import Aggregator, DistributedDotGAT, ReconDecoder
from utils.misc import count_parameters, unique_filename
from utils.plotting import plot_stats
from utils.training_lrmc import evaluate, init_stats, init_weights, train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=32)
    parser.add_argument('--m', type=int, default=32)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--density', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--num_agents', type=int, default=30)
    parser.add_argument('--agentdistrib', type=str, default='uniform', choices=['all-see-all', 'uniform'])
    parser.add_argument('--sampling_scheme', type=str, default='random', choices=['constant', 'random'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--adjacency_mode', type=str, default='none', choices=['none', 'learned'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--theta', type=float, default=0.95, help='Weight for the known entry loss vs penalty')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--steps', type=int, default=5, help='Number of message passing steps. If 0, the model reduces to an encoder-decoder.')
    parser.add_argument('--train_n', type=int, default=500, help='Number of training matrices')
    parser.add_argument('--val_n', type=int, default=64, help='Number of validation matrices')
    parser.add_argument('--test_n', type=int, default=64, help='Number of test matrices')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_set = AgentMatrixReconstructionDataset(
        num_matrices=args.train_n, n=args.n, m=args.m, r=args.r, 
        num_agents=args.num_agents, agentdistrib=args.agentdistrib, sampling_scheme=args.sampling_scheme,
        density=args.density, sigma=args.sigma
    )
    shared_cache = None
    if args.sampling_scheme == 'constant':
        shared_cache = train_set.get_masks_cache()
        
    val_set = AgentMatrixReconstructionDataset(
        num_matrices=args.val_n, n=args.n, m=args.m, r=args.r, 
        num_agents=args.num_agents, agentdistrib=args.agentdistrib, sampling_scheme=args.sampling_scheme,
        density=args.density, sigma=args.sigma, verbose=False, masks_cache=shared_cache
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=torch.cuda.is_available(), 
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, 
        pin_memory=torch.cuda.is_available()
    )

    network = DistributedDotGAT(
        input_dim=train_set.input_dim,
        hidden_dim=args.hidden_dim,
        n=args.n, m=args.m,
        num_agents=args.num_agents,
        num_heads=args.num_heads,
        dropout=args.dropout,
        message_steps=args.steps,
        adjacency_mode=args.adjacency_mode
    )
    network.apply(init_weights)
    decoder = ReconDecoder(
        hidden_dim=args.hidden_dim,
        n=args.n, m=args.m,
        num_agents=args.num_agents
    )
    decoder.apply(init_weights)
    
    model = nn.Sequential(network, decoder).to(device)
    count_parameters(model)
    print("--------------------------")
    
    aggregator = Aggregator(
        num_agents=args.num_agents, output_dim=args.n * args.m
    ).to(device)
    aggregator.apply(init_weights)
    count_parameters(aggregator)
    print("--------------------------")
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(aggregator.parameters()), lr=args.lr
    )
    scaler = GradScaler(device=device.type) if torch.cuda.is_available() else None
    criterion = nn.MSELoss()
    
    stats = init_stats()
    file_base = unique_filename()
    checkpoint_path = f"{file_base}_checkpoint.pt"
    
    best_loss = float('inf')
    patience_counter = 0
    
    with torch.no_grad():
        batch = next(iter(train_loader))
        x = batch.x.view(batch.num_graphs, args.num_agents, -1).to(device, non_blocking=True)
        out = model(x)
        recon = aggregator(out)
        print("Initial output variance:", recon.var().item())

    # print time at beginning of training
    start = datetime.now()
    print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model, aggregator, train_loader, optimizer, args.theta, criterion, 
            args.n, args.m, args.r, device, scaler
        )
        val_batches = len(val_loader)
        t_known, t_unknown, t_nuc, t_var, t_gap = evaluate(
            model, aggregator, train_loader, criterion, 
            args.n, args.m, args.r, device, tag="train", max_batches=val_batches
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
        
        if epoch == 1:
            t1 = datetime.now()
            print(f"Time elapsed for first epoch: {(t1 - start).total_seconds()} seconds.")
        
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
        
    end = datetime.now()
    print(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training time: {(end - start).total_seconds() / 60} minutes.")
    
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
        num_agents=args.num_agents, agentdistrib=args.agentdistrib, sampling_scheme=args.sampling_scheme,
        density=args.density, sigma=args.sigma, verbose=False, masks_cache=shared_cache
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
    