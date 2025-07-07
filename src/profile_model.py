
import argparse

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.loader import DataLoader

from datagen import AgentMatrixReconstructionDataset
from dotGAT import Aggregator, DistributedDotGAT
from utils.misc import count_parameters
from utils.training import evaluate, init_weights, train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--m', type=int, default=64)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--density', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--num_agents', type=int, default=30)
    parser.add_argument('--agentdistrib', type=str, default='uniform')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--adjacency_mode', type=str, default='none', choices=['none', 'learned'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--theta', type=float, default=0.95, help='Weight for the known entry loss vs penalty')
    parser.add_argument('--steps', type=int, default=5, help='Number of message passing steps. If 0, the model reduces to an encoder-decoder.')
    parser.add_argument('--train_n', type=int, default=100, help='Number of training matrices')
    
    # Setup
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() 
                          #else 'mps' if torch.backends.mps.is_available() 
                          else 'cpu')
    train_set = AgentMatrixReconstructionDataset(
        num_matrices=args.train_n, n=args.n, m=args.m, r=args.r, 
        num_agents=args.num_agents, agentdistrib=args.agentdistrib,
        density=args.density, sigma=args.sigma
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size)

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
    
    # Warm up
    batch = next(iter(train_loader))
    x = batch.x.view(batch.num_graphs, model.num_agents, -1).to(device)
    out = model(x)
    recon = aggregator(out)
    
    # Forward pass profiler
    activities = [ProfilerActivity.CPU]
    devicename = 'cpu'
    if torch.cuda.is_available():
        devicename = 'cuda'
        activities += [ProfilerActivity.CUDA]
        
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            batch = next(iter(train_loader))
            x = batch.x.view(batch.num_graphs, model.num_agents, -1).to(device)
            out = model(x)
            recon = aggregator(out)
    
    print(f"Profiler results for forward pass with {devicename}:")
    print(prof.key_averages().table(sort_by=devicename + "_time_total", row_limit=15))
    #print(prof.key_averages(group_by_input_shape=True).table(sort_by=devicename + "_time_total", row_limit=50))
    
    with profile(activities=activities, profile_memory=True) as prof:
        with record_function("model_inference_mem"):
            batch = next(iter(train_loader))
            x = batch.x.view(batch.num_graphs, model.num_agents, -1).to(device)
            out = model(x)
            recon = aggregator(out)
    
    print(f"Profiler results for forward pass memory usage with {devicename}:")
    print(prof.key_averages().table(sort_by="self_" + devicename + "_memory_usage", row_limit=5))
    
    # Training setup
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(aggregator.parameters()), lr=args.lr
    )
    criterion = nn.MSELoss()
    
    # Warmup
    train_loss = train(
        model, aggregator, train_loader, optimizer, args.theta, criterion, 
        args.n, args.m, args.r, device, None
    )
    t_known, t_unknown, t_nuc, t_var, t_gap = evaluate(
        model, aggregator, train_loader, criterion, 
        args.n, args.m, args.r, device, tag="train"
    )
    
    # Training profiler
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_training"):
            train_loss = train(
                model, aggregator, train_loader, optimizer, args.theta, criterion, 
                args.n, args.m, args.r, device, None
            )
            t_known, t_unknown, t_nuc, t_var, t_gap = evaluate(
                model, aggregator, train_loader, criterion, 
                args.n, args.m, args.r, device, tag="train"
            )

            print(f"Trainin stats. L: {train_loss:.4f} | Kn: {t_known:.4f} | "+
                f"Unkn: {t_unknown:.4f} | NN: {t_nuc:.2f} | Gap: {t_gap:.2f} | Var: {t_var:.4f}")

    print(f"Profiler results for training with {devicename}:")
    print(prof.key_averages().table(sort_by=devicename + "_time_total", row_limit=15))
    #print(prof.key_averages(group_by_input_shape=True).table(sort_by=devicename + "_time_total", row_limit=50))
    