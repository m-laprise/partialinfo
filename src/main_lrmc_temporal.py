"""
Dot product graph attention network for distributed inductive matrix completion, 
with learned agent-based message passing setup.
"""

import argparse
import gc
from datetime import datetime

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from datagen_temporal import GTMatrices, SensingMasks, TemporalData
from dotGAT import CollectiveClassifier, DistributedDotGAT
from utils.misc import count_parameters, unique_filename
from utils.plotting import plot_classif
from utils.training_temporal import evaluate, init_stats, init_weights, train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=int, default=32)
    parser.add_argument('--m', type=int, default=32)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--density', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--num_agents', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--adjacency_mode', type=str, default='none', choices=['none', 'learned'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=200, help='Early stopping patience')
    parser.add_argument('--steps', type=int, default=5, help='Number of message passing steps. If 0, the model reduces to an encoder-decoder.')
    parser.add_argument('--train_n', type=int, default=500, help='Number of training matrices')
    parser.add_argument('--val_n', type=int, default=64, help='Number of validation matrices')
    parser.add_argument('--test_n', type=int, default=64, help='Number of test matrices')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_GT = GTMatrices(num_matrices=args.train_n, t=args.t, m=args.m, r=args.r)
    val_GT = GTMatrices(num_matrices=args.val_n, t=args.t, m=args.m, r=args.r)
    test_GT = GTMatrices(num_matrices=args.test_n, t=args.t, m=args.m, r=args.r)
    
    train_data = TemporalData(train_GT)
    val_data = TemporalData(val_GT)
    test_data = TemporalData(test_GT)
    
    sensingmasks = SensingMasks(train_data, args.r, args.num_agents, args.density)
    
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=torch.cuda.is_available(), 
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, 
        pin_memory=torch.cuda.is_available()
    )

    model = DistributedDotGAT(
        input_dim=args.t * args.m,
        hidden_dim=args.hidden_dim,
        n=args.t, m=args.m,
        num_agents=args.num_agents,
        num_heads=args.num_heads,
        dropout=args.dropout,
        message_steps=args.steps,
        adjacency_mode=args.adjacency_mode,
        sensing_masks=sensingmasks
    ).to(device)
    model.apply(init_weights)
    count_parameters(model)
    
    aggregator = CollectiveClassifier(
        num_agents=args.num_agents, agent_outputs_dim=args.hidden_dim, m = args.m
    ).to(device)
    aggregator.apply(init_weights)
    count_parameters(aggregator)
    print("--------------------------")
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(aggregator.parameters()), lr=args.lr
    )
    scaler = GradScaler(device=device.type) if torch.cuda.is_available() else None
    criterion = nn.CrossEntropyLoss()
    
    stats = init_stats()
    file_base = unique_filename()
    checkpoint_path = f"{file_base}_checkpoint.pt"
    
    best_loss = float('inf')
    patience_counter = 0

    # print time at beginning of training
    start = datetime.now()
    print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model, aggregator, train_loader, optimizer, criterion, 
            args.t, args.m, args.r, device, scaler
        )
        val_batches = len(val_loader)
        _, t_accuracy = evaluate(
            model, aggregator, train_loader, criterion, 
            args.t, args.m, args.r, device, tag="train", #max_batches=val_batches
        )
        val_loss, val_accuracy = evaluate(
            model, aggregator, val_loader, criterion, 
            args.t, args.m, args.r, device, tag="val"
        )
        
        stats["train_loss"].append(train_loss)
        stats["t_accuracy"].append(t_accuracy)
        stats["val_loss"].append(val_loss)
        stats["val_accuracy"].append(val_accuracy)
        
        if epoch == 1:
            t1 = datetime.now()
            print(f"Time elapsed for first epoch: {(t1 - start).total_seconds()} seconds.")
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Ep {epoch:03d}. T loss: {train_loss:.2e} | T acc: {t_accuracy:.2f} | V loss: {val_loss:.2e} | V acc: {val_accuracy:.2f}")
        
        # Save connectivity matrix for visualization
        #adj_matrix = model.connectivity.detach().cpu().numpy()
        #np.save(f"{file_base}_adj_epoch{epoch}.npy", adj_matrix)
        
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            
            if val_accuracy == 1.0:
                print(f"Early stopping at epoch {epoch}; accuracy on validation set is 100%.")
                break
  
        else:
           patience_counter += 1
           if val_loss > 10:
               print(f"Early stopping at epoch {epoch}; loss on validation set is diverging.")
               break
        #    if patience_counter >= args.patience:
        #        print(f"Early stopping at epoch {epoch}; no improvement for {args.patience} epochs.")
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

    plot_classif(stats, file_base)
    
    # Final test evaluation on fresh data

    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    test_loss, test_accuracy = evaluate(
        model, aggregator, test_loader, criterion, 
        args.t, args.m, args.r, device, tag="test"
    )
    print(f"Test Set Performance | Loss: {test_loss:.2e}, Accuracy: {test_accuracy:.2f}")

    #file_prefix = Path(file_base).name  # Extracts just 'run_YYYYMMDD_HHMMSS'
    #plot_connectivity_matrices("results", prefix=file_prefix, cmap="coolwarm")
    