"""
Dot product graph attention network for distributed inductive matrix completion or prediction tasks, 
with learned agent-based message passing setup.
"""

import argparse
import gc
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datagen_temporal import GTMatrices, SensingMasks, TemporalData
from dotGAT import CollectiveClassifier, DistributedDotGAT
from utils.logging import log_training_run
from utils.misc import count_parameters, unique_filename
from utils.plotting import plot_classif
from utils.training_temporal import (
    evaluate,
    init_stats,
    init_weights,
    stacked_cross_entropy_loss,
    train,
)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Ground truth and sensing hyperparameters
    parser.add_argument('--t', type=int, default=50, 
                        help='Number of rows in each ground truth matrix')
    parser.add_argument('--m', type=int, default=25, 
                        help='Number of columns in each ground truth matrix')
    parser.add_argument('--r', type=int, default=25, help='Rank of each ground truth matrix')
    parser.add_argument('--density', type=float, default=0.5, 
                        help='Target proportion of known entries in each ground truth matrix')
    parser.add_argument('--num_agents', type=int, default=20, help='Number of agents')
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the model')
    # Message passing hyperparameters
    parser.add_argument('--att_heads', type=int, default=4, 
                        help='Number of attention heads in the message passing layers') 
    parser.add_argument('--adjacency_mode', type=str, default='learned', choices=['none', 'learned'], 
                        help='Whether adjacency matrix for message-passing is all-to-all or learned')
    parser.add_argument('--steps', type=int, default=5, 
                        help='Number of message passing steps. If 0, the model reduces to an encoder-decoder.')
    # Training hyperparameters
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability during training')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience')
    parser.add_argument('--train_n', type=int, default=800, help='Number of training matrices')
    parser.add_argument('--val_n', type=int, default=200, help='Number of validation matrices')
    parser.add_argument('--test_n', type=int, default=200, help='Number of test matrices')
    parser.add_argument('--nres', type=int, default=10, help='Number of realizations per DGP')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}") # type: ignore
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        torch.backends.cudnn.benchmark = True
    
    with torch.no_grad():  
        train_GT = GTMatrices(N=args.train_n, t=args.t, m=args.m, r=args.r, realizations = args.nres)
        val_GT = GTMatrices(N=args.val_n, t=args.t, m=args.m, r=args.r, realizations = args.nres)
        test_GT = GTMatrices(N=args.test_n, t=args.t, m=args.m, r=args.r, realizations = args.nres)
        
        train_data = TemporalData(train_GT)
        val_data = TemporalData(val_GT, verbose=False)
        test_data = TemporalData(test_GT, verbose=False)
    
        sensingmasks = SensingMasks(train_data, args.r, args.num_agents, args.density).to(device)
    
    num_workers = min(os.cpu_count() // 2, 4) if torch.cuda.is_available() else 0 # type: ignore
    print(f"Number of workers: {num_workers}")
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=torch.cuda.is_available(), num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, 
        pin_memory=torch.cuda.is_available(), num_workers=num_workers,
    )

    model = DistributedDotGAT(
        device=device, input_dim=args.t * args.m,  hidden_dim=args.hidden_dim, n=args.t, m=args.m,
        num_agents=args.num_agents, num_heads=args.att_heads, dropout=args.dropout, 
        message_steps=args.steps, adjacency_mode=args.adjacency_mode, sensing_masks=sensingmasks
    ).to(device)
    model.apply(init_weights)
    count_parameters(model)
    
    aggregator = CollectiveClassifier(
        num_agents=args.num_agents, agent_outputs_dim=args.hidden_dim, m = args.m
    ).to(device)
    aggregator.apply(init_weights)
    count_parameters(aggregator)
    print("--------------------------")
    
    if torch.cuda.is_available():
        print("Compiling model and aggregator with torch.compile...")  
        model = torch.compile(model, mode='reduce-overhead', fullgraph=True) # also should try: "max-autotune"
        aggregator = torch.compile(aggregator, mode='reduce-overhead', fullgraph=True)
        print("torch.compile done.")
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(aggregator.parameters()), lr=args.lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(device=device.type) if torch.cuda.is_available() else None
    criterion = stacked_cross_entropy_loss
    
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
        scheduler.step()
        _, t_accuracy, t_agreement = evaluate(
            model, aggregator, train_loader, criterion, args.t, args.m, args.r, device, tag="train", 
        )
        val_loss, val_accuracy, val_agreement = evaluate(
            model, aggregator, val_loader, criterion, args.t, args.m, args.r, device, tag="val"
        )
        
        stats["train_loss"].append(train_loss)
        stats["t_accuracy"].append(t_accuracy)
        stats["t_agreement"].append(t_agreement)
        stats["val_loss"].append(val_loss)
        stats["val_accuracy"].append(val_accuracy)
        stats["val_agreement"].append(val_agreement)
        
        if epoch == 1:
            t1 = datetime.now()
            print(f"Time elapsed for first epoch: {(t1 - start).total_seconds()} seconds.")        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Ep {epoch:03d}. ",
                  f"T loss: {train_loss:.2e} | T acc: {t_accuracy:.2f} | T % maj: {t_agreement:.2f} | ",
                  f"V loss: {val_loss:.2e} | V acc: {val_accuracy:.2f} | V % maj: {val_agreement:.2f}")
        
        # Save connectivity matrix for visualization
        """netxmask = model.connect.learn_mask.detach().cpu().numpy().astype(int)
        netx = model.connect()[0].detach().cpu().numpy()
        netx[netx == float('-inf')] = 0.0
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #ax[0].matshow(netxmask, cmap='gray', vmin=0, vmax=1)
        maxv = abs(netx).max()
        ax[1].matshow(netx, cmap='PiYG', vmin=-maxv, vmax=maxv)
        fig.show()"""
        #np.save(f"{file_base}_adj_epoch{epoch}.npy", netx)
        
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

    random_accuracy = 1.0 / args.m
    plot_classif(stats, file_base, random_accuracy)
    
    # Final test evaluation on fresh data
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, 
        pin_memory=torch.cuda.is_available(), num_workers=num_workers, 
    )
    test_loss, test_accuracy, test_agreement = evaluate(
        model, aggregator, test_loader, criterion, args.t, args.m, args.r, device, tag="test"
    )
    print("Test Set Performance | ",
          f"Loss: {test_loss:.2e}, Accuracy: {test_accuracy:.2f}, % maj: {test_agreement:.2f}")

    log_training_run(
        file_base, args, stats, test_loss, test_accuracy, test_agreement, 
        start, end, model, aggregator
    )
    #file_prefix = Path(file_base).name  # Extracts just 'run_YYYYMMDD_HHMMSS'
    #plot_connectivity_matrices("results", prefix=file_prefix, cmap="coolwarm")
    