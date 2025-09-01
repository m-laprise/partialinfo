"""
Dot product graph attention network for distributed inductive matrix completion or prediction tasks, 
with learned agent-based message passing setup.
"""

import argparse
import gc
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datagen_temporal import GTMatrices, SensingMasks, TemporalData
from dotGAT import CollectiveClassifier, DistributedDotGAT
from utils.logging import atomic_save, log_training_run, snapshot
from utils.misc import count_parameters, unique_filename
from utils.plotting import plot_classif
from utils.training_temporal import (
    evaluate,
    init_stats,
    stacked_cross_entropy_loss,
    train,
)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    parser.add_argument('--num_agents', type=int, default=128, help='Number of agents')
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the model')
    # Message passing hyperparameters
    parser.add_argument('--att_heads', type=int, default=4, 
                        help='Number of attention heads in the message passing layers') 
    parser.add_argument('--adjacency_mode', type=str, default='learned', choices=['none', 'learned'], 
                        help='Whether adjacency matrix for message-passing is all-to-all or learned')
    # Training hyperparameters
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability during training')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience')
    parser.add_argument('--train_n', type=int, default=1000, help='Number of training matrices')
    parser.add_argument('--val_n', type=int, default=400, help='Number of validation matrices')
    parser.add_argument('--test_n', type=int, default=400, help='Number of test matrices')
    parser.add_argument('--nres', type=int, default=10, help='Number of realizations per DGP')
    parser.add_argument("--sharedv", dest="sharedv", action="store_true", 
                        help="Agents share a V embedding matrix")
    parser.add_argument("--no-sharedv", dest="sharedv", action="store_false", 
                        help="Agents have their own V embedding matrices")
    parser.set_defaults(sharedv=False)
    
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
        sensingmasks = SensingMasks(train_data, args.r, args.num_agents, args.density)
    
    num_workers = min(os.cpu_count() // 2, 4) if torch.cuda.is_available() else 0 # type: ignore
    print(f"Number of workers: {num_workers}")
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=pin, 
        num_workers=num_workers,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        pin_memory=pin, 
        num_workers=num_workers,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        pin_memory=pin, 
        num_workers=num_workers
    )
    
    random_accuracy = 1.0 / args.m
    step_sizes_to_test = [0, 1, 2, 3, 4, 5, 10, 15]
    
    results_train_loss = []
    results_train_accuracy = []
    results_train_agreement = []
    results_test_loss = []
    results_test_accuracy = []
    results_test_agreement = []
    
    for stepsize in step_sizes_to_test:
        torch._dynamo.reset()
          
        model = DistributedDotGAT(
            device=device, input_dim=args.t * args.m, hidden_dim=args.hidden_dim, n=args.t, m=args.m,
            num_agents=args.num_agents, num_heads=args.att_heads, sharedV=args.sharedv, dropout=args.dropout, 
            message_steps=stepsize, adjacency_mode=args.adjacency_mode, sensing_masks=sensingmasks
        ).to(device)
        aggregator = CollectiveClassifier(
            num_agents=args.num_agents, agent_outputs_dim=args.hidden_dim, m = args.m
        ).to(device)
        count_parameters(model)
        count_parameters(aggregator)
        print("--------------------------")
    
        if torch.cuda.is_available():
            print("Compiling model and aggregator with torch.compile...")  
            model = torch.compile(model, mode='reduce-overhead') # also should try: "max-autotune"
            aggregator = torch.compile(aggregator, mode='reduce-overhead')
            print("torch.compile done.")
        
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(aggregator.parameters()), lr=args.lr
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        scaler = GradScaler(enabled=(device.type == 'cuda'))
        criterion = stacked_cross_entropy_loss

        stats = init_stats()
        file_base = unique_filename()
        checkpoint_path = f"{file_base}_checkpoint.pt"
        
        best = {"loss": float('inf'), "acc": float('-inf')}
        patience_counter = 0
        
        start = datetime.now()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, aggregator, train_loader, optimizer, criterion, device, scaler)
            scheduler.step()
            
            _, t_acc, t_agree = evaluate(model, aggregator, train_loader, criterion, device)
            val_loss, val_acc, val_agree = evaluate(model, aggregator, val_loader, criterion, device)
            
            stats["train_loss"].append(train_loss)
            stats["t_accuracy"].append(t_acc)
            stats["t_agreement"].append(t_agree)
            stats["val_loss"].append(val_loss)
            stats["val_accuracy"].append(val_acc)
            stats["val_agreement"].append(val_agree)
            
            if epoch == 1:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = datetime.now()
                print(f"Time elapsed for first epoch: {(t1 - start).total_seconds():.4f} seconds.")
                
            if epoch % 10 == 0 or epoch == 1:
                print(f"Ep {epoch:03d}. ",
                    f"T loss: {train_loss:.2e} | T acc: {t_acc:.2f} | T % maj: {t_agree:.2f} | ",
                    f"V loss: {val_loss:.2e} | V acc: {val_acc:.2f} | V % maj: {val_agree:.2f}")
            
            improved = (val_acc > best["acc"] + 1e-5) | (val_acc >= best["acc"] - 1e-2 and val_loss < best["loss"] - 1e-5)
            
            if improved or epoch == 1:
                best.update(loss=val_loss, acc=val_acc, agree=val_agree, epoch=epoch)
                atomic_save(snapshot(model, aggregator, epoch, args), checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                
            if val_acc == 1.0:
                print(f"Early stopping at epoch {epoch}; validation accuracy is 100%.")
                break
            if val_loss > 10:
                print(f"Early stopping at epoch {epoch}; validation loss is diverging.")
                break
            
        end = datetime.now()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        print(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training time for step size {stepsize}: {(end - start).total_seconds() / 60:.4f} minutes.")
        
        # Clear memory (avoid OOM) and load best model
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state["model"])
            aggregator.load_state_dict(state["aggregator"])
            model.to(device)
            aggregator.to(device)
            print(f"Loaded best model (epoch {state['epoch']}) from checkpoint.")
        else:
            print("No checkpoint found; using current in-memory weights.")

        plot_classif(stats, file_base, random_accuracy)
        
        # Final test evaluation on fresh data
        test_loss, test_acc, test_agree = evaluate(model, aggregator, test_loader, criterion, device)
        print(f"Test Set Performance for step size {stepsize}: | ",
              f"Loss: {test_loss:.2e}, Accuracy: {test_acc:.2f}, % maj: {test_agree:.2f}")

        log_training_run(
            file_base, args, stats, test_loss, test_acc, test_agree, 
            start, end, model, aggregator
        )
        
        results_train_loss.append(stats["train_loss"][-1])
        results_train_accuracy.append(stats["t_accuracy"][-1])
        results_train_agreement.append(stats["t_agreement"][-1])
        results_test_loss.append(test_loss)
        results_test_accuracy.append(test_acc)
        results_test_agreement.append(test_agree)
    
    # Plot results
    plt.style.use('bmh')
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4))
    axs[0].plot(step_sizes_to_test, np.log(results_train_loss), label="Train Loss", marker="o")
    axs[0].plot(step_sizes_to_test, np.log(results_test_loss), label="Test Loss", marker="o")
    axs[1].plot(step_sizes_to_test, results_train_accuracy, label="Train Accuracy", marker="o")
    axs[1].plot(step_sizes_to_test, results_test_accuracy, label="Test Accuracy", marker="o")
    axs[2].plot(step_sizes_to_test, results_train_agreement, label="Train Agreement", marker="o", linestyle='dotted')
    axs[2].plot(step_sizes_to_test, results_test_agreement, label="Test Agreement", marker="o", linestyle='dotted')
    axs[1].axhline(y=random_accuracy, label="Random guessing", color='tab:grey', linestyle='--')
    axs[0].set_ylabel("Log Loss")
    axs[1].set_ylabel("Accuracy")
    axs[2].set_ylabel("Proportion of agents in plurality")
    #axs[0].set_ylim(0, None)
    axs[1].set_ylim(0, 1)
    axs[2].set_ylim(0, 1)
    for ax in axs:
        ax.set_xlabel("Message Passing Steps")
        ax.legend()
        ax.grid(True)
    fig.suptitle(f"Training performance after {args.epochs} epochs, by number of message-passing steps")
    txt = f"Hyperparameters: " \
        f"T = {args.t}, m = {args.m}, r = {args.r}, density = {args.density}, " \
        f"num_agents = {args.num_agents}, hidden_dim = {args.hidden_dim}, att_heads = {args.att_heads}\n" \
        f"dropout = {args.dropout}, init_lr = {args.lr}, batch_size = {args.batch_size}, " \
        f"train_n = {args.train_n}, test_n = {args.test_n}, nres = {args.nres}."
    plt.figtext(0.5, 0.0, txt, ha='center', va ='top', fontsize=9)
    fig.subplots_adjust(bottom=0.3)
    fig.tight_layout()
    
    fig.savefig(f"results/nsteps_experiment_{args.lr:.0e}.png", bbox_inches="tight")
    plt.close(fig)
    