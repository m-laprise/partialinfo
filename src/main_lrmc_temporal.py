"""
Dot product graph attention network for distributed inductive matrix completion or prediction tasks, 
with learned agent-based message passing setup.

Example usage:
```
uv run ./src/main_lrmc_temporal.py \
  --t 50 --m 25 --r 6 --density 0.5 \
  --num_agents 64 --nb_ties 4 --hidden_dim 64 \
  --lr 1e-4 --epochs 150 --steps 5 \
  --batch_size 100 --train_n 1000 --no-sharedv \
  --gt_mode 'value' --kernel 'cauchy' --vtype 'random' --task 'argmax'
```
"""

import argparse
import gc
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.logging import atomic_save, init_stats, log_training_run, printlog, snapshot
from utils.misc import count_parameters, unique_filename
from utils.plotting import plot_classif, plot_regression
from utils.setup import create_data, setup_model
from utils.training_temporal import (
    evaluate,
    final_test,
    stacked_cross_entropy_loss,
    stacked_MSE,
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
    parser.add_argument('--gt_mode', type=str, default='value', choices=['value', 'return'], 
                        help='Kind of ground truth matrices (absolute value or relative return)')
    parser.add_argument('--kernel', type=str, default='cauchy', choices=['matern', 'cauchy', 'whitenoise'], 
                        help='Type of vcov for columns of U factors')
    parser.add_argument('--vtype', type=str, default='random', choices=['random', 'block'], 
                        help='Random or block diagonal V factors')
    parser.add_argument('--task', type=str, default='nonlinear', choices=['argmax', 'nonlinear'], 
                        help='Prediction task: argmax or arbitrary nonlinear function of row t+1')
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
    parser.add_argument('--nb_ties', type=int, default=4, help='Number of ties for each agent (k)')
    parser.add_argument('--steps', type=int, default=5, 
                        help='Number of message passing steps. If 0, the model reduces to an encoder-decoder.')
    # Training hyperparameters
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability during training')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience')
    parser.add_argument('--train_n', type=int, default=800, help='Number of training matrices')
    parser.add_argument('--val_n', type=int, default=400, help='Number of validation matrices')
    parser.add_argument('--test_n', type=int, default=400, help='Number of test matrices')
    parser.add_argument('--nres', type=int, default=10, help='Number of realizations per DGP')
    parser.add_argument("--sharedv", dest="sharedv", action="store_true", 
                        help="Agents share a V embedding matrix")
    parser.add_argument("--no-sharedv", dest="sharedv", action="store_false", 
                        help="Agents have their own V embedding matrices")
    parser.set_defaults(sharedv=True)
    args = parser.parse_args()
    
    if args.task == 'argmax':
        task_cat = 'classif'
    elif args.task == 'nonlinear':
        task_cat = 'regression'
    else:
        raise NotImplementedError(f"Task {args.task} not implemented")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}") # type: ignore
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        torch.backends.cudnn.benchmark = True
    
    # SETUP DATA
    train_loader, val_loader, test_loader, sensingmasks = create_data(args)
    
    # SETUP MODEL
    model, aggregator = setup_model(args, sensingmasks, device, task_cat)
    count_parameters(model)
    count_parameters(aggregator)
    print("--------------------------")
    
    model = model.to(device)
    aggregator = aggregator.to(device)
    
    if torch.cuda.is_available():
        print("Compiling model and aggregator with torch.compile...")  
        model = torch.compile(model, mode='reduce-overhead', fullgraph=True) # also should try: "max-autotune"
        aggregator = torch.compile(aggregator, mode='reduce-overhead', fullgraph=True)
        print("torch.compile done.")
    
    # SET UP TRAINING
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aggregator.parameters()), lr=args.lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    criterion = stacked_cross_entropy_loss if task_cat == 'classif' else stacked_MSE
    
    # SET UP LOGGING, CHECKPOINTING, AND EARLY STOPPING
    METRIC_KEYS = {
        "classif": ["loss", "accuracy", "agreement"],
        "regression": ["loss", "mse_m", "diversity_m", "mse_y", "diversity_y"],
    }
    def pack_metrics(values_tuple, keys):
        return dict(zip(keys, values_tuple))
    
    stats = init_stats(task_cat, METRIC_KEYS)
    file_base = unique_filename()
    checkpoint_path = f"{file_base}_checkpoint.pt"
    best = {"loss": float('inf'), "acc": float('-inf')}
    patience_counter = 0
    val_acc = 0.0

    # PRINT TIME
    start = datetime.now()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # TRAINING LOOP
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, aggregator, train_loader, optimizer, criterion, device, scaler)
        scheduler.step()
        stats["train_loss"].append(train_loss)
        
        keyset = METRIC_KEYS["classif"] if task_cat == "classif" else METRIC_KEYS["regression"]
        train_eval = evaluate(model, aggregator, train_loader, criterion, device, task=task_cat)
        val_eval   = evaluate(model, aggregator, val_loader,   criterion, device, task=task_cat)
        train_metrics = pack_metrics(train_eval, keyset)
        val_metrics   = pack_metrics(val_eval,   keyset)
        
        for metric_name in keyset:
            if metric_name == "loss":
                stats["val_loss"].append(val_metrics["loss"])
                continue
            stats[f"t_{metric_name}"].append(train_metrics[metric_name])
            stats[f"val_{metric_name}"].append(val_metrics[metric_name])
        
        """ if task_cat == 'classif':
            _, t_acc, t_agree = evaluate(                                       # type: ignore
                model, aggregator, train_loader, criterion, device, task=task_cat
            ) 
            val_loss, val_acc, val_agree = evaluate(                            # type: ignore
                model, aggregator, val_loader, criterion, device, task=task_cat
            ) 
            stats["t_accuracy"].append(t_acc)
            stats["t_agreement"].append(t_agree)
            stats["val_loss"].append(val_loss)
            stats["val_accuracy"].append(val_acc)
            stats["val_agreement"].append(val_agree)
        else:
            _, t_mse_m, t_diversity_m, t_mse_y, t_diversity_y = evaluate(            # type: ignore
                model, aggregator, train_loader, criterion, device, task=task_cat
            )
            val_loss, val_mse_m, val_diversity_m, val_mse_y, val_diversity_y = evaluate(  # type: ignore
                model, aggregator, val_loader, criterion, device, task=task_cat
            )
            stats["t_mse_m"].append(t_mse_m)
            stats["t_diversity_m"].append(t_diversity_m)
            stats["t_mse_y"].append(t_mse_y)
            stats["t_diversity_y"].append(t_diversity_y)
            stats["val_loss"].append(val_loss)
            stats["val_mse_m"].append(val_mse_m)
            stats["val_diversity_m"].append(val_diversity_m)
            stats["val_mse_y"].append(val_mse_y)
            stats["val_diversity_y"].append(val_diversity_y) """
        
        if epoch == 1:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = datetime.now()
            print(f"Time elapsed for first epoch: {(t1 - start).total_seconds():.4f} seconds.")
            
        if epoch % 10 == 0 or epoch == 1:
            printlog(task_cat, epoch, stats, METRIC_KEYS)
        
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"] if task_cat == 'classif' else 0.0
        
        if task_cat == 'classif':
            improved = (val_acc > best["acc"] + 1e-5) | \
                (val_acc >= best["acc"] - 1e-2 and val_loss < best["loss"] - 1e-5)
        else:
            improved = val_loss < best["loss"] - 1e-5
        
        if improved or epoch == 1:
            if task_cat == 'classif':
                best.update(loss=val_loss, acc=val_acc, epoch=epoch)
            else:
                best.update(loss=val_loss, epoch=epoch)
            atomic_save(snapshot(model, aggregator, epoch, args), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if val_acc == 1.0:
            print(f"Early stopping at epoch {epoch}; validation accuracy is 100%.")
            break
        if val_loss > (10 * stats['val_loss'][0]):
            print(f"Early stopping at epoch {epoch}; validation loss is diverging.")
            break
    #    if args.patience > 0 and patience_counter >= args.patience:
    #        print(f"Early stopping at epoch {epoch}; no improvement for {args.patience} epochs.")
    #        break
        
    end = datetime.now()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training time: {(end - start).total_seconds() / 60:.4f} minutes.")
    
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

    # SAVE TRAINING CURVE PLOT
    if task_cat == 'classif':
        random_accuracy = 1.0 / args.m
        plot_classif(stats, file_base, random_accuracy)
    else:
        plot_regression(stats, file_base)
    
    # EVALUATE ON TEST DATA
    test_stats = final_test(model, aggregator, test_loader, criterion, device, task_cat)

    # SAVE LOGS
    log_training_run(
        file_base, args, stats, test_stats, 
        start, end, model, aggregator, task_cat
    )

    #file_prefix = Path(file_base).name  # Extracts just 'run_YYYYMMDD_HHMMSS'
    #plot_connectivity_matrices("results", prefix=file_prefix, cmap="coolwarm")
    