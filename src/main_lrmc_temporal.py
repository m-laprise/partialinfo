"""
Dot product graph attention network for distributed inductive matrix completion or prediction tasks, 
with learned agent-based message passing setup.

Example usage:
```
uv run ./src/main_lrmc_temporal.py \
  --t 50 --m 25 --r 6 --density 0.5 \
  --num-agents 64 --nb_ties 4 --hidden-dim 64 \
  --lr 1e-4 --epochs 150 --steps 5 \
  --batch-size 100 --train-n 1000 --no-sharedv \
  --gt-mode 'value' --kernel 'cauchy' --vtype 'random' --task 'argmax'
```
"""

import gc
import os
from dataclasses import asdict
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from cli_config import Config, build_parser_from_dataclass, load_config
from utils.logging import atomic_save, init_stats, log_training_run, printlog, snapshot
from utils.misc import count_parameters, unique_filename
from utils.plotting import plot_classif, plot_regression
from utils.setup import create_data, setup_model
from utils.training_temporal import (
    baseline_classif,
    baseline_mse_m,
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

if __name__ == "__main__":
    parser = build_parser_from_dataclass(Config)
    parsed = parser.parse_args()
    cfg = load_config(parsed, Config)

    print("Effective config:", asdict(cfg))
    
    if cfg.task == 'argmax':
        task_cat = 'classif'
    elif cfg.task == 'nonlinear':
        task_cat = 'regression'
    else:
        raise NotImplementedError(f"Task {cfg.task} not implemented")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}") # type: ignore
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        torch.backends.cudnn.benchmark = True
    
    # SETUP DATA
    train_loader, val_loader, test_loader, sensingmasks = create_data(cfg)
    
    # SETUP MODEL
    model, aggregator = setup_model(cfg, sensingmasks, device, task_cat)
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
        list(model.parameters()) + list(aggregator.parameters()), lr=cfg.lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
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
    
    if task_cat == 'classif':
        naive_partial, naive_full = baseline_classif(
            val_loader, 
            sensingmasks.global_known, 
            cfg.t, cfg.m
        )

        print(f"Accuracy for naive prediction on val set with full information: {naive_full:.2f}")
        print(f"Accuracy for naive prediction on val set with partial information: {naive_partial:.2f}")
    else:
        naive_partial, naive_full = baseline_mse_m(
            val_loader, 
            sensingmasks.global_known, 
            cfg.t, cfg.m
        )
        print(f"MSE_m for naive prediction on val set with full information: {naive_full:.4f}")
        print(f"MSE_m for naive prediction on val set with partial information: {naive_partial:.4f}")

    # PRINT TIME
    start = datetime.now()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # TRAINING LOOP
    for epoch in range(1, cfg.epochs + 1):
        # TRAIN
        train_loss = train(model, aggregator, train_loader, optimizer, criterion, device, scaler)
        scheduler.step()
        stats["train_loss"].append(train_loss)
        
        # EVALUATE
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
        
        # PRINT
        if epoch == 1:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = datetime.now()
            print(f"Time elapsed for first epoch: {(t1 - start).total_seconds():.4f} seconds.")
            
        if epoch % 10 == 0 or epoch == 1:
            printlog(task_cat, epoch, stats, METRIC_KEYS)
        
        # CHECKPOINT AND EARLY STOPPING
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
            atomic_save(snapshot(model, aggregator, epoch, cfg), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if val_acc == 1.0 or val_loss < 1e-5:
            print(f"Early stopping at epoch {epoch}; validation accuracy is 100%.")
            break
        if val_loss > (10 * stats['val_loss'][0]):
            print(f"Early stopping at epoch {epoch}; validation loss is diverging.")
            break
    #    if cfg.patience > 0 and patience_counter >= args.patience:
    #        print(f"Early stopping at epoch {epoch}; no improvement for {cfg.patience} epochs.")
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
        random_accuracy = 1.0 / cfg.m
        plot_classif(stats, file_base, random_accuracy)
    else:
        plot_regression(stats, file_base)
    
    # EVALUATE ON TEST DATA
    test_stats = final_test(model, aggregator, test_loader, sensingmasks.global_known,
                            criterion, device, task_cat, cfg)

    # SAVE LOGS
    log_training_run(
        file_base, cfg, stats, test_stats, 
        start, end, model, aggregator, task_cat
    )

    #file_prefix = Path(file_base).name  # Extracts just 'run_YYYYMMDD_HHMMSS'
    #plot_connectivity_matrices("results", prefix=file_prefix, cmap="coolwarm")
    