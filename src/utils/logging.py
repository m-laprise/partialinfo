
import os
import platform
import subprocess
import tempfile
from typing import Dict, List, Mapping, Sequence, Union

import psutil
import torch


def atomic_save(state, path: str):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_ckpt_", suffix=".pt")
    os.close(fd)
    torch.save(state, tmp)
    os.replace(tmp, path)


def snapshot(model, aggregator, epoch, args):
    state = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": epoch,
        "args": vars(args),
    }
    if aggregator is not None:
        state["aggregator"] = {k: v.detach().cpu() for k, v in aggregator.state_dict().items()}
    else:
        state["aggregator"] = {}
    return state
    

def init_stats(
    task_cat: str,
    metric_keys: Union[Mapping[str, Sequence[str]], Sequence[str]]
) -> Dict[str, List[float]]:
    """
    Create the `stats` dict for `task_cat`.

    The function always creates:
      - "train_loss" (filled by train())
      - "val_loss"   (filled by evaluate on val_loader)
      - "t_<metric>" and "val_<metric>" for every metric in the list

    Example:
      METRIC_KEYS = {"classif": ["accuracy","agreement"], "regression": ["mse","diversity", ...]}
      stats = init_stats("classif", METRIC_KEYS)
    """
    if isinstance(metric_keys, Mapping):
        try:
            metrics = list(metric_keys[task_cat])
        except KeyError:
            raise ValueError(f"task_cat '{task_cat}' not found in metric_keys mapping")
    else:
        metrics = list(metric_keys)
    stats: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": []
    }
    for m in metrics:
        stats[f"t_{m}"] = []
        stats[f"val_{m}"] = []
    return stats


def printlog(task_cat, epoch, stats, METRIC_KEYS):
    if task_cat == 'classif':
        print(
            f"Ep {epoch:03d}. ",
            f"T loss: {stats['train_loss'][-1]:.2e} | T acc: {stats['t_accuracy'][-1]:.2f} | T % maj: {stats['t_agreement'][-1]:.2f} | ",
            f"V loss: {stats['val_loss'][-1]:.2e} | V acc: {stats['val_accuracy'][-1]:.2f} | V % maj: {stats['val_agreement'][-1]:.2f}"
        )
    elif task_cat == 'regression':
        # Regression: if mse/diversity keys exist, print the rich format; otherwise, fallback to simple losses
        has_mse = 'val_mse' in stats and len(stats['val_mse']) > 0
        has_div = 'val_diversity' in stats and len(stats['val_diversity']) > 0
        if has_mse and has_div:
            print(
                f"Ep {epoch:03d}. ",
                f"Loss: {stats['train_loss'][-1]:.4f} | T var: {stats.get('t_diversity', [0])[-1]:.2f} || V loss: {stats['val_loss'][-1]:.4f} |",
                f"V mse: {stats['val_mse'][-1]:.4f} | V var: {stats['val_diversity'][-1]:.2f}"
            )
        else:
            print(
                f"Ep {epoch:03d}. ",
                f"T loss: {stats['train_loss'][-1]:.4f} | V loss: {stats['val_loss'][-1]:.4f}"
            )
    elif task_cat == 'reconstruction':
        print(
            f"Ep {epoch:03d}. ",
            f"T loss: {stats['train_loss'][-1]:.4f} | pnlt: {stats['t_penalty'][-1]:.2f} | kn: {stats['t_mse_known'][-1]:.3f} | unk: {stats['t_mse_unknown'][-1]:.3f} ||",
            f"V kn: {stats['val_mse_known'][-1]:.4f} | unk: {stats['val_mse_unknown'][-1]:.4f} ||",
            f"var: {stats['val_variance'][-1]:.2f} | gap: {stats['val_gap'][-1]:.2f} | NN: {stats['val_nucnorm'][-1]:.2f}"
        )
    pass


def log_training_run(
    filename_base, args, stats, test_stats, start_time, end_time, model, aggregator, task_cat
):
    log_file = f"{filename_base}_log.txt"
    if task_cat == 'classif':
        test_loss, test_accuracy, test_agreement = test_stats
    elif task_cat == 'regression':
        # Accept flexible test_stats shapes
        if isinstance(test_stats, (list, tuple)):
            if len(test_stats) == 3:
                test_loss, test_mse, test_diversity = test_stats
            elif len(test_stats) == 1:
                test_loss = test_stats[0]
                test_mse = float('nan')
                test_diversity = float('nan')
            else:
                # Fallback
                test_loss = test_stats[0]
                test_mse = float('nan')
                test_diversity = float('nan')
        else:
            test_loss = float(test_stats)
            test_mse = float('nan')
            test_diversity = float('nan')
    elif task_cat == 'reconstruction':
        test_loss, _, test_mse, test_diversity, _, test_gap, test_nucnorm = test_stats
    else:
        raise NotImplementedError(f"Task {task_cat} not implemented.")
 
    # Git commit hash
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        commit_hash = f"Could not retrieve commit hash: {e}"

    # Hyperparameter groups
    grouped_hparams = {
        "Ground Truth Hyperparameters": ['t', 'm', 'r', 'density', 'nres'],
        "Model Hyperparameters": ['num_agents', 'hidden_dim', 'sharedv'],
        "Message Passing Hyperparameters": ['att_heads', 'nb_ties', 'steps'],
        "Training Hyperparameters": ['dropout', 'lr', 'epochs', 'batch_size', 'patience', 'train_n', 'val_n', 'test_n'],
        "Task Hyperparameters": ['task', 'gt_mode', 'kernel', 'vtype']
    }

    # Hardware Info
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"       # type: ignore
    num_gpus = torch.cuda.device_count()
    cpu_name = platform.processor()
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    python_version = platform.python_version()
    pytorch_version = torch.__version__

    with open(log_file, 'w') as f:
        f.write("Training Run Log\n")
        f.write("================\n\n")
        f.write(f"Timestamp (start): {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Git Commit: {commit_hash}\n\n")

        # Hardware Specs
        f.write("Hardware Specs\n")
        f.write("--------------\n")
        f.write(f"Device: {device_name}\n")
        f.write(f"CUDA Version: {cuda_version}\n")
        f.write(f"Number of GPUs: {num_gpus}\n")
        f.write(f"CPU: {cpu_name}\n")
        f.write(f"System RAM: {ram_gb} GB\n")
        f.write(f"Python Version: {python_version}\n")
        f.write(f"PyTorch Version: {pytorch_version}\n")
        f.write(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})\n\n")

        # Hyperparameters
        for section, keys in grouped_hparams.items():
            f.write(f"{section}\n")
            f.write(f"{'-' * len(section)}\n")
            for key in keys:
                value = getattr(args, key, 'N/A')
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
        # Model parameter counts
        f.write("Trainable Parameters\n")
        f.write("---------------------\n")
        modules = [("Main Model", model)] + (([("Aggregator", aggregator)] if aggregator is not None else []))
        for name, module in modules:
            total = sum(p.numel() for p in module.parameters() if p.requires_grad)
            f.write(f"{name} - Total: {total:,} parameters\n")
            f.write("Breakdown by layer:\n")
            for pname, param in module.named_parameters():
                if param.requires_grad:
                    f.write(f"  {pname:50} {param.numel():>10}\n")
            f.write("\n")

        # Training Results
        f.write("Training Results\n")
        f.write("----------------\n")
        training_time = (end_time - start_time).total_seconds() / 60
        f.write(f"Total Training Time: {training_time:.2f} minutes\n")
        f.write(f"Final Test Loss: {test_loss:.2e}\n")
        if task_cat == 'classif':
            f.write(f"Final Test Accuracy: {test_accuracy:.2f}\n")
            f.write(f"Final Test % in Majority: {test_agreement:.2f}\n\n")

            f.write("Epoch Performance\n")
            f.write("-----------------\n")
            f.write("Epoch | Train Loss | Train Acc | T % maj | Val Loss | Val Acc | V % maj\n")
            f.write("------|------------|-----------|---------|----------|---------|---------\n")
            for i, (tl, ta, tm, vl, va, vm) in enumerate(
                zip(stats["train_loss"], stats["t_accuracy"], stats["t_agreement"],
                    stats["val_loss"], stats["val_accuracy"], stats["val_agreement"]), 1):
                f.write(f"{i:5d} | {tl:.2e}   | {ta:.2f}      | {tm:.2f}    | {vl:.2e} | {va:.2f}    | {vm:.2f}\n")
        
        elif task_cat == 'regression':
            if not (torch.isnan(torch.tensor(test_mse)) or torch.isnan(torch.tensor(test_diversity))):
                f.write(f"Final Test MSE: {test_mse:.2e}\n")
                f.write(f"Final Test Diversity: {test_diversity:.2f}\n\n")
            else:
                f.write("\n")

            f.write("Epoch Performance\n")
            f.write("-----------------\n")
            if "val_mse" in stats and "val_diversity" in stats:
                f.write("Epoch | Train Loss | V loss   | V mse    | V var   \n")
                f.write("------|------------|----------|----------|---------\n")
                for i, (tl, vl, vmm, vvm) in enumerate(
                    zip(stats["train_loss"], stats["val_loss"], stats["val_mse"], stats["val_diversity"]), 1
                ):
                    f.write(f"{i:5d} | {tl:.2e}   | {vl:.2e} | {vmm:.2e} | {vvm:.4f}\n")
            else:
                f.write("Epoch | Train Loss | Val Loss\n")
                f.write("------|------------|---------\n")
                for i, (tl, vl) in enumerate(zip(stats["train_loss"], stats["val_loss"]), 1):
                    f.write(f"{i:5d} | {tl:.2e}   | {vl:.2e}\n")
        elif task_cat == 'reconstruction':
            f.write(f"Final Test Loss: {test_loss:.2e}\n")
            f.write(f"Final Test MSE (unknown): {test_mse:.3f}\n")
            f.write(f"Final Test Diversity: {test_diversity:.2f}\n")
            f.write(f"Final Test Mean Spectral Gap: {test_gap:.2f}\n")
            f.write(f"Final Test Mean Nuclear Norm: {test_nucnorm:.2f}\n\n")
            
            f.write("Epoch Performance\n")
            f.write("-----------------\n")
            f.write("Epoch | Train Loss | Pnlty | kn    | unk   | var  | gap  | NN   | Val unk\n")
            f.write("------|------------|-------|-------|-------|------|------|------|-------\n")
            for i, (tl, pn, kn, unk, var, gap, nn, val_unk) in enumerate(
                zip(stats["train_loss"], stats["t_penalty"], stats["t_mse_known"], stats["t_mse_unknown"],
                    stats["val_variance"], stats["val_gap"], stats["val_nucnorm"], stats["val_mse_unknown"]), 1
            ):
                f.write(f"{i:5d} | {tl:.2e}   | {pn:.2f}  | {kn:.3f} | {unk:.3f} | {var:.2f} | {gap:.2f} | {nn:.2f} | {val_unk:.3f}\n")

    print(f"Training log saved to: {log_file}")
