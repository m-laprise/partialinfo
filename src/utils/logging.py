
import platform
import subprocess

import psutil
import torch


def log_training_run(filename_base, args, stats, test_loss, test_accuracy, test_agreement, start_time, end_time, model, aggregator):
    log_file = f"{filename_base}_log.txt"

    # Git commit hash
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        commit_hash = f"Could not retrieve commit hash: {e}"

    # Hyperparameter groups
    grouped_hparams = {
        "Ground Truth and Sensing Hyperparameters": ['t', 'm', 'r', 'density', 'num_agents'],
        "Model Hyperparameters": ['hidden_dim'],
        "Message Passing Hyperparameters": ['att_heads', 'adjacency_mode', 'steps'],
        "Training Hyperparameters": ['dropout', 'lr', 'epochs', 'batch_size', 'patience', 'train_n', 'val_n', 'test_n']
    }

    # Hardware Info
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
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
        for name, module in [("Main Model", model), ("Aggregator", aggregator)]:
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
        f.write(f"Final Test Accuracy: {test_accuracy:.2f}\n")
        f.write(f"Final Test % in Majority: {test_agreement:.2f}\n\n")

        f.write("Epoch Performance\n")
        f.write("-----------------\n")
        f.write("Epoch | Train Loss | Train Acc | T % maj | Val Loss | Val Acc | V % maj\n")
        f.write("------|------------|-----------|---------|----------|---------|---------\n")
        for i, (tl, ta, tm, vl, va, vm) in enumerate(zip(stats["train_loss"], stats["t_accuracy"], stats["t_agreement"],
                                                 stats["val_loss"], stats["val_accuracy"], stats["val_agreement"]), 1):
            f.write(f"{i:5d} | {tl:.2e}   | {ta:.2f}      | {tm:.2f}    | {vl:.2e} | {va:.2f}    | {vm:.2f}\n")

    print(f"Training log saved to: {log_file}")