import glob
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_stats(stats, filename_base, true_nuclear_mean, true_gap_mean, true_variance):
    epochs = np.arange(1, len(stats["train_loss"]) + 1)
    # Plot loss-related metrics in two panels
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), dpi=320)
    axs[0].plot(epochs, stats["train_loss"], label="Train Loss", color='tab:blue')
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True)
    axs[0].set_ylim(0, 1)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].plot(epochs, stats["t_known_mse"], 
                label="Train MSE Known Entries", color='tab:blue')
    axs[1].plot(epochs, stats["t_unknown_mse"], 
                label="Train MSE Unknown Entries", color='tab:purple')
    axs[1].plot(epochs, stats["t_variance"],
                label="Train Var of Entries", color='tab:grey')
    axs[1].axhline(y=true_variance, label="Mean True Var of Entries", 
                   color='tab:grey', linestyle='--')
    axs[1].plot(epochs, stats["val_known_mse"], linestyle='dotted', 
                label="Val MSE Known Entries", color='tab:green')
    axs[1].plot(epochs, stats["val_unknown_mse"], linestyle='dotted', 
                label="Val MSE Unknown Entries", color='tab:orange')
    axs[1].plot(epochs, stats["val_variance"], linestyle='dotted', 
                label="Val Var of Entries", color='tab:red')
    axs[1].set_title("Training & Validation Loss & Variance")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Metric")
    axs[1].grid(True)
    axs[1].set_ylim(0, 1)
    axs[1].legend()
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(f"{filename_base}_loss_metrics.png")
    plt.close(fig)
    # Plot spectral diagnostics
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=320)
    ax.plot(epochs, stats["t_nuclear_norm"], label="Train Nuclear Norm", color='tab:purple')
    ax.plot(epochs, stats["t_spectral_gap"], label="Train Spectral Gap", color='tab:orange')
    ax.plot(epochs, stats["val_nuclear_norm"], label="Val Nuclear Norm", color='tab:blue', linestyle='dotted')
    ax.plot(epochs, stats["val_spectral_gap"], label="Val Spectral Gap", color='tab:red', linestyle='dotted')
    ax.axhline(y=true_nuclear_mean, color='tab:purple', linestyle='--', 
               label="Mean True Nuclear Norm")
    ax.axhline(y=true_gap_mean, color='tab:orange', linestyle='--', 
               label="Mean True Spectral Gap")
    ax.set_title("Spectral Properties Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Singular Value Scale")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(f"{filename_base}_spectral_diagnostics.png")
    plt.close(fig)
    
    
def plot_connectivity_matrices(directory, prefix, max_cols=5, cmap='coolwarm'):
    # Gather all saved connectivity matrices
    files = sorted(glob.glob(os.path.join(directory, f"{prefix}_adj_epoch*.npy")))
    if not files:
        print("No connectivity files found.")
        return

    num_matrices = len(files)
    num_rows = (num_matrices + max_cols - 1) // max_cols

    fig, axs = plt.subplots(num_rows, max_cols, figsize=(4 * max_cols, 4 * num_rows))
    fig, axs = plt.subplots(num_rows, max_cols, figsize=(4 * max_cols, 4 * num_rows))

    # Normalize axs to always be a flat list of Axes objects
    if num_matrices == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    vmin, vmax = -1.0, 1.0  # clamp color range for consistency
    for i, file in enumerate(files):
        mat = np.load(file)
        im = axs[i].imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        epoch_str = os.path.basename(file).split("epoch")[-1].split(".")[0]
        axs[i].set_title(f"Epoch {epoch_str}")
        axs[i].set_xlabel("Agent")
        axs[i].set_ylabel("Agent")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    cbar = fig.colorbar(im, ax=axs[:i+1], orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Connectivity Strength")
    plt.tight_layout()
    plt.suptitle("Evolution of Agent Connectivity Over Epochs", fontsize=16, y=1.02)
    plt.savefig(os.path.join(directory, f"{prefix}_connectivity_evolution.png"))
    plt.show()

