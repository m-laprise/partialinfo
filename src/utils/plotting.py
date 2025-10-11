#import glob
#import os

#import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_stats(stats, filename_base, true_nuclear_mean, true_gap_mean, true_variance):
    epochs = np.arange(1, len(stats["train_loss"]) + 1)
    # Plot loss-related metrics in two panels
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=320)
    if min(stats["train_loss"]) >= 0:
        axs[0].plot(epochs, np.log(stats["train_loss"]), label="Train Loss", color='tab:blue')
    else:
        log_offset_train_loss = np.log(stats["train_loss"] - min(stats["train_loss"]))
        axs[0].plot(epochs, log_offset_train_loss, label="Train Loss", color='tab:blue')
    axs[0].plot(epochs, np.log(stats["t_penalty"]), label="Penalty component", 
                color='tab:grey', linestyle='--')
    axs[0].set_title("Log Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("")
    axs[0].grid(True)
    axs[0].legend()
    #axs[0].set_ylim(0, 2)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].plot(epochs, stats["t_mse_known"], 
                label="Train MSE Known Entries", color='tab:blue')
    axs[1].plot(epochs, stats["t_mse_unknown"], 
                label="Train MSE Unknown Entries", color='tab:purple')
    #axs[1].axhline(y=true_variance, label="Ref Var of Entries", 
    #               color='tab:grey', linestyle='--')
    axs[1].plot(epochs, stats["val_mse_known"], linestyle='dotted', 
                label="Val MSE Known Entries", color='tab:green')
    axs[1].plot(epochs, stats["val_mse_unknown"], linestyle='dotted', 
                label="Val MSE Unknown Entries", color='tab:orange')
    axs[1].set_title("Training & Validation MSE")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("")
    axs[1].grid(True)
    #axs[1].set_ylim(0, 2)
    axs[1].legend()
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].plot(epochs, stats["t_variance"],
                label="Train Var between Agents", color='tab:grey')
    axs[2].plot(epochs, stats["val_variance"], linestyle='dotted', 
                label="Val Var between Agents", color='tab:red')
    axs[2].set_title("Training & Validation Diversity")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("")
    axs[2].grid(True)
    axs[2].legend()
    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(f"{filename_base}_loss_metrics.png")
    plt.close(fig)
    # Plot spectral diagnostics
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=320)
    ax.plot(epochs, stats["t_nucnorm"], label="Train Nuclear Norm", color='tab:purple')
    ax.plot(epochs, stats["t_gap"], label="Train Spectral Gap", color='tab:orange')
    ax.plot(epochs, stats["val_nucnorm"], label="Val Nuclear Norm", color='tab:blue', linestyle='dotted')
    ax.plot(epochs, stats["val_gap"], label="Val Spectral Gap", color='tab:red', linestyle='dotted')
    ax.axhline(y=true_nuclear_mean, color='tab:purple', linestyle='--', 
               label="Ref Nuclear Norm")
    ax.axhline(y=true_gap_mean, color='tab:orange', linestyle='--', 
               label="Ref Spectral Gap")
    ax.set_title("Spectral Properties Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Singular Value Scale")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(f"{filename_base}_spectral_diagnostics.png")
    plt.close(fig)
    

def plot_classif(stats, filename_base, random_accuracy, naive_full, naive_partial):
    epochs = np.arange(1, len(stats["train_loss"]) + 1)
    # Plot loss-related metrics in two panels
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    axs[0].plot(epochs, np.log(stats["train_loss"]), label="Train Loss", color='tab:blue')
    axs[0].plot(epochs, np.log(stats["val_loss"]), label="Val Loss", color='tab:orange')
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Log Cross-Entropy Loss")
    axs[0].grid(True)
    axs[0].legend()
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[1].plot(epochs, stats["t_accuracy"], label="Train Accuracy", color='tab:blue')
    axs[1].plot(epochs, stats["val_accuracy"], label="Val Accuracy", color='tab:orange')
    axs[1].plot(epochs, stats["t_agreement"], label="Train Agreement", color='tab:blue', linestyle='dotted')
    axs[1].plot(epochs, stats["val_agreement"], label="Val Agreement", color='tab:orange', linestyle='dotted')
    axs[1].axhline(y=random_accuracy, label="Random guessing", color='tab:grey', linestyle='--')
    axs[1].axhline(y=naive_partial, label="Naive pred., partial info", color='tab:red', linestyle='--')
    axs[1].set_title("Classification Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("% Accuracy")
    axs[1].grid(True)
    axs[1].legend()
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(f"{filename_base}_classif_metrics.png")
    plt.close(fig)
    

def plot_regression(stats, filename_base, naive_full, naive_partial):
    xmin = 1
    xmax = len(stats["train_loss"]) + 1
    epochs = np.arange(xmin, xmax)
    # Plot loss-related metrics in two panels
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    axs[0].plot(epochs, np.log(stats["train_loss"]), 
                label="Train Loss (all agents)", color='tab:blue')
    axs[0].plot(epochs, np.log(stats["val_loss"]), 
                label="Val Loss (all agents)", color='tab:orange')
    axs[0].plot(epochs, np.log(stats["val_mse"]), 
                label="Val MSE (collective pred.)", color='tab:green', linestyle='--')
    if naive_full != naive_partial:    
        axs[0].axhline(y=np.log(naive_partial),
                    label="Naive pred., partial info", 
                    color='tab:red', linestyle='dotted')
    axs[0].axhline(y=np.log(naive_full),
                   label="Naive pred., full info", 
                   color='tab:purple', linestyle='dotted')
    axs[0].set_title("Average of prediction error (training loss)\nand error of average prediction")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Log MSE")
    axs[0].grid(True)
    axs[0].legend()
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[1].plot(epochs, stats["t_diversity"], label="Train Diversity", color='tab:blue')
    axs[1].plot(epochs, stats["val_diversity"], label="Val Diversity", color='tab:orange')
    axs[1].set_title("Diversity of prediction across agents")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Variance")
    axs[1].set_ylim(0, None)
    axs[1].grid(True)
    axs[1].legend()
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(f"{filename_base}_regression_metrics.png")
    plt.close(fig)

"""    
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
    # for j in range(i + 1, len(axs)):
    #     axs[j].axis("off")

    # cbar = fig.colorbar(im, ax=axs[:i+1], orientation="vertical", fraction=0.02, pad=0.04)
    # cbar.set_label("Connectivity Strength")
    plt.tight_layout()
    plt.suptitle("Evolution of Agent Connectivity Over Epochs", fontsize=16, y=1.02)
    plt.savefig(os.path.join(directory, f"{prefix}_connectivity_evolution.png"))
    plt.show()
"""
