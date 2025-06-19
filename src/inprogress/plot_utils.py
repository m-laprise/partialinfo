import glob
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


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

