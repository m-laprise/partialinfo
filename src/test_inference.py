"""
This script generates testing data, loads the best checkpoint of a saved model from the /results/ 
folder and runs it for one step of inference on the first matrix of the testing data.

usage:
❯ uv run ./src/test_inference.py \
  --t 24 --m 24 --r 1 --density 0.7 \
  --num-agents 10 --sensing-rho 0 --nb-ties 4 --hidden-dim 296 \
  --lr 1e-3 --epochs 300 --steps 2 --no-sharedv \
  --batch-size 256 --train-n 6000 --task 'lrmc'

"""
import os
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from cli_config import Config, build_parser_from_dataclass, load_config
from utils.setup import create_data, setup_model

checkpoint_path = "/Users/mlaprise/dev/partialinfo/results/run_20251208_123847_checkpoint.pt"
task_cat = 'reconstruction'

parser = build_parser_from_dataclass(Config)
parsed = parser.parse_args()
cfg = load_config(parsed, Config)

cfg.t = 24
cfg.m = 24
cfg.r = 1
cfg.density = 0.7
cfg.num_agents = 10
cfg.sensing_rho = 0
cfg.nb_ties = 4
cfg.hidden_dim = 296
cfg.lr = 1e-3
cfg.epochs = 300
cfg.steps = 2
cfg.sharedv = False
cfg.batch_size = 256
cfg.train_n = 6000
cfg.task = 'lrmc'
print("Effective config:", asdict(cfg))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SETUP DATA
_, _, test_loader, sensingmasks, refnuc, refgap, refvar = create_data(cfg)
globalmask = sensingmasks.global_known      # type: ignore

# SETUP MODEL
model, aggregator = setup_model(cfg, sensingmasks, device, task_cat)
model = model.to(device)
aggregator = aggregator.to(device)  # type: ignore

if os.path.exists(checkpoint_path):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    if aggregator is not None:
        aggregator.load_state_dict(state["aggregator"])
        aggregator.to(device)
    print(f"Loaded best model (epoch {state['epoch']}) from checkpoint.")
else:
    print("Checkpoint not found.")
    
# Extract first testing example
test_data = next(test_loader.__iter__())
test_matrix = test_data['matrix'][0].to(device)
# Run forward pass
with torch.no_grad():
    ground_truth = test_matrix.reshape(cfg.t, cfg.m)
    agents_outputs = aggregator(model(test_matrix)).squeeze(0) # shape [num_agents, t * m]
    collective_output = torch.mean(agents_outputs, dim=0).reshape(cfg.t, cfg.m)
    err = (ground_truth - collective_output).abs()

gt_np   = ground_truth.detach().cpu().numpy()
coll_np = collective_output.detach().cpu().numpy()
err_np = err.detach().cpu().numpy()

# choose 2 random agents
agts = torch.randperm(agents_outputs.shape[0])[:2]
# sensingmasks.masks is [num_agents, t * m]
agts_masks = sensingmasks.masks[agts, :] #type:ignore
agent_masks_np = []
for i in range(len(agts)):
    if isinstance(agts_masks, torch.Tensor):
        mask_flat = agts_masks[i].detach().cpu().numpy()
    else:
        mask_flat = np.asarray(agts_masks[i])
    mask_2d = mask_flat.reshape(cfg.t, cfg.m).astype(bool)
    agent_masks_np.append(mask_2d)

agent_recons_np = []
agent_errs_np   = []
for idx in agts:
    recon = agents_outputs[idx].reshape(cfg.t, cfg.m)
    err   = (ground_truth - recon).abs()
    agent_recons_np.append(recon.detach().cpu().numpy())
    agent_errs_np.append(err.detach().cpu().numpy())

# collective error
err_collective     = (ground_truth - collective_output).abs()
err_collective_np  = err_collective.detach().cpu().numpy()

# PLOT

# --- Shared normalization for all error plots ---
all_err = np.stack([err_collective_np] + agent_errs_np, axis=0)
#vmin = 0.0
#vmax = np.percentile(all_err, 99)  # clip outliers; or use all_err.max()

fig, axs = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)
axs[0, 0].imshow(gt_np, cmap='viridis')
axs[0, 0].set_title('Ground Truth')
axs[0, 0].axis('off')
axs[0, 1].imshow(coll_np, cmap='viridis')
axs[0, 1].set_title('Collective Reconstruction')
axs[0, 1].axis('off')
im_err0 = axs[0, 2].imshow(
    err_collective_np,
    cmap='magma',
    #vmin=vmin, vmax=vmax,
)
axs[0, 2].set_title('Collective Abs. Error')
axs[0, 2].axis('off')
for row in range(2):
    mask = agent_masks_np[row]
    known = int(mask.sum())
    # Mask: black (True) / white (False)
    axs[row + 1, 0].imshow(mask, cmap='gray_r', vmin=0, vmax=1)
    axs[row + 1, 0].set_title(f'Agent {agts[row].item()} Mask\n{known} known entries')
    axs[row + 1, 0].axis('off')
    # agent reconstruction
    axs[row + 1, 1].imshow(agent_recons_np[row], cmap='viridis')
    axs[row + 1, 1].set_title(f'Agent {agts[row].item()} Reconstruction')
    axs[row + 1, 1].axis('off')
    # agent-specific error
    axs[row + 1, 2].imshow(
        agent_errs_np[row],
        cmap='magma',
        #vmin=vmin, vmax=vmax,
    )
    axs[row + 1, 2].set_title(f'Agent {agts[row].item()} Abs. Error')
    axs[row + 1, 2].axis('off')
cbar = fig.colorbar(
    im_err0,
    ax=axs[:, 2].ravel().tolist(),
    location='right',     # explicit side
    fraction=0.046,
    pad=0.04,
)
cbar.set_label('Error magnitude')

# save plot to /results/ folder
plt.savefig(f"results/inference_plot_{cfg.task}.png")
plt.close()
print("...")