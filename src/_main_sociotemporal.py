"""
Socio-temporal training script: agents exchange messages over time with causal memory
and social attention per time step. At each time step, agents predict a label (argmax over m).

Inputs can be provided as TemporalData matrices; masking is column-wise per agent via
SensingMasksTemporal. Memory attention is causal (no future leakage).

Example:
uv run ./src/main_sociotemporal.py \
  --t 50 --m 25 --r 6 --density 0.5 \
  --num-agents 16 --nb_ties 4 --hidden-dim 64 \
  --lr 3e-4 --epochs 50 --steps 3 \
  --batch-size 64 --train-n 1000 --no-sharedv \
  --gt-mode 'value' --kernel 'cauchy' --vtype 'random' --task 'argmax'
"""

from dataclasses import asdict
from datetime import datetime
import os
import gc

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from cli_config import Config, build_parser_from_dataclass, load_config
from datautils.datagen_temporal import GTMatrices, TemporalData
from datautils.sensing import SensingMasksTemporal
from dotGAT import DistributedDoTGATTimeSeries
from utils.logging import atomic_save, init_stats, log_training_run, printlog, snapshot
from utils.misc import count_parameters, unique_filename


def _btm_from_batch(batch, t: int, m: int, device: torch.device) -> torch.Tensor:
    x = batch['matrix'].to(device, non_blocking=True)
    # Accept [B, t*m] or [B, 1, t*m] -> [B, t, m]
    if x.dim() == 2 and x.shape[1] == t * m:
        return x.view(-1, t, m)
    if x.dim() == 3 and x.shape[-1] == t * m:
        B = x.shape[0]
        return x.view(B, t, m)
    if x.dim() == 3 and x.shape[1:] == (t, m):
        return x
    raise ValueError(f"Unexpected matrix shape {tuple(x.shape)}; expected [B,t*m], [B,1,t*m], or [B,t,m]")


def _timewise_cross_entropy(logits_btam: torch.Tensor, targets_bt: torch.Tensor) -> torch.Tensor:
    """Compute CE over time and agents.
    logits_btam: [B, T, A, m]
    targets_bt:  [B, T] (class indices 0..m-1)
    Returns scalar mean loss.
    """
    if logits_btam.dim() != 4:
        raise ValueError(f"Expected logits [B,T,A,m], got {logits_btam.shape}")
    B, T, A, m = logits_btam.shape
    if targets_bt.shape != (B, T):
        raise ValueError(f"Expected targets [B,T], got {targets_bt.shape}")
    targets_bt = targets_bt.long()
    # Expand to [B,T,A]
    targets_bta = targets_bt.unsqueeze(-1).expand(-1, -1, A)
    logits_flat = logits_btam.reshape(B * T * A, m)
    targets_flat = targets_bta.reshape(B * T * A)
    return nn.CrossEntropyLoss(reduction='mean')(logits_flat, targets_flat)


def classif_vote_metrics(logits_btam: torch.Tensor, targets_bt: torch.Tensor):
    """Majority-vote accuracy and agreement over time.
    logits_btam: [B, T, A, m]
    targets_bt:  [B, T]
    Returns: (acc, agree)
    """
    B, T, A, m = logits_btam.shape
    preds_bta = logits_btam.argmax(dim=-1)  # [B,T,A]
    targets_bt = targets_bt.long()

    # counts per class per (B,T)
    counts = torch.zeros(B, T, m, device=logits_btam.device, dtype=torch.int32)
    ones = torch.ones_like(preds_bta, dtype=torch.int32)
    counts.scatter_add_(2, preds_bta, ones)  # [B,T,m]
    maj_class = counts.argmax(dim=2)         # [B,T]

    acc = (maj_class == targets_bt).float().mean().item()
    agree = (counts.max(dim=2).values.float() / A).mean().item()
    return acc, agree


def create_temporal_loaders(cfg: Config):
    with torch.no_grad():
        totN = cfg.train_n + cfg.val_n + cfg.test_n
        all_GT = GTMatrices(
            N=totN, t=cfg.t, m=cfg.m, r=cfg.r,
            realizations=cfg.nres, mode=cfg.gt_mode, kernel=cfg.kernel, vtype=cfg.vtype
        )
        all_data = TemporalData(all_GT, task=cfg.task, verbose=True)
        train_data, val_data, test_data = torch.utils.data.random_split(
            all_data, [cfg.train_n, cfg.val_n, cfg.test_n]
        )

    import os as _os
    num_workers = min(_os.cpu_count() // 2, 4) if torch.cuda.is_available() else 0  # type: ignore
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    def _dl(ds, shuffle=False):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            pin_memory=pin,
            num_workers=num_workers,
            persistent_workers=persistent,
            prefetch_factor=2 if persistent else None,
            drop_last=shuffle,
        )

    return _dl(train_data, True), _dl(val_data), _dl(test_data), all_data


if __name__ == "__main__":
    parser = build_parser_from_dataclass(Config)
    parsed = parser.parse_args()
    cfg = load_config(parsed, Config)

    if cfg.task != 'argmax':
        raise NotImplementedError("This script currently supports 'argmax' classification only.")

    print("Effective config:", asdict(cfg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")  # type: ignore
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Data and loaders
    train_loader, val_loader, test_loader, full_data = create_temporal_loaders(cfg)

    # Column-wise per-agent masks for time-series
    smt = SensingMasksTemporal(full_data, num_agents=cfg.num_agents, rho=cfg.sensing_rho)

    # Model
    model = DistributedDoTGATTimeSeries(
        device=device,
        m=cfg.m,
        num_agents=cfg.num_agents,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.att_heads,
        message_steps=cfg.steps,
        dropout=cfg.dropout,
        adjacency_mode=cfg.adjacency_mode,
        sharedV=cfg.sharedv,
        k=cfg.nb_ties,
        sensing_masks_temporal=smt,
        y_dim=cfg.m,  # per-timestep classification over m classes
    ).to(device)

    count_parameters(model)
    print("--------------------------")

    if torch.cuda.is_available():
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead', fullgraph=True)  # type: ignore
        print("torch.compile done.")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    METRIC_KEYS = ["loss", "accuracy", "agreement"]
    def pack_metrics(values_tuple):
        return dict(zip(METRIC_KEYS, values_tuple))

    stats = init_stats('classif', { 'classif': METRIC_KEYS })
    file_base = unique_filename()
    checkpoint_path = f"{file_base}_checkpoint.pt"
    best = {"loss": float('inf'), "acc": float('-inf')}

    # Training/eval loops
    def run_epoch(loader, train_mode: bool):
        if train_mode:
            model.train()
        else:
            model.eval()
        use_amp = (device.type == 'cuda') and (scaler is not None)
        autocast_ctx = autocast(device_type='cuda') if use_amp else torch.no_grad()

        total_loss = 0.0
        total_examples = 0
        total_acc = 0.0
        total_agree = 0.0

        for batch in loader:
            if device.type == 'cuda' and train_mode:
                try:
                    torch.compiler.cudagraph_mark_step_begin()
                except Exception:
                    pass

            x_btm = _btm_from_batch(batch, cfg.t, cfg.m, device)  # [B,T,M]
            targets_bt = batch['matrix'].to(device, non_blocking=True)
            # Labels per time: argmax over m on ground-truth rows (no future leakage when computing labels)
            if targets_bt.dim() == 2 and targets_bt.shape[1] == cfg.t * cfg.m:
                targets_bt = targets_bt.view(-1, cfg.t, cfg.m).argmax(dim=-1)  # [B,T]
            elif targets_bt.dim() == 3 and targets_bt.shape[-1] == cfg.t * cfg.m:
                B = targets_bt.shape[0]
                targets_bt = targets_bt.view(B, cfg.t, cfg.m).argmax(dim=-1)
            elif targets_bt.dim() == 3 and targets_bt.shape[1:] == (cfg.t, cfg.m):
                targets_bt = targets_bt.argmax(dim=-1)
            else:
                raise ValueError(f"Unexpected matrix shape for labels {tuple(targets_bt.shape)}")

            B = targets_bt.size(0)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                h, y = model(x_btm)                  # y: [B,T,A,m]
                if y is None:
                    raise RuntimeError("Model was not configured with y_dim; expected per-timestep logits.")
                loss = _timewise_cross_entropy(y, targets_bt)

            if train_mode:
                if use_amp:
                    scaler.scale(loss).backward()    # type: ignore
                    scaler.unscale_(optimizer)       # type: ignore
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)           # type: ignore
                    scaler.update()                  # type: ignore
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            acc, agree = classif_vote_metrics(y.detach(), targets_bt)
            total_loss += loss.item() * B
            total_acc += acc * B
            total_agree += agree * B
            total_examples += B

        mean_loss = total_loss / max(total_examples, 1)
        mean_acc = total_acc / max(total_examples, 1)
        mean_agree = total_agree / max(total_examples, 1)
        return mean_loss, mean_acc, mean_agree

    # Baseline (optional): skipped here to keep script focused on the socio-temporal model

    start = datetime.now()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(1, cfg.epochs + 1):
        train_vals = run_epoch(train_loader, True)
        scheduler.step()
        stats["train_loss"].append(train_vals[0])

        val_vals = run_epoch(val_loader, False)
        train_metrics = pack_metrics(train_vals)
        val_metrics = pack_metrics(val_vals)

        stats["val_loss"].append(val_metrics["loss"])
        stats["t_accuracy"].append(train_metrics["accuracy"])  # reuse keys
        stats["t_agreement"].append(train_metrics["agreement"])  # reuse keys
        stats["val_accuracy"].append(val_metrics["accuracy"])
        stats["val_agreement"].append(val_metrics["agreement"])

        if epoch == 1:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = datetime.now()
            print(f"Time elapsed for first epoch: {(t1 - start).total_seconds():.4f} seconds.")

        if epoch % 10 == 0 or epoch == 1:
            printlog('classif', epoch, stats, { 'classif': ["loss", "accuracy", "agreement"] })

        # checkpointing
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        improved = (val_acc > best["acc"] + 1e-5) or (
            val_acc >= best["acc"] - 1e-2 and val_loss < best["loss"] - 1e-5
        )
        if improved or epoch == 1:
            best.update(loss=val_loss, acc=val_acc, epoch=epoch)  # type: ignore
            atomic_save(snapshot(model, None, epoch, cfg), checkpoint_path)

        if val_acc == 1.0 or val_loss < 1e-5:
            print(f"Early stopping at epoch {epoch}; validation accuracy is 100% or loss is tiny.")
            break
        if val_loss > (10 * stats['val_loss'][0]):
            print(f"Early stopping at epoch {epoch}; validation loss is diverging.")
            break

    end = datetime.now()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training time: {(end - start).total_seconds() / 60:.4f} minutes.")

    # Load best
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])  # type: ignore
        model.to(device)
        print(f"Loaded best model (epoch {state['epoch']}) from checkpoint.")
    else:
        print("No checkpoint found; using current in-memory weights.")

    # Simple test evaluation
    test_loss, test_acc, test_agree = run_epoch(test_loader, False)
    print("Test Set Performance | ",
          f"Loss: {test_loss:.2e}, Accuracy: {test_acc:.2f}, % maj: {test_agree:.2f}")

    # Save logs
    log_training_run(
        unique_filename(prefix="socio"), cfg, {
            'train_loss': stats['train_loss'],
            'val_loss': stats['val_loss'],
            't_accuracy': stats['t_accuracy'],
            'val_accuracy': stats['val_accuracy'],
            't_agreement': stats['t_agreement'],
            'val_agreement': stats['val_agreement'],
        }, (start, end)[0], (start, end)[1], model, None, 'classif'
    )
