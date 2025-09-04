"""
Socio-temporal regression training: each agent outputs a scalar per time step; loss is
per-agent MSE to labels, averaged over time and then averaged across agents for backprop.

Task: cfg.task must be 'nonlin_function' (TemporalData yields labels [T,1]).
"""

import gc
import os
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from cli_config import Config, build_parser_from_dataclass, load_config
from datautils.datagen_temporal import GTMatrices, TemporalData
from datautils.sensing import SensingMasksTemporal
from dotGAT import DistributedDoTGATTimeSeries
from utils.logging import atomic_save, init_stats, log_training_run, printlog, snapshot
from utils.misc import count_parameters, unique_filename


def _btm_from_batch(batch, t: int, m: int, device: torch.device) -> torch.Tensor:
    x = batch['matrix'].to(device, non_blocking=True)
    if x.dim() == 2 and x.shape[1] == t * m:
        return x.view(-1, t, m)
    if x.dim() == 3 and x.shape[-1] == t * m:
        B = x.shape[0]
        return x.view(B, t, m)
    if x.dim() == 3 and x.shape[1:] == (t, m):
        return x
    raise ValueError(f"Unexpected matrix shape {tuple(x.shape)}; expected [B,t*m], [B,1,t*m], or [B,t,m]")


def _bt_labels_from_batch(batch, t: int, device: torch.device) -> torch.Tensor:
    y = batch['label'].to(device, non_blocking=True)  # [B,T,1] or [T,1]
    if y.dim() == 2 and y.shape == (t, 1):
        return y.unsqueeze(0).squeeze(-1)  # [1,T]
    if y.dim() == 3 and y.shape[-1] == 1:
        return y.squeeze(-1)               # [B,T]
    raise ValueError(f"Unexpected label shape {tuple(y.shape)}; expected [B,T,1] or [T,1]")


def _agent_averaged_mse(y_pred_bta: torch.Tensor, y_true_bt: torch.Tensor) -> torch.Tensor:
    """Compute per-agent MSE over time, then mean across agents.
    y_pred_bta: [B,T,A]
    y_true_bt:  [B,T]
    returns scalar loss
    """
    sqerr = (y_pred_bta - y_true_bt.unsqueeze(-1)) ** 2  # [B,T,A]
    per_agent_mse = sqerr.mean(dim=(0, 1))               # [A]
    return per_agent_mse.mean()                           # scalar


def create_temporal_loaders(cfg: Config):
    with torch.no_grad():
        totN = cfg.train_n + cfg.val_n + cfg.test_n
        all_GT = GTMatrices(
            N=totN, t=cfg.t, m=cfg.m, r=cfg.r,
            realizations=cfg.nres, mode=cfg.gt_mode, kernel=cfg.kernel, vtype=cfg.vtype,
            U_only=cfg.u_only
        )
        all_data = TemporalData(all_GT, task=cfg.task, verbose=True)
        train_data, val_data, test_data = torch.utils.data.random_split(
            all_data, [cfg.train_n, cfg.val_n, cfg.test_n]
        )

    num_workers = min(os.cpu_count() // 2, 4) if torch.cuda.is_available() else 0  # type: ignore
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

    if cfg.task != 'nonlin_function':
        raise NotImplementedError("Set cfg.task to 'nonlin_function' for this regression trainer.")

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

    # Data
    train_loader, val_loader, test_loader, full_data = create_temporal_loaders(cfg)
    # Column-wise per-agent masks
    smt = SensingMasksTemporal(full_data, num_agents=cfg.num_agents, rho=cfg.sensing_rho)

    # Model (time-series, causal). y_dim=1 for scalar per agent per time step.
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
        y_dim=1,
    ).to(device)

    count_parameters(model)
    print("--------------------------")
    if torch.cuda.is_available():
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead', fullgraph=True)  # type: ignore
        print("torch.compile done.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    METRICS = ["loss"]
    stats = init_stats('regression', { 'regression': METRICS })
    file_base = unique_filename(prefix="socio_reg")
    checkpoint_path = f"{file_base}_checkpoint.pt"
    best = {"loss": float('inf')}

    def run_epoch(loader, train_mode: bool):
        if train_mode:
            model.train()
        else:
            model.eval()
        use_amp = (device.type == 'cuda') and (scaler is not None)
        #autocast_ctx = autocast(device_type='cuda') if use_amp else torch.no_grad()
        autocast_ctx = autocast(device_type='cuda') if use_amp else nullcontext()
        
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            if device.type == 'cuda' and train_mode:
                try:
                    torch.compiler.cudagraph_mark_step_begin()
                except Exception:
                    pass

            x_btm = _btm_from_batch(batch, cfg.t, cfg.m, device)           # [B,T,M]
            y_bt = _bt_labels_from_batch(batch, cfg.t, device)              # [B,T]
            B = y_bt.size(0)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                h, y_btay = model(x_btm)        # y: [B,T,A,1]
                if y_btay is None:
                    raise RuntimeError("Model must be configured with y_dim=1.")
                y_bta = y_btay.squeeze(-1)      # [B,T,A]
                loss = _agent_averaged_mse(y_bta, y_bt)

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

            total_loss += loss.item() * B
            total_examples += B

        return total_loss / max(total_examples, 1)

    start = datetime.now()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(train_loader, True)
        scheduler.step()
        stats['train_loss'].append(train_loss)

        val_loss = run_epoch(val_loader, False)
        stats['val_loss'].append(val_loss)

        if epoch == 1:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = datetime.now()
            print(f"Time elapsed for first epoch: {(t1 - start).total_seconds():.4f} seconds.")

        if epoch % 10 == 0 or epoch == 1:
            printlog('regression', epoch, stats, { 'regression': METRICS })

        if val_loss < best['loss'] - 1e-5 or epoch == 1:
            best.update(loss=val_loss, epoch=epoch)
            atomic_save(snapshot(model, None, epoch, cfg), checkpoint_path)

        if val_loss < 1e-6:
            print(f"Early stopping at epoch {epoch}; validation loss is tiny.")
            break
        if val_loss > (10 * stats['val_loss'][0]):
            print(f"Early stopping at epoch {epoch}; validation loss is diverging.")
            break

    end = datetime.now()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training time: {(end - start).total_seconds() / 60:.4f} minutes.")

    # Load best and test
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state['model'])  # type: ignore
        model.to(device)
        print(f"Loaded best model (epoch {state['epoch']}) from checkpoint.")
    else:
        print("No checkpoint found; using current in-memory weights.")

    test_loss = run_epoch(test_loader, False)
    print("Test Set Performance | ", f"Loss: {test_loss:.4f}")

    # Save logs
    log_training_run(
        unique_filename(prefix="socio_reg"), cfg, 
        stats={
            'train_loss': stats['train_loss'],
            'val_loss': stats['val_loss'],
        }, 
        test_stats={
            'test_loss': test_loss,
        }, 
        start_time=start, end_time=end, 
        model=model, aggregator=None, task_cat='regression'
    )
