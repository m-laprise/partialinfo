# cli_config.py
import argparse
import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict


@dataclass
class Config:
    # Ground truth 
    t: int = field(default=50, metadata={"help": "Number of rows in each ground truth matrix"})
    m: int = field(default=25, metadata={"help": "Number of columns in each ground truth matrix"})
    r: int = field(default=25, metadata={"help": "Rank of each ground truth matrix"})
    gt_mode: str = field(
        default="value", 
        metadata={"choices": ["value", "return"], 
                  "help": "Kind of ground truth matrices (absolute value or relative return)"})
    kernel: str = field(
        default="cauchy", 
        metadata={"choices": ["matern", "cauchy", "whitenoise"], 
                  "help": "Type of vcov for columns of U factors"})
    vtype: str = field(
        default="random", 
        metadata={"choices": ["random", "block"], "help": "Random or block diagonal V factors"})
    nres: int = field(default=10, metadata={"help": "Number of realizations per DGP"})
    u_only: bool = field(default=False, metadata={"help": "If true, GTMatrices returns U only (U_only mode)"})
    
    # Task and sensing hyperparameters
    task: str = field(
        default="nonlin_function", 
        metadata={"choices": ["argmax", "lastrow", "nextrow", "nonlinear_seq", "nonlin_function"], 
                  "help": "Prediction task: argmax (classification), nextrow (regression), nonlinear_seq, or nonlin_function (timewise regression)"})
    density: float = field(default=0.5, metadata={"help": "Target proportion of known entries"})
    num_agents: int = field(default=20, metadata={"help": "Number of agents"})
    sensing_rho: float = field(default=0.25, 
                               metadata={"help": "Controls degree of overlap in sensing mask"})
    sensing_gamma: float = field(default=0.0, 
                                 metadata={"help": "Controls degree of inequality in sensing mask"})

    # Model hyperparameters
    hidden_dim: int = field(default=128, metadata={"help": "Hidden dimension of the model"})
    sharedv: bool = field(      # boolean toggle: sharedv / no-sharedv
        default=True, 
        metadata={"help": "Agents share a V embedding matrix (use --no-sharedv to disable)"})
    
    # Message passing hyperparameters
    memory: bool = field(
        default=True,
        metadata={
            "help": "Dynamic version where agents with memory process rows one by one (use --no-memory to disable)"
        }
    )
    att_heads: int = field(default=4, metadata={"help": "Number of attention heads"})
    adjacency_mode: str = field(
        default="learned", 
        metadata={"choices": ["none", "learned"], "help": "Adjacency mode (all-to-all or learned)"})
    nb_ties: int = field(default=4, metadata={"help": "Number of ties per agent"})
    steps: int = field(default=5, metadata={"help": "Number of message passing steps"})

    # Training hyperparameters
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability"})
    lr: float = field(default=0.001, metadata={"help": "Initial learning rate"})
    epochs: int = field(default=50, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=64, metadata={"help": "Batch size"})
    patience: int = field(default=0, metadata={"help": "Early stopping patience"})
    train_n: int = field(default=800, metadata={"help": "Number of training matrices"})
    val_n: int = field(default=400, metadata={"help": "Number of validation matrices"})
    test_n: int = field(default=400, metadata={"help": "Number of test matrices"})


def build_parser_from_dataclass(cfg_cls) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (overrides defaults)")

    for f in fields(cfg_cls):
        name = f"--{f.name.replace('_','-')}"
        meta = f.metadata or {}
        if f.type == bool:
            # Add paired flags: --no-foo when default True, otherwise --foo to enable
            if f.default is True:
                parser.add_argument(f"--no-{f.name.replace('_','-')}", dest=f.name, action="store_false", help=meta.get("help"))
                parser.set_defaults(**{f.name: True})
            else:
                parser.add_argument(name, dest=f.name, action="store_true", help=meta.get("help"))
                parser.set_defaults(**{f.name: f.default})
            continue

        kwargs: Dict[str, Any] = {"type": f.type, "default": f.default, "help": meta.get("help", "")}
        if "choices" in meta:
            kwargs["choices"] = meta["choices"]
        parser.add_argument(name, **kwargs)

    return parser

def load_config(args_namespace, cfg_cls):
    # If --config JSON passed, merge it (JSON keys override defaults)
    args = vars(args_namespace)
    cfg_path = args.pop("config", None)
    if cfg_path:
        with open(cfg_path, "r") as fh:
            file_cfg = json.load(fh)
        # update args with values from file_cfg
        args.update({k: v for k, v in file_cfg.items() if k in {f.name for f in fields(cfg_cls)}})
    # build dataclass from args
    return cfg_cls(**{k: v for k, v in args.items() if k in {f.name for f in fields(cfg_cls)}})
