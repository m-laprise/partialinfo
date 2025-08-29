import os

import torch
from torch.utils.data import DataLoader

from datagen_temporal import GTMatrices, SensingMasks, TemporalData


def create_data(args):
    if args.train_n // args.nres == 0:
        raise ValueError("One data-generating process will be split between training and validation sets.",
                         "Set nres to be a divisor of train_n to avoid leakage.")
    with torch.no_grad():  
        totN = args.train_n + args.val_n + args.test_n
        all_GT = GTMatrices(N=totN, t=args.t, m=args.m, r=args.r, 
                              realizations = args.nres, mode=args.gt_mode, kernel=args.kernel, vtype=args.vtype)
        all_data = TemporalData(all_GT, task=args.task, verbose=True)
        sensingmasks = SensingMasks(all_data, args.r, args.num_agents, args.density)
        train_data, val_data, test_data = torch.utils.data.random_split(
            all_data, [args.train_n, args.val_n, args.test_n]
        )
    
    num_workers = min(os.cpu_count() // 2, 4) if torch.cuda.is_available() else 0 # type: ignore
    print(f"Number of workers: {num_workers}")
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=pin, 
        num_workers=num_workers,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        pin_memory=pin, 
        num_workers=num_workers,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        pin_memory=pin, 
        num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, sensingmasks