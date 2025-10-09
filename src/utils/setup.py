import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datautils.datagen_temporal import GTMatrices, TemporalData
from datautils.sensing import SensingMasks, SensingMasksTemporal
from dotGAT import (
    Aggregator,
    CollectiveClassifier,
    CollectiveInferPredict,
    DistributedDotGAT,
    DynamicDotGAT,
    ReconDecoder,
)
from utils.misc import sequential_split


def create_data(args):
    if args.task == 'lrmc':
        structured = False
    else:
        structured = True
        if args.train_n // args.nres == 0:
            if args.nres != (args.train_n + args.val_n + args.test_n):
                raise ValueError("One data-generating process will be split between training and validation sets.",
                                "Set nres to be a divisor of train_n to avoid leakage.")
        if args.nres == (args.train_n + args.val_n + args.test_n):
            print("Different realizations of a single data-generating process will be used for training, validation, and test.")
    with torch.no_grad():  
        totN = args.train_n + args.val_n + args.test_n
        all_GT = GTMatrices(N=totN, t=args.t, m=args.m, r=args.r, structured=structured,
                            realizations=args.nres, mode=args.gt_mode, 
                            kernel=args.kernel, vtype=args.vtype, U_only=args.u_only)
        all_data = TemporalData(all_GT, task=args.task, verbose=True)
        if args.memory is True:
            sensingmasks = SensingMasksTemporal(all_data, args.num_agents, args.sensing_rho)
        else:
            hide_future = False if args.task == 'lrmc' else True
            sensingmasks = SensingMasks(
                all_data, args.r, args.num_agents, args.density, 
                hide_future=hide_future, rho=args.sensing_rho, gamma=args.sensing_gamma)
        train_data, val_data, test_data = sequential_split(
            all_data, [args.train_n, args.val_n, args.test_n]
        )
    
    num_workers = min(os.cpu_count() // 2, 4) if torch.cuda.is_available() else 0 # type: ignore
    print(f"Number of workers: {num_workers}")
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    if args.memory is True:
        shuffle = False
    else:
        shuffle = True
    
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=shuffle,
        pin_memory=pin, 
        num_workers=num_workers,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
        drop_last=shuffle,
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
    
    return (
        train_loader, val_loader, test_loader, sensingmasks, 
        all_data.nuc, all_data.gap, all_data.var
    )




def setup_model(args, sensingmasks, device, task_cat):
    if args.memory is True:
        if args.task == 'nonlin_function':
            y_dim = 1
        elif args.task == 'lastrow':
            y_dim = args.m
        elif args.task == 'nextrow':
            y_dim = (args.t - 1, args.m)
        else:
            raise NotImplementedError(f"Task {args.task} not implemented for memory network")

        model = DynamicDotGAT(
            device=device, m=args.m,
            num_agents=args.num_agents,
            hidden_dim=args.hidden_dim,
            num_heads=args.att_heads,
            message_steps=args.steps,
            dropout=args.dropout,
            adjacency_mode=args.adjacency_mode,
            sharedV=args.sharedv,
            k=args.nb_ties,
            sensing_masks_temporal=sensingmasks, 
            y_dim=y_dim,
        )
        aggregator = None
    else:
        model = DistributedDotGAT(
            device=device, input_dim=args.t * args.m, hidden_dim=args.hidden_dim, n=args.t, m=args.m,
            num_agents=args.num_agents, num_heads=args.att_heads, sharedV=args.sharedv, 
            dropout=args.dropout, message_steps=args.steps, adjacency_mode=args.adjacency_mode, 
            k=args.nb_ties, sensing_masks=sensingmasks
        )
        if task_cat == 'classif':
            aggregator = CollectiveClassifier(
                num_agents=args.num_agents, agent_outputs_dim=args.hidden_dim, m = args.m
            )
            
        elif task_cat == 'regression':
            aggregator = CollectiveInferPredict(
                num_agents=args.num_agents, agent_outputs_dim=args.hidden_dim, 
                m=args.m, y_dim=args.m
            )
        
        elif task_cat == 'reconstruction':
            """aggregator = nn.Sequential(
                ReconDecoder(
                    hidden_dim=args.hidden_dim,
                    n=args.t, m=args.m,
                    num_agents=args.num_agents
                ),
                Aggregator(
                    num_agents=args.num_agents, output_dim=args.t * args.m
                )
            )"""
            #aggregator = ReconDecoder(
            #    hidden_dim=args.hidden_dim,
            #    n=args.t, m=args.m,
            #    num_agents=args.num_agents
            #)
            aggregator = CollectiveInferPredict(
                num_agents=args.num_agents, agent_outputs_dim=args.hidden_dim, 
                m=args.m, y_dim=args.t * args.m
            )
        
    return model, aggregator
