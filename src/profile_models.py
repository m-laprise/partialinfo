"""
Detailed profiling script to identify bottlenecks in model operations.
Uses PyTorch profiler to trace CUDA kernels and operations.
"""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

from datautils.datagen_temporal import GTMatrices, TemporalData
from datautils.sensing import SensingMasksTemporal
from dotGAT import DynamicDotGAT as OriginalModel
from dotGAT_optimized import OptimizedDynamicDotGAT as OptimizedModel


def setup_model_and_data(model_class, device, compile_model=False):
    """Setup model and create sample data"""
    # Model parameters
    t, m, r = 50, 25, 25
    num_agents = 20
    hidden_dim = 128
    batch_size = 32

    # Create data
    with torch.no_grad():
        gt_matrices = GTMatrices(
            N=10, t=t, m=m, r=r,
            realizations=1, mode='value',
            kernel='cauchy', vtype='random',
            U_only=False
        )
        temporal_data = TemporalData(gt_matrices, task='nonlin_function', verbose=False)
        sensing_masks = SensingMasksTemporal(temporal_data, num_agents=num_agents, rho=0.25)

    # Create model
    model = model_class(
        device=device,
        m=m,
        num_agents=num_agents,
        hidden_dim=hidden_dim,
        num_heads=4,
        message_steps=5,
        dropout=0.0,
        adjacency_mode='socnet',
        sharedV=True,
        k=4,
        sensing_masks_temporal=sensing_masks,
        y_dim=1
    ).to(device)

    if compile_model and device.type == 'cuda':
        model = torch.compile(model, mode='max-autotune', fullgraph=True)

    # Create sample input
    x = torch.randn(batch_size, t, m, device=device)
    y_true = torch.randn(batch_size, t, device=device)

    return model, x, y_true


def profile_model(model, x, y_true, model_name, device, trace_path=None):
    """Profile a model's forward and backward pass"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Warmup
    print(f"Warming up {model_name}...")
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        h, y_pred = model(x)
        loss = ((y_pred.squeeze(-1) - y_true.unsqueeze(-1)) ** 2).mean()
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Profile
    print(f"Profiling {model_name}...")
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path) if trace_path else None
    ) as prof:
        with record_function(f"{model_name}_training_step"):
            optimizer.zero_grad(set_to_none=True)

            with record_function("forward"):
                h, y_pred = model(x)
                loss = ((y_pred.squeeze(-1) - y_true.unsqueeze(-1)) ** 2).mean()

            with record_function("backward"):
                loss.backward()

            with record_function("optimizer_step"):
                optimizer.step()

    return prof


def analyze_profile(prof, model_name):
    """Analyze and print profiling results"""
    print(f"\n{'='*80}")
    print(f"Profile Analysis: {model_name}")
    print(f"{'='*80}")

    # Top operations by CUDA time
    if torch.cuda.is_available():
        print("\nTop 10 Operations by CUDA Time:")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=10
        ))

        print("\nTop 10 Operations by CUDA Memory:")
        print(prof.key_averages().table(
            sort_by="cuda_memory_usage",
            row_limit=10
        ))

    # Top operations by CPU time
    print("\nTop 10 Operations by CPU Time:")
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10
    ))

    # Get specific operation stats
    print("\nKey Operation Statistics:")
    key_ops = ['aten::einsum', 'aten::matmul', 'aten::bmm',
               'aten::scaled_dot_product_attention', 'aten::linear']

    for event in prof.key_averages():
        for op in key_ops:
            if op in event.key:
                cuda_time = event.cuda_time_total / 1000 if torch.cuda.is_available() else 0
                cpu_time = event.cpu_time_total / 1000
                count = event.count

                print(f"\n{event.key}:")
                print(f"  Count: {count}")
                print(f"  CPU Time: {cpu_time:.2f} ms")
                if cuda_time > 0:
                    print(f"  CUDA Time: {cuda_time:.2f} ms")
                    print(f"  CUDA Mem: {event.cuda_memory_usage / 1024**2:.2f} MB" if event.cuda_memory_usage else "  CUDA Mem: 0 MB")


def compare_models(device, save_traces=False):
    """Compare original and optimized models"""

    trace_dir = None
    if save_traces:
        trace_dir = f"./profiler_traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(trace_dir, exist_ok=True)
        print(f"Saving traces to {trace_dir}")

    # Profile original model
    print("\n" + "="*80)
    print("PROFILING ORIGINAL MODEL")
    print("="*80)

    model_orig, x, y_true = setup_model_and_data(OriginalModel, device, compile_model=False)
    prof_orig = profile_model(
        model_orig, x, y_true, "Original", device,
        trace_path=f"{trace_dir}/original" if trace_dir else None
    )
    analyze_profile(prof_orig, "Original Model")

    # Clear memory
    del model_orig
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Profile optimized model
    print("\n" + "="*80)
    print("PROFILING OPTIMIZED MODEL")
    print("="*80)

    model_opt, x, y_true = setup_model_and_data(OptimizedModel, device, compile_model=False)
    prof_opt = profile_model(
        model_opt, x, y_true, "Optimized", device,
        trace_path=f"{trace_dir}/optimized" if trace_dir else None
    )
    analyze_profile(prof_opt, "Optimized Model")

    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    def get_total_time(prof, time_type='cuda_time_total'):
        total = 0
        for event in prof.key_averages():
            if hasattr(event, time_type):
                total += getattr(event, time_type)
        return total / 1000  # Convert to ms

    if device.type == 'cuda':
        orig_cuda_time = get_total_time(prof_orig, 'cuda_time_total')
        opt_cuda_time = get_total_time(prof_opt, 'cuda_time_total')

        print(f"\nTotal CUDA Time:")
        print(f"  Original:  {orig_cuda_time:.2f} ms")
        print(f"  Optimized: {opt_cuda_time:.2f} ms")
        print(f"  Speedup:   {orig_cuda_time / opt_cuda_time:.2f}x")

    orig_cpu_time = get_total_time(prof_orig, 'cpu_time_total')
    opt_cpu_time = get_total_time(prof_opt, 'cpu_time_total')

    print(f"\nTotal CPU Time:")
    print(f"  Original:  {orig_cpu_time:.2f} ms")
    print(f"  Optimized: {opt_cpu_time:.2f} ms")
    print(f"  Speedup:   {orig_cpu_time / opt_cpu_time:.2f}x")

    if save_traces:
        print(f"\nProfiler traces saved to {trace_dir}")
        print("View with: tensorboard --logdir={trace_dir}")


def main():
    parser = argparse.ArgumentParser(description="Profile and compare model implementations")
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of CUDA')
    parser.add_argument('--save-traces', action='store_true', help='Save profiler traces for TensorBoard')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU")

    compare_models(device, save_traces=args.save_traces)


if __name__ == "__main__":
    main()