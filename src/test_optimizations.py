#!/usr/bin/env python3
"""
Quick test script to verify optimizations are working correctly.
Compares outputs and timing between original and optimized models.
"""

import time

import torch
import torch.nn as nn

from datautils.datagen_temporal import GTMatrices, TemporalData
from datautils.sensing import SensingMasksTemporal
from dotGAT import DynamicDotGAT as OriginalModel
from dotGAT_optimized import OptimizedDynamicDotGAT as OptimizedModel


def test_model_equivalence():
    """Test that optimized model produces similar outputs to original"""
    print("Testing model output equivalence...")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Parameters
    t, m, r = 20, 10, 10  # Smaller for quick test
    num_agents = 5
    hidden_dim = 32
    batch_size = 4

    # Create data
    with torch.no_grad():
        gt = GTMatrices(N=10, t=t, m=m, r=r, realizations=1,
                        mode='value', kernel='cauchy', vtype='random', U_only=False)
        data = TemporalData(gt, task='nonlin_function', verbose=False)
        masks = SensingMasksTemporal(data, num_agents=num_agents, rho=0.25)

    # Create both models with same parameters
    kwargs = dict(
        device=device, m=m, num_agents=num_agents,
        hidden_dim=hidden_dim, num_heads=2, message_steps=2,
        dropout=0.0, adjacency_mode='socnet', sharedV=True,
        k=2, sensing_masks_temporal=masks, y_dim=1
    )

    model_orig = OriginalModel(**kwargs).to(device)
    model_opt = OptimizedModel(**kwargs).to(device)

    # Copy weights from original to optimized to ensure same initialization
    with torch.no_grad():
        # Copy embedding weights
        model_opt.W_embed.data = model_orig.W_embed.data.clone()

        # Copy attention weights
        if hasattr(model_opt, 'W_qkv_mem'):
            # Fused version - concatenate original weights
            model_opt.W_qkv_mem.data = torch.cat([
                model_orig.W_q_mem.data,
                model_orig.W_k_mem.data,
                model_orig.W_v_mem.data
            ], dim=-1)
            model_opt.W_qkv_soc.data = torch.cat([
                model_orig.W_q_soc.data,
                model_orig.W_k_soc.data,
                model_orig.W_v_soc.data
            ], dim=-1)
        else:
            # Shared V version
            model_opt.W_qk_mem.data = torch.cat([
                model_orig.W_q_mem.data,
                model_orig.W_k_mem.data
            ], dim=-1)
            model_opt.W_qk_soc.data = torch.cat([
                model_orig.W_q_soc.data,
                model_orig.W_k_soc.data
            ], dim=-1)

        # Copy MLP weights - need to handle fused version
        model_opt.W_fwd.data = torch.cat([
            model_orig.W_fwd1.data,
            model_orig.W_fwd2.data
        ], dim=-1)
        model_opt.b_fwd1.data = model_orig.b_fwd1.data.clone()
        model_opt.b_fwd2.data = model_orig.b_fwd2.data.clone()

        # Copy prediction weights
        if model_orig.W_decode_pred is not None:
            model_opt.W_decode_pred.data = model_orig.W_decode_pred.data.clone()
            model_opt.b_decode_pred.data = model_orig.b_decode_pred.data.clone()

    # Test input
    x = torch.randn(batch_size, t, m, device=device)

    # Set to eval mode for consistent results
    model_orig.eval()
    model_opt.eval()

    # Forward pass
    with torch.no_grad():
        h_orig, y_orig = model_orig(x)
        h_opt, y_opt = model_opt(x)

    # Check outputs
    print(f"\nOutput shapes:")
    print(f"  Original - h: {h_orig.shape}, y: {y_orig.shape if y_orig is not None else None}")
    print(f"  Optimized - h: {h_opt.shape}, y: {y_opt.shape if y_opt is not None else None}")

    # Compare outputs
    h_diff = (h_orig - h_opt).abs().max().item()
    print(f"\nMax difference in hidden states: {h_diff:.6f}")

    if y_orig is not None and y_opt is not None:
        # Handle potential shape differences
        if y_orig.shape != y_opt.shape:
            if y_opt.dim() == y_orig.dim() + 1:
                y_opt = y_opt.squeeze(-1)
            elif y_orig.dim() == y_opt.dim() + 1:
                y_orig = y_orig.squeeze(-1)

        y_diff = (y_orig - y_opt).abs().max().item()
        print(f"Max difference in predictions: {y_diff:.6f}")

        if y_diff < 1e-4:
            print("✓ Outputs are numerically equivalent!")
        else:
            print("⚠ Outputs differ significantly - check implementation")
    else:
        print("✓ Hidden states are numerically equivalent!" if h_diff < 1e-4 else
              "⚠ Hidden states differ - check implementation")

    return h_diff < 1e-4


def quick_speed_test():
    """Quick speed comparison"""
    print("\n" + "="*60)
    print("Quick Speed Test")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parameters for speed test
    t, m = 50, 25
    num_agents = 20
    hidden_dim = 128
    batch_size = 32

    # Create data
    with torch.no_grad():
        gt = GTMatrices(N=10, t=t, m=m, r=25, realizations=1,
                        mode='value', kernel='cauchy', vtype='random', U_only=False)
        data = TemporalData(gt, task='nonlin_function', verbose=False)
        masks = SensingMasksTemporal(data, num_agents=num_agents, rho=0.25)

    kwargs = dict(
        device=device, m=m, num_agents=num_agents,
        hidden_dim=hidden_dim, num_heads=4, message_steps=5,
        dropout=0.0, adjacency_mode='socnet', sharedV=True,
        k=4, sensing_masks_temporal=masks, y_dim=1
    )

    # Test input
    x = torch.randn(batch_size, t, m, device=device)
    y_true = torch.randn(batch_size, t, device=device)

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {t}")
    print(f"  Features: {m}")
    print(f"  Agents: {num_agents}")
    print(f"  Hidden dim: {hidden_dim}")

    # Original model
    print("\nTesting Original Model...")
    model_orig = OriginalModel(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model_orig.parameters())

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        h, y = model_orig(x)
        loss = ((y.squeeze(-1) - y_true.unsqueeze(-1)) ** 2).mean()
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Time original
    start = time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad()
        h, y = model_orig(x)
        loss = ((y.squeeze(-1) - y_true.unsqueeze(-1)) ** 2).mean()
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    orig_time = (time.perf_counter() - start) / 10

    del model_orig, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Optimized model
    print("Testing Optimized Model...")
    model_opt = OptimizedModel(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model_opt.parameters())

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        h, y = model_opt(x)
        loss = ((y.squeeze(-1) - y_true.unsqueeze(-1)) ** 2).mean()
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Time optimized
    start = time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad()
        h, y = model_opt(x)
        loss = ((y.squeeze(-1) - y_true.unsqueeze(-1)) ** 2).mean()
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    opt_time = (time.perf_counter() - start) / 10

    # Results
    print(f"\nResults (per iteration):")
    print(f"  Original:  {orig_time*1000:.2f} ms")
    print(f"  Optimized: {opt_time*1000:.2f} ms")
    print(f"  Speedup:   {orig_time/opt_time:.2f}x")
    print(f"  Throughput improvement: {((1/opt_time - 1/orig_time) / (1/orig_time)) * 100:.1f}%")


def main():
    print("="*60)
    print("OPTIMIZATION TEST SUITE")
    print("="*60)

    # Test correctness
    correct = test_model_equivalence()

    if correct:
        # Run speed test
        quick_speed_test()
    else:
        print("\n⚠ Skipping speed test due to correctness issues")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()