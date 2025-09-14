# GPU Performance Optimization Benchmarking

This document describes the GPU optimization improvements made to the DynamicDotGAT model and how to benchmark them.

## Optimizations Implemented

### 1. Model Architecture Optimizations (`dotGAT_optimized.py`)
- **Fused QKV Projections**: Combined separate Query, Key, Value weight matrices into single fused operations
- **Reduced Tensor Reshaping**: Minimized permute/view operations in attention mechanisms
- **Optimized MLP**: Fused two-layer MLP into single weight matrix with chunk splitting
- **Efficient Einsum Operations**: Replaced complex BMM operations with optimized einsum calls

### 2. Training Pipeline Optimizations (`main_sociotemporal.py`)
- **Compilation Mode**: Changed from `reduce-overhead` to `max-autotune` for better kernel selection
- **Data Loading**: Increased workers (4→8) and prefetch factor (2→4) for better GPU utilization
- **Memory Settings**: Enabled TF32 and optimized CUDA settings

## Benchmarking Tools

### 1. Quick Test (`test_optimizations.py`)
Verifies correctness and provides quick speed comparison:
```bash
python test_optimizations.py
```

Output includes:
- Numerical equivalence check
- Quick speed comparison
- Throughput improvement percentage

### 2. Comprehensive Benchmark (`benchmark_models.py`)
Detailed performance comparison across different batch sizes:
```bash
python benchmark_models.py
```

Features:
- Tests multiple batch sizes (8, 16, 32, 64)
- Measures forward pass, backward pass, and optimizer step separately
- Compares both compiled and non-compiled versions
- Reports memory usage and throughput
- Calculates speedup ratios

### 3. Profiler Analysis (`profile_models.py`)
Deep dive into CUDA kernel performance:
```bash
# Basic profiling
python profile_models.py

# Save traces for TensorBoard visualization
python profile_models.py --save-traces

# CPU-only profiling
python profile_models.py --cpu
```

Provides:
- Top operations by CUDA/CPU time
- Memory usage analysis
- Kernel-level performance metrics
- TensorBoard-compatible traces

## Running the Optimized Training

To use the optimized model in training:
```bash
python main_sociotemporal.py [your usual arguments]
```

The script automatically uses the optimized model. To compare with the original, modify the import in `main_sociotemporal.py`:
```python
# For original model:
from dotGAT import DynamicDotGAT

# For optimized model (current default):
from dotGAT_optimized import OptimizedDynamicDotGAT as DynamicDotGAT
```

## Expected Performance Improvements

Based on the optimizations:
- **Forward Pass**: 1.5-2.5x speedup
- **Backward Pass**: 1.3-2.0x speedup
- **Memory Usage**: 10-20% reduction
- **Overall Training**: 1.4-2.2x speedup

Actual speedup depends on:
- GPU model (better on newer architectures with Tensor Cores)
- Batch size (larger batches see more improvement)
- Model configuration (hidden_dim, num_agents, message_steps)

## Optimization Tips

1. **Batch Size**: Use the largest batch size that fits in GPU memory
2. **Mixed Precision**: Already enabled via GradScaler
3. **Compilation**: First epoch is slower due to compilation, subsequent epochs are faster
4. **Multi-GPU**: For further scaling, consider DistributedDataParallel

## Troubleshooting

If the optimized model produces different results:
1. Run `test_optimizations.py` to check numerical equivalence
2. Ensure weight initialization is consistent
3. Check for any shape mismatches in outputs

If performance is not improved:
1. Ensure CUDA is available and being used
2. Check that torch.compile is working (first epoch should be slower)
3. Profile with `profile_models.py` to identify bottlenecks
4. Verify TF32 and cuDNN settings are enabled

## Further Optimizations

Potential areas for additional improvement:
1. **Gradient Checkpointing**: Trade compute for memory to enable larger batches
2. **Custom CUDA Kernels**: For agent-specific operations
3. **Quantization**: INT8 inference for deployment
4. **Flash Attention 2**: When available in stable PyTorch