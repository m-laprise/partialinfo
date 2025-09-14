"""
Benchmark script to compare original DynamicDotGAT vs OptimizedDynamicDotGAT
Measures forward pass, backward pass, and full training iteration performance.
"""

import gc
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from datautils.datagen_temporal import GTMatrices, TemporalData
from datautils.sensing import SensingMasksTemporal
from dotGAT import DynamicDotGAT as OriginalModel
from dotGAT_optimized import OptimizedDynamicDotGAT as OptimizedModel


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    # Model dimensions
    t: int = 50          # Time steps
    m: int = 25          # Features
    r: int = 25          # Rank
    num_agents: int = 20
    hidden_dim: int = 128
    att_heads: int = 4
    message_steps: int = 5

    # Batch sizes to test
    batch_sizes: List[int] = None

    # Number of warmup iterations
    warmup_iters: int = 10
    # Number of benchmark iterations
    benchmark_iters: int = 50

    # Device
    use_cuda: bool = True
    use_amp: bool = True
    compile_model: bool = True

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [8, 16, 32, 64]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    model_name: str
    batch_size: int
    forward_time: float
    backward_time: float
    optimizer_time: float
    total_time: float
    memory_allocated: float
    memory_reserved: float
    throughput: float  # samples per second


class ModelBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device('cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA version: {torch.version.cuda}")
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Create data
        self._setup_data()

    def _setup_data(self):
        """Create synthetic data for benchmarking"""
        with torch.no_grad():
            # Create ground truth matrices
            gt_matrices = GTMatrices(
                N=100,  # Just need a small dataset for benchmarking
                t=self.config.t,
                m=self.config.m,
                r=self.config.r,
                realizations=1,
                mode='value',
                kernel='cauchy',
                vtype='random',
                U_only=False
            )

            # Create temporal data
            temporal_data = TemporalData(gt_matrices, task='nonlin_function', verbose=False)

            # Create sensing masks
            self.sensing_masks = SensingMasksTemporal(
                temporal_data,
                num_agents=self.config.num_agents,
                rho=0.25
            )

    def create_model(self, model_class, compile: bool = False):
        """Create and optionally compile a model"""
        model = model_class(
            device=self.device,
            m=self.config.m,
            num_agents=self.config.num_agents,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.att_heads,
            message_steps=self.config.message_steps,
            dropout=0.0,  # No dropout for benchmarking
            adjacency_mode='socnet',
            sharedV=True,
            k=4,
            sensing_masks_temporal=self.sensing_masks,
            y_dim=1
        ).to(self.device)

        if compile and self.device.type == 'cuda':
            model = torch.compile(model, mode='max-autotune', fullgraph=True)

        return model

    def benchmark_model(
        self,
        model: nn.Module,
        model_name: str,
        batch_size: int
    ) -> BenchmarkResult:
        """Benchmark a single model configuration"""

        # Create synthetic input
        x = torch.randn(batch_size, self.config.t, self.config.m, device=self.device)
        y_true = torch.randn(batch_size, self.config.t, device=self.device)

        # Setup optimizer and scaler
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = GradScaler(enabled=(self.config.use_amp and self.device.type == 'cuda'))

        # Warmup
        print(f"  Warming up {model_name} (batch_size={batch_size})...")
        for _ in range(self.config.warmup_iters):
            self._training_step(model, x, y_true, optimizer, scaler)

        # Clear cache and synchronize
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Benchmark
        print(f"  Benchmarking {model_name} (batch_size={batch_size})...")
        forward_times = []
        backward_times = []
        optimizer_times = []
        total_times = []

        for _ in range(self.config.benchmark_iters):
            times = self._timed_training_step(model, x, y_true, optimizer, scaler)
            forward_times.append(times['forward'])
            backward_times.append(times['backward'])
            optimizer_times.append(times['optimizer'])
            total_times.append(times['total'])

        # Get memory stats
        if self.device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        else:
            memory_allocated = 0
            memory_reserved = 0

        # Calculate statistics
        result = BenchmarkResult(
            model_name=model_name,
            batch_size=batch_size,
            forward_time=np.median(forward_times) * 1000,  # Convert to ms
            backward_time=np.median(backward_times) * 1000,
            optimizer_time=np.median(optimizer_times) * 1000,
            total_time=np.median(total_times) * 1000,
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved,
            throughput=batch_size / np.median(total_times)
        )

        return result

    def _training_step(self, model, x, y_true, optimizer, scaler):
        """Single training step"""
        optimizer.zero_grad(set_to_none=True)

        use_amp = self.config.use_amp and self.device.type == 'cuda'

        with autocast(device_type='cuda', enabled=use_amp):
            h, y_pred = model(x)
            if y_pred is None:
                raise RuntimeError("Model must output predictions")

            # Simple MSE loss
            y_pred = y_pred.squeeze(-1)  # [B, T, A]
            loss = ((y_pred - y_true.unsqueeze(-1)) ** 2).mean()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    def _timed_training_step(self, model, x, y_true, optimizer, scaler) -> Dict[str, float]:
        """Training step with detailed timing"""
        times = {}

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Total time
        total_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        use_amp = self.config.use_amp and self.device.type == 'cuda'

        # Forward pass
        forward_start = time.perf_counter()
        with autocast(device_type='cuda', enabled=use_amp):
            h, y_pred = model(x)
            if y_pred is None:
                raise RuntimeError("Model must output predictions")
            y_pred = y_pred.squeeze(-1)
            loss = ((y_pred - y_true.unsqueeze(-1)) ** 2).mean()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        times['forward'] = time.perf_counter() - forward_start

        # Backward pass
        backward_start = time.perf_counter()
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        times['backward'] = time.perf_counter() - backward_start

        # Optimizer step
        optimizer_start = time.perf_counter()
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        times['optimizer'] = time.perf_counter() - optimizer_start

        times['total'] = time.perf_counter() - total_start

        return times

    def run_comparison(self) -> Dict[str, List[BenchmarkResult]]:
        """Run full comparison between original and optimized models"""
        results = {
            'original': [],
            'optimized': [],
            'original_compiled': [],
            'optimized_compiled': []
        }

        for batch_size in self.config.batch_sizes:
            print(f"\nBatch size: {batch_size}")
            print("-" * 50)

            # Test original model
            print("Testing Original Model...")
            model = self.create_model(OriginalModel, compile=False)
            result = self.benchmark_model(model, "Original", batch_size)
            results['original'].append(result)
            del model
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # Test optimized model
            print("Testing Optimized Model...")
            model = self.create_model(OptimizedModel, compile=False)
            result = self.benchmark_model(model, "Optimized", batch_size)
            results['optimized'].append(result)
            del model
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # Test compiled versions if requested
            if self.config.compile_model and self.device.type == 'cuda':
                # Original compiled
                print("Testing Original Model (Compiled)...")
                model = self.create_model(OriginalModel, compile=True)
                result = self.benchmark_model(model, "Original+Compiled", batch_size)
                results['original_compiled'].append(result)
                del model
                gc.collect()
                torch.cuda.empty_cache()

                # Optimized compiled
                print("Testing Optimized Model (Compiled)...")
                model = self.create_model(OptimizedModel, compile=True)
                result = self.benchmark_model(model, "Optimized+Compiled", batch_size)
                results['optimized_compiled'].append(result)
                del model
                gc.collect()
                torch.cuda.empty_cache()

        return results

    def print_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Print formatted benchmark results"""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        # Compare for each batch size
        for i, batch_size in enumerate(self.config.batch_sizes):
            print(f"\nBatch Size: {batch_size}")
            print("-" * 60)

            # Create comparison table
            print(f"{'Model':<25} {'Forward(ms)':<12} {'Backward(ms)':<12} {'Total(ms)':<12} {'Throughput':<12} {'Memory(GB)':<10}")
            print("-" * 60)

            # Get results for this batch size
            if results['original']:
                orig = results['original'][i]
                print(f"{'Original':<25} {orig.forward_time:<12.2f} {orig.backward_time:<12.2f} "
                      f"{orig.total_time:<12.2f} {orig.throughput:<12.1f} {orig.memory_allocated:<10.2f}")

            if results['optimized']:
                opt = results['optimized'][i]
                print(f"{'Optimized':<25} {opt.forward_time:<12.2f} {opt.backward_time:<12.2f} "
                      f"{opt.total_time:<12.2f} {opt.throughput:<12.1f} {opt.memory_allocated:<10.2f}")

            if results['original_compiled']:
                orig_c = results['original_compiled'][i]
                print(f"{'Original+Compiled':<25} {orig_c.forward_time:<12.2f} {orig_c.backward_time:<12.2f} "
                      f"{orig_c.total_time:<12.2f} {orig_c.throughput:<12.1f} {orig_c.memory_allocated:<10.2f}")

            if results['optimized_compiled']:
                opt_c = results['optimized_compiled'][i]
                print(f"{'Optimized+Compiled':<25} {opt_c.forward_time:<12.2f} {opt_c.backward_time:<12.2f} "
                      f"{opt_c.total_time:<12.2f} {opt_c.throughput:<12.1f} {opt_c.memory_allocated:<10.2f}")

            # Calculate speedups
            if results['original'] and results['optimized']:
                print(f"\nSpeedup (Optimized vs Original):")
                print(f"  Forward:  {orig.forward_time / opt.forward_time:.2f}x")
                print(f"  Backward: {orig.backward_time / opt.backward_time:.2f}x")
                print(f"  Total:    {orig.total_time / opt.total_time:.2f}x")
                print(f"  Memory:   {(1 - opt.memory_allocated / orig.memory_allocated) * 100:.1f}% reduction")

            if results['original_compiled'] and results['optimized_compiled']:
                print(f"\nSpeedup (Optimized+Compiled vs Original+Compiled):")
                print(f"  Forward:  {orig_c.forward_time / opt_c.forward_time:.2f}x")
                print(f"  Backward: {orig_c.backward_time / opt_c.backward_time:.2f}x")
                print(f"  Total:    {orig_c.total_time / opt_c.total_time:.2f}x")

        # Overall summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        if results['optimized'] and results['original']:
            avg_speedup = np.mean([
                results['original'][i].total_time / results['optimized'][i].total_time
                for i in range(len(self.config.batch_sizes))
            ])
            print(f"Average speedup (Optimized vs Original): {avg_speedup:.2f}x")

        if results['optimized_compiled'] and results['original_compiled']:
            avg_speedup_compiled = np.mean([
                results['original_compiled'][i].total_time / results['optimized_compiled'][i].total_time
                for i in range(len(self.config.batch_sizes))
            ])
            print(f"Average speedup (Compiled versions): {avg_speedup_compiled:.2f}x")


def main():
    """Run the benchmark"""
    # Configure benchmark
    config = BenchmarkConfig(
        t=50,
        m=25,
        num_agents=20,
        hidden_dim=128,
        att_heads=4,
        message_steps=5,
        batch_sizes=[8, 16, 32, 64],
        warmup_iters=10,
        benchmark_iters=50,
        use_cuda=True,
        use_amp=True,
        compile_model=True  # Test both compiled and non-compiled
    )

    # Run benchmark
    benchmark = ModelBenchmark(config)
    results = benchmark.run_comparison()

    # Print results
    benchmark.print_results(results)


if __name__ == "__main__":
    main()