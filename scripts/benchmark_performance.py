#!/usr/bin/env python3
"""
Benchmark DataLoader performance to verify Linux-level speed.

This script tests the performance of the optimized DataLoader
on different platforms (Windows, WSL2, Linux).
"""
import sys
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import time
import platform
import argparse
from torch.utils.data import TensorDataset, DataLoader

from panoptic_bev.utils.windows_performance import (
    check_performance_mode, 
    setup_for_maximum_performance,
    print_system_info
)
from panoptic_bev.utils.windows_dataloader import create_fast_dataloader


def benchmark_dataloader(num_samples=1000, batch_size=16, num_workers=None):
    """Benchmark DataLoader throughput."""
    print(f"\n{'='*60}")
    print(f"Platform: {platform.system()}")
    check_performance_mode()
    print(f"{'='*60}")
    
    # Create dummy dataset simulating image loading
    print(f"\nCreating dummy dataset with {num_samples} samples...")
    images = torch.randn(num_samples, 3, 512, 512)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    
    # Test configurations
    configs = [
        ('Standard (workers=0)', {'num_workers': 0}),
    ]
    
    if num_workers is not None:
        configs.append((f'Custom (workers={num_workers})', {'num_workers': num_workers}))
    
    # Add auto-optimized config
    configs.append(('Optimized (auto)', {'num_workers': 'auto'}))
    
    results = []
    
    for name, config in configs:
        print(f"\nTesting: {name}")
        
        try:
            if config['num_workers'] == 'auto':
                loader = create_fast_dataloader(
                    dataset, 
                    batch_size=batch_size
                )
            else:
                loader = create_fast_dataloader(
                    dataset, 
                    batch_size=batch_size,
                    **config
                )
            
            # Warmup
            print("  Warming up...")
            for _ in range(2):
                for batch in loader:
                    pass
            
            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            
            total_samples = 0
            for batch in loader:
                total_samples += batch[0].size(0)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start
            
            throughput = total_samples / elapsed
            results.append((name, throughput, elapsed))
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {throughput:.1f} samples/sec")
            
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((name, 0, 0))
    
    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Configuration':<30} {'Throughput':>15} {'Time':>10}")
    print("-" * 60)
    for name, throughput, elapsed in results:
        print(f"{name:<30} {throughput:>14.1f} {elapsed:>10.2f}s")
    
    if len(results) > 1:
        baseline = results[0][1] if results[0][1] > 0 else 1
        best = max(r[1] for r in results)
        speedup = best / baseline
        print(f"\nSpeedup vs baseline: {speedup:.2f}x")
    
    return results


def benchmark_gpu_transfer(num_samples=100, batch_size=8):
    """Benchmark GPU transfer speed with different pin_memory settings."""
    if not torch.cuda.is_available():
        print("\nGPU not available, skipping GPU transfer benchmark")
        return
    
    print(f"\n{'='*60}")
    print("GPU TRANSFER BENCHMARK")
    print(f"{'='*60}")
    
    images = torch.randn(num_samples, 3, 512, 512)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    
    configs = [
        ('pin_memory=False', {'pin_memory': False, 'num_workers': 0}),
        ('pin_memory=True', {'pin_memory': True, 'num_workers': 0}),
    ]
    
    results = []
    
    for name, config in configs:
        print(f"\nTesting: {name}")
        
        loader = DataLoader(dataset, batch_size=batch_size, **config)
        
        # Warmup
        for batch in loader:
            _ = batch[0].to('cuda', non_blocking=True)
            break
        
        torch.cuda.synchronize()
        start = time.time()
        
        for batch in loader:
            _ = batch[0].to('cuda', non_blocking=True)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = num_samples / elapsed
        results.append((name, throughput, elapsed))
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.1f} samples/sec")
    
    print(f"\n{'='*60}")
    print("GPU TRANSFER SUMMARY")
    print(f"{'='*60}")
    for name, throughput, elapsed in results:
        print(f"{name:<30} {throughput:>14.1f} {elapsed:>10.2f}s")


def benchmark_multiprocessing(num_samples=500, batch_size=8):
    """Benchmark different multiprocessing configurations."""
    print(f"\n{'='*60}")
    print("MULTIPROCESSING BENCHMARK")
    print(f"{'='*60}")
    
    images = torch.randn(num_samples, 3, 256, 256)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    
    worker_counts = [0, 1, 2, 4]
    results = []
    
    for workers in worker_counts:
        print(f"\nTesting with {workers} workers...")
        
        try:
            loader = create_fast_dataloader(
                dataset,
                batch_size=batch_size,
                num_workers=workers,
                pin_memory=torch.cuda.is_available()
            )
            
            # Warmup
            for _ in range(1):
                for batch in loader:
                    pass
            
            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            
            for batch in loader:
                pass
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start
            
            throughput = num_samples / elapsed
            results.append((workers, throughput, elapsed))
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {throughput:.1f} samples/sec")
            
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((workers, 0, 0))
    
    print(f"\n{'='*60}")
    print("MULTIPROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Workers':<10} {'Throughput':>15} {'Time':>10}")
    print("-" * 40)
    for workers, throughput, elapsed in results:
        print(f"{workers:<10} {throughput:>14.1f} {elapsed:>10.2f}s")
    
    # Find optimal
    if results:
        best = max(results, key=lambda x: x[1])
        print(f"\nOptimal worker count: {best[0]} ({best[1]:.1f} samples/sec)")


def main():
    parser = argparse.ArgumentParser(description='Benchmark DataLoader performance')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers to test')
    parser.add_argument('--gpu-transfer', action='store_true', help='Run GPU transfer benchmark')
    parser.add_argument('--multiprocessing', action='store_true', help='Run multiprocessing benchmark')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    args = parser.parse_args()
    
    # Setup optimizations
    setup_for_maximum_performance()
    
    # Print system info
    print_system_info()
    
    # Run benchmarks
    if args.all:
        benchmark_dataloader(args.samples, args.batch_size, args.workers)
        benchmark_gpu_transfer(min(args.samples, 100), args.batch_size)
        benchmark_multiprocessing(args.samples, args.batch_size)
    elif args.gpu_transfer:
        benchmark_gpu_transfer(min(args.samples, 100), args.batch_size)
    elif args.multiprocessing:
        benchmark_multiprocessing(args.samples, args.batch_size)
    else:
        benchmark_dataloader(args.samples, args.batch_size, args.workers)
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)


if __name__ == '__main__':
    main()
