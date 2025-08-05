#!/usr/bin/env python3
"""
Lazy DataLoader Random Access Performance Benchmark

This script evaluates the performance of the lazy dataloader for random access
to 50K subset sampling, which is critical for efficient PyTorch training workflows.

Key Performance Metrics:
- Random access latency per sample
- Batch loading throughput with random indices
- Memory efficiency during random access patterns
- Cache hit rates and effectiveness
- Large dataset subset sampling performance

Usage:
    python examples/random_access_evaluation/lazy_dataloader_random_access_benchmark.py
"""

import sys
import time
import random
import tempfile
import statistics
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import polars as pl
import torch

# Add represent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from represent.lazy_dataloader import LazyParquetDataset

# Performance targets from CLAUDE.md
TARGET_SINGLE_SAMPLE_MS = 1.0  # <1ms per sample
TARGET_BATCH_MS = 50.0  # <50ms per batch
TARGET_THROUGHPUT_SAMPLES_SEC = 10000  # 10K+ samples/second


class RandomAccessBenchmark:
    """Comprehensive benchmark for random access performance."""

    def __init__(self, dataset_size: int = 100000, cache_sizes: List[int] = None):
        """
        Initialize benchmark.

        Args:
            dataset_size: Size of synthetic dataset to create
            cache_sizes: List of cache sizes to test
        """
        self.dataset_size = dataset_size
        self.cache_sizes = cache_sizes or [50, 100, 500, 1000, 5000]
        self.temp_dir = None
        self.parquet_path = None

        print("🚀 Random Access Benchmark Initialized")
        print(f"   Dataset size: {dataset_size:,} samples")
        print(f"   Cache sizes to test: {self.cache_sizes}")
        print("=" * 60)

    def setup_test_dataset(self) -> Path:
        """Create a large synthetic parquet dataset for testing."""
        print(f"📊 Creating synthetic dataset with {self.dataset_size:,} samples...")
        start_time = time.perf_counter()

        self.temp_dir = tempfile.mkdtemp()
        temp_path = Path(self.temp_dir)

        # Use chunked creation for memory efficiency
        chunk_size = 10000

        for chunk_start in range(0, self.dataset_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.dataset_size)
            chunk_data = []

            for i in range(chunk_start, chunk_end):
                # Create realistic market depth tensor with some variation
                base_values = np.random.normal(0.5, 0.1, (402, 500)).astype(np.float32)
                # Add some structure to make it more realistic
                base_values[:200, :] *= 0.8  # Bid side slightly different
                base_values[202:, :] *= 1.2  # Ask side slightly different

                mock_tensor = torch.from_numpy(base_values)

                chunk_data.append(
                    {
                        "market_depth_features": mock_tensor.numpy().tobytes(),
                        "classification_label": i % 13,  # 13-class classification
                        "feature_shape": str(mock_tensor.shape),
                        "start_timestamp": i * 1000000,
                        "end_timestamp": (i + 1) * 1000000,
                        "sample_id": f"benchmark_{i}",
                        "symbol": "BENCHMARK",
                    }
                )

            # Convert chunk to DataFrame and append
            chunk_df = pl.DataFrame(chunk_data)

            if chunk_start == 0:
                # First chunk - create file
                chunk_df.write_parquet(temp_path / "benchmark_dataset.parquet")
            else:
                # Subsequent chunks - append
                existing_df = pl.read_parquet(temp_path / "benchmark_dataset.parquet")
                combined_df = pl.concat([existing_df, chunk_df])
                combined_df.write_parquet(temp_path / "benchmark_dataset.parquet")

            if (chunk_start // chunk_size + 1) % 5 == 0:
                progress = (chunk_end / self.dataset_size) * 100
                print(f"   Progress: {progress:.1f}% ({chunk_end:,} / {self.dataset_size:,})")

        self.parquet_path = temp_path / "benchmark_dataset.parquet"
        creation_time = time.perf_counter() - start_time
        file_size_mb = self.parquet_path.stat().st_size / 1024 / 1024

        print(f"   ✅ Dataset created in {creation_time:.1f}s")
        print(f"   ✅ File size: {file_size_mb:.1f}MB")
        print(f"   ✅ Saved to: {self.parquet_path}")

        return self.parquet_path

    def benchmark_single_sample_access(self, cache_size: int = 100) -> Dict[str, float]:
        """Benchmark single sample random access performance."""
        print(f"\n🎯 Single Sample Random Access (cache_size={cache_size})")
        print("-" * 50)

        dataset = LazyParquetDataset(
            self.parquet_path,
            cache_size=cache_size,
            sample_fraction=1.0,  # Use full dataset
        )

        # Generate random indices for 50K subset
        total_samples = len(dataset)
        subset_size = min(50000, total_samples)
        random_indices = random.sample(range(total_samples), subset_size)

        print(f"   Dataset size: {total_samples:,} samples")
        print(f"   Testing with {subset_size:,} random samples")

        # Warm up cache with some samples
        warmup_samples = min(cache_size, 100)
        for i in range(warmup_samples):
            _ = dataset[random_indices[i]]

        # Benchmark random access
        test_samples = min(1000, len(random_indices))  # Test 1000 random samples
        sample_times = []

        for i in range(test_samples):
            idx = random_indices[i]

            start_time = time.perf_counter()
            features, label = dataset[idx]
            end_time = time.perf_counter()

            # Validate data integrity
            assert isinstance(features, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            assert features.shape == (402, 500)

            sample_time_ms = (end_time - start_time) * 1000
            sample_times.append(sample_time_ms)

        # Calculate statistics
        avg_time = statistics.mean(sample_times)
        median_time = statistics.median(sample_times)
        p95_time = np.percentile(sample_times, 95)
        p99_time = np.percentile(sample_times, 99)
        min_time = min(sample_times)
        max_time = max(sample_times)

        # Cache statistics
        cache_info = {
            "cache_size": len(dataset._sample_cache),
            "cache_limit": dataset.cache_size,
            "cache_utilization": len(dataset._sample_cache) / dataset.cache_size,
        }

        results = {
            "avg_time_ms": avg_time,
            "median_time_ms": median_time,
            "p95_time_ms": p95_time,
            "p99_time_ms": p99_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "samples_tested": test_samples,
            "cache_info": cache_info,
        }

        # Performance assessment
        meets_target = avg_time < TARGET_SINGLE_SAMPLE_MS
        performance_rating = "✅ EXCELLENT" if meets_target else "⚠️  NEEDS OPTIMIZATION"

        print(f"   Average time: {avg_time:.2f}ms")
        print(f"   Median time: {median_time:.2f}ms")
        print(f"   95th percentile: {p95_time:.2f}ms")
        print(f"   99th percentile: {p99_time:.2f}ms")
        print(f"   Range: {min_time:.2f}ms - {max_time:.2f}ms")
        print(f"   Cache utilization: {cache_info['cache_utilization']:.1%}")
        print(f"   Performance: {performance_rating} (target: <{TARGET_SINGLE_SAMPLE_MS}ms)")

        return results

    def benchmark_random_batch_loading(
        self, batch_size: int = 32, cache_size: int = 500
    ) -> Dict[str, float]:
        """Benchmark random batch loading performance."""
        print(f"\n📦 Random Batch Loading (batch_size={batch_size}, cache_size={cache_size})")
        print("-" * 60)

        # Create dataloader with random sampling
        dataloader = create_market_depth_dataloader(
            parquet_path=self.parquet_path,
            batch_size=batch_size,
            shuffle=True,  # Enable random sampling
            sample_fraction=0.5,  # Use 50% of dataset for 50K subset simulation
            cache_size=cache_size,
            num_workers=0,  # Single process for consistent measurement
        )

        total_samples = len(dataloader.dataset)
        num_batches = len(dataloader)

        print(f"   Dataset samples: {total_samples:,}")
        print(f"   Number of batches: {num_batches:,}")
        print(f"   Expected samples per epoch: {num_batches * batch_size:,}")

        # Benchmark batch loading
        batch_times = []
        tensor_process_times = []
        total_samples_processed = 0

        max_batches_to_test = min(100, num_batches)  # Test up to 100 batches

        start_total = time.perf_counter()

        for i, (features, labels) in enumerate(dataloader):
            if i >= max_batches_to_test:
                break

            batch_start = time.perf_counter()

            # Validate batch data
            assert isinstance(features, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert features.shape[0] == labels.shape[0]
            assert features.shape[1:] == (402, 500)

            # Simulate tensor processing (typical ML operations)
            tensor_start = time.perf_counter()
            _ = features.mean()
            _ = features.std()
            _ = labels.sum()
            tensor_time = (time.perf_counter() - tensor_start) * 1000
            tensor_process_times.append(tensor_time)

            batch_time = (time.perf_counter() - batch_start) * 1000
            batch_times.append(batch_time)
            total_samples_processed += features.shape[0]

        total_time = time.perf_counter() - start_total

        # Calculate metrics
        avg_batch_time = statistics.mean(batch_times)
        median_batch_time = statistics.median(batch_times)
        p95_batch_time = np.percentile(batch_times, 95)
        throughput = total_samples_processed / total_time
        avg_tensor_time = statistics.mean(tensor_process_times)

        results = {
            "avg_batch_time_ms": avg_batch_time,
            "median_batch_time_ms": median_batch_time,
            "p95_batch_time_ms": p95_batch_time,
            "throughput_samples_sec": throughput,
            "avg_tensor_process_ms": avg_tensor_time,
            "total_samples_processed": total_samples_processed,
            "batches_tested": len(batch_times),
            "total_time_sec": total_time,
        }

        # Performance assessment
        batch_meets_target = avg_batch_time < TARGET_BATCH_MS
        throughput_meets_target = throughput > TARGET_THROUGHPUT_SAMPLES_SEC

        batch_rating = "✅ EXCELLENT" if batch_meets_target else "⚠️  NEEDS OPTIMIZATION"
        throughput_rating = "✅ EXCELLENT" if throughput_meets_target else "⚠️  NEEDS OPTIMIZATION"

        print(f"   Average batch time: {avg_batch_time:.2f}ms")
        print(f"   Median batch time: {median_batch_time:.2f}ms")
        print(f"   95th percentile: {p95_batch_time:.2f}ms")
        print(f"   Throughput: {throughput:.0f} samples/sec")
        print(f"   Tensor processing: {avg_tensor_time:.2f}ms")
        print(f"   Batch performance: {batch_rating} (target: <{TARGET_BATCH_MS}ms)")
        print(
            f"   Throughput performance: {throughput_rating} (target: >{TARGET_THROUGHPUT_SAMPLES_SEC:,}/sec)"
        )

        return results

    def benchmark_50k_subset_sampling(self) -> Dict[str, Any]:
        """Benchmark 50K subset sampling patterns typical in ML training."""
        print("\n🎲 50K Subset Sampling Benchmark")
        print("-" * 40)

        total_samples = min(self.dataset_size, 200000)  # Limit for reasonable test time
        subset_size = 50000

        # Test different sampling strategies
        strategies = [
            ("random", "Random sampling"),
            ("stratified", "Stratified sampling (every Nth sample)"),
            ("blocked", "Blocked sampling (consecutive chunks)"),
        ]

        results = {}

        for strategy_name, strategy_desc in strategies:
            print(f"\n   Testing: {strategy_desc}")

            # Generate indices based on strategy
            if strategy_name == "random":
                indices = random.sample(range(total_samples), subset_size)
            elif strategy_name == "stratified":
                step = total_samples // subset_size
                indices = list(range(0, total_samples, step))[:subset_size]
            else:  # blocked
                chunk_size = subset_size // 10  # 10 blocks
                indices = []
                for i in range(10):
                    start = i * total_samples // 10
                    indices.extend(range(start, start + chunk_size))

            # Create dataset with reasonable cache
            dataset = LazyParquetDataset(self.parquet_path, cache_size=1000, sample_fraction=1.0)

            # Benchmark access time for this strategy
            test_samples = min(1000, len(indices))  # Test subset
            sample_times = []

            start_time = time.perf_counter()

            for i in range(test_samples):
                idx = indices[i]
                access_start = time.perf_counter()
                features, label = dataset[idx]
                access_time = (time.perf_counter() - access_start) * 1000
                sample_times.append(access_time)

            total_time = time.perf_counter() - start_time

            strategy_results = {
                "avg_access_time_ms": statistics.mean(sample_times),
                "median_access_time_ms": statistics.median(sample_times),
                "total_time_sec": total_time,
                "throughput_samples_sec": test_samples / total_time,
                "cache_utilization": len(dataset._sample_cache) / dataset.cache_size,
            }

            results[strategy_name] = strategy_results

            print(f"     Average access: {strategy_results['avg_access_time_ms']:.2f}ms")
            print(f"     Throughput: {strategy_results['throughput_samples_sec']:.0f} samples/sec")
            print(f"     Cache utilization: {strategy_results['cache_utilization']:.1%}")

        return results

    def benchmark_cache_effectiveness(self) -> Dict[str, Any]:
        """Benchmark cache effectiveness with different sizes and access patterns."""
        print("\n🗄️  Cache Effectiveness Benchmark")
        print("-" * 40)

        results = {}

        # Test each cache size
        for cache_size in self.cache_sizes:
            print(f"\n   Testing cache size: {cache_size}")

            dataset = LazyParquetDataset(
                self.parquet_path, cache_size=cache_size, sample_fraction=1.0
            )

            total_samples = min(len(dataset), 50000)

            # Test 1: Sequential access (should have poor cache performance)
            sequential_indices = list(range(0, total_samples, total_samples // 1000))[:500]
            sequential_times = []

            for idx in sequential_indices:
                start = time.perf_counter()
                _ = dataset[idx]
                sequential_times.append((time.perf_counter() - start) * 1000)

            # Test 2: Repeated access (should have excellent cache performance)
            repeated_indices = sequential_indices[: cache_size // 2]  # Within cache size
            repeated_times = []

            # First pass - populate cache
            for idx in repeated_indices:
                _ = dataset[idx]

            # Second pass - should hit cache
            for idx in repeated_indices:
                start = time.perf_counter()
                _ = dataset[idx]
                repeated_times.append((time.perf_counter() - start) * 1000)

            # Test 3: Random access within cache size
            if cache_size < total_samples:
                random_indices = random.sample(range(cache_size * 2), cache_size // 2)
                random_times = []

                for idx in random_indices:
                    start = time.perf_counter()
                    _ = dataset[idx]
                    random_times.append((time.perf_counter() - start) * 1000)
            else:
                random_times = [0]  # Skip if cache larger than test set

            cache_results = {
                "cache_size": cache_size,
                "sequential_avg_ms": statistics.mean(sequential_times),
                "repeated_avg_ms": statistics.mean(repeated_times),
                "random_avg_ms": statistics.mean(random_times),
                "speedup_ratio": statistics.mean(sequential_times)
                / statistics.mean(repeated_times),
                "final_cache_utilization": len(dataset._sample_cache) / cache_size,
            }

            results[str(cache_size)] = cache_results

            print(f"     Sequential access: {cache_results['sequential_avg_ms']:.2f}ms")
            print(f"     Repeated access: {cache_results['repeated_avg_ms']:.2f}ms")
            print(f"     Random access: {cache_results['random_avg_ms']:.2f}ms")
            print(f"     Cache speedup: {cache_results['speedup_ratio']:.1f}x")
            print(f"     Cache utilization: {cache_results['final_cache_utilization']:.1%}")

        return results

    def benchmark_memory_efficiency(self) -> Dict[str, float]:
        """Benchmark memory usage during random access operations."""
        print("\n💾 Memory Efficiency Benchmark")
        print("-" * 35)

        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            print(f"   Initial memory: {initial_memory:.1f}MB")

            # Test with large cache and many random accesses
            large_cache_size = 5000
            dataset = LazyParquetDataset(
                self.parquet_path, cache_size=large_cache_size, sample_fraction=1.0
            )

            total_samples = len(dataset)
            test_samples = min(10000, total_samples)
            random_indices = random.sample(range(total_samples), test_samples)

            memory_measurements = []

            # Measure memory at different points
            for i, idx in enumerate(random_indices):
                _ = dataset[idx]

                if i % 1000 == 0:  # Measure every 1000 samples
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_measurements.append(current_memory)

            final_memory = process.memory_info().rss / 1024 / 1024
            max_memory = max(memory_measurements) if memory_measurements else final_memory
            memory_increase = final_memory - initial_memory
            peak_increase = max_memory - initial_memory

            results = {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "max_memory_mb": max_memory,
                "memory_increase_mb": memory_increase,
                "peak_increase_mb": peak_increase,
                "samples_tested": test_samples,
                "cache_size": large_cache_size,
            }

            print(f"   Final memory: {final_memory:.1f}MB")
            print(f"   Peak memory: {max_memory:.1f}MB")
            print(f"   Memory increase: {memory_increase:.1f}MB")
            print(f"   Peak increase: {peak_increase:.1f}MB")
            print(f"   Samples tested: {test_samples:,}")

            # Memory efficiency assessment
            efficient = memory_increase < 500  # Less than 500MB increase
            rating = "✅ EFFICIENT" if efficient else "⚠️  HIGH USAGE"
            print(f"   Memory efficiency: {rating}")

            return results

        except ImportError:
            print("   ⚠️  psutil not available - skipping memory benchmark")
            return {"error": "psutil not available"}

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("🚀 LAZY DATALOADER RANDOM ACCESS BENCHMARK")
        print("=" * 60)
        print("Testing random access performance for 50K subset sampling")
        print("Critical for efficient PyTorch training workflows")
        print("=" * 60)

        # Setup test dataset
        self.setup_test_dataset()

        # Run all benchmarks
        all_results = {}

        # 1. Single sample access with different cache sizes
        print("\n📊 PHASE 1: Single Sample Random Access")
        single_sample_results = {}
        for cache_size in [100, 500, 1000]:
            single_sample_results[str(cache_size)] = self.benchmark_single_sample_access(cache_size)
        all_results["single_sample_access"] = single_sample_results

        # 2. Batch loading performance
        print("\n📊 PHASE 2: Random Batch Loading")
        batch_results = {}
        for batch_size in [16, 32, 64]:
            batch_results[str(batch_size)] = self.benchmark_random_batch_loading(batch_size)
        all_results["batch_loading"] = batch_results

        # 3. 50K subset sampling strategies
        print("\n📊 PHASE 3: 50K Subset Sampling")
        all_results["subset_sampling"] = self.benchmark_50k_subset_sampling()

        # 4. Cache effectiveness
        print("\n📊 PHASE 4: Cache Effectiveness")
        all_results["cache_effectiveness"] = self.benchmark_cache_effectiveness()

        # 5. Memory efficiency
        print("\n📊 PHASE 5: Memory Efficiency")
        all_results["memory_efficiency"] = self.benchmark_memory_efficiency()

        # Generate summary report
        self.generate_summary_report(all_results)

        return all_results

    def generate_summary_report(self, results: Dict[str, Any]):
        """Generate a comprehensive summary report."""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY REPORT")
        print("=" * 60)

        # Single Sample Access Summary
        if "single_sample_access" in results:
            print("\n🎯 Single Sample Random Access:")
            for cache_size, metrics in results["single_sample_access"].items():
                avg_time = metrics["avg_time_ms"]
                meets_target = avg_time < TARGET_SINGLE_SAMPLE_MS
                status = "✅" if meets_target else "❌"
                print(f"   Cache {cache_size}: {avg_time:.2f}ms {status}")

        # Batch Loading Summary
        if "batch_loading" in results:
            print("\n📦 Random Batch Loading:")
            for batch_size, metrics in results["batch_loading"].items():
                avg_time = metrics["avg_batch_time_ms"]
                throughput = metrics["throughput_samples_sec"]
                batch_ok = avg_time < TARGET_BATCH_MS
                throughput_ok = throughput > TARGET_THROUGHPUT_SAMPLES_SEC
                batch_status = "✅" if batch_ok else "❌"
                throughput_status = "✅" if throughput_ok else "❌"
                print(
                    f"   Batch {batch_size}: {avg_time:.2f}ms {batch_status}, {throughput:.0f} samples/sec {throughput_status}"
                )

        # Subset Sampling Summary
        if "subset_sampling" in results:
            print("\n🎲 50K Subset Sampling:")
            for strategy, metrics in results["subset_sampling"].items():
                throughput = metrics["throughput_samples_sec"]
                print(f"   {strategy.title()}: {throughput:.0f} samples/sec")

        # Cache Effectiveness Summary
        if "cache_effectiveness" in results:
            print("\n🗄️  Cache Effectiveness:")
            best_cache = None
            best_speedup = 0
            for cache_size, metrics in results["cache_effectiveness"].items():
                speedup = metrics["speedup_ratio"]
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_cache = cache_size
                print(f"   Cache {cache_size}: {speedup:.1f}x speedup")
            print(f"   Best cache size: {best_cache} ({best_speedup:.1f}x speedup)")

        # Memory Efficiency Summary
        if "memory_efficiency" in results and "error" not in results["memory_efficiency"]:
            metrics = results["memory_efficiency"]
            memory_increase = metrics["memory_increase_mb"]
            efficient = memory_increase < 500
            status = "✅ EFFICIENT" if efficient else "⚠️  HIGH"
            print(f"\n💾 Memory Usage: {memory_increase:.1f}MB increase {status}")

        # Overall Performance Assessment
        print("\n🏆 OVERALL PERFORMANCE ASSESSMENT:")
        print("-" * 40)

        # Check if key targets are met
        targets_met = 0
        total_targets = 0

        if "single_sample_access" in results:
            best_single = min(
                metrics["avg_time_ms"] for metrics in results["single_sample_access"].values()
            )
            total_targets += 1
            if best_single < TARGET_SINGLE_SAMPLE_MS:
                targets_met += 1
                print("✅ Single sample access: MEETS TARGET")
            else:
                print(
                    f"❌ Single sample access: {best_single:.2f}ms (target: <{TARGET_SINGLE_SAMPLE_MS}ms)"
                )

        if "batch_loading" in results:
            best_batch = min(
                metrics["avg_batch_time_ms"] for metrics in results["batch_loading"].values()
            )
            best_throughput = max(
                metrics["throughput_samples_sec"] for metrics in results["batch_loading"].values()
            )

            total_targets += 2
            if best_batch < TARGET_BATCH_MS:
                targets_met += 1
                print("✅ Batch loading time: MEETS TARGET")
            else:
                print(f"❌ Batch loading time: {best_batch:.2f}ms (target: <{TARGET_BATCH_MS}ms)")

            if best_throughput > TARGET_THROUGHPUT_SAMPLES_SEC:
                targets_met += 1
                print("✅ Throughput: MEETS TARGET")
            else:
                print(
                    f"❌ Throughput: {best_throughput:.0f} samples/sec (target: >{TARGET_THROUGHPUT_SAMPLES_SEC:,}/sec)"
                )

        # Final score
        score_pct = (targets_met / total_targets * 100) if total_targets > 0 else 0

        print(
            f"\n🎯 Performance Score: {targets_met}/{total_targets} targets met ({score_pct:.0f}%)"
        )

        if score_pct >= 80:
            print("🎉 EXCELLENT: Ready for production PyTorch workflows!")
        elif score_pct >= 60:
            print("⚡ GOOD: Suitable for most training scenarios")
        else:
            print("⚠️  NEEDS OPTIMIZATION: Performance improvements recommended")

        print("\n" + "=" * 60)
        print("Benchmark complete! See results above for detailed analysis.")
        print("=" * 60)

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir:
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f"🧹 Cleaned up temporary files: {self.temp_dir}")


def main():
    """Run the random access benchmark."""
    # Configuration - use smaller dataset for initial testing
    # Increase dataset_size to 100000+ for full production benchmarking
    dataset_size = 10000  # 10K sample dataset for quick testing (increase for full benchmark)
    cache_sizes = [50, 100, 500, 1000]  # Reduced cache sizes for quicker testing

    # Initialize and run benchmark
    benchmark = RandomAccessBenchmark(dataset_size=dataset_size, cache_sizes=cache_sizes)

    try:
        results = benchmark.run_full_benchmark()

        # Optionally save results to file
        import json

        results_file = Path(__file__).parent / "random_access_benchmark_results.json"

        # Convert numpy types to JSON serializable
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Recursively convert all numpy types
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)

        serializable_results = recursive_convert(results)

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()
