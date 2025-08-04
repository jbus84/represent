#!/usr/bin/env python3
"""
Usage examples for the lazy dataloader random access benchmark.

This shows different ways to configure and run the benchmark for various scenarios.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lazy_dataloader_random_access_benchmark import RandomAccessBenchmark


def example_quick_test():
    """Example 1: Quick test with small dataset."""
    print("📊 EXAMPLE 1: Quick Test")
    print("=" * 30)

    benchmark = RandomAccessBenchmark(
        dataset_size=1000,  # Small dataset for quick test
        cache_sizes=[50, 100],  # Just test 2 cache sizes
    )

    try:
        # Setup dataset
        benchmark.setup_test_dataset()

        # Run single test
        results = benchmark.benchmark_single_sample_access(cache_size=100)
        print(f"Average access time: {results['avg_time_ms']:.2f}ms")

        print("✅ Quick test completed!")

    finally:
        benchmark.cleanup()


def example_cache_optimization():
    """Example 2: Find optimal cache size."""
    print("\n📊 EXAMPLE 2: Cache Optimization")
    print("=" * 35)

    benchmark = RandomAccessBenchmark(
        dataset_size=5000,
        cache_sizes=[10, 50, 100, 250, 500, 1000],  # Wide range of cache sizes
    )

    try:
        benchmark.setup_test_dataset()

        # Test cache effectiveness
        cache_results = benchmark.benchmark_cache_effectiveness()

        # Find best cache size
        best_cache = None
        best_speedup = 0

        for cache_size, metrics in cache_results.items():
            speedup = metrics["speedup_ratio"]
            if speedup > best_speedup:
                best_speedup = speedup
                best_cache = cache_size

        print(f"🏆 Optimal cache size: {best_cache} ({best_speedup:.1f}x speedup)")

    finally:
        benchmark.cleanup()


def example_batch_size_tuning():
    """Example 3: Optimize batch size for throughput."""
    print("\n📊 EXAMPLE 3: Batch Size Tuning")
    print("=" * 35)

    benchmark = RandomAccessBenchmark(
        dataset_size=5000,
        cache_sizes=[500],  # Use reasonable cache
    )

    try:
        benchmark.setup_test_dataset()

        # Test different batch sizes
        batch_sizes = [8, 16, 32, 64, 128]
        results = {}

        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            batch_results = benchmark.benchmark_random_batch_loading(
                batch_size=batch_size, cache_size=500
            )
            results[batch_size] = batch_results["throughput_samples_sec"]

        # Find best batch size
        best_batch_size = max(results.keys(), key=lambda k: results[k])
        best_throughput = results[best_batch_size]

        print(f"\n🏆 Optimal batch size: {best_batch_size} ({best_throughput:.0f} samples/sec)")

    finally:
        benchmark.cleanup()


def example_sampling_strategy_comparison():
    """Example 4: Compare sampling strategies."""
    print("\n📊 EXAMPLE 4: Sampling Strategy Comparison")
    print("=" * 45)

    benchmark = RandomAccessBenchmark(dataset_size=10000, cache_sizes=[500])

    try:
        benchmark.setup_test_dataset()

        # Test different sampling strategies
        sampling_results = benchmark.benchmark_50k_subset_sampling()

        print("\nSampling Strategy Performance:")
        for strategy, metrics in sampling_results.items():
            throughput = metrics["throughput_samples_sec"]
            cache_util = metrics["cache_utilization"]
            print(f"  {strategy.title()}: {throughput:.0f} samples/sec (cache: {cache_util:.1%})")

        # Recommend best strategy
        best_strategy = max(
            sampling_results.keys(), key=lambda k: sampling_results[k]["throughput_samples_sec"]
        )
        print(f"\n🏆 Best strategy for throughput: {best_strategy}")

    finally:
        benchmark.cleanup()


def example_production_benchmark():
    """Example 5: Full production benchmark (commented out due to time)."""
    print("\n📊 EXAMPLE 5: Production Benchmark")
    print("=" * 35)
    print("This would run a full benchmark with large dataset:")
    print("- Dataset size: 100,000+ samples")
    print("- Multiple cache sizes: [50, 100, 500, 1000, 5000]")
    print("- All benchmark phases")
    print("- Complete performance analysis")
    print()
    print("To run this benchmark, uncomment the code below and run:")

    # Uncomment for full production benchmark (takes ~15-30 minutes)
    """
    benchmark = RandomAccessBenchmark(
        dataset_size=100000,  # Large dataset
        cache_sizes=[50, 100, 500, 1000, 5000]
    )
    
    try:
        results = benchmark.run_full_benchmark()
        print("🎉 Production benchmark completed!")
        print("See random_access_benchmark_results.json for detailed results")
    finally:
        benchmark.cleanup()
    """


def main():
    """Run all examples."""
    print("🚀 LAZY DATALOADER BENCHMARK USAGE EXAMPLES")
    print("=" * 50)
    print("These examples show different ways to use the benchmark")
    print("for various performance testing scenarios.")
    print("=" * 50)

    try:
        # Run quick examples
        example_quick_test()
        example_cache_optimization()
        example_batch_size_tuning()
        example_sampling_strategy_comparison()
        example_production_benchmark()

        print("\n" + "=" * 50)
        print("🎉 All examples completed!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Run minimal_test.py for basic functionality check")
        print("2. Run lazy_dataloader_random_access_benchmark.py for full benchmark")
        print("3. Adjust dataset_size in benchmark for production testing")
        print("4. Use results to optimize your PyTorch training configuration")

    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
