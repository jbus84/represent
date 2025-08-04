#!/usr/bin/env python3
"""
Quick test of the lazy dataloader random access benchmark.
This runs a smaller, faster version to verify functionality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lazy_dataloader_random_access_benchmark import RandomAccessBenchmark


def main():
    print("🚀 Quick Random Access Test")
    print("=" * 40)

    # Use smaller dataset for quick test
    benchmark = RandomAccessBenchmark(
        dataset_size=1000,  # Much smaller for quick test
        cache_sizes=[50, 100],  # Fewer cache sizes
    )

    try:
        # Setup dataset
        parquet_path = benchmark.setup_test_dataset()
        print(f"✅ Test dataset created: {parquet_path}")

        # Run single sample access test
        print("\n🎯 Testing single sample access...")
        results = benchmark.benchmark_single_sample_access(cache_size=100)
        print(f"✅ Average access time: {results['avg_time_ms']:.2f}ms")

        # Run batch loading test
        print("\n📦 Testing batch loading...")
        batch_results = benchmark.benchmark_random_batch_loading(batch_size=16, cache_size=100)
        print(f"✅ Average batch time: {batch_results['avg_batch_time_ms']:.2f}ms")
        print(f"✅ Throughput: {batch_results['throughput_samples_sec']:.0f} samples/sec")

        print("\n🎉 Quick test completed successfully!")
        print("Run the full benchmark with larger dataset for complete evaluation.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()
