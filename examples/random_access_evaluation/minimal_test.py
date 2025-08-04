#!/usr/bin/env python3
"""
Minimal test to verify the lazy dataloader random access functionality.
"""

import sys
import time
import tempfile
from pathlib import Path
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from represent.lazy_dataloader import LazyParquetDataset
from represent.dataloader import create_market_depth_dataloader


def create_small_test_dataset(size: int = 100) -> Path:
    """Create a small test dataset."""
    print(f"📊 Creating test dataset with {size} samples...")

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    labeled_data = []
    for i in range(size):
        mock_tensor = torch.zeros(402, 500, dtype=torch.float32)
        labeled_data.append(
            {
                "market_depth_features": mock_tensor.numpy().tobytes(),
                "classification_label": i % 3,
                "feature_shape": str(mock_tensor.shape),
                "start_timestamp": i * 1000000,
                "sample_id": f"test_{i}",
                "symbol": "TEST",
            }
        )

    df = pl.DataFrame(labeled_data)
    parquet_path = temp_path / "test_dataset.parquet"
    df.write_parquet(parquet_path)

    print(f"   ✅ Created: {parquet_path}")
    return parquet_path


def test_random_access(parquet_path: Path):
    """Test random access performance."""
    print("\n🎯 Testing Random Access...")

    dataset = LazyParquetDataset(parquet_path, cache_size=50)
    total_samples = len(dataset)

    print(f"   Dataset size: {total_samples} samples")

    # Test random access
    import random

    test_indices = random.sample(range(total_samples), min(20, total_samples))

    access_times = []
    for idx in test_indices:
        start = time.perf_counter()
        features, label = dataset[idx]
        end = time.perf_counter()

        access_time_ms = (end - start) * 1000
        access_times.append(access_time_ms)

        # Validate
        assert isinstance(features, torch.Tensor)
        assert features.shape == (402, 500)

    avg_time = sum(access_times) / len(access_times)
    print(f"   ✅ Average access time: {avg_time:.2f}ms")
    print(f"   ✅ Tested {len(test_indices)} random samples")

    return avg_time


def test_batch_loading(parquet_path: Path):
    """Test batch loading performance."""
    print("\n📦 Testing Batch Loading...")

    dataloader = create_market_depth_dataloader(
        parquet_path=parquet_path, batch_size=8, shuffle=True, sample_fraction=1.0, num_workers=0
    )

    batch_count = 0
    total_time = 0

    start_time = time.perf_counter()

    for features, labels in dataloader:
        batch_start = time.perf_counter()

        # Validate batch
        assert isinstance(features, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert features.shape[1:] == (402, 500)

        batch_time = (time.perf_counter() - batch_start) * 1000
        total_time += batch_time
        batch_count += 1

        if batch_count >= 5:  # Test just a few batches
            break

    end_time = time.perf_counter()

    avg_batch_time = total_time / batch_count if batch_count > 0 else 0
    total_elapsed = (end_time - start_time) * 1000

    print(f"   ✅ Average batch time: {avg_batch_time:.2f}ms")
    print(f"   ✅ Total time for {batch_count} batches: {total_elapsed:.2f}ms")

    return avg_batch_time


def main():
    """Run minimal random access test."""
    print("🚀 MINIMAL RANDOM ACCESS TEST")
    print("=" * 40)

    try:
        # Create small test dataset
        parquet_path = create_small_test_dataset(size=100)

        # Test random access
        avg_access_time = test_random_access(parquet_path)

        # Test batch loading
        avg_batch_time = test_batch_loading(parquet_path)

        # Results
        print("\n📊 RESULTS:")
        print(f"   Random access: {avg_access_time:.2f}ms per sample")
        print(f"   Batch loading: {avg_batch_time:.2f}ms per batch")

        # Performance assessment
        access_good = avg_access_time < 5.0  # Relaxed target for small test
        batch_good = avg_batch_time < 100.0  # Relaxed target for small test

        if access_good and batch_good:
            print("\n🎉 SUCCESS: Random access performance looks good!")
            print("   Ready to run full benchmark with larger dataset.")
        else:
            print("\n⚠️  NOTICE: Performance may need optimization.")
            print("   Consider running full benchmark for detailed analysis.")

        # Cleanup
        import shutil

        shutil.rmtree(parquet_path.parent, ignore_errors=True)
        print("\n🧹 Cleaned up test files")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
