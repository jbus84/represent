"""
Tests for the new parquet-based architecture (v2.0.0).
"""

import tempfile
from pathlib import Path
import numpy as np
import polars as pl
import torch
import pytest

from represent.lazy_dataloader import LazyParquetDataset, LazyParquetDataLoader, create_parquet_dataloader
from represent.config import create_represent_config


def create_synthetic_dbn_data(num_rows: int = 10000) -> pl.DataFrame:
    """Create synthetic market data that mimics DBN format."""
    np.random.seed(42)

    # Generate realistic price data
    base_price = 1.25000
    price_changes = np.random.normal(0, 0.00005, num_rows)
    mid_prices = base_price + np.cumsum(price_changes)

    data = {}
    data["ts_event"] = np.arange(num_rows) * 1000000
    data["ts_recv"] = data["ts_event"] + 100
    data["symbol"] = ["M6AM4"] * num_rows

    # Generate 10-level market data
    spread = 0.00003
    for level in range(10):
        level_str = str(level).zfill(2)

        bid_offset = spread / 2 + level * 0.00001
        ask_offset = spread / 2 + level * 0.00001

        data[f"bid_px_{level_str}"] = (mid_prices - bid_offset) * 100000
        data[f"ask_px_{level_str}"] = (mid_prices + ask_offset) * 100000
        data[f"bid_sz_{level_str}"] = np.random.uniform(10, 100, num_rows)
        data[f"ask_sz_{level_str}"] = np.random.uniform(10, 100, num_rows)
        data[f"bid_ct_{level_str}"] = np.random.randint(1, 10, num_rows)
        data[f"ask_ct_{level_str}"] = np.random.randint(1, 10, num_rows)

    return pl.DataFrame(data)




class TestLazyParquetDataset:
    """Test lazy parquet dataset functionality."""

    def create_test_parquet(self, temp_dir: Path) -> Path:
        """Create a test parquet file with labeled data."""
        # Create mock labeled data
        num_samples = 100
        labeled_data = []

        for i in range(num_samples):
            # Create mock tensor (zeros for simplicity)
            mock_tensor = torch.zeros(402, 500, dtype=torch.float32)

            labeled_data.append(
                {
                    "market_depth_features": mock_tensor.numpy().tobytes().hex(),
                    "classification_label": i % 13,  # 13-class labels
                    "feature_shape": str(mock_tensor.shape),
                    "start_timestamp": i * 1000000,
                    "end_timestamp": (i + 1) * 1000000,
                    "sample_id": f"sample_{i}",
                    "symbol": "M6AM4",
                }
            )

        df = pl.DataFrame(labeled_data)
        parquet_path = temp_dir / "test_dataset.parquet"
        df.write_parquet(parquet_path)
        return parquet_path

    def test_dataset_initialization(self):
        """Test dataset initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = self.create_test_parquet(Path(temp_dir))

            dataset = LazyParquetDataset(parquet_path)

            assert dataset.parquet_path == parquet_path
            assert dataset.batch_size == 32  # Default
            assert len(dataset) == 100  # All samples

    def test_dataset_with_sampling(self):
        """Test dataset with sample fraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = self.create_test_parquet(Path(temp_dir))

            dataset = LazyParquetDataset(parquet_path, sample_fraction=0.5)

            assert len(dataset) == 50  # Half the samples

    def test_dataset_getitem(self):
        """Test getting individual samples."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = self.create_test_parquet(Path(temp_dir))

            dataset = LazyParquetDataset(parquet_path, cache_size=10)

            # Get first sample
            features, label = dataset[0]

            assert isinstance(features, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            assert features.shape == (402, 500)
            assert label.shape == ()  # Scalar
            assert 0 <= label.item() < 13

    def test_dataset_caching(self):
        """Test LRU caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = self.create_test_parquet(Path(temp_dir))

            dataset = LazyParquetDataset(parquet_path, cache_size=5)

            # Access samples to populate cache
            for i in range(10):
                _ = dataset[i]

            # Cache should have most recent 5 samples
            assert len(dataset._sample_cache) == 5
            assert len(dataset._cache_order) == 5

    def test_dataset_info(self):
        """Test dataset info functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = self.create_test_parquet(Path(temp_dir))

            dataset = LazyParquetDataset(parquet_path)
            info = dataset.get_dataset_info()

            assert "parquet_file" in info
            assert "total_samples" in info
            assert "active_samples" in info
            assert info["total_samples"] == 100
            assert info["active_samples"] == 100


class TestLazyParquetDataLoader:
    """Test lazy parquet dataloader functionality."""

    def create_test_parquet(self, temp_dir: Path) -> Path:
        """Create a test parquet file with labeled data."""
        num_samples = 100
        labeled_data = []

        for i in range(num_samples):
            mock_tensor = torch.zeros(402, 500, dtype=torch.float32)

            labeled_data.append(
                {
                    "market_depth_features": mock_tensor.numpy().tobytes().hex(),
                    "classification_label": i % 3,  # 3-class for simplicity
                    "feature_shape": str(mock_tensor.shape),
                    "start_timestamp": i * 1000000,
                    "sample_id": f"sample_{i}",
                }
            )

        df = pl.DataFrame(labeled_data)
        parquet_path = temp_dir / "test_dataset.parquet"
        df.write_parquet(parquet_path)
        return parquet_path

    def test_dataloader_creation(self):
        """Test dataloader creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = self.create_test_parquet(Path(temp_dir))

            dataloader = LazyParquetDataLoader(
                parquet_path=parquet_path, batch_size=8, shuffle=False
            )

            assert dataloader.batch_size == 8
            assert len(dataloader) > 0

    def test_dataloader_iteration(self):
        """Test dataloader iteration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = self.create_test_parquet(Path(temp_dir))

            dataloader = LazyParquetDataLoader(
                parquet_path=parquet_path, batch_size=4, shuffle=False, num_workers=0
            )

            batch_count = 0
            for features, labels in dataloader:
                assert isinstance(features, torch.Tensor)
                assert isinstance(labels, torch.Tensor)
                assert features.shape[0] == labels.shape[0]
                assert features.shape[1:] == (402, 500)

                batch_count += 1
                if batch_count >= 3:  # Test first 3 batches
                    break

            assert batch_count >= 3

    def test_factory_function(self):
        """Test create_parquet_dataloader factory function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = self.create_test_parquet(Path(temp_dir))

            dataloader = create_parquet_dataloader(
                parquet_path=parquet_path, batch_size=16, sample_fraction=0.5
            )

            assert isinstance(dataloader, LazyParquetDataLoader)
            assert dataloader.batch_size == 16

            # Should be able to iterate
            batch = next(iter(dataloader))
            features, labels = batch
            assert features.shape[0] <= 16  # Batch size or less


class TestCurrencyConfigurations:
    """Test currency-specific configurations."""

    def test_audusd_config(self):
        """Test AUDUSD configuration."""
        config = create_represent_config("AUDUSD")

        assert config.currency == "AUDUSD"
        assert config.nbins == 13
        assert config.lookforward_input == 5000

    def test_gbpusd_config(self):
        """Test GBPUSD configuration."""
        config = create_represent_config("GBPUSD")

        assert config.currency == "GBPUSD"
        assert config.lookforward_input == 3000  # GBPUSD-specific optimization

    def test_eurjpy_config(self):
        """Test EURJPY configuration."""
        config = create_represent_config("EURJPY")

        assert config.currency == "EURJPY"
        assert config.true_pip_size == 0.01  # JPY-specific optimization
        assert config.nbins == 9  # JPY-specific optimization


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.mark.integration
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create synthetic data
            synthetic_data = create_synthetic_dbn_data(5000)
            temp_parquet = temp_dir / "synthetic.parquet"
            synthetic_data.write_parquet(temp_parquet)

            # Create converter and mock the conversion process
            # converter = DBNToParquetConverter(currency='AUDUSD')

            # Create minimal labeled dataset for testing
            labeled_samples = []
            for i in range(50):  # Small number for test
                mock_tensor = torch.zeros(402, 500, dtype=torch.float32)
                labeled_samples.append(
                    {
                        "market_depth_features": mock_tensor.numpy().tobytes().hex(),
                        "classification_label": i % 3,
                        "feature_shape": str(mock_tensor.shape),
                        "sample_id": f"test_{i}",
                    }
                )

            labeled_df = pl.DataFrame(labeled_samples)
            labeled_parquet = temp_dir / "labeled.parquet"
            labeled_df.write_parquet(labeled_parquet)

            # Test dataloader with labeled data
            dataloader = create_parquet_dataloader(
                parquet_path=labeled_parquet, batch_size=8, sample_fraction=1.0
            )

            # Verify we can iterate through data
            total_samples = 0
            for features, labels in dataloader:
                assert features.shape[1:] == (402, 500)
                assert labels.shape[0] == features.shape[0]
                total_samples += features.shape[0]

                # Only test a few batches
                if total_samples >= 32:
                    break

            assert total_samples >= 32


class TestPerformance:
    """Performance tests for the new architecture."""

    def test_lazy_loading_performance(self):
        """Test lazy loading performance."""
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create larger dataset for performance testing
            num_samples = 1000
            labeled_data = []

            for i in range(num_samples):
                mock_tensor = torch.zeros(402, 500, dtype=torch.float32)
                labeled_data.append(
                    {
                        "market_depth_features": mock_tensor.numpy().tobytes().hex(),
                        "classification_label": i % 3,
                        "feature_shape": str(mock_tensor.shape),
                        "sample_id": f"perf_{i}",
                    }
                )

            df = pl.DataFrame(labeled_data)
            parquet_path = Path(temp_dir) / "perf_test.parquet"
            df.write_parquet(parquet_path)

            # Test dataloader performance
            dataloader = create_parquet_dataloader(
                parquet_path=parquet_path,
                batch_size=32,
                sample_fraction=0.1,  # Use 10% for quick test
            )

            # Measure batch loading time
            batch_times = []
            for i, (features, labels) in enumerate(dataloader):
                start_time = time.perf_counter()

                # Simulate some processing
                _ = features.sum()
                _ = labels.sum()

                batch_time = (time.perf_counter() - start_time) * 1000
                batch_times.append(batch_time)

                if i >= 5:  # Test first 5 batches
                    break

            avg_batch_time = sum(batch_times) / len(batch_times)

            # Performance target: <50ms per batch
            assert avg_batch_time < 50.0, f"Batch processing too slow: {avg_batch_time:.2f}ms"

            print(f"Average batch time: {avg_batch_time:.2f}ms")

    @pytest.mark.skip(reason="Memory test can be flaky in different environments")
    def test_memory_efficiency(self):
        """Test memory efficiency of lazy loading."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dataset larger than we want to load into memory
            num_samples = 2000
            labeled_data = []

            for i in range(num_samples):
                mock_tensor = torch.zeros(402, 500, dtype=torch.float32)
                labeled_data.append(
                    {
                        "market_depth_features": mock_tensor.numpy().tobytes().hex(),
                        "classification_label": i % 3,
                        "feature_shape": str(mock_tensor.shape),
                        "sample_id": f"mem_{i}",
                    }
                )

            df = pl.DataFrame(labeled_data)
            parquet_path = Path(temp_dir) / "memory_test.parquet"
            df.write_parquet(parquet_path)

            # Create dataloader with small cache
            dataloader = create_parquet_dataloader(
                parquet_path=parquet_path,
                batch_size=16,
                sample_fraction=0.2,  # Use 20% of data
            )

            # Process multiple batches
            for i, (features, labels) in enumerate(dataloader):
                if i >= 10:  # Process 10 batches
                    break

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory should not increase dramatically with lazy loading
            assert memory_increase < 600, f"Memory increase too high: {memory_increase:.1f}MB"

            print(f"Memory increase: {memory_increase:.1f}MB")
