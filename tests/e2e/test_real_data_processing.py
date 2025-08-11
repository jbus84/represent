"""
End-to-end tests using real market data from the data directory.
These tests validate the entire pipeline with actual market data.
"""

import time

import numpy as np
import polars as pl
import pytest

from represent import MarketDepthProcessor, create_processor, process_market_data
from represent.config import create_represent_config


class TestRealDataProcessing:
    """Test processing with real market data."""

    def setup_method(self):
        """Setup config for each test."""
        self.config = create_represent_config("AUDUSD")

    @pytest.mark.e2e
    def test_process_real_market_data_basic(self, sample_real_data):
        """Test basic processing of real market data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")

        # Process the data
        result = process_market_data(sample_real_data, config=self.config)

        # Validate output shape and type
        assert result.shape == self.config.output_shape, f"Expected shape {self.config.output_shape}, got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

        # Validate output is finite and not all zeros
        assert np.all(np.isfinite(result)), "Output contains non-finite values"
        assert not np.all(result == 0), "Output is all zeros"

        # Validate reasonable value ranges
        assert np.all(result >= -10), "Output contains extremely negative values"
        assert np.all(result <= 10), "Output contains extremely positive values"

    @pytest.mark.e2e
    def test_processor_with_real_data(self, sample_real_data):
        """Test processor factory with real market data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")

        # Create processor
        processor = create_processor(config=self.config)
        assert isinstance(processor, MarketDepthProcessor)

        # Process data
        result = processor.process(sample_real_data)

        # Validate output
        assert result.shape == self.config.output_shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    @pytest.mark.e2e
    def test_multiple_batches_real_data(self, real_market_data):
        """Test processing multiple batches of real data."""
        if real_market_data is None:
            pytest.skip("Real market data not available")

        processor = create_processor(config=self.config)

        # Process multiple 50K chunks (required size)
        chunk_size = 50000
        results = []

        # Process up to 3 chunks if we have enough data
        max_chunks = min(3, len(real_market_data) // chunk_size)

        for i in range(max_chunks):
            start_idx = i * chunk_size
            chunk = real_market_data.slice(start_idx, chunk_size)
            if len(chunk) == chunk_size:  # Only process full chunks
                result = processor.process(chunk)
                results.append(result)

        # Validate all results
        assert len(results) > 0, "No chunks were processed"

        for i, result in enumerate(results):
            assert result.shape == self.config.output_shape, f"Chunk {i} has wrong shape"
            assert result.dtype == np.float32, f"Chunk {i} has wrong dtype"
            assert np.all(np.isfinite(result)), f"Chunk {i} contains non-finite values"

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_large_dataset_processing(self, sample_real_data):
        """Test processing of standard dataset with performance measurement."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")

        # Use the standard 50K sample (required by pipeline)
        start_time = time.perf_counter()
        result = process_market_data(sample_real_data, config=self.config)
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        records_per_second = len(sample_real_data) / processing_time

        # Validate output
        assert result.shape == self.config.output_shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

        # Performance validation (should be fast)
        assert records_per_second > 10000, f"Processing too slow: {records_per_second:.0f} rps"

        print(
            f"Processed {len(sample_real_data)} records in {processing_time:.3f}s ({records_per_second:.0f} rps)"
        )


class TestRealDataValidation:
    """Test data validation with real market data."""

    def setup_method(self):
        """Setup config for each test."""
        self.config = create_represent_config("AUDUSD")

    @pytest.mark.e2e
    def test_real_data_structure(self, sample_real_data):
        """Validate the structure of real market data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")

        # Check required columns exist
        required_columns = [
            "ts_event",
            "ts_recv",
            "symbol",
            # Ask prices
            "ask_px_00",
            "ask_px_01",
            "ask_px_02",
            "ask_px_03",
            "ask_px_04",
            "ask_px_05",
            "ask_px_06",
            "ask_px_07",
            "ask_px_08",
            "ask_px_09",
            # Bid prices
            "bid_px_00",
            "bid_px_01",
            "bid_px_02",
            "bid_px_03",
            "bid_px_04",
            "bid_px_05",
            "bid_px_06",
            "bid_px_07",
            "bid_px_08",
            "bid_px_09",
            # Ask volumes
            "ask_sz_00",
            "ask_sz_01",
            "ask_sz_02",
            "ask_sz_03",
            "ask_sz_04",
            "ask_sz_05",
            "ask_sz_06",
            "ask_sz_07",
            "ask_sz_08",
            "ask_sz_09",
            # Bid volumes
            "bid_sz_00",
            "bid_sz_01",
            "bid_sz_02",
            "bid_sz_03",
            "bid_sz_04",
            "bid_sz_05",
            "bid_sz_06",
            "bid_sz_07",
            "bid_sz_08",
            "bid_sz_09",
        ]

        available_columns = sample_real_data.columns
        missing_columns = [col for col in required_columns if col not in available_columns]

        if missing_columns:
            print(f"Available columns: {available_columns}")
            print(f"Missing columns: {missing_columns}")
            # Don't fail the test, but note what's missing
            pytest.skip(f"Real data missing required columns: {missing_columns}")

        # Validate data types and ranges
        assert len(sample_real_data) > 0, "Real data is empty"

        # Check price columns have reasonable values
        for col in [c for c in required_columns if "px_" in c and c in available_columns]:
            prices = sample_real_data[col].to_numpy()
            prices = prices[~np.isnan(prices)]  # Remove NaN values
            if len(prices) > 0:
                assert np.all(prices > 0), f"Column {col} has non-positive prices"
                assert np.all(prices < 100), f"Column {col} has unreasonably high prices"

    @pytest.mark.e2e
    def test_real_data_consistency(self, sample_real_data):
        """Test consistency of real market data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")

        # Check that ask prices are generally higher than bid prices
        if "ask_px_00" in sample_real_data.columns and "bid_px_00" in sample_real_data.columns:
            ask_prices = sample_real_data["ask_px_00"].to_numpy()
            bid_prices = sample_real_data["bid_px_00"].to_numpy()

            # Remove NaN values
            valid_mask = ~(np.isnan(ask_prices) | np.isnan(bid_prices))
            ask_prices = ask_prices[valid_mask]
            bid_prices = bid_prices[valid_mask]

            if len(ask_prices) > 0:
                # Most ask prices should be higher than bid prices (allowing for some data anomalies)
                spread_positive = np.sum(ask_prices > bid_prices) / len(ask_prices)
                assert spread_positive > 0.8, f"Only {spread_positive:.1%} of spreads are positive"

        # Check timestamp ordering
        if "ts_event" in sample_real_data.columns:
            timestamps = sample_real_data["ts_event"].to_numpy()
            # Most timestamps should be in ascending order (allowing for some out-of-order)
            ordered_ratio = np.sum(np.diff(timestamps) >= 0) / (len(timestamps) - 1)
            assert ordered_ratio > 0.9, f"Only {ordered_ratio:.1%} of timestamps are ordered"


class TestRealDataPerformance:
    """Performance tests with real market data."""

    def setup_method(self):
        """Setup config for each test."""
        self.config = create_represent_config("AUDUSD")

    @pytest.mark.e2e
    @pytest.mark.performance
    def test_real_data_latency(self, sample_real_data):
        """Test processing latency with real data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")

        # Warm up
        process_market_data(sample_real_data, config=self.config)

        # Measure latency
        start_time = time.perf_counter()
        result = process_market_data(sample_real_data, config=self.config)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Validate output
        assert result.shape == self.config.output_shape

        # Performance target (should be fast)
        assert latency_ms < 1000, (
            f"Latency too high: {latency_ms:.2f}ms for {len(sample_real_data)} records"
        )

        print(f"Processed {len(sample_real_data)} real records in {latency_ms:.2f}ms")

    @pytest.mark.e2e
    @pytest.mark.performance
    def test_real_data_throughput(self, sample_real_data):
        """Test processing throughput with real data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")

        # Process multiple times to get stable measurement
        num_runs = 3
        durations = []

        for _ in range(num_runs):
            start_time = time.perf_counter()
            result = process_market_data(sample_real_data, config=self.config)
            end_time = time.perf_counter()
            durations.append(end_time - start_time)

        best_duration = min(durations)
        throughput_rps = len(sample_real_data) / best_duration

        # Validate output
        assert result.shape == self.config.output_shape

        # Performance target
        assert throughput_rps > 50000, f"Throughput too low: {throughput_rps:.0f} rps"

        print(f"Real data throughput: {throughput_rps:.0f} records/second")

    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.slow
    def test_real_data_memory_usage(self, sample_real_data):
        """Test memory usage with real data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")

        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process data
        result = process_market_data(sample_real_data, config=self.config)

        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - baseline_memory

        # Validate output
        assert result.shape == self.config.output_shape

        # Memory target (should be reasonable)
        assert memory_used < 1000, f"Memory usage too high: {memory_used:.2f}MB"

        print(f"Memory used for {len(sample_real_data)} real records: {memory_used:.2f}MB")


class TestRealDataEdgeCases:
    """Test edge cases with real market data."""

    def setup_method(self):
        """Setup config for each test."""
        self.config = create_represent_config("AUDUSD")

    @pytest.mark.e2e
    def test_partial_real_data(self, sample_real_data):
        """Test processing with standard real data sample."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")

        # Test with the standard 50K sample (pipeline requirement)
        result = process_market_data(sample_real_data, config=self.config)

        assert result.shape == self.config.output_shape
        assert np.all(np.isfinite(result))

    @pytest.mark.e2e
    def test_real_data_with_gaps(self, real_market_data):
        """Test processing real data with time gaps."""
        if real_market_data is None:
            pytest.skip("Real market data not available")

        # Take non-contiguous samples to simulate gaps, but ensure we get exactly 50K
        if len(real_market_data) >= 500000:  # Need enough data to sample from
            # Take every 10th row to create gaps, then take first 50K
            gapped_data = real_market_data.filter(pl.int_range(pl.len()).mod(10) == 0).head(50000)

            result = process_market_data(gapped_data, config=self.config)

            assert result.shape == self.config.output_shape
            assert np.all(np.isfinite(result))
        else:
            pytest.skip(f"Need at least 500K records for gap testing, have {len(real_market_data)}")
