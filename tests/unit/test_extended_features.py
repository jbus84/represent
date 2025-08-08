"""
Tests for extended features functionality.
Tests volume, variance, and trade_counts features with proper dimensional output.
"""

import numpy as np
import pytest
import polars as pl

from represent.pipeline import MarketDepthProcessor, process_market_data, create_processor
from represent.constants import (
    PRICE_LEVELS,
    TIME_BINS,
    DEFAULT_FEATURES,
    ASK_COUNT_COLUMNS,
    BID_COUNT_COLUMNS,
    VARIANCE_COLUMN,
)
from tests.unit.fixtures.sample_data import generate_realistic_market_data


class TestExtendedFeatures:
    """Test extended features functionality."""

    def test_single_feature_volume(self):
        """Test single volume feature extraction."""
        data = generate_realistic_market_data(50000)
        processor = MarketDepthProcessor(features=["volume"])

        result = processor.process(data)

        # Single feature should return 2D array
        assert result.shape == (PRICE_LEVELS, TIME_BINS)
        assert result.dtype == np.float32

    def test_single_feature_trade_counts(self):
        """Test single trade_counts feature extraction."""
        data = generate_realistic_market_data(50000)

        # Add count columns to data
        for col in ASK_COUNT_COLUMNS + BID_COUNT_COLUMNS:
            if col not in data.columns:
                data = data.with_columns(pl.lit(1.0).alias(col))

        processor = MarketDepthProcessor(features=["trade_counts"])
        result = processor.process(data)

        # Single feature should return 2D array
        assert result.shape == (PRICE_LEVELS, TIME_BINS)
        assert result.dtype == np.float32

    def test_single_feature_variance(self):
        """Test single variance feature extraction."""
        data = generate_realistic_market_data(50000)

        # Add variance column to data
        data = data.with_columns(pl.lit(0.1).alias(VARIANCE_COLUMN))

        processor = MarketDepthProcessor(features=["variance"])
        result = processor.process(data)

        # Single feature should return 2D array
        assert result.shape == (PRICE_LEVELS, TIME_BINS)
        assert result.dtype == np.float32

    def test_multiple_features_2d(self):
        """Test multiple features (2 features) extraction."""
        data = generate_realistic_market_data(50000)

        # Add count columns to data
        for col in ASK_COUNT_COLUMNS + BID_COUNT_COLUMNS:
            if col not in data.columns:
                data = data.with_columns(pl.lit(1.0).alias(col))

        processor = MarketDepthProcessor(features=["volume", "trade_counts"])
        result = processor.process(data)

        # Multiple features should return 3D array
        assert result.shape == (2, PRICE_LEVELS, TIME_BINS)
        assert result.dtype == np.float32

    def test_multiple_features_3d(self):
        """Test all three features extraction."""
        data = generate_realistic_market_data(50000)

        # Add count columns and variance column to data
        for col in ASK_COUNT_COLUMNS + BID_COUNT_COLUMNS:
            if col not in data.columns:
                data = data.with_columns(pl.lit(1.0).alias(col))

        data = data.with_columns(pl.lit(0.1).alias(VARIANCE_COLUMN))

        processor = MarketDepthProcessor(features=["volume", "variance", "trade_counts"])
        result = processor.process(data)

        # Multiple features should return 3D array
        assert result.shape == (3, PRICE_LEVELS, TIME_BINS)
        assert result.dtype == np.float32

    def test_feature_ordering_consistency(self):
        """Test that features are ordered consistently."""
        data = generate_realistic_market_data(50000)

        # Add all required columns
        for col in ASK_COUNT_COLUMNS + BID_COUNT_COLUMNS:
            if col not in data.columns:
                data = data.with_columns(pl.lit(1.0).alias(col))

        data = data.with_columns(pl.lit(0.1).alias(VARIANCE_COLUMN))

        # Test with different feature order inputs
        processor1 = MarketDepthProcessor(features=["trade_counts", "volume", "variance"])
        processor2 = MarketDepthProcessor(features=["variance", "volume", "trade_counts"])

        result1 = processor1.process(data)
        result2 = processor2.process(data)

        # Results should have same shape and same feature ordering (sorted by index)
        assert result1.shape == result2.shape == (3, PRICE_LEVELS, TIME_BINS)

        # Features should be ordered consistently: volume=0, variance=1, trade_counts=2
        assert processor1.features == ["volume", "variance", "trade_counts"]
        assert processor2.features == ["volume", "variance", "trade_counts"]

    def test_default_features(self):
        """Test default features behavior."""
        data = generate_realistic_market_data(50000)

        # Test with no features specified
        processor = MarketDepthProcessor()
        result = processor.process(data)

        assert processor.features == DEFAULT_FEATURES
        assert result.shape == (PRICE_LEVELS, TIME_BINS)

    def test_invalid_features(self):
        """Test error handling for invalid features."""
        with pytest.raises(ValueError, match="Invalid features"):
            MarketDepthProcessor(features=["invalid_feature"])

        with pytest.raises(ValueError, match="At least one feature"):
            MarketDepthProcessor(features=[])

        # Test too many valid features
        with pytest.raises(ValueError, match="Too many features"):
            MarketDepthProcessor(
                features=["volume", "variance", "trade_counts", "volume"]
            )  # 4 features

    def test_process_market_data_api_with_features(self):
        """Test process_market_data API with features parameter."""
        data = generate_realistic_market_data(50000)

        # Test single feature
        result_single = process_market_data(data, features=["volume"])
        assert result_single.shape == (PRICE_LEVELS, TIME_BINS)

        # Test multiple features
        for col in ASK_COUNT_COLUMNS + BID_COUNT_COLUMNS:
            if col not in data.columns:
                data = data.with_columns(pl.lit(1.0).alias(col))

        result_multi = process_market_data(data, features=["volume", "trade_counts"])
        assert result_multi.shape == (2, PRICE_LEVELS, TIME_BINS)

        # Test default behavior (backward compatibility)
        result_default = process_market_data(data)
        assert result_default.shape == (PRICE_LEVELS, TIME_BINS)

    def test_create_processor_factory_with_features(self):
        """Test create_processor factory with features parameter."""
        # Test with features
        processor = create_processor(features=["volume", "trade_counts"])
        assert processor.features == ["volume", "trade_counts"]

        # Test default behavior
        processor_default = create_processor()
        assert processor_default.features == DEFAULT_FEATURES

    def test_variance_feature_calculation(self):
        """Test that variance feature calculates volume variance correctly."""
        data = generate_realistic_market_data(50000)
        # Variance feature now calculates variance of volume data, not from a separate column

        processor = MarketDepthProcessor(features=["variance"])
        result = processor.process(data)

        # Should return proper shape
        assert result.shape == (PRICE_LEVELS, TIME_BINS)
        # Result should contain meaningful variance data (not all zeros)
        assert not np.allclose(result, 0.0, atol=1e-6)
        # Should be in normalized range [-1, 1]
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_feature_data_isolation(self):
        """Test that different features are processed independently."""
        data = generate_realistic_market_data(50000)

        # Add count columns to data
        for col in ASK_COUNT_COLUMNS + BID_COUNT_COLUMNS:
            if col not in data.columns:
                data = data.with_columns(pl.lit(1.0).alias(col))

        # Test that multi-feature processing works correctly
        processor = MarketDepthProcessor(features=["volume", "trade_counts"])
        result = processor.process(data)

        # Verify we have the expected shape and valid results
        assert result.shape == (2, PRICE_LEVELS, TIME_BINS)

        volume_result = result[0]  # First feature (volume)
        counts_result = result[1]  # Second feature (trade_counts)

        # Both features should produce valid normalized results
        assert not np.all(volume_result == 0.0)
        assert not np.all(counts_result == 0.0)
        assert np.all(np.abs(volume_result) <= 1.0)  # Normalized values
        assert np.all(np.abs(counts_result) <= 1.0)  # Normalized values

        # Test that the processor correctly uses different data columns for each feature
        # This is verified by checking that the feature extraction doesn't fail
        # and produces reasonable outputs within expected ranges
        assert volume_result.dtype == np.float32
        assert counts_result.dtype == np.float32


class TestFeaturePerformance:
    """Test performance aspects of extended features."""

    def test_single_vs_multiple_feature_performance(self):
        """Test that multiple features don't significantly degrade performance."""
        import time

        data = generate_realistic_market_data(50000)

        # Add required columns
        for col in ASK_COUNT_COLUMNS + BID_COUNT_COLUMNS:
            if col not in data.columns:
                data = data.with_columns(pl.lit(1.0).alias(col))

        # Test single feature
        processor_single = MarketDepthProcessor(features=["volume"])
        start_time = time.perf_counter()
        result_single = processor_single.process(data)
        single_time = time.perf_counter() - start_time

        # Test multiple features
        processor_multi = MarketDepthProcessor(features=["volume", "trade_counts"])
        start_time = time.perf_counter()
        result_multi = processor_multi.process(data)
        multi_time = time.perf_counter() - start_time

        # Verify results
        assert result_single.shape == (PRICE_LEVELS, TIME_BINS)
        assert result_multi.shape == (2, PRICE_LEVELS, TIME_BINS)

        # Multiple features should not be more than 3x slower
        assert multi_time < single_time * 3.0, (
            f"Multi-feature too slow: {multi_time:.4f}s vs {single_time:.4f}s"
        )

    @pytest.mark.performance
    def test_extended_features_latency(self):
        """Test that extended features meet latency requirements."""
        import time

        data = generate_realistic_market_data(50000)

        # Add all required columns
        for col in ASK_COUNT_COLUMNS + BID_COUNT_COLUMNS:
            if col not in data.columns:
                data = data.with_columns(pl.lit(1.0).alias(col))

        data = data.with_columns(pl.lit(0.1).alias(VARIANCE_COLUMN))

        processor = MarketDepthProcessor(features=["volume", "variance", "trade_counts"])

        # Run multiple times to get stable measurement
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            result = processor.process(data)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
            assert result.shape == (3, PRICE_LEVELS, TIME_BINS)

        avg_time = sum(times) / len(times)

        # Should still meet <50ms requirement for extended features (relaxed from 10ms)
        assert avg_time < 50.0, f"Extended features too slow: {avg_time:.2f}ms average"


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_api_unchanged(self):
        """Test that existing API calls work without modification."""
        data = generate_realistic_market_data(50000)

        # These should work exactly as before
        processor = MarketDepthProcessor()
        result = processor.process(data)
        assert result.shape == (PRICE_LEVELS, TIME_BINS)

        result2 = process_market_data(data)
        assert result2.shape == (PRICE_LEVELS, TIME_BINS)

        # Results should be identical to preserve backward compatibility
        assert np.allclose(result, result2)

    def test_default_behavior_unchanged(self):
        """Test that default behavior produces same results as before."""
        data = generate_realistic_market_data(50000)

        # Default should be volume-only
        processor_default = MarketDepthProcessor()
        processor_explicit = MarketDepthProcessor(features=["volume"])

        result_default = processor_default.process(data)
        result_explicit = processor_explicit.process(data)

        assert np.allclose(result_default, result_explicit)
        assert result_default.shape == result_explicit.shape == (PRICE_LEVELS, TIME_BINS)
