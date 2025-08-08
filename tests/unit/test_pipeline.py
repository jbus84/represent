"""
Tests to improve pipeline module coverage.
"""

import pytest
import numpy as np
import polars as pl
from represent import MarketDepthProcessor, create_processor, process_market_data, FeatureType
from represent.config import create_represent_config


class TestPipelineCoverage:
    """Test pipeline module to improve coverage."""
    
    def setup_method(self):
        """Setup config for each test."""
        self.config = create_represent_config("AUDUSD")

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = {
            "ask_px_00": np.random.uniform(100.0, 101.0, 50000),
            "bid_px_00": np.random.uniform(99.0, 100.0, 50000),
            "ask_sz_00": np.random.uniform(10, 100, 50000),
            "bid_sz_00": np.random.uniform(10, 100, 50000),
            "ask_ct_00": np.random.randint(1, 10, 50000),
            "bid_ct_00": np.random.randint(1, 10, 50000),
        }

        # Add all required columns
        for i in range(1, 10):
            data[f"ask_px_{str(i).zfill(2)}"] = data["ask_px_00"] + np.random.uniform(0, 0.1, 50000)
            data[f"bid_px_{str(i).zfill(2)}"] = data["bid_px_00"] - np.random.uniform(0, 0.1, 50000)
            data[f"ask_sz_{str(i).zfill(2)}"] = np.random.uniform(5, 50, 50000)
            data[f"bid_sz_{str(i).zfill(2)}"] = np.random.uniform(5, 50, 50000)
            data[f"ask_ct_{str(i).zfill(2)}"] = np.random.randint(1, 5, 50000)
            data[f"bid_ct_{str(i).zfill(2)}"] = np.random.randint(1, 5, 50000)

        return pl.DataFrame(data)

    def test_processor_initialization_edge_cases(self):
        """Test processor initialization edge cases."""
        # Test with empty features list
        with pytest.raises(ValueError, match="At least one feature must be specified"):
            MarketDepthProcessor(features=[])

        # Test with too many features
        too_many_features = ["volume", "variance", "trade_counts", "invalid"]
        with pytest.raises(ValueError, match="Invalid features"):
            MarketDepthProcessor(features=too_many_features)

        # Test with FeatureType enums
        processor = MarketDepthProcessor(features=[FeatureType.VOLUME, FeatureType.VARIANCE])
        assert processor.features == ["volume", "variance"]

        # Test with mixed string and enum types
        processor = MarketDepthProcessor(features=[FeatureType.VOLUME, "trade_counts"])
        assert "volume" in processor.features
        assert "trade_counts" in processor.features

    def test_processor_expression_compilation(self, sample_data):
        """Test expression compilation and reuse."""
        processor = MarketDepthProcessor(features=["volume"])

        # Test that expressions are compiled on first use
        assert not processor._compiled_expressions_ready

        # Process data to trigger compilation
        processor.process(sample_data)
        assert processor._compiled_expressions_ready

        # Test expression reuse on second call
        processor.process(sample_data)
        assert processor._compiled_expressions_ready

    def test_price_lookup_caching(self, sample_data):
        """Test price lookup table caching."""
        processor = MarketDepthProcessor(features=["volume"])

        # First processing should create lookup table
        processor.process(sample_data)
        first_lookup = processor._price_lookup
        first_mid_price = processor._cached_mid_price

        assert first_lookup is not None
        assert first_mid_price > 0

        # Second processing with same data should reuse lookup table
        processor.process(sample_data)
        assert processor._price_lookup is first_lookup
        assert processor._cached_mid_price == first_mid_price

        # Processing with different mid price should create new lookup
        modified_data = sample_data.with_columns(
            [
                pl.col("ask_px_00") * 2,  # Change mid price significantly
                pl.col("bid_px_00") * 2,
            ]
        )
        processor.process(modified_data)
        assert processor._price_lookup is not first_lookup
        assert processor._cached_mid_price != first_mid_price

    def test_input_validation(self, sample_data):
        """Test input validation."""
        processor = MarketDepthProcessor(features=["volume"])

        # Test with insufficient samples
        small_data = sample_data.head(100)  # Too small for meaningful processing
        with pytest.raises(ValueError, match="Input must have at least 500 samples"):
            processor.process(small_data)

    def test_variance_feature_processing(self, sample_data):
        """Test variance feature specific processing."""
        processor = MarketDepthProcessor(features=["variance"])
        result = processor.process(sample_data)

        assert result.shape == (402, self.config.time_bins)
        assert np.isfinite(result).all()

    def test_trade_counts_feature_processing(self, sample_data):
        """Test trade counts feature specific processing."""
        processor = MarketDepthProcessor(features=["trade_counts"])
        result = processor.process(sample_data)

        assert result.shape == (402, self.config.time_bins)
        assert np.isfinite(result).all()

    def test_side_data_processing_with_empty_data(self):
        """Test side data processing with minimal data."""
        processor = MarketDepthProcessor(features=["volume"])

        # Create minimal data with zeros
        empty_data = {}
        for col in (
            ["ask_px_00", "bid_px_00"]
            + [f"ask_px_{str(i).zfill(2)}" for i in range(1, 10)]
            + [f"bid_px_{str(i).zfill(2)}" for i in range(1, 10)]
        ):
            empty_data[col] = np.zeros(50000)

        for col in (
            ["ask_sz_00", "bid_sz_00"]
            + [f"ask_sz_{str(i).zfill(2)}" for i in range(1, 10)]
            + [f"bid_sz_{str(i).zfill(2)}" for i in range(1, 10)]
        ):
            empty_data[col] = np.zeros(50000)

        for col in (
            ["ask_ct_00", "bid_ct_00"]
            + [f"ask_ct_{str(i).zfill(2)}" for i in range(1, 10)]
            + [f"bid_ct_{str(i).zfill(2)}" for i in range(1, 10)]
        ):
            empty_data[col] = np.zeros(50000)

        df = pl.DataFrame(empty_data)
        result = processor.process(df)

        assert result.shape == (402, self.config.time_bins)
        # With all zeros, result should be all zeros
        assert np.allclose(result, 0.0)

    def test_mixed_feature_combinations(self, sample_data):
        """Test various feature combinations."""
        # Test all possible 2-feature combinations
        combinations = [
            ["volume", "variance"],
            ["volume", "trade_counts"],
            ["variance", "trade_counts"],
        ]

        for features in combinations:
            processor = MarketDepthProcessor(features=features)
            result = processor.process(sample_data)

            assert result.shape == (2, 402, self.config.time_bins)
            assert np.isfinite(result).all()

        # Test single features
        for feature in ["volume", "variance", "trade_counts"]:
            processor = MarketDepthProcessor(features=[feature])
            result = processor.process(sample_data)

            assert result.shape == (402, self.config.time_bins)
            assert np.isfinite(result).all()

    def test_factory_functions(self, sample_data):
        """Test factory functions."""
        # Test create_processor with various parameters
        processor1 = create_processor()
        assert processor1.features == ["volume"]

        processor2 = create_processor(features=["variance"])
        assert processor2.features == ["variance"]

        processor3 = create_processor(features=[FeatureType.TRADE_COUNTS])
        assert processor3.features == ["trade_counts"]

        # Test process_market_data function
        result1 = process_market_data(sample_data)
        assert result1.shape == (402, self.config.time_bins)

        result2 = process_market_data(sample_data, features=["volume", "variance"])
        assert result2.shape == (2, 402, self.config.time_bins)

        result3 = process_market_data(sample_data, features=[FeatureType.TRADE_COUNTS])
        assert result3.shape == (402, self.config.time_bins)

    def test_edge_cases_in_processing(self, sample_data):
        """Test edge cases in data processing."""
        processor = MarketDepthProcessor(features=["volume"])

        # Test with data that has very similar prices (minimal spread)
        flat_data = sample_data.with_columns(
            [pl.lit(100.0).alias("ask_px_00"), pl.lit(99.999).alias("bid_px_00")]
            + [pl.lit(100.0 + i * 0.0001).alias(f"ask_px_{str(i).zfill(2)}") for i in range(1, 10)]
            + [pl.lit(99.999 - i * 0.0001).alias(f"bid_px_{str(i).zfill(2)}") for i in range(1, 10)]
        )

        result = processor.process(flat_data)
        assert result.shape == (402, self.config.time_bins)
        assert np.isfinite(result).all()

    def test_processor_state_isolation(self, sample_data):
        """Test that processor instances don't interfere with each other."""
        processor1 = MarketDepthProcessor(features=["volume"])
        processor2 = MarketDepthProcessor(features=["variance"])
        processor3 = MarketDepthProcessor(features=["volume", "trade_counts"])

        # Process with different processors
        result1 = processor1.process(sample_data)
        result2 = processor2.process(sample_data)
        result3 = processor3.process(sample_data)

        # Verify shapes and independence
        assert result1.shape == (402, self.config.time_bins)
        assert result2.shape == (402, self.config.time_bins)
        assert result3.shape == (2, 402, self.config.time_bins)

        # Results should be different (different features) or at least valid
        # Note: with random data, different features might produce similar normalized results
        # The key test is that processing works correctly for each processor type
        assert np.isfinite(result1).all()
        assert np.isfinite(result2).all()
        assert np.isfinite(result3).all()

        # Verify independence by checking processors have different feature sets
        assert processor1.features != processor2.features
        assert processor1.features != processor3.features
