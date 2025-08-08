"""
Integration tests to verify the complete pipeline works as expected.
"""

import numpy as np
import polars as pl

from represent.constants import PRICE_LEVELS
from represent.config import create_represent_config
from represent.pipeline import process_market_data
from tests.unit.fixtures.sample_data import generate_realistic_market_data


class TestIntegration:
    """Integration tests for the complete system."""

    def setup_method(self):
        """Setup config for each test."""
        self.config = create_represent_config("AUDUSD")

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate realistic test data
        data = generate_realistic_market_data(n_samples=50000, seed=42)

        # Run the current pipeline
        result = process_market_data(data, currency="AUDUSD")

        # Validate output with current config
        expected_shape = self.config.output_shape
        assert result.shape == expected_shape
        assert result.shape == (PRICE_LEVELS, self.config.time_bins)
        assert np.all(np.isfinite(result))
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

        # Check that we have meaningful variation
        assert np.std(result) > 0.01
        assert not np.all(result == 0)

    def test_pipeline_reproducibility(self):
        """Test that pipeline produces reproducible results."""
        # Generate same data twice
        data1 = generate_realistic_market_data(n_samples=50000, seed=123)
        data2 = generate_realistic_market_data(n_samples=50000, seed=123)

        # Run pipeline
        result1 = process_market_data(data1, currency="AUDUSD")
        result2 = process_market_data(data2, currency="AUDUSD")

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_pipeline_with_different_market_conditions(self):
        """Test pipeline under different market conditions."""
        # Test with different base prices and spreads
        conditions = [
            {"base_price": 0.6600, "spread": 0.0002},  # Normal AUDUSD
            {"base_price": 1.2000, "spread": 0.0003},  # Higher price, wider spread
            {"base_price": 0.9000, "spread": 0.0001},  # Mid price, tight spread
        ]

        results = []
        for condition in conditions:
            data = generate_realistic_market_data(
                n_samples=50000,
                base_price=condition["base_price"],
                spread=condition["spread"],
                seed=42,
            )
            result = process_market_data(data, currency="AUDUSD")
            results.append(result)

            # Each result should be valid
            assert result.shape == self.config.output_shape
            assert np.all(np.isfinite(result))

        # Different market conditions should produce different results
        assert not np.array_equal(results[0], results[1])
        assert not np.array_equal(results[1], results[2])

    def test_pipeline_stability(self):
        """Test pipeline stability with edge cases."""
        # Test with constant prices (no variation)
        constant_data = generate_realistic_market_data(n_samples=50000, seed=42)
        
        # Make prices constant
        for col in constant_data.columns:
            if "px_" in col:
                constant_data = constant_data.with_columns(
                    pl.lit(1.0).alias(col)
                )

        result = process_market_data(constant_data, currency="AUDUSD")

        # Should still produce valid output
        assert result.shape == self.config.output_shape
        assert np.all(np.isfinite(result))

    def test_pipeline_multi_feature(self):
        """Test pipeline with multiple features."""
        data = generate_realistic_market_data(n_samples=50000, seed=42)

        # Test single feature
        result_single = process_market_data(data, features=["volume"], currency="AUDUSD")
        assert result_single.shape == (PRICE_LEVELS, self.config.time_bins)

        # Test multiple features
        result_multi = process_market_data(data, features=["volume", "variance"], currency="AUDUSD")
        assert result_multi.shape == (2, PRICE_LEVELS, self.config.time_bins)

        # Results might be the same if only using volume feature
        # But shapes should be different
        assert result_single.ndim == 2
        assert result_multi.ndim == 3

    def test_pipeline_different_currencies(self):
        """Test pipeline with different currency configurations."""
        data = generate_realistic_market_data(n_samples=50000, seed=42)

        currencies = ["AUDUSD", "EURUSD", "GBPUSD"]
        results = {}

        for currency in currencies:
            config = create_represent_config(currency)
            result = process_market_data(data, currency=currency)
            results[currency] = result

            # Each should have correct shape for that currency
            assert result.shape == config.output_shape
            assert result.shape == (PRICE_LEVELS, config.time_bins)
            assert np.all(np.isfinite(result))

        # All currencies should use same time_bins (250) but may have different processing
        for currency, result in results.items():
            assert result.shape[1] == 250  # time_bins = 25000 // 100 = 250

    def test_pipeline_performance(self):
        """Test pipeline meets performance requirements."""
        import time
        
        data = generate_realistic_market_data(n_samples=50000, seed=42)

        # Measure processing time
        start_time = time.perf_counter()
        result = process_market_data(data, currency="AUDUSD")
        end_time = time.perf_counter()

        processing_time_ms = (end_time - start_time) * 1000

        # Validate result
        assert result.shape == self.config.output_shape
        assert np.all(np.isfinite(result))

        # Performance requirement: should be reasonably fast
        assert processing_time_ms < 5000  # Less than 5 seconds for 50K samples

    def test_pipeline_memory_efficiency(self):
        """Test pipeline memory usage."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process multiple datasets
        for i in range(5):
            data = generate_realistic_market_data(n_samples=25000, seed=i)
            result = process_market_data(data, currency="AUDUSD")
            assert result.shape == (PRICE_LEVELS, 250)  # Verify correct time_bins

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory
        assert memory_increase < 500  # Less than 500MB increase


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_config_consistency_across_components(self):
        """Test that all components use consistent configuration."""
        from represent.pipeline import MarketDepthProcessor
        
        currency = "AUDUSD"
        config = create_represent_config(currency)
        processor = MarketDepthProcessor(currency=currency)

        # All should use same configuration values
        assert processor.config.time_bins == config.time_bins
        assert processor.config.micro_pip_size == config.micro_pip_size
        assert processor.config.output_shape == config.output_shape

    def test_time_bins_calculation(self):
        """Test that TIME_BINS is calculated correctly."""
        config = create_represent_config("AUDUSD")
        
        # Verify the calculation
        expected_time_bins = config.samples // config.ticks_per_bin
        assert config.time_bins == expected_time_bins
        assert config.time_bins == 250  # 25000 // 100 = 250

    def test_output_shape_consistency(self):
        """Test that output_shape matches actual processing results."""
        data = generate_realistic_market_data(n_samples=50000, seed=42)
        config = create_represent_config("AUDUSD")
        
        result = process_market_data(data, currency="AUDUSD")
        
        # Result shape should match config output_shape
        assert result.shape == config.output_shape
        assert result.shape == (PRICE_LEVELS, config.time_bins)