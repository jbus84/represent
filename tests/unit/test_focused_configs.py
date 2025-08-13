"""
Tests for the new focused configuration system.
"""

import pytest

from represent.configs import (
    DatasetBuilderConfig,
    GlobalThresholdConfig,
    MarketDepthProcessorConfig,
    create_compatible_configs,
    create_dataset_builder_config,
    create_processor_config,
    create_represent_config,
    create_threshold_config,
    list_available_currencies,
)


class TestDatasetBuilderConfig:
    """Test DatasetBuilderConfig focused configuration."""

    def test_default_dataset_builder_config(self):
        """Test creating DatasetBuilderConfig with default values."""
        config = DatasetBuilderConfig()

        assert config.currency == "AUDUSD"
        assert config.lookback_rows == 5000
        assert config.lookforward_input == 5000
        assert config.lookforward_offset == 500
        assert config.min_required_samples == 10500  # Computed field

    def test_custom_dataset_builder_config(self):
        """Test creating DatasetBuilderConfig with custom values."""
        config = DatasetBuilderConfig(
            currency="EURUSD",
            lookback_rows=3000,
            lookforward_input=4000,
            lookforward_offset=600
        )

        assert config.currency == "EURUSD"
        assert config.lookback_rows == 3000
        assert config.lookforward_input == 4000
        assert config.lookforward_offset == 600
        assert config.min_required_samples == 7600  # 3000 + 4000 + 600

    def test_currency_validation(self):
        """Test currency pair validation."""
        # Valid currency
        config = DatasetBuilderConfig(currency="GBPUSD")
        assert config.currency == "GBPUSD"

        # Invalid currencies should raise error
        with pytest.raises(ValueError, match="Currency pair must be 6 alphabetic characters"):
            DatasetBuilderConfig(currency="INVALID")

        with pytest.raises(ValueError, match="Currency pair must be 6 alphabetic characters"):
            DatasetBuilderConfig(currency="USD")  # Too short


class TestGlobalThresholdConfig:
    """Test GlobalThresholdConfig focused configuration."""

    def test_default_threshold_config(self):
        """Test creating GlobalThresholdConfig with default values."""
        config = GlobalThresholdConfig()

        assert config.currency == "AUDUSD"
        assert config.nbins == 13
        assert config.lookback_rows == 5000
        assert config.lookforward_input == 5000
        assert config.lookforward_offset == 500
        assert config.max_samples_per_file == 10000
        assert config.sample_fraction == 0.5
        assert config.jump_size == 100

    def test_custom_threshold_config(self):
        """Test creating GlobalThresholdConfig with custom values."""
        config = GlobalThresholdConfig(
            currency="EURUSD",
            nbins=9,
            sample_fraction=0.3,
            jump_size=50
        )

        assert config.currency == "EURUSD"
        assert config.nbins == 9
        assert config.sample_fraction == 0.3
        assert config.jump_size == 50

    def test_nbins_validation(self):
        """Test nbins validation."""
        # Valid nbins
        for nbins in [3, 5, 7, 9, 13]:
            config = GlobalThresholdConfig(nbins=nbins)
            assert config.nbins == nbins

        # Invalid nbins should raise error
        with pytest.raises(ValueError, match="Unsupported nbins value"):
            GlobalThresholdConfig(nbins=11)


class TestMarketDepthProcessorConfig:
    """Test MarketDepthProcessorConfig focused configuration."""

    def test_default_processor_config(self):
        """Test creating MarketDepthProcessorConfig with default values."""
        config = MarketDepthProcessorConfig()

        assert config.features == ["volume"]
        assert config.samples == 50000
        assert config.ticks_per_bin == 100
        assert config.micro_pip_size == 0.00001
        assert config.time_bins == 500  # Computed: 50000 // 100
        assert config.output_shape == (402, 500)  # Single feature: 2D

    def test_custom_processor_config(self):
        """Test creating MarketDepthProcessorConfig with custom values."""
        config = MarketDepthProcessorConfig(
            features=["volume", "variance"],
            samples=25000,
            ticks_per_bin=50
        )

        assert config.features == ["volume", "variance"]
        assert config.samples == 25000
        assert config.ticks_per_bin == 50
        assert config.time_bins == 500  # 25000 // 50
        assert config.output_shape == (2, 402, 500)  # Multiple features: 3D

    def test_feature_validation(self):
        """Test feature validation."""
        # Valid features
        config = MarketDepthProcessorConfig(features=["volume", "variance", "trade_counts"])
        assert config.features == ["volume", "variance", "trade_counts"]

        # Invalid features should raise error
        with pytest.raises(ValueError, match="Invalid features"):
            MarketDepthProcessorConfig(features=["invalid_feature"])


class TestFactoryFunctions:
    """Test configuration factory functions."""

    def test_create_dataset_builder_config(self):
        """Test dataset builder config factory."""
        config = create_dataset_builder_config(
            currency="GBPUSD",
            lookback_rows=3000
        )

        assert isinstance(config, DatasetBuilderConfig)
        assert config.currency == "GBPUSD"
        assert config.lookback_rows == 3000

    def test_create_threshold_config(self):
        """Test threshold config factory."""
        config = create_threshold_config(
            currency="EURUSD",
            nbins=9,
            jump_size=50
        )

        assert isinstance(config, GlobalThresholdConfig)
        assert config.currency == "EURUSD"
        assert config.nbins == 9
        assert config.jump_size == 50

    def test_create_processor_config(self):
        """Test processor config factory."""
        config = create_processor_config(
            features=["volume", "variance"],
            samples=30000
        )

        assert isinstance(config, MarketDepthProcessorConfig)
        assert config.features == ["volume", "variance"]
        assert config.samples == 30000

    def test_create_compatible_configs(self):
        """Test creating compatible configs for all modules."""
        dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
            currency="GBPUSD",
            features=["volume", "variance"],
            nbins=9,
            samples=25000
        )

        # Check types
        assert isinstance(dataset_cfg, DatasetBuilderConfig)
        assert isinstance(threshold_cfg, GlobalThresholdConfig)
        assert isinstance(processor_cfg, MarketDepthProcessorConfig)

        # Check consistency
        assert dataset_cfg.currency == threshold_cfg.currency == "GBPUSD"
        assert dataset_cfg.lookback_rows == threshold_cfg.lookback_rows
        assert dataset_cfg.lookforward_input == threshold_cfg.lookforward_input
        assert dataset_cfg.lookforward_offset == threshold_cfg.lookforward_offset
        assert threshold_cfg.nbins == 9
        assert processor_cfg.features == ["volume", "variance"]
        assert processor_cfg.samples == 25000


class TestLegacyCompatibility:
    """Test legacy compatibility functions."""

    def test_create_represent_config_returns_tuple(self):
        """Test that create_represent_config returns tuple of configs."""
        result = create_represent_config()

        # Should return tuple of 3 configs
        assert isinstance(result, tuple)
        assert len(result) == 3

        dataset_cfg, threshold_cfg, processor_cfg = result
        assert isinstance(dataset_cfg, DatasetBuilderConfig)
        assert isinstance(threshold_cfg, GlobalThresholdConfig)
        assert isinstance(processor_cfg, MarketDepthProcessorConfig)

    def test_create_represent_config_currency_optimizations(self):
        """Test that currency-specific optimizations are applied."""
        # GBPUSD should get shorter lookforward for volatility
        dataset_cfg, threshold_cfg, processor_cfg = create_represent_config(currency="GBPUSD")
        assert dataset_cfg.lookforward_input == 3000
        assert threshold_cfg.lookforward_input == 3000

        # JPY pairs should get different pip sizes and fewer bins
        dataset_cfg, threshold_cfg, processor_cfg = create_represent_config(currency="USDJPY")
        assert threshold_cfg.nbins == 9
        assert processor_cfg.micro_pip_size == 0.001

    def test_override_currency_optimizations(self):
        """Test that explicit parameters override currency-specific optimizations."""
        dataset_cfg, threshold_cfg, processor_cfg = create_represent_config(
            currency="GBPUSD",
            lookforward_input=5000,  # Override the GBPUSD default of 3000
            nbins=13
        )

        assert dataset_cfg.lookforward_input == 5000  # Should use explicit value
        assert threshold_cfg.lookforward_input == 5000
        assert threshold_cfg.nbins == 13

    def test_list_available_currencies(self):
        """Test listing available currencies."""
        currencies = list_available_currencies()

        assert isinstance(currencies, list)
        assert len(currencies) > 0
        assert "AUDUSD" in currencies
        assert "EURUSD" in currencies
        assert "GBPUSD" in currencies


class TestConfigValidation:
    """Test configuration validation scenarios."""

    def test_positive_values_required(self):
        """Test that positive values are required for numeric fields."""
        with pytest.raises(ValueError):
            DatasetBuilderConfig(lookback_rows=0)

        with pytest.raises(ValueError):
            DatasetBuilderConfig(lookforward_input=-1)

        with pytest.raises(ValueError):
            GlobalThresholdConfig(max_samples_per_file=0)

        with pytest.raises(ValueError):
            MarketDepthProcessorConfig(samples=0)

    def test_range_validation(self):
        """Test range validation for specific fields."""
        with pytest.raises(ValueError):
            GlobalThresholdConfig(sample_fraction=0.0)  # Must be > 0

        with pytest.raises(ValueError):
            GlobalThresholdConfig(sample_fraction=1.5)  # Must be <= 1

        with pytest.raises(ValueError):
            MarketDepthProcessorConfig(samples=10000)  # Must be >= 25000


class TestConfigIntegration:
    """Test that configs work together properly."""

    def test_compatible_price_movement_params(self):
        """Test that dataset and threshold configs have compatible price movement params."""
        dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
            lookback_rows=3000,
            lookforward_input=4000,
            lookforward_offset=600
        )

        # Price movement parameters should match
        assert dataset_cfg.lookback_rows == threshold_cfg.lookback_rows == 3000
        assert dataset_cfg.lookforward_input == threshold_cfg.lookforward_input == 4000
        assert dataset_cfg.lookforward_offset == threshold_cfg.lookforward_offset == 600

        # Min required samples should be consistent
        expected_min_samples = 3000 + 4000 + 600  # 7600
        assert dataset_cfg.min_required_samples == expected_min_samples

    def test_feature_consistency(self):
        """Test that features are handled consistently."""
        dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
            features=["volume", "variance", "trade_counts"]
        )

        # Only processor config should have features
        assert processor_cfg.features == ["volume", "variance", "trade_counts"]
        # Dataset and threshold configs don't need features

        # Output shape should be computed correctly
        assert processor_cfg.output_shape == (3, 402, 500)  # 3 features = 3D tensor
