"""
Efficient tests for Pydantic configuration system.
Tests the new ClassificationConfig, SamplingConfig, and CurrencyConfig models.
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from represent.config import (
    ClassificationConfig,
    SamplingConfig,
    CurrencyConfig,
    load_currency_config,
    get_default_currency_config,
    save_currency_config,
    list_available_currencies,
)


class TestClassificationConfig:
    """Test ClassificationConfig validation and functionality."""

    def test_default_classification_config(self):
        """Test default configuration values."""
        config = ClassificationConfig()

        assert config.micro_pip_size == 0.00001
        assert config.true_pip_size == 0.0001
        assert config.ticks_per_bin == 100
        assert config.lookforward_offset == 500
        assert config.lookforward_input == 5000
        assert config.lookback_rows == 5000
        assert config.nbins == 13
        assert isinstance(config.bin_thresholds, dict)
        assert 13 in config.bin_thresholds

    def test_custom_classification_config(self):
        """Test custom configuration creation."""
        config = ClassificationConfig(
            true_pip_size=0.01,  # JPY pair
            nbins=9,
            lookforward_input=3000,
        )

        assert config.true_pip_size == 0.01
        assert config.nbins == 9
        assert config.lookforward_input == 3000
        # Other values should remain default
        assert config.ticks_per_bin == 100

    def test_classification_validation(self):
        """Test validation of configuration values."""
        # Test valid nbins values
        valid_nbins = [3, 5, 7, 9, 13]
        for nbins in valid_nbins:
            config = ClassificationConfig(nbins=nbins)
            assert config.nbins == nbins

        # Test invalid nbins values
        with pytest.raises(ValidationError):
            ClassificationConfig(nbins=4)  # Not in allowed values

        with pytest.raises(ValidationError):
            ClassificationConfig(nbins=20)  # Too high

    def test_get_thresholds_method(self):
        """Test the get_thresholds method for different configurations."""
        config = ClassificationConfig()

        # Test default thresholds
        thresholds = config.get_thresholds()
        assert "bin_1" in thresholds
        assert "bin_2" in thresholds
        assert isinstance(thresholds["bin_1"], float)

        # Test specific configuration
        thresholds = config.get_thresholds(ticks_per_bin=100, lookforward_input=5000)
        assert len(thresholds) >= 3

        # Test fallback behavior
        thresholds = config.get_thresholds(ticks_per_bin=999, lookforward_input=999)
        assert len(thresholds) >= 3  # Should fallback to default

    def test_pip_size_validation(self):
        """Test pip size validation."""
        # Valid pip sizes
        ClassificationConfig(true_pip_size=0.0001)
        ClassificationConfig(true_pip_size=0.01)  # JPY pairs

        # Invalid pip sizes
        with pytest.raises(ValidationError):
            ClassificationConfig(true_pip_size=0)  # Must be > 0

        with pytest.raises(ValidationError):
            ClassificationConfig(true_pip_size=-0.1)  # Must be positive


class TestSamplingConfig:
    """Test SamplingConfig validation and functionality."""

    def test_default_sampling_config(self):
        """Test default sampling configuration."""
        config = SamplingConfig()

        assert config.sampling_mode == "consecutive"
        assert config.coverage_percentage == 1.0
        assert config.end_tick_strategy == "uniform_random"
        assert config.min_tick_spacing == 100
        assert config.seed == 42
        assert config.max_samples is None

    def test_custom_sampling_config(self):
        """Test custom sampling configuration."""
        config = SamplingConfig(
            sampling_mode="random", coverage_percentage=0.5, seed=123, max_samples=1000
        )

        assert config.sampling_mode == "random"
        assert config.coverage_percentage == 0.5
        assert config.seed == 123
        assert config.max_samples == 1000

    def test_sampling_validation(self):
        """Test validation of sampling parameters."""
        # Valid sampling modes
        for mode in ["consecutive", "random", "stratified_random"]:
            config = SamplingConfig(sampling_mode=mode)
            assert config.sampling_mode == mode

        # Invalid sampling mode
        with pytest.raises(ValidationError):
            SamplingConfig(sampling_mode="invalid_mode")

        # Coverage percentage validation
        SamplingConfig(coverage_percentage=0.0)  # Valid
        SamplingConfig(coverage_percentage=1.0)  # Valid
        SamplingConfig(coverage_percentage=0.5)  # Valid

        with pytest.raises(ValidationError):
            SamplingConfig(coverage_percentage=-0.1)  # Invalid

        with pytest.raises(ValidationError):
            SamplingConfig(coverage_percentage=1.5)  # Invalid

    def test_end_tick_strategy_validation(self):
        """Test end tick strategy validation."""
        valid_strategies = ["uniform_random", "weighted_random", "temporal_distribution"]
        for strategy in valid_strategies:
            config = SamplingConfig(end_tick_strategy=strategy)
            assert config.end_tick_strategy == strategy

        with pytest.raises(ValidationError):
            SamplingConfig(end_tick_strategy="invalid_strategy")


class TestCurrencyConfig:
    """Test CurrencyConfig validation and functionality."""

    def test_default_currency_config(self):
        """Test currency configuration creation."""
        config = CurrencyConfig(currency_pair="AUDUSD")

        assert config.currency_pair == "AUDUSD"
        assert isinstance(config.classification, ClassificationConfig)
        assert isinstance(config.sampling, SamplingConfig)
        assert config.description is None

    def test_currency_pair_validation(self):
        """Test currency pair validation."""
        # Valid currency pairs
        valid_pairs = ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY"]
        for pair in valid_pairs:
            config = CurrencyConfig(currency_pair=pair)
            assert config.currency_pair == pair.upper()

        # Test lowercase input (should be converted to uppercase)
        config = CurrencyConfig(currency_pair="audusd")
        assert config.currency_pair == "AUDUSD"

        # Invalid currency pairs
        with pytest.raises(ValidationError):
            CurrencyConfig(currency_pair="INVALID")  # Too short

        with pytest.raises(ValidationError):
            CurrencyConfig(currency_pair="AUDUSDD")  # Too long

        with pytest.raises(ValidationError):
            CurrencyConfig(currency_pair="AUD123")  # Contains numbers

    def test_nested_configuration(self):
        """Test nested classification and sampling configs."""
        config = CurrencyConfig(
            currency_pair="EURUSD",
            classification=ClassificationConfig(nbins=9),
            sampling=SamplingConfig(coverage_percentage=0.8),
        )

        assert config.currency_pair == "EURUSD"
        assert config.classification.nbins == 9
        assert config.sampling.coverage_percentage == 0.8


class TestCurrencyConfigFunctions:
    """Test currency configuration loading and management functions."""

    def test_get_default_currency_config(self):
        """Test default currency configuration generation."""
        # Test AUDUSD
        config = get_default_currency_config("AUDUSD")
        assert config.currency_pair == "AUDUSD"
        assert config.classification.true_pip_size == 0.0001
        assert config.classification.nbins == 13
        assert config.sampling.coverage_percentage == 0.8

        # Test JPY pair
        config = get_default_currency_config("USDJPY")
        assert config.currency_pair == "USDJPY"
        assert config.classification.true_pip_size == 0.01  # JPY pip size
        assert config.classification.nbins == 9  # Fewer bins for JPY

        # Test unknown pair (should get generic config)
        config = get_default_currency_config("NZDUSD")  # Valid but not specifically configured
        assert config.currency_pair == "NZDUSD"
        assert isinstance(config.classification, ClassificationConfig)

    def test_save_and_load_currency_config(self):
        """Test saving and loading currency configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create test configuration
            original_config = CurrencyConfig(
                currency_pair="CHFUSD",  # Valid 6-character currency pair
                classification=ClassificationConfig(nbins=7),
                sampling=SamplingConfig(coverage_percentage=0.6),
                description="Test configuration",
            )

            # Save configuration
            config_file = save_currency_config(original_config, config_dir)
            assert config_file.exists()
            assert config_file.name == "chfusd.json"

            # Load configuration
            loaded_config = load_currency_config("CHFUSD", config_dir)
            assert loaded_config.currency_pair == "CHFUSD"
            assert loaded_config.classification.nbins == 7
            assert loaded_config.sampling.coverage_percentage == 0.6
            assert loaded_config.description == "Test configuration"

    def test_load_nonexistent_currency_config(self):
        """Test loading non-existent currency configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Should return default configuration
            config = load_currency_config("USDCAD", config_dir)  # Valid but non-existent
            assert config.currency_pair == "USDCAD"
            assert isinstance(config.classification, ClassificationConfig)

    def test_list_available_currencies(self):
        """Test listing available currency configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Initially empty
            currencies = list_available_currencies(config_dir)
            assert currencies == []

            # Save some configurations
            config1 = CurrencyConfig(currency_pair="AUDUSD")
            config2 = CurrencyConfig(currency_pair="EURUSD")

            save_currency_config(config1, config_dir)
            save_currency_config(config2, config_dir)

            # Should list both
            currencies = list_available_currencies(config_dir)
            assert sorted(currencies) == ["AUDUSD", "EURUSD"]

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create invalid JSON file
            invalid_file = config_dir / "invalid.json"
            invalid_file.write_text("{ invalid json }")

            # Should raise ValueError
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_currency_config("INVALID", config_dir)


class TestConfigurationIntegration:
    """Test integration between different configuration components."""

    def test_config_serialization_roundtrip(self):
        """Test that configurations can be serialized and deserialized correctly."""
        original_config = CurrencyConfig(
            currency_pair="CADCHF",  # Valid 6-character currency pair
            classification=ClassificationConfig(
                nbins=9, true_pip_size=0.01, lookforward_input=3000
            ),
            sampling=SamplingConfig(sampling_mode="random", coverage_percentage=0.7, seed=999),
            description="Integration test configuration",
        )

        # Serialize to dict
        config_dict = original_config.model_dump()

        # Deserialize from dict
        restored_config = CurrencyConfig(**config_dict)

        # Verify all fields match
        assert restored_config.currency_pair == original_config.currency_pair
        assert restored_config.classification.nbins == original_config.classification.nbins
        assert (
            restored_config.classification.true_pip_size
            == original_config.classification.true_pip_size
        )
        assert restored_config.sampling.sampling_mode == original_config.sampling.sampling_mode
        assert (
            restored_config.sampling.coverage_percentage
            == original_config.sampling.coverage_percentage
        )
        assert restored_config.description == original_config.description

    def test_partial_config_updates(self):
        """Test updating configurations with partial data."""
        base_config = ClassificationConfig()

        # Update with partial data
        updated_data = {"nbins": 7, "lookforward_input": 3000}
        updated_config = ClassificationConfig(**{**base_config.model_dump(), **updated_data})

        # Should have updated fields
        assert updated_config.nbins == 7
        assert updated_config.lookforward_input == 3000

        # Should retain other defaults
        assert updated_config.true_pip_size == base_config.true_pip_size
        assert updated_config.ticks_per_bin == base_config.ticks_per_bin

    def test_threshold_consistency(self):
        """Test that threshold configurations are consistent."""
        # Test that all supported nbins have thresholds
        for nbins in [3, 5, 7, 9, 13]:
            config_with_bins = ClassificationConfig(nbins=nbins)
            thresholds = config_with_bins.get_thresholds()
            assert len(thresholds) > 0
            assert all(isinstance(v, float) for v in thresholds.values())

    def test_real_currency_configs(self):
        """Test the actual pre-configured currency files."""
        # Test AUDUSD configuration exists and is valid
        try:
            audusd_config = load_currency_config("AUDUSD")
            assert audusd_config.currency_pair == "AUDUSD"
            assert audusd_config.classification.nbins == 13
            assert audusd_config.sampling.coverage_percentage == 0.8
        except FileNotFoundError:
            # Config file might not exist in test environment, that's ok
            pass

        # Test that default generation works for all major pairs
        major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
        for pair in major_pairs:
            config = get_default_currency_config(pair)
            assert config.currency_pair == pair
            assert isinstance(config.classification, ClassificationConfig)
            assert isinstance(config.sampling, SamplingConfig)
