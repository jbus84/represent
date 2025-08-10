"""
Tests for the simplified RepresentConfig system.
"""

import pytest
from represent.config import RepresentConfig, create_represent_config


class TestRepresentConfig:
    """Test the new simplified configuration system."""
    
    def test_default_represent_config(self):
        """Test creating RepresentConfig with default values."""
        config = RepresentConfig()
        
        assert config.currency == "AUDUSD"
        assert config.nbins == 13
        assert config.samples == 25000
        assert config.features == ["volume"]
        assert config.lookback_rows == 5000
        assert config.lookforward_input == 5000
        assert config.lookforward_offset == 500
        assert config.batch_size == 1000
        assert config.ticks_per_bin == 100
        
    def test_custom_represent_config(self):
        """Test creating RepresentConfig with custom values."""
        config = RepresentConfig(
            currency="EURUSD",
            nbins=9,
            samples=50000,
            features=["volume", "variance"],
            lookback_rows=3000,
            lookforward_input=4000,
            batch_size=2000,
        )
        
        assert config.currency == "EURUSD"
        assert config.nbins == 9
        assert config.samples == 50000
        assert config.features == ["volume", "variance"]
        assert config.lookback_rows == 3000
        assert config.lookforward_input == 4000
        assert config.batch_size == 2000
        
    def test_configurable_lookback_lookforward(self):
        """Test that lookback and lookforward parameters are fully configurable."""
        # Test different combinations
        configs = [
            {"lookback_rows": 1000, "lookforward_input": 2000},
            {"lookback_rows": 3000, "lookforward_input": 5000}, 
            {"lookback_rows": 10000, "lookforward_input": 1000},
        ]
        
        for config_data in configs:
            config = RepresentConfig(**config_data)
            assert config.lookback_rows == config_data["lookback_rows"]
            assert config.lookforward_input == config_data["lookforward_input"]
            
    def test_computed_fields(self):
        """Test auto-computed fields."""
        config = RepresentConfig(samples=10000, ticks_per_bin=50)
        
        # time_bins should be computed as samples // ticks_per_bin
        assert config.time_bins == 10000 // 50  # = 200
        
        # min_symbol_samples is now configurable with default 100 (lowered from 1000)
        assert config.min_symbol_samples == 100  # default value
        
        # Test that min_symbol_samples can be configured
        config_custom = RepresentConfig(samples=10000, ticks_per_bin=50, min_symbol_samples=400)
        assert config_custom.min_symbol_samples == 400  # custom value
        
    def test_feature_validation(self):
        """Test feature validation."""
        # Valid features
        config = RepresentConfig(features=["volume", "variance", "trade_counts"])
        assert config.features == ["volume", "variance", "trade_counts"]
        
        # Invalid features should raise error
        with pytest.raises(ValueError, match="Invalid features"):
            RepresentConfig(features=["invalid_feature"])
            
    def test_currency_validation(self):
        """Test currency pair validation."""
        # Valid currency
        config = RepresentConfig(currency="GBPUSD")
        assert config.currency == "GBPUSD"
        
        # Invalid currencies should raise error
        with pytest.raises(ValueError, match="Currency pair must be 6 alphabetic characters"):
            RepresentConfig(currency="INVALID")
            
        with pytest.raises(ValueError, match="Currency pair must be 6 alphabetic characters"):
            RepresentConfig(currency="USD")  # Too short
            
    def test_nbins_validation(self):
        """Test nbins validation."""
        # Valid nbins
        for nbins in [3, 5, 7, 9, 13]:
            config = RepresentConfig(nbins=nbins)
            assert config.nbins == nbins
            
        # Invalid nbins should raise error
        with pytest.raises(ValueError, match="Unsupported nbins value"):
            RepresentConfig(nbins=11)
            
    def test_config_simplicity(self):
        """Test that configuration is now simple and direct."""
        config = RepresentConfig(
            currency="GBPUSD",
            nbins=9,
            lookback_rows=3000,
            lookforward_input=4000,
        )
        
        # Direct access to all parameters - no nested structures
        assert config.currency == "GBPUSD"
        assert config.nbins == 9
        assert config.lookback_rows == 3000
        assert config.lookforward_input == 4000


class TestCreateRepresentConfig:
    """Test the create_represent_config convenience function."""
    
    def test_create_simple_config(self):
        """Test creating simple config with convenience function."""
        config = create_represent_config()
        
        assert config.currency == "AUDUSD"
        assert config.nbins == 13
        assert config.samples == 25000
        assert config.features == ["volume"]
        
    def test_create_custom_config(self):
        """Test creating custom config with convenience function."""
        config = create_represent_config(
            currency="GBPUSD",
            nbins=9,
            samples=15000,
            features=["volume", "variance"],
            lookback_rows=2000,
        )
        
        assert config.currency == "GBPUSD"
        assert config.nbins == 9
        assert config.samples == 15000
        assert config.features == ["volume", "variance"]
        assert config.lookback_rows == 2000
        
    def test_currency_specific_optimizations(self):
        """Test that currency-specific optimizations are applied."""
        # GBPUSD should get shorter lookforward for volatility
        gbp_config = create_represent_config(currency="GBPUSD")
        assert gbp_config.lookforward_input == 3000
        
        # JPY pairs should get different pip sizes and fewer bins
        jpy_config = create_represent_config(currency="USDJPY")
        assert jpy_config.true_pip_size == 0.01
        assert jpy_config.micro_pip_size == 0.001
        assert jpy_config.nbins == 9
        
    def test_override_currency_optimizations(self):
        """Test that explicit parameters override currency-specific optimizations."""
        config = create_represent_config(
            currency="GBPUSD",
            lookforward_input=5000,  # Override the GBPUSD default of 3000
            nbins=13
        )
        
        assert config.lookforward_input == 5000  # Should use explicit value
        assert config.nbins == 13


class TestConfigErrorHandling:
    """Test configuration error handling scenarios."""
    
    def test_invalid_currency_error(self):
        """Test error handling for invalid currency pairs."""
        with pytest.raises(ValueError, match="Currency pair must be 6 alphabetic characters"):
            RepresentConfig(currency="INVALID")
            
        with pytest.raises(ValueError, match="Currency pair must be 6 alphabetic characters"):
            RepresentConfig(currency="USD")  # Too short
            
    def test_invalid_features_error(self):
        """Test error handling for invalid features."""
        with pytest.raises(ValueError, match="Invalid features"):
            RepresentConfig(features=["invalid_feature"])
    
    def test_create_config_error_handling(self):
        """Test error handling in create_represent_config."""
        with pytest.raises(ValueError):
            create_represent_config("INVALID")


class TestCurrencySpecificConfigs:
    """Test currency-specific configuration optimizations."""
    
    def test_supported_currencies(self):
        """Test all supported currencies can create configs."""
        currencies = ["AUDUSD", "GBPUSD", "EURJPY", "USDJPY", "EURUSD"]
        
        for currency in currencies:
            config = create_represent_config(currency)
            assert config.currency == currency
            assert config.nbins > 0
            assert config.lookforward_input > 0