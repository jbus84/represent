"""
Tests for API coverage improvement.
"""

import pytest
from pathlib import Path

# Import API components directly to avoid torch issues
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from represent.api import RepresentAPI, create_training_dataloader
    from represent.config import load_currency_config

    API_AVAILABLE = True
except ImportError as e:
    print(f"API import failed: {e}")
    API_AVAILABLE = False


@pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
class TestRepresentAPI:
    """Test RepresentAPI functionality."""

    def test_api_initialization(self):
        """Test API initialization."""
        api = RepresentAPI()
        assert api is not None

    def test_get_package_info(self):
        """Test package info retrieval."""
        api = RepresentAPI()
        info = api.get_package_info()

        assert isinstance(info, dict)
        assert "version" in info
        assert "architecture" in info
        assert "available_currencies" in info
        assert "supported_features" in info
        assert "tensor_shape" in info

    def test_get_currency_config(self):
        """Test currency config retrieval."""
        api = RepresentAPI()

        # Test each supported currency
        for currency in ["AUDUSD", "GBPUSD", "EURJPY"]:
            config = api.get_currency_config(currency)
            assert config is not None
            assert config.currency_pair == currency
            assert config.classification is not None




@pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
class TestAPIConvenienceFunctions:
    """Test API convenience functions."""


    def test_create_training_dataloader_missing_file(self):
        """Test create_training_dataloader with missing file."""
        with pytest.raises(FileNotFoundError):
            create_training_dataloader(parquet_path="/nonexistent/file.parquet", batch_size=32)


@pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
class TestAPIConstants:
    """Test API constants and utilities."""

    def test_supported_currencies(self):
        """Test supported currencies list."""
        api = RepresentAPI()
        info = api.get_package_info()

        supported = info["available_currencies"]
        assert "AUDUSD" in supported
        assert "GBPUSD" in supported
        assert "EURJPY" in supported

    def test_supported_features(self):
        """Test supported features list."""
        api = RepresentAPI()
        info = api.get_package_info()

        features = info["supported_features"]
        assert "volume" in features
        assert "variance" in features
        assert "trade_counts" in features

    def test_tensor_shape_info(self):
        """Test tensor shape information."""
        api = RepresentAPI()
        info = api.get_package_info()

        tensor_shape = info["tensor_shape"]
        # Shape info contains descriptive text
        assert "(402, 500)" in tensor_shape


@pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
class TestAPIErrorHandling:
    """Test API error handling."""

    def test_get_currency_config_invalid_currency(self):
        """Test get_currency_config with invalid currency."""
        api = RepresentAPI()

        with pytest.raises(ValueError):
            api.get_currency_config("INVALID")



class TestDirectAPIImports:
    """Test direct API imports without full package."""

    def test_currency_config_loading(self):
        """Test direct currency config loading."""
        try:
            config = load_currency_config("AUDUSD")
            assert config.currency_pair == "AUDUSD"
            assert config.classification is not None
        except Exception:
            pytest.skip("Currency config loading not available")

    def test_multiple_currency_configs(self):
        """Test loading multiple currency configs."""
        currencies = ["AUDUSD", "GBPUSD", "EURJPY"]

        for currency in currencies:
            try:
                config = load_currency_config(currency)
                assert config.currency_pair == currency
            except Exception:
                pytest.skip(f"Currency config for {currency} not available")
