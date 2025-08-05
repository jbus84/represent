"""
Simple tests to boost coverage for core functionality.
These tests avoid the PyTorch import issues by testing components in isolation.
"""

import pytest
import sys
from pathlib import Path

# Add represent to path to avoid package import issues
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_constants_import():
    """Test constants can be imported."""
    from represent.constants import PRICE_LEVELS, TIME_BINS, SAMPLES, TICKS_PER_BIN, MICRO_PIP_SIZE

    assert PRICE_LEVELS == 402
    assert TIME_BINS == 500
    assert SAMPLES == 50000
    assert TICKS_PER_BIN == 100
    assert MICRO_PIP_SIZE == 0.00001


def test_config_currency_loading():
    """Test currency config loading."""
    from represent.config import load_currency_config

    # Test each supported currency
    currencies = ["AUDUSD", "GBPUSD", "EURJPY"]

    for currency in currencies:
        config = load_currency_config(currency)
        assert config.currency_pair == currency
        assert config.classification is not None
        assert config.classification.nbins > 0
        assert config.classification.lookforward_input > 0


def test_config_error_handling():
    """Test config error handling."""
    from represent.config import load_currency_config

    with pytest.raises(ValueError):
        load_currency_config("INVALID_CURRENCY")


def test_data_structures_creation():
    """Test data structure creation."""
    from represent.data_structures import PriceLookupTable, VolumeGrid, OutputBuffer

    # Test PriceLookupTable
    lookup = PriceLookupTable(125000.0)  # Mid price in micro pips
    assert lookup is not None

    # Test valid bid price lookup
    bid_price = 125000 - 1  # Slightly below mid
    bid_index = lookup.get_bid_index(bid_price)
    assert bid_index >= 0

    # Test valid ask price lookup
    ask_price = 125000 + 1  # Slightly above mid
    ask_index = lookup.get_ask_index(ask_price)
    assert ask_index >= 0

    # Test VolumeGrid
    grid = VolumeGrid()
    assert grid is not None
    assert grid.grid.shape == (402, 500)

    # Test OutputBuffer
    buffer = OutputBuffer()
    assert buffer is not None
    assert buffer.buffer.shape == (402, 500)


def test_pipeline_feature_validation():
    """Test pipeline feature validation."""
    from represent.pipeline import MarketDepthProcessor

    # Test valid features
    valid_features = ["volume"]
    processor = MarketDepthProcessor(features=valid_features)
    assert processor.features == valid_features

    # Test default features
    processor_default = MarketDepthProcessor()
    assert processor_default.features == ["volume"]

    # Test invalid features should raise error
    with pytest.raises(ValueError, match="Invalid features"):
        MarketDepthProcessor(features=["invalid_feature"])


def test_core_functions():
    """Test core module functions."""
    from represent.core import process_market_data, reference_pipeline

    # These functions should be importable
    assert callable(process_market_data)
    assert callable(reference_pipeline)


def test_dataloader_factory():
    """Test dataloader factory function."""
    from represent.dataloader import create_market_depth_dataloader

    # Function should be importable
    assert callable(create_market_depth_dataloader)

    # Should raise error with missing file
    with pytest.raises((FileNotFoundError, OSError)):
        create_market_depth_dataloader(parquet_path="/nonexistent/file.parquet", batch_size=32)


def test_config_file_loading():
    """Test config file loading functions."""
    from represent.config import load_config_from_file

    # Should raise error with missing file
    with pytest.raises(FileNotFoundError):
        load_config_from_file("/nonexistent/config.yaml")


def test_api_components():
    """Test API components that don't require PyTorch."""
    try:
        from represent.api import RepresentAPI

        api = RepresentAPI()

        # Test package info
        info = api.get_package_info()
        assert isinstance(info, dict)
        assert "version" in info

        # Test currency config retrieval
        config = api.get_currency_config("AUDUSD")
        assert config.currency_pair == "AUDUSD"

        # Test invalid currency
        with pytest.raises(ValueError):
            api.get_currency_config("INVALID")

    except ImportError:
        # Skip if API can't import due to PyTorch issues
        pytest.skip("API not importable due to dependencies")


def test_converter_components():
    """Test converter components - DEPRECATED: converter removed in v3.0.0."""
    # Converter has been removed in v3.0.0 - use new 3-stage architecture
    pytest.skip("Converter removed in v3.0.0 - use UnlabeledDBNConverter + ParquetClassifier instead")


def test_feature_types():
    """Test feature type constants."""
    from represent.constants import FEATURE_TYPES, DEFAULT_FEATURES, FEATURE_INDEX_MAP

    assert "volume" in FEATURE_TYPES
    assert "variance" in FEATURE_TYPES
    assert "trade_counts" in FEATURE_TYPES

    assert DEFAULT_FEATURES == ["volume"]

    assert FEATURE_INDEX_MAP["volume"] == 0
    assert FEATURE_INDEX_MAP["variance"] == 1
    assert FEATURE_INDEX_MAP["trade_counts"] == 2


def test_column_definitions():
    """Test column definition constants."""
    from represent.constants import (
        ASK_PRICE_COLUMNS,
        BID_PRICE_COLUMNS,
        ASK_VOL_COLUMNS,
        BID_VOL_COLUMNS,
        ASK_COUNT_COLUMNS,
        BID_COUNT_COLUMNS,
    )

    # Should have 10 levels each
    assert len(ASK_PRICE_COLUMNS) == 10
    assert len(BID_PRICE_COLUMNS) == 10
    assert len(ASK_VOL_COLUMNS) == 10
    assert len(BID_VOL_COLUMNS) == 10
    assert len(ASK_COUNT_COLUMNS) == 10
    assert len(BID_COUNT_COLUMNS) == 10

    # First columns should be correctly formatted
    assert ASK_PRICE_COLUMNS[0] == "ask_px_00"
    assert BID_PRICE_COLUMNS[0] == "bid_px_00"
    assert ASK_VOL_COLUMNS[0] == "ask_sz_00"
    assert BID_VOL_COLUMNS[0] == "bid_sz_00"
