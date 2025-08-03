"""
Represent: High-performance LOB feature extraction package.

This package provides ultra-fast market depth processing capabilities
optimized for real-time trading applications.
"""

__version__ = "1.7.2"

# Import main API functions
from .pipeline import process_market_data, create_processor, MarketDepthProcessor
from .constants import (
    MICRO_PIP_SIZE, TICKS_PER_BIN, SAMPLES, PRICE_LEVELS, TIME_BINS,
    ASK_PRICE_COLUMNS, BID_PRICE_COLUMNS, ASK_VOL_COLUMNS, BID_VOL_COLUMNS,
    FeatureType, FEATURE_TYPES, DEFAULT_FEATURES, FEATURE_INDEX_MAP, MAX_FEATURES, get_output_shape
)
from .dataloader import (
    MarketDepthDataset, 
    HighPerformanceDataLoader, BackgroundBatchProducer,
    create_streaming_dataloader, create_file_dataloader, create_high_performance_dataloader
)
from .config import (
    ClassificationConfig,
    SamplingConfig,
    CurrencyConfig,
    load_currency_config,
    get_default_currency_config,
    save_currency_config,
    list_available_currencies
)

# Public API
__all__ = [
    "process_market_data",
    "create_processor", 
    "MarketDepthProcessor",
    "MarketDepthDataset",
    "HighPerformanceDataLoader",
    "BackgroundBatchProducer",
    "create_streaming_dataloader",
    "create_file_dataloader",
    "create_high_performance_dataloader",
    "ClassificationConfig",
    "SamplingConfig",
    "CurrencyConfig",
    "load_currency_config",
    "get_default_currency_config",
    "save_currency_config",
    "list_available_currencies",
    "MICRO_PIP_SIZE",
    "TICKS_PER_BIN", 
    "SAMPLES",
    "PRICE_LEVELS",
    "TIME_BINS",
    "ASK_PRICE_COLUMNS",
    "BID_PRICE_COLUMNS", 
    "ASK_VOL_COLUMNS",
    "BID_VOL_COLUMNS",
    "FeatureType",
    "FEATURE_TYPES",
    "DEFAULT_FEATURES", 
    "FEATURE_INDEX_MAP",
    "MAX_FEATURES",
    "get_output_shape",
]