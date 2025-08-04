"""
Represent: High-performance market depth ML pipeline.

This package provides:
1. DBN to labeled parquet conversion with classification
2. Lazy loading parquet dataloader for ML training
3. Currency-specific market configurations
4. High-performance PyTorch integration
"""

__version__ = "1.9.0"

# New primary API - DBN to Parquet conversion and lazy loading
from .converter import DBNToParquetConverter, convert_dbn_file, batch_convert_dbn_files
from .dataloader import (
    LazyParquetDataset,
    LazyParquetDataLoader,
    create_market_depth_dataloader,
    MarketDepthDataLoader,
)

# Core processing and configuration
from .pipeline import process_market_data, create_processor, MarketDepthProcessor
from .constants import (
    MICRO_PIP_SIZE,
    TICKS_PER_BIN,
    SAMPLES,
    PRICE_LEVELS,
    TIME_BINS,
    ASK_PRICE_COLUMNS,
    BID_PRICE_COLUMNS,
    ASK_VOL_COLUMNS,
    BID_VOL_COLUMNS,
    FeatureType,
    FEATURE_TYPES,
    DEFAULT_FEATURES,
    FEATURE_INDEX_MAP,
    MAX_FEATURES,
    get_output_shape,
)
from .config import (
    ClassificationConfig,
    SamplingConfig,
    CurrencyConfig,
    load_currency_config,
    load_config_from_file,
    get_default_currency_config,
    save_currency_config,
    list_available_currencies,
)

# Public API
__all__ = [
    # Primary new API
    "DBNToParquetConverter",
    "convert_dbn_file",
    "batch_convert_dbn_files",
    "LazyParquetDataset",
    "LazyParquetDataLoader",
    "create_market_depth_dataloader",
    "MarketDepthDataLoader",
    # Core processing
    "process_market_data",
    "create_processor",
    "MarketDepthProcessor",
    # Configuration
    "ClassificationConfig",
    "SamplingConfig",
    "CurrencyConfig",
    "load_currency_config",
    "load_config_from_file",
    "get_default_currency_config",
    "save_currency_config",
    "list_available_currencies",
    # Constants
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
    # High-level convenience API
    "RepresentAPI",
    "api",
    "convert_to_training_data",
    "load_training_dataset",
    "create_training_dataloader",
]


# High-level API imports
from .api import (
    RepresentAPI,
    api,
    convert_to_training_data,
    create_training_dataloader,
    load_training_dataset,
)
