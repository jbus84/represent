"""
Represent: High-performance market depth ML pipeline.

This package provides a clean 3-stage architecture (v3.0.0):
1. DBN to unlabeled symbol-grouped parquet conversion
2. Post-processing classification with uniform distribution
3. Lazy loading parquet dataloader for ML training
4. Currency-specific market configurations
5. High-performance PyTorch integration
"""

__version__ = "1.10.0"

# V2.0.0 - 3-Stage Architecture API
from .unlabeled_converter import convert_dbn_to_parquet, batch_convert_dbn_files as batch_convert_unlabeled
from .parquet_classifier import classify_parquet_file, batch_classify_parquet_files
from .lazy_dataloader import (
    LazyParquetDataset,
    LazyParquetDataLoader,
    create_parquet_dataloader,
)

# Dynamic classification configuration
from .classification_config_generator import (
    ClassificationConfigGenerator,
    generate_classification_config_from_parquet,
    classify_with_generated_config,
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
    RepresentConfig,
    create_represent_config,
    list_available_currencies,
)

# Public API
__all__ = [
    # V2.0.0 - 3-Stage Architecture API
    "convert_dbn_to_parquet",  # Stage 1: Unlabeled conversion
    "batch_convert_unlabeled", 
    "classify_parquet_file",   # Stage 2: Post-processing classification
    "batch_classify_parquet_files",
    "create_parquet_dataloader",  # Stage 3: ML training
    "LazyParquetDataset",
    "LazyParquetDataLoader",
    # Core processing
    "process_market_data",
    "create_processor",
    "MarketDepthProcessor",
    # Configuration
    "RepresentConfig",
    "create_represent_config",
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
    # Dynamic classification configuration  
    "ClassificationConfigGenerator",
    "generate_classification_config_from_parquet",
    "classify_with_generated_config",
    # High-level convenience API
    "create_training_dataloader",
    "RepresentAPI",
    "api",
    "load_training_dataset",
    "create_parquet_dataloader",
]


# High-level API imports
from .api import (
    RepresentAPI,
    api,
    create_training_dataloader,
    load_training_dataset,
)
