"""
Represent: High-performance market depth ML pipeline.

This package provides flexible multi-stage architectures (v4.0.0):

STREAMLINED 2-STAGE APPROACH (NEW):
1. DBN → Classified Parquet (Direct, Symbol-by-Symbol)
2. ML Training with On-demand Feature Generation

CLASSIC 3-STAGE APPROACH:
1. DBN to unlabeled symbol-grouped parquet conversion
2. Post-processing classification with uniform distribution  
3. Lazy loading parquet dataloader for ML training

Both approaches support:
- Currency-specific market configurations
- High-performance PyTorch integration
- Quantile-based uniform distribution
- Multi-feature extraction (volume, variance, trade_counts)
"""

__version__ = "1.11.0"

# V4.0.0 - Multi-approach Architecture API

# Streamlined DBN-to-Classified-Parquet Approach
from .parquet_classifier import (
    ParquetClassifier,
    classify_parquet_file, 
    batch_classify_parquet_files,
    process_dbn_to_classified_parquets
)

# Global Threshold Calculation
from .global_threshold_calculator import (
    GlobalThresholds,
    GlobalThresholdCalculator,
    calculate_global_thresholds
)

# Alternative: Unlabeled conversion approach
from .unlabeled_converter import convert_dbn_to_parquet, batch_convert_dbn_files as batch_convert_unlabeled
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
    PRICE_LEVELS,
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
    # Streamlined DBN-to-Classified-Parquet Approach
    "ParquetClassifier",
    "process_dbn_to_classified_parquets",
    "classify_parquet_file",
    "batch_classify_parquet_files",
    # Global Threshold Calculation
    "GlobalThresholds",
    "GlobalThresholdCalculator",
    "calculate_global_thresholds",
    # Alternative: Unlabeled conversion approach
    "convert_dbn_to_parquet",
    "batch_convert_unlabeled",
    "create_parquet_dataloader",
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
    # Constants (TIME_BINS moved to RepresentConfig.time_bins)
    "PRICE_LEVELS",
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
