"""
Represent: High-performance market depth ML pipeline.

This package provides a symbol-split-merge architecture (v5.0.0) for creating
comprehensive symbol datasets from multiple DBN files:

SYMBOL-SPLIT-MERGE APPROACH:
1. Split multiple DBN files by symbol into intermediate files
2. Merge each symbol across all files into comprehensive datasets
3. ML Training with comprehensive symbol-specific datasets

Key Features:
- Currency-specific market configurations
- High-performance PyTorch integration
- Quantile-based uniform distribution
- Multi-feature extraction (volume, variance, trade_counts)
- Comprehensive symbol coverage across multiple files
- Performance-optimized two-phase processing
"""

__version__ = "1.13.0"

# V5.0.0 - Symbol-Split-Merge Architecture API

# Symbol-Split-Merge Dataset Building (Primary Approach v5.0.0+)
from .dataset_builder import (
    DatasetBuilder,
    DatasetBuildConfig, 
    build_datasets_from_dbn_files,
    batch_build_datasets_from_directory
)

# Global Threshold Calculation
from .global_threshold_calculator import (
    GlobalThresholds,
    GlobalThresholdCalculator,
    calculate_global_thresholds
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
    # Symbol-Split-Merge Dataset Building (Primary Approach)
    "DatasetBuilder",
    "DatasetBuildConfig",
    "build_datasets_from_dbn_files", 
    "batch_build_datasets_from_directory",
    # Global Threshold Calculation
    "GlobalThresholds",
    "GlobalThresholdCalculator",
    "calculate_global_thresholds",
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
    "RepresentAPI",
    "api",
    "load_training_dataset",
]


# High-level API imports
from .api import (
    RepresentAPI,
    api,
    load_training_dataset,
)
