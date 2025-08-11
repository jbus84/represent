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
# Dynamic classification functionality moved to global_threshold_calculator
from .config import (
    RepresentConfig,
    create_represent_config,
    list_available_currencies,
)
from .constants import (
    ASK_PRICE_COLUMNS,
    ASK_VOL_COLUMNS,
    BID_PRICE_COLUMNS,
    BID_VOL_COLUMNS,
    DEFAULT_FEATURES,
    FEATURE_INDEX_MAP,
    FEATURE_TYPES,
    MAX_FEATURES,
    PRICE_LEVELS,
    FeatureType,
    get_output_shape,
)
from .dataset_builder import (
    DatasetBuildConfig,
    DatasetBuilder,
    batch_build_datasets_from_directory,
    build_datasets_from_dbn_files,
)

# Global Threshold Calculation
from .global_threshold_calculator import (
    GlobalThresholdCalculator,
    GlobalThresholds,
    calculate_global_thresholds,
)

# Core processing and configuration
from .market_depth_processor import MarketDepthProcessor, create_processor, process_market_data

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
    # Dynamic classification functionality moved to global_threshold_calculator
    # High-level convenience API removed - use direct function imports instead
]


# Removed redundant high-level API wrapper - use direct imports instead:
# - build_datasets_from_dbn_files() for symbol dataset building
# - DatasetBuilder() for advanced processing
# - calculate_global_thresholds() for threshold calculation
