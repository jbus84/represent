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

__version__ = "1.15.0"

# V5.0.0 - Symbol-Split-Merge Architecture API

# New focused configuration architecture
from .configs import (
    DatasetBuilderConfig,
    GlobalThresholdConfig,
    MarketDepthProcessorConfig,
    create_compatible_configs,
    create_dataset_builder_config,
    create_processor_config,
    create_represent_config,  # Legacy compatibility function
    create_threshold_config,
    list_available_currencies,  # Legacy compatibility function
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

# Legacy dataset builder removed - use ModularDatasetBuilder instead
# Legacy directional MFE system removed - use modular target generators instead
# Global Threshold Calculation
from .global_threshold_calculator import (
    GlobalThresholdCalculator,
    GlobalThresholds,
    calculate_global_thresholds,
)

# Core processing and configuration
from .market_depth_processor import MarketDepthProcessor, create_processor, process_market_data
from .modular_dataset_builder import ModularDatasetBuilder, create_modular_builder

# Parameter Optimization (optional stubs)
from .parameter_optimization import (
    OPTIMIZATION_AVAILABLE,
    ParameterOptimizer,
    optimize_all_methods,
)

# Target-Only Generation API
from .target_api import (
    batch_generate_targets,
    create_target_config_template,
    generate_targets_from_dataframe,
    generate_targets_from_parquet,
    load_targets_and_join,
)

# Modular Target Generation System
from .target_generators import (
    CumulativeReturnsGenerator,
    DirectionalMFEGenerator,
    GALabelingGenerator,
    GlobalThresholdClassificationGenerator,
    LogReturnHorizonsGenerator,
    PriceMovementGenerator,
    QuantileClassificationGenerator,
    RemainingValueTunerGenerator,
    TargetGenerator,
    TargetGeneratorFactory,
    VolatilityGenerator,
    VolatilityScaledReturnsGenerator,
)

# Public API
__all__ = [
    # Legacy dataset building removed - use ModularDatasetBuilder instead
    # Global Threshold Calculation
    "GlobalThresholds",
    "GlobalThresholdCalculator",
    "calculate_global_thresholds",
    # Legacy directional MFE system removed - use modular target generators instead
    # Core processing
    "process_market_data",
    "create_processor",
    "MarketDepthProcessor",
    # Focused configurations
    "DatasetBuilderConfig",
    "GlobalThresholdConfig",
    "MarketDepthProcessorConfig",
    "create_compatible_configs",
    "create_dataset_builder_config",
    "create_threshold_config",
    "create_processor_config",
    # Legacy compatibility functions (deprecated - use modular system)
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
    # Legacy functionality moved to modular target generation system
    # Modular Target Generation System
    "TargetGenerator",
    "TargetGeneratorFactory",
    "QuantileClassificationGenerator",
    "GlobalThresholdClassificationGenerator",
    "DirectionalMFEGenerator",
    "GALabelingGenerator",
    "LogReturnHorizonsGenerator",
    "PriceMovementGenerator",
    "VolatilityGenerator",
    "CumulativeReturnsGenerator",
    "VolatilityScaledReturnsGenerator",
    "RemainingValueTunerGenerator",
    "ModularDatasetBuilder",
    "create_modular_builder",
    # Target-Only Generation API
    "generate_targets_from_parquet",
    "generate_targets_from_dataframe",
    "batch_generate_targets",
    "create_target_config_template",
    "load_targets_and_join",
    # Parameter Optimization (if available)
    "ParameterOptimizer",
    "optimize_all_methods",
    "OPTIMIZATION_AVAILABLE",
]

# Use modular target generation system instead:
# - ModularDatasetBuilder() for dataset building with pluggable targets
# - TargetGeneratorFactory.create() for creating target generators
# - calculate_global_thresholds() for threshold calculation
