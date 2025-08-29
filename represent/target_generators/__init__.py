"""
Modular Target Generation System

This package provides a pluggable architecture for generating both classification
and regression targets from market data. New labeling logic can be added by
implementing the TargetGenerator interface.

See docs/MODULAR_TARGET_ARCHITECTURE.md for complete documentation.
"""

from .base import TargetGenerator
from .factory import TargetGeneratorFactory
from .classification import (
    QuantileClassificationGenerator,
    GlobalThresholdClassificationGenerator,
)
from .regression import (
    DirectionalMFEGenerator,
    PriceMovementGenerator,
    VolatilityGenerator,
)

# TStrends-based generators (optional - requires tstrends library)
try:
    from .tstrends_labeling import (
        BinaryCTLGenerator,
        TernaryCTLGenerator,
        OracleBinaryTrendGenerator,
        OracleTernaryTrendGenerator,
        TunedTrendGenerator,
    )
    TSTRENDS_GENERATORS_AVAILABLE = True
except ImportError:
    TSTRENDS_GENERATORS_AVAILABLE = False

__all__ = [
    # Core interface
    "TargetGenerator",
    "TargetGeneratorFactory",
    # Classification generators
    "QuantileClassificationGenerator", 
    "GlobalThresholdClassificationGenerator",
    # Regression generators
    "DirectionalMFEGenerator",
    "PriceMovementGenerator", 
    "VolatilityGenerator",
]

# Add TStrends generators to exports if available
if TSTRENDS_GENERATORS_AVAILABLE:
    __all__.extend([
        "BinaryCTLGenerator",
        "TernaryCTLGenerator", 
        "OracleBinaryTrendGenerator",
        "OracleTernaryTrendGenerator",
        "TunedTrendGenerator",
    ])