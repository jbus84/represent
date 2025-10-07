"""
Modular Target Generation System

This package provides a pluggable architecture for generating both classification
and regression targets from market data. New labeling logic can be added by
implementing the TargetGenerator interface.

See docs/MODULAR_TARGET_ARCHITECTURE.md for complete documentation.
"""

from .base import TargetGenerator
from .classification import (
    GlobalThresholdClassificationGenerator,
    QuantileClassificationGenerator,
)
from .factory import TargetGeneratorFactory
from .regression import (
    CumulativeReturnsGenerator,
    DirectionalMFEGenerator,
    LogReturnHorizonsGenerator,
    PriceMovementGenerator,
    RemainingValueTunerGenerator,
    VolatilityGenerator,
    VolatilityScaledReturnsGenerator,
)

# GA Labeling generator
try:
    from .ga_labeling import GALabelingGenerator  # noqa: F401

    GA_LABELING_AVAILABLE = True
except ImportError:
    GA_LABELING_AVAILABLE = False

__all__ = [
    # Core interface
    "TargetGenerator",
    "TargetGeneratorFactory",
    # Classification generators
    "QuantileClassificationGenerator",
    "GlobalThresholdClassificationGenerator",
    # Regression generators
    "DirectionalMFEGenerator",
    "LogReturnHorizonsGenerator",
    "PriceMovementGenerator",
    "VolatilityGenerator",
    "CumulativeReturnsGenerator",
    "VolatilityScaledReturnsGenerator",
    "RemainingValueTunerGenerator",
]

# Add GA labeling generator to exports if available
if GA_LABELING_AVAILABLE:
    __all__.extend(["GALabelingGenerator"])
