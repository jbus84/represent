"""
Performance-critical constants for market depth processing.
These values are tuned for optimal cache performance and vectorization.
"""

from enum import Enum
from typing import Final

import numpy as np

# NOTE: Core processing constants have been moved to RepresentConfig:
# - MICRO_PIP_SIZE → config.micro_pip_size
# - TICKS_PER_BIN → config.ticks_per_bin
# - SAMPLES → use config.samples for processing batch size, or calculate expected size dynamically
# - TIME_BINS → config.time_bins (computed from samples and ticks_per_bin)
# - OUTPUT_SHAPE → config.output_shape (computed from PRICE_LEVELS and time_bins)
# Use create_represent_config() to access these values.
PRICE_LEVELS: Final[int] = 402  # 200 bid + 200 ask + 2 mid


# Column definitions for 10-level market data
ASK_PRICE_COLUMNS: Final[list[str]] = [f"ask_px_{str(i).zfill(2)}" for i in range(10)]
ASK_VOL_COLUMNS: Final[list[str]] = [f"ask_sz_{str(i).zfill(2)}" for i in range(10)]
ASK_COUNT_COLUMNS: Final[list[str]] = [f"ask_ct_{str(i).zfill(2)}" for i in range(10)]
ASK_ANCHOR_COLUMN: Final[str] = "ask_px_00"

BID_PRICE_COLUMNS: Final[list[str]] = [f"bid_px_{str(i).zfill(2)}" for i in range(10)]
BID_VOL_COLUMNS: Final[list[str]] = [f"bid_sz_{str(i).zfill(2)}" for i in range(10)]
BID_COUNT_COLUMNS: Final[list[str]] = [f"bid_ct_{str(i).zfill(2)}" for i in range(10)]
BID_ANCHOR_COLUMN: Final[str] = "bid_px_00"

# Performance tuning constants
PRICE_RANGE: Final[int] = 200  # Price levels on each side of mid

# NOTE: MICRO_PIP_MULTIPLIER moved to RepresentConfig - use (1.0 / config.micro_pip_size)
# NOTE: OUTPUT_SHAPE moved to RepresentConfig - use config.output_shape

# NumPy data types for optimal performance
PRICE_DTYPE: Final[np.dtype] = np.dtype(np.int64)  # For price calculations
VOLUME_DTYPE: Final[np.dtype] = np.dtype(np.float64)  # For volume calculations
INDEX_DTYPE: Final[np.dtype] = np.dtype(np.int32)  # For indexing operations
OUTPUT_DTYPE: Final[np.dtype] = np.dtype(np.float32)  # For final output array


# Extended Features Enum
class FeatureType(Enum):
    """Available feature types for market depth processing."""

    VOLUME = "volume"
    VARIANCE = "variance"
    TRADE_COUNTS = "trade_counts"

    @classmethod
    def get_all_values(cls) -> list[str]:
        """Get all enum values as strings."""
        return [item.value for item in cls]

    @classmethod
    def from_string(cls, value: str) -> "FeatureType":
        """Create FeatureType from string value."""
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"Invalid feature type: {value}. Valid options: {cls.get_all_values()}")


# Extended Features Constants
FEATURE_TYPES: Final[list[str]] = FeatureType.get_all_values()
DEFAULT_FEATURES: Final[list[str]] = [FeatureType.VOLUME.value]
MAX_FEATURES: Final[int] = len(FeatureType)

# Feature Index Mapping (consistent ordering in multi-feature tensors)
FEATURE_INDEX_MAP: Final[dict[str, int]] = {
    FeatureType.VOLUME.value: 0,
    FeatureType.VARIANCE.value: 1,
    FeatureType.TRADE_COUNTS.value: 2,
}

# NOTE: Variance feature is calculated dynamically from volume data in market_depth_processor.py
# No separate variance column is needed - variance is computed via .var() on volume columns


# Extended output shapes
def get_output_shape(features: list[str] | list[FeatureType], time_bins: int = 500) -> tuple[int, ...]:
    """Get output shape based on feature selection.

    Args:
        features: List of features
        time_bins: Number of time bins (defaults to 500 for backward compatibility)
    """
    if len(features) == 1:
        return (PRICE_LEVELS, time_bins)  # (402, time_bins)
    else:
        return (len(features), PRICE_LEVELS, time_bins)  # (N, 402, time_bins)
