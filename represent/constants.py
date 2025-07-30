"""
Performance-critical constants for market depth processing.
These values are tuned for optimal cache performance and vectorization.
"""
import numpy as np
from typing import Final, Union
from enum import Enum

# Core processing constants (from notebook analysis)
MICRO_PIP_SIZE: Final[float] = 0.00001
TICKS_PER_BIN: Final[int] = 100
SAMPLES: Final[int] = 50000  # 500 * TICKS_PER_BIN
PRICE_LEVELS: Final[int] = 402  # 200 bid + 200 ask + 2 mid
TIME_BINS: Final[int] = 500

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
CACHE_LINE_SIZE: Final[int] = 64  # For cache-aligned allocations
MAX_PRICE_DEVIATION: Final[int] = 1000  # Maximum price deviation in micro-pips

# Pre-computed constants for performance
MICRO_PIP_MULTIPLIER: Final[float] = 1.0 / MICRO_PIP_SIZE  # 100000.0
OUTPUT_SHAPE: Final[tuple[int, int]] = (PRICE_LEVELS, TIME_BINS)

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
    def from_string(cls, value: str) -> 'FeatureType':
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
    FeatureType.TRADE_COUNTS.value: 2
}

# Variance column name from DBN files
VARIANCE_COLUMN: Final[str] = 'market_depth_extraction_micro_pips_var'

# Extended output shapes
def get_output_shape(features: Union[list[str], list[FeatureType]]) -> tuple[int, ...]:
    """Get output shape based on feature selection."""
    if len(features) == 1:
        return OUTPUT_SHAPE  # (402, 500)
    else:
        return (len(features), PRICE_LEVELS, TIME_BINS)  # (N, 402, 500)