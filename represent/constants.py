"""
Performance-critical constants for market depth processing.
These values are tuned for optimal cache performance and vectorization.
"""
import numpy as np
from typing import Final

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
PRICE_DTYPE: Final[np.dtype] = np.int64  # For price calculations
VOLUME_DTYPE: Final[np.dtype] = np.float64  # For volume calculations
INDEX_DTYPE: Final[np.dtype] = np.int32  # For indexing operations
OUTPUT_DTYPE: Final[np.dtype] = np.float32  # For final output array