"""
Minimal data structures for pipeline compatibility.
The new architecture primarily uses lazy loading from parquet,
but these structures are still needed for the core processing pipeline.
"""

import numpy as np
from .constants import PRICE_LEVELS, TIME_BINS, VOLUME_DTYPE, OUTPUT_DTYPE


class PriceLookupTable:
    """Ultra-fast price-to-index lookup using pre-computed arrays."""

    def __init__(self, mid_price: float, price_range: int = 200):
        """Initialize lookup table centered on mid price."""
        self._mid_price_int = int(mid_price)
        self._price_range = price_range

        # Create lookup arrays for bid and ask
        self._bid_indices = np.arange(price_range, dtype=np.int32)
        self._ask_indices = np.arange(price_range + 2, price_range * 2 + 2, dtype=np.int32)

        # Price boundaries for fast lookup
        self._min_bid_price = self._mid_price_int - price_range
        self._max_bid_price = self._mid_price_int - 1
        self._min_ask_price = self._mid_price_int + 1
        self._max_ask_price = self._mid_price_int + price_range

    def get_bid_index(self, price_int: int) -> int:
        """Get bid index for price (higher price = higher index)."""
        if price_int < self._min_bid_price or price_int > self._max_bid_price:
            return -1
        return self._max_bid_price - price_int

    def get_ask_index(self, price_int: int) -> int:
        """Get ask index for price (lower price = lower index)."""
        if price_int < self._min_ask_price or price_int > self._max_ask_price:
            return -1
        return self._price_range + 2 + (price_int - self._min_ask_price)

    def price_to_index(self, price: int) -> int:
        """Get price index for any price (bid or ask)."""
        # First try bid side
        bid_idx = self.get_bid_index(price)
        if bid_idx != -1:
            return bid_idx

        # Then try ask side
        ask_idx = self.get_ask_index(price)
        if ask_idx != -1:
            return ask_idx

        return -1  # Out of bounds

    def vectorized_lookup(self, prices: np.ndarray) -> np.ndarray:
        """Vectorized price lookup for multiple prices."""
        result = np.full(len(prices), -1, dtype=np.int32)

        for i, price in enumerate(prices):
            result[i] = self.price_to_index(int(price))

        return result


class VolumeGrid:
    """Pre-allocated 2D grid for volume mapping."""

    def __init__(self):
        """Initialize grid with pre-allocated memory."""
        self._grid = np.zeros((PRICE_LEVELS, TIME_BINS), dtype=VOLUME_DTYPE)

    def clear(self):
        """Reset grid to zero."""
        self._grid.fill(0)

    def add_volume(self, price_idx: int, time_idx: int, volume: float):
        """Add volume at specific grid position."""
        if 0 <= price_idx < PRICE_LEVELS and 0 <= time_idx < TIME_BINS:
            self._grid[price_idx, time_idx] += volume

    @property
    def grid(self) -> np.ndarray:
        """Get the volume grid array."""
        return self._grid

    @property
    def data(self) -> np.ndarray:
        """Get the volume grid data (alias for grid)."""
        return self._grid

    def set_volumes(self, y_coords: np.ndarray, x_coords: np.ndarray, volumes: np.ndarray):
        """Set volumes at multiple grid positions."""
        # Filter valid coordinates
        valid_mask = (
            (y_coords >= 0) & (y_coords < PRICE_LEVELS) & (x_coords >= 0) & (x_coords < TIME_BINS)
        )

        valid_y = y_coords[valid_mask]
        valid_x = x_coords[valid_mask]
        valid_volumes = volumes[valid_mask]

        # Set values at valid positions
        self._grid[valid_y, valid_x] = valid_volumes

    def get_cumulative_volume(self, reverse: bool = False) -> np.ndarray:
        """Get cumulative volume along price axis."""
        if reverse:
            return np.flip(np.cumsum(np.flip(self._grid, axis=0), axis=0), axis=0)
        else:
            return np.cumsum(self._grid, axis=0)


class OutputBuffer:
    """Pre-allocated buffer for final normalized output."""

    def __init__(self):
        """Initialize output buffer."""
        self._buffer = np.zeros((PRICE_LEVELS, TIME_BINS), dtype=OUTPUT_DTYPE)
        self._temp_combined = np.empty((PRICE_LEVELS, TIME_BINS), dtype=VOLUME_DTYPE)
        self._temp_abs = np.empty((PRICE_LEVELS, TIME_BINS), dtype=VOLUME_DTYPE)

    def prepare_output(self, ask_grid: np.ndarray, bid_grid: np.ndarray) -> np.ndarray:
        """Prepare normalized combined output."""
        # Calculate combined volume (ask - bid)
        np.subtract(ask_grid, bid_grid, out=self._temp_combined)

        # Take absolute value
        np.abs(self._temp_combined, out=self._temp_abs)

        # Normalize to [0, 1] range
        max_val = np.max(self._temp_abs)
        if max_val > 0:
            np.divide(self._temp_abs, max_val, out=self._buffer)
        else:
            self._buffer.fill(0)

        return self._buffer

    @property
    def buffer(self) -> np.ndarray:
        """Get the output buffer."""
        return self._buffer

    @property
    def data(self) -> np.ndarray:
        """Get the output buffer data (alias for buffer)."""
        return self._buffer

    def compute_normalized_difference(
        self, ask_volume: np.ndarray, bid_volume: np.ndarray
    ) -> np.ndarray:
        """Compute normalized difference between ask and bid volumes."""
        return self.prepare_output(ask_volume, bid_volume)

    def get_copy(self) -> np.ndarray:
        """Get a copy of the output buffer."""
        return self._buffer.copy()
