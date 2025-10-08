"""
Minimal data structures for pipeline compatibility.
The new architecture primarily uses lazy loading from parquet,
but these structures are still needed for the core processing pipeline.
"""

from collections.abc import Sequence

import numpy as np

from .constants import OUTPUT_DTYPE, PRICE_LEVELS, VOLUME_DTYPE


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

    def __init__(self, time_bins: int = 500, price_levels: int = PRICE_LEVELS):
        """Initialize grid with pre-allocated memory.

        Args:
            time_bins: Number of time bins (defaults to 500 for backward compatibility)
            price_levels: Number of price levels in the ladder
        """
        self.time_bins = time_bins
        self.price_levels = price_levels
        self._grid = np.zeros((price_levels, time_bins), dtype=VOLUME_DTYPE)

    def clear(self):
        """Reset grid to zero."""
        self._grid.fill(0)

    def add_volume(self, price_idx: int, time_idx: int, volume: float):
        """Add volume at specific grid position."""
        if 0 <= price_idx < self.price_levels and 0 <= time_idx < self.time_bins:
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
            (y_coords >= 0)
            & (y_coords < self.price_levels)
            & (x_coords >= 0)
            & (x_coords < self.time_bins)
        )

        valid_y = y_coords[valid_mask]
        valid_x = x_coords[valid_mask]
        valid_volumes = volumes[valid_mask]

        # Set values at valid positions
        self._grid[valid_y, valid_x] = valid_volumes

    def _collapse_side(
        self, side: np.ndarray, stride: int, groups: Sequence[np.ndarray] | None
    ) -> np.ndarray:
        """Collapse consecutive rows on one side of the book."""
        if groups is not None:
            return np.stack([side[group].sum(axis=0) for group in groups], axis=0)

        if stride == 1:
            return side

        if side.shape[0] % stride != 0:
            raise ValueError("price_range must be divisible by collapse stride")

        collapsed_rows = side.reshape(side.shape[0] // stride, stride, self.time_bins).sum(axis=1)
        return collapsed_rows

    def _collapse_grid(self, stride: int, groups: Sequence[np.ndarray] | None) -> np.ndarray:
        """Collapse bid/ask ladders while keeping the two mid rows intact."""
        if stride == 1 and groups is None:
            return self._grid

        price_levels = self.price_levels
        price_range = (price_levels - 2) // 2

        bid = self._grid[:price_range]
        mid = self._grid[price_range : price_range + 2]
        ask = self._grid[price_range + 2 :]

        bid_collapsed = self._collapse_side(bid, stride, groups)
        ask_collapsed = self._collapse_side(ask, stride, groups)

        return np.concatenate((bid_collapsed, mid, ask_collapsed), axis=0)

    def get_cumulative_volume(
        self, reverse: bool = False, stride: int = 1, groups: Sequence[np.ndarray] | None = None
    ) -> np.ndarray:
        """Get cumulative volume along price axis with optional collapse."""
        grid = self._collapse_grid(stride, groups)
        if reverse:
            return np.flip(np.cumsum(np.flip(grid, axis=0), axis=0), axis=0)
        return np.cumsum(grid, axis=0)


class OutputBuffer:
    """Pre-allocated buffer for final normalized output."""

    def __init__(self, time_bins: int = 500, price_levels: int = PRICE_LEVELS):
        """Initialize output buffer.

        Args:
            time_bins: Number of time bins (defaults to 500 for backward compatibility)
            price_levels: Number of price levels represented in the output tensor
        """
        self.time_bins = time_bins
        self.price_levels = price_levels
        self._buffer = np.zeros((price_levels, time_bins), dtype=OUTPUT_DTYPE)
        self._temp_combined = np.empty((price_levels, time_bins), dtype=VOLUME_DTYPE)
        self._temp_abs = np.empty((price_levels, time_bins), dtype=VOLUME_DTYPE)

    def prepare_output(self, ask_grid: np.ndarray, bid_grid: np.ndarray) -> np.ndarray:
        """Prepare normalized combined output using notebook approach."""
        # Calculate combined volume (ask - bid)
        np.subtract(ask_grid, bid_grid, out=self._temp_combined)

        # Create mask for negative values (bid > ask)
        neg_mask = self._temp_combined < 0

        # Take absolute value
        np.abs(self._temp_combined, out=self._temp_abs)

        # Normalize: (abs_combined - 0) / (abs_combined.max() - 0)
        # min is always 0 volume
        max_val = np.max(self._temp_abs)
        if max_val > 0:
            np.divide(self._temp_abs, max_val, out=self._buffer)
        else:
            self._buffer.fill(0)

        # Restore negative sign for values where bid > ask
        self._buffer[neg_mask] *= -1

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
