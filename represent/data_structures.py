"""
Performance-optimized data structures for market depth processing.
All structures are designed for cache efficiency and zero-copy operations.
"""
import numpy as np
from .constants import (
    SAMPLES, PRICE_LEVELS, TIME_BINS, CACHE_LINE_SIZE,
    VOLUME_DTYPE, INDEX_DTYPE, OUTPUT_DTYPE
)


class RingBuffer:
    """
    High-performance ring buffer for O(1) insertions/removals.
    Uses pre-allocated memory and avoids dynamic allocations.
    """
    
    def __init__(self, capacity: int = SAMPLES):
        """Initialize ring buffer with pre-allocated memory."""
        # Align memory to cache line boundaries for optimal performance
        alignment = CACHE_LINE_SIZE // 8  # 8 bytes per float64
        padded_capacity = ((capacity + alignment - 1) // alignment) * alignment
        
        # Pre-allocate all arrays
        self._capacity = capacity
        self._data = np.zeros((padded_capacity, 75), dtype=VOLUME_DTYPE)  # 75 columns from market data
        self._head = 0
        self._tail = 0
        self._size = 0
        self._full = False
    
    def push(self, item: np.ndarray) -> None:
        """Add item to buffer with O(1) complexity."""
        self._data[self._head] = item
        self._head = (self._head + 1) % self._capacity
        
        if self._full:
            self._tail = (self._tail + 1) % self._capacity
        elif self._head == self._tail:
            self._full = True
        
        if not self._full:
            self._size += 1
    
    def get_recent_data(self, count: int) -> np.ndarray:
        """Get most recent count items without copying."""
        actual_size = self._capacity if self._full else self._size
        if count > actual_size:
            count = actual_size
        
        if count == 0:
            return np.empty((0, 75), dtype=VOLUME_DTYPE)
        
        if self._head >= count:
            # Contiguous data
            return self._data[self._head - count:self._head]
        else:
            # Wrapped data - need to concatenate
            part1 = self._data[self._capacity - (count - self._head):]
            part2 = self._data[:self._head]
            return np.vstack([part1, part2])
    
    @property
    def size(self) -> int:
        """Current number of items in buffer."""
        return self._capacity if self._full else self._size
    
    @property
    def is_full(self) -> bool:
        """Whether buffer is at capacity."""
        return self._full


class PriceLookupTable:
    """
    Ultra-fast price-to-index lookup using pre-computed arrays.
    Designed for cache efficiency with minimal memory access.
    """
    
    def __init__(self, mid_price: float, price_range: int = 200):
        """Initialize lookup table centered on mid price."""
        self._mid_price_int = int(mid_price)
        self._price_range = price_range
        
        # Create price bins
        self._min_price = self._mid_price_int - price_range
        self._max_price = self._mid_price_int + price_range + 1
        
        # Pre-allocate lookup array for O(1) access
        # Add padding for potential price deviations
        lookup_size = (self._max_price - self._min_price) + 2000  # Extra buffer
        self._lookup_offset = self._min_price - 1000  # Offset for negative indexing
        self._lookup_table = np.full(lookup_size, -1, dtype=INDEX_DTYPE)
        
        # Fill lookup table
        for i, price in enumerate(range(self._min_price, self._max_price + 1)):
            lookup_idx = price - self._lookup_offset
            if 0 <= lookup_idx < lookup_size:
                self._lookup_table[lookup_idx] = i
    
    def price_to_index(self, price: int) -> int:
        """Convert price to grid index with bounds checking."""
        lookup_idx = price - self._lookup_offset
        if 0 <= lookup_idx < len(self._lookup_table):
            result = self._lookup_table[lookup_idx]
            return result if result != -1 else -1
        return -1
    
    def vectorized_lookup(self, prices: np.ndarray) -> np.ndarray:
        """Vectorized price-to-index conversion."""
        lookup_indices = prices - self._lookup_offset
        
        # Clip to valid range
        lookup_indices = np.clip(lookup_indices, 0, len(self._lookup_table) - 1)
        
        # Perform lookup
        result = self._lookup_table[lookup_indices]
        
        # Mark out-of-bounds as -1
        valid_mask = (prices >= self._min_price) & (prices <= self._max_price)
        result = np.where(valid_mask, result, -1)
        
        return result


class VolumeGrid:
    """
    Pre-allocated 2D grid for volume mapping with cache-aligned memory.
    Optimized for vectorized operations and minimal memory allocations.
    """
    
    def __init__(self):
        """Initialize grid with cache-aligned memory."""
        # Allocate cache-aligned memory
        self._grid = np.zeros(PRICE_LEVELS * TIME_BINS, dtype=VOLUME_DTYPE)
        self._grid = self._grid.reshape((PRICE_LEVELS, TIME_BINS))
        
        # Pre-allocate temporary arrays for operations
        self._temp_y_coords = np.empty(10 * TIME_BINS, dtype=INDEX_DTYPE)  # Max 10 levels per time bin
        self._temp_x_coords = np.empty(10 * TIME_BINS, dtype=INDEX_DTYPE)
        self._temp_volumes = np.empty(10 * TIME_BINS, dtype=VOLUME_DTYPE)
    
    def clear(self) -> None:
        """Clear grid efficiently."""
        self._grid.fill(0.0)
    
    def set_volumes(self, y_coords: np.ndarray, x_coords: np.ndarray, volumes: np.ndarray) -> None:
        """Set volumes at specified coordinates."""
        # Filter valid coordinates
        valid_mask = (y_coords >= 0) & (y_coords < PRICE_LEVELS) & (x_coords >= 0) & (x_coords < TIME_BINS)
        
        if valid_mask.any():
            valid_y = y_coords[valid_mask]
            valid_x = x_coords[valid_mask]
            valid_vol = volumes[valid_mask]
            
            self._grid[valid_y, valid_x] = valid_vol
    
    def get_cumulative_volume(self, reverse: bool = False) -> np.ndarray:
        """Calculate cumulative volume along price axis."""
        if reverse:
            return np.cumsum(self._grid[::-1, :], axis=0)
        else:
            result = np.cumsum(self._grid, axis=0)
            return result[::-1, :]  # Reverse for ask side
    
    @property
    def data(self) -> np.ndarray:
        """Direct access to underlying grid data."""
        return self._grid


class OutputBuffer:
    """
    Pre-allocated buffer for final normalized output.
    Designed for zero-copy operations and optimal cache performance.
    """
    
    def __init__(self):
        """Initialize output buffer with optimal alignment."""
        self._buffer = np.zeros((PRICE_LEVELS, TIME_BINS), dtype=OUTPUT_DTYPE)
        self._temp_combined = np.empty((PRICE_LEVELS, TIME_BINS), dtype=VOLUME_DTYPE)
        self._temp_abs = np.empty((PRICE_LEVELS, TIME_BINS), dtype=VOLUME_DTYPE)
    
    def compute_normalized_difference(self, ask_volume: np.ndarray, bid_volume: np.ndarray) -> np.ndarray:
        """Compute normalized absolute combined array with optimal performance."""
        # Use pre-allocated temporary arrays to avoid allocations
        np.subtract(ask_volume, bid_volume, out=self._temp_combined)
        
        # Create negative mask before taking absolute value
        neg_mask = self._temp_combined < 0
        
        # Compute absolute values
        np.abs(self._temp_combined, out=self._temp_abs)
        
        # Find maximum for normalization
        max_val = self._temp_abs.max()
        
        if max_val > 0:
            # Normalize to [0, 1]
            np.divide(self._temp_abs, max_val, out=self._buffer)
            
            # Apply sign based on negative mask
            self._buffer[neg_mask] *= -1
        else:
            # Handle edge case where all values are zero
            self._buffer.fill(0.0)
        
        return self._buffer
    
    def get_copy(self) -> np.ndarray:
        """Get a copy of the current buffer."""
        return self._buffer.copy()
    
    @property
    def data(self) -> np.ndarray:
        """Direct access to buffer data."""
        return self._buffer