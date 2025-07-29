"""
High-performance market depth processing pipeline.
Optimized for <10ms array generation and zero-copy operations.
"""
import numpy as np
import polars as pl
from typing import Optional

from .constants import (
    MICRO_PIP_MULTIPLIER, TICKS_PER_BIN, SAMPLES,
    ASK_PRICE_COLUMNS, BID_PRICE_COLUMNS, ASK_VOL_COLUMNS, BID_VOL_COLUMNS,
    ASK_ANCHOR_COLUMN, BID_ANCHOR_COLUMN, PRICE_LEVELS, TIME_BINS,
    VOLUME_DTYPE
)
from .data_structures import PriceLookupTable, VolumeGrid, OutputBuffer


class MarketDepthProcessor:
    """
    Ultra-high-performance market depth processor.
    Designed to meet <10ms array generation requirements.
    """
    
    def __init__(self):
        """Initialize processor with pre-allocated structures."""
        # Pre-allocate all data structures to avoid runtime allocations
        self._ask_grid = VolumeGrid()
        self._bid_grid = VolumeGrid()
        self._output_buffer = OutputBuffer()
        
        # Pre-allocate temporary arrays for processing
        self._temp_ask_volumes = np.empty((PRICE_LEVELS, TIME_BINS), dtype=VOLUME_DTYPE)
        self._temp_bid_volumes = np.empty((PRICE_LEVELS, TIME_BINS), dtype=VOLUME_DTYPE)
        
        # Cache for price lookup table (will be created when needed)
        self._price_lookup: Optional[PriceLookupTable] = None
        self._cached_mid_price: float = 0.0
        
        # Pre-compile Polars expressions for performance
        self._price_conversion_expressions = None
        self._time_bin_expression = None
        self._compiled_expressions_ready = False
    
    def _prepare_expressions(self) -> None:
        """Pre-compile Polars expressions for optimal performance."""
        if self._compiled_expressions_ready:
            return
        
        # Pre-compile price conversion expressions
        ask_exprs = [
            (pl.col(col) * MICRO_PIP_MULTIPLIER).round().cast(pl.Int64).alias(col)
            for col in ASK_PRICE_COLUMNS
        ]
        bid_exprs = [
            (pl.col(col) * MICRO_PIP_MULTIPLIER).round().cast(pl.Int64).alias(col)
            for col in BID_PRICE_COLUMNS
        ]
        self._price_conversion_expressions = ask_exprs + bid_exprs
        
        # Pre-compile time bin expression
        self._time_bin_expression = (pl.int_range(0, SAMPLES) // TICKS_PER_BIN).alias("tick_bin")
        
        self._compiled_expressions_ready = True
    
    def _convert_prices_to_micro_pips(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert prices to integer micro-pip format with vectorized operations."""
        self._prepare_expressions()
        return df.with_columns(self._price_conversion_expressions)
    
    def _add_time_bins(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add time bin column using pre-compiled expression."""
        return df.with_columns(self._time_bin_expression)
    
    def _calculate_mid_price(self, df: pl.DataFrame) -> float:
        """Calculate mid price from last row with minimal overhead."""
        last_row = df.tail(1)
        ask_price = last_row[ASK_ANCHOR_COLUMN][0]
        bid_price = last_row[BID_ANCHOR_COLUMN][0]
        return float((ask_price + bid_price) / 2)
    
    def _get_or_create_price_lookup(self, mid_price: float) -> PriceLookupTable:
        """Get cached price lookup table or create new one if mid price changed."""
        if self._price_lookup is None or abs(mid_price - self._cached_mid_price) > 0.5:
            self._price_lookup = PriceLookupTable(mid_price)
            self._cached_mid_price = mid_price
        return self._price_lookup
    
    def _process_side_data_vectorized(
        self, 
        df: pl.DataFrame, 
        price_columns: list[str], 
        volume_columns: list[str],
        lookup_table: PriceLookupTable,
        grid: VolumeGrid
    ) -> None:
        """Process ask or bid side data with full vectorization."""
        # Clear grid first
        grid.clear()
        
        # Group by time bins using lazy evaluation
        grouped_prices = (
            df.lazy()
            .select([*price_columns, "tick_bin"])
            .group_by("tick_bin")
            .agg([pl.col(col).mean().floor().cast(pl.Int64) for col in price_columns])
            .sort("tick_bin")
            .collect()
        )
        
        grouped_volumes = (
            df.lazy()
            .select([*volume_columns, "tick_bin"])
            .group_by("tick_bin")
            .agg([pl.col(col).median() for col in volume_columns])
            .sort("tick_bin")
            .collect()
        )
        
        # Convert to numpy arrays for vectorized processing
        prices_array = grouped_prices.select(price_columns).to_numpy()  # Shape: (time_bins, 10)
        volumes_array = grouped_volumes.select(volume_columns).to_numpy()  # Shape: (time_bins, 10)
        
        # Vectorized price-to-index conversion
        # Flatten for vectorized lookup, then reshape
        prices_flat = prices_array.flatten()
        indices_flat = lookup_table.vectorized_lookup(prices_flat)
        indices_array = indices_flat.reshape(prices_array.shape)
        
        # Create coordinate arrays for valid mappings
        valid_mask = indices_array >= 0
        
        if valid_mask.any():
            # Get coordinates of valid entries
            time_indices, level_indices = np.where(valid_mask)
            y_coords = indices_array[valid_mask]
            x_coords = time_indices
            volumes = volumes_array[valid_mask]
            
            # Set volumes in grid
            grid.set_volumes(y_coords, x_coords, volumes)
    
    def process(self, df: pl.DataFrame) -> np.ndarray:
        """
        Main processing pipeline optimized for <10ms execution.
        
        Args:
            df: Input DataFrame with market data
            
        Returns:
            Normalized absolute combined array (402x500)
        """
        # Validate input size
        if len(df) != SAMPLES:
            raise ValueError(f"Input must have exactly {SAMPLES} samples, got {len(df)}")
        
        # Step 1: Convert prices to micro-pips (vectorized)
        df_processed = self._convert_prices_to_micro_pips(df)
        
        # Step 2: Add time bins (pre-compiled expression)
        df_processed = self._add_time_bins(df_processed)
        
        # Step 3: Calculate mid price and get lookup table
        mid_price = self._calculate_mid_price(df_processed)
        lookup_table = self._get_or_create_price_lookup(mid_price)
        
        # Step 4: Process ask side (vectorized)
        self._process_side_data_vectorized(
            df_processed, ASK_PRICE_COLUMNS, ASK_VOL_COLUMNS, lookup_table, self._ask_grid
        )
        
        # Step 5: Process bid side (vectorized)
        self._process_side_data_vectorized(
            df_processed, BID_PRICE_COLUMNS, BID_VOL_COLUMNS, lookup_table, self._bid_grid
        )
        
        # Step 6: Calculate cumulative volumes (cache-friendly)
        ask_cumulative = self._ask_grid.get_cumulative_volume(reverse=False)
        bid_cumulative = self._bid_grid.get_cumulative_volume(reverse=True)
        
        # Step 7: Generate final normalized output (zero-copy where possible)
        result = self._output_buffer.compute_normalized_difference(ask_cumulative, bid_cumulative)
        
        return result.copy()  # Return copy to ensure data integrity


# Factory function for easy instantiation
def create_processor() -> MarketDepthProcessor:
    """Create a new market depth processor instance."""
    return MarketDepthProcessor()


# Main API function that matches the reference implementation interface
def process_market_data(df: pl.DataFrame) -> np.ndarray:
    """
    Process market data and return normalized depth representation.
    
    This function provides the same interface as the reference implementation
    but with significant performance optimizations.
    
    Args:
        df: Polars DataFrame with market data
        
    Returns:
        numpy array of shape (402, 500) with normalized market depth
    """
    processor = create_processor()
    return processor.process(df)