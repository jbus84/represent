"""
Reference implementation extracted from market_depth_extraction_micro_pips.ipynb
Used for testing against optimized implementations.
"""
import numpy as np
import polars as pl
from typing import Dict, Tuple

# Constants from notebook
MICRO_PIP_SIZE = 0.00001
TICKS_PER_BIN = 100
SAMPLES = 50000  # 500 * TICKS_PER_BIN
PRICE_LEVELS = 402  # 200 bid + 200 ask + 2 mid
TIME_BINS = 500

# Column definitions from notebook
ASK_PRICE_COLUMNS = [f"ask_px_{str(i).zfill(2)}" for i in range(10)]
ASK_VOL_COLUMNS = [f"ask_sz_{str(i).zfill(2)}" for i in range(10)]
ASK_COUNT_COLUMNS = [f"ask_ct_{str(i).zfill(2)}" for i in range(10)]
ASK_ANCHOR_COLUMN = ["ask_px_00"]

BID_PRICE_COLUMNS = [f"bid_px_{str(i).zfill(2)}" for i in range(10)]
BID_VOL_COLUMNS = [f"bid_sz_{str(i).zfill(2)}" for i in range(10)]
BID_COUNT_COLUMNS = [f"bid_ct_{str(i).zfill(2)}" for i in range(10)]
BID_ANCHOR_COLUMN = ["bid_px_00"]


def convert_prices_to_micro_pips(df: pl.DataFrame) -> pl.DataFrame:
    """Convert prices to integer micro-pip format."""
    # Create expressions for converting price columns
    ask_expressions = [
        (pl.col(col) / MICRO_PIP_SIZE).round().cast(pl.Int64).alias(col)
        for col in ASK_PRICE_COLUMNS
    ]
    bid_expressions = [
        (pl.col(col) / MICRO_PIP_SIZE).round().cast(pl.Int64).alias(col)
        for col in BID_PRICE_COLUMNS
    ]
    
    return df.with_columns(ask_expressions + bid_expressions)


def create_time_bins(df: pl.DataFrame, samples: int = SAMPLES) -> pl.DataFrame:
    """Add time bin column to dataframe."""
    return df.with_columns(
        (pl.int_range(0, samples) // TICKS_PER_BIN).alias("tick_bin")
    )


def calculate_mid_price(df: pl.DataFrame) -> float:
    """Calculate the most recent mid price."""
    last_row = df.tail(1)
    ask_price = last_row[ASK_ANCHOR_COLUMN[0]][0]
    bid_price = last_row[BID_ANCHOR_COLUMN[0]][0]
    return (ask_price + bid_price) / 2


def create_price_bins(mid_price: float, price_range: int = 200) -> Tuple[np.ndarray, Dict[int, int]]:
    """Create price bins centered on mid price."""
    ask_bin_start = mid_price + 0.5
    bid_bin_start = mid_price - 0.5
    
    ask_price_bins = np.arange(ask_bin_start, ask_bin_start + (price_range + 1), 1)
    bid_price_bins = np.arange(bid_bin_start, bid_bin_start - (price_range + 1), -1)
    
    price_bins = np.array(list(bid_price_bins[::-1]) + list(ask_price_bins))
    price_to_index = {int(price): idx for idx, price in enumerate(price_bins)}
    
    return price_bins, price_to_index


def process_side_data(
    df: pl.DataFrame, 
    price_columns: list, 
    volume_columns: list, 
    price_to_index: Dict[int, int]
) -> np.ndarray:
    """Process ask or bid side data into market volume array."""
    # Group by time bins and calculate means/medians
    grouped_prices = df[price_columns + ["tick_bin"]].group_by(["tick_bin"]).mean().sort(by="tick_bin")
    grouped_prices = grouped_prices.with_columns([pl.col(col) // 1 for col in price_columns])  # Floor division
    
    grouped_volumes = df[volume_columns + ["tick_bin"]].group_by(["tick_bin"]).median().sort(by="tick_bin")[volume_columns]
    
    # Create index columns
    idx_columns = []
    for col in price_columns:
        idx_column = f"{col}_idx"
        grouped_prices = grouped_prices.with_columns(
            pl.col(col).replace_strict(price_to_index, default=None).alias(idx_column)
        )
        idx_columns.append(idx_column)
    
    grouped_prices = grouped_prices[idx_columns]
    
    # Map to 2D grid
    y_coords = grouped_prices.to_numpy().T
    x_coords = np.tile(np.arange(y_coords.shape[1]), (y_coords.shape[0], 1))
    mapped_volumes = np.zeros((len(price_to_index), y_coords.shape[1])) * np.nan
    
    # Fill in the mapped volumes
    null_mask = np.isnan(y_coords)
    y_coords_clean = y_coords[~null_mask].flatten().astype(int)
    x_coords_clean = x_coords[~null_mask].flatten().astype(int)
    volume_clean = grouped_volumes.to_numpy().T[~null_mask].flatten()
    
    mapped_volumes[y_coords_clean, x_coords_clean] = volume_clean
    
    # Replace NaN with 0
    nan_mask = np.isnan(mapped_volumes)
    mapped_volumes[nan_mask] = 0
    
    return mapped_volumes


def calculate_market_volume(mapped_volumes: np.ndarray, is_ask: bool = True) -> np.ndarray:
    """Calculate cumulative market volume."""
    if is_ask:
        # For ask side: cumsum and reverse
        market_volume = np.cumsum(mapped_volumes, axis=0)
        return market_volume[::-1, :]
    else:
        # For bid side: reverse then cumsum
        mapped_volumes_reversed = mapped_volumes[::-1, :]
        return np.cumsum(mapped_volumes_reversed, axis=0)


def create_normed_abs_combined(ask_market_volume: np.ndarray, bid_market_volume: np.ndarray) -> np.ndarray:
    """Create the final normalized absolute combined array."""
    combined = ask_market_volume - bid_market_volume
    neg_mask = combined < 0
    
    abs_combined = np.abs(combined)
    
    # Normalize (min is always 0 volume)
    normed_abs_combined = (abs_combined - 0) / (abs_combined.max() - 0)
    normed_abs_combined[neg_mask] *= -1
    
    return normed_abs_combined


def reference_pipeline(df: pl.DataFrame) -> np.ndarray:
    """
    Complete reference pipeline to generate normed_abs_combined array.
    This matches the exact logic from the notebook.
    """
    # Step 1: Convert prices to micro-pips (already done in notebook)
    df_processed = convert_prices_to_micro_pips(df)
    
    # Step 2: Create time bins
    df_processed = create_time_bins(df_processed)
    
    # Step 3: Calculate mid price and create price bins
    mid_price = calculate_mid_price(df_processed)
    price_bins, price_to_index = create_price_bins(mid_price)
    
    # Step 4: Process ask side
    ask_mapped_volumes = process_side_data(
        df_processed, ASK_PRICE_COLUMNS, ASK_VOL_COLUMNS, price_to_index
    )
    ask_market_volume = calculate_market_volume(ask_mapped_volumes, is_ask=True)
    
    # Step 5: Process bid side
    bid_mapped_volumes = process_side_data(
        df_processed, BID_PRICE_COLUMNS, BID_VOL_COLUMNS, price_to_index
    )
    bid_market_volume = calculate_market_volume(bid_mapped_volumes, is_ask=False)
    
    # Step 6: Create final normalized array
    normed_abs_combined = create_normed_abs_combined(ask_market_volume, bid_market_volume)
    
    return normed_abs_combined


def validate_output_shape(normed_abs_combined: np.ndarray) -> bool:
    """Validate that output has expected shape."""
    return normed_abs_combined.shape == (PRICE_LEVELS, TIME_BINS)


def validate_output_range(normed_abs_combined: np.ndarray) -> bool:
    """Validate that output values are in expected range [-1, 1]."""
    return (normed_abs_combined >= -1.0).all() and (normed_abs_combined <= 1.0).all()