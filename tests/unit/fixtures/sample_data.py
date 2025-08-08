"""
Sample data generation for testing based on the notebook patterns.
"""

import numpy as np
import polars as pl
from datetime import datetime, timezone
from typing import Optional

from tests.unit.reference_implementation import (
    ASK_PRICE_COLUMNS,
    BID_PRICE_COLUMNS,
    ASK_VOL_COLUMNS,
    BID_VOL_COLUMNS,
    ASK_COUNT_COLUMNS,
    BID_COUNT_COLUMNS,
)


def generate_realistic_market_data(
    n_samples: int = 50000,  # Standard expected dataset size
    base_price: float = 0.6600,
    spread: float = 0.0002,
    seed: Optional[int] = 42,
) -> pl.DataFrame:
    """
    Generate realistic market data similar to the notebook's AUDUSD data.

    Args:
        n_samples: Number of samples to generate
        base_price: Base price around which to generate data
        spread: Typical bid-ask spread
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate timestamps
    start_time = datetime(2024, 4, 5, tzinfo=timezone.utc)
    timestamps = [
        start_time.timestamp() * 1_000_000_000 + i * 1_000_000  # nanoseconds
        for i in range(n_samples)
    ]

    # Generate realistic price walks
    price_changes = np.random.normal(0, 0.00001, n_samples).cumsum()
    mid_prices = base_price + price_changes

    # Generate ask prices (10 levels)
    ask_data = {}
    for i, col in enumerate(ASK_PRICE_COLUMNS):
        level_spread = spread / 2 + i * 0.00001  # Increasing spread by level
        ask_data[col] = mid_prices + level_spread + np.random.normal(0, 0.000005, n_samples)

    # Generate bid prices (10 levels)
    bid_data = {}
    for i, col in enumerate(BID_PRICE_COLUMNS):
        level_spread = spread / 2 + i * 0.00001  # Increasing spread by level
        bid_data[col] = mid_prices - level_spread + np.random.normal(0, 0.000005, n_samples)

    # Generate volumes (decreasing by level typically)
    ask_vol_data = {}
    for i, col in enumerate(ASK_VOL_COLUMNS):
        base_volume = np.random.exponential(1000000, n_samples)  # Exponential distribution
        level_factor = np.exp(-i * 0.2)  # Decreasing volume by level
        ask_vol_data[col] = (base_volume * level_factor).astype(int)

    bid_vol_data = {}
    for i, col in enumerate(BID_VOL_COLUMNS):
        base_volume = np.random.exponential(1000000, n_samples)
        level_factor = np.exp(-i * 0.2)
        bid_vol_data[col] = (base_volume * level_factor).astype(int)

    # Generate counts (smaller numbers)
    ask_count_data = {}
    for i, col in enumerate(ASK_COUNT_COLUMNS):
        base_count = np.random.poisson(10, n_samples)  # Poisson distribution
        level_factor = max(0.3, 1 - i * 0.1)  # Decreasing count by level
        ask_count_data[col] = (base_count * level_factor).astype(int)

    bid_count_data = {}
    for i, col in enumerate(BID_COUNT_COLUMNS):
        base_count = np.random.poisson(10, n_samples)
        level_factor = max(0.3, 1 - i * 0.1)
        bid_count_data[col] = (base_count * level_factor).astype(int)

    # Combine all data
    data = {
        "ts_event": timestamps,
        "ts_recv": timestamps,
        "rtype": [10] * n_samples,  # Market by price data type
        "publisher_id": [1] * n_samples,
        "symbol": ["M6AM4"] * n_samples,
        **ask_data,
        **bid_data,
        **ask_vol_data,
        **bid_vol_data,
        **ask_count_data,
        **bid_count_data,
    }

    return pl.DataFrame(data)


def create_simple_test_data() -> pl.DataFrame:
    """
    Create simple, predictable test data for unit tests.
    """
    n_samples = 1000  # Small dataset for fast testing

    # Simple linear prices
    base_ask = 66000  # In micro-pip format (0.6600 * 100000)
    base_bid = 65998  # 2 pip spread

    data = {
        "ts_event": list(range(n_samples)),
        "ts_recv": list(range(n_samples)),
        "rtype": [10] * n_samples,
        "publisher_id": [1] * n_samples,
        "symbol": ["TEST"] * n_samples,
    }

    # Simple ask prices (constant spread)
    for i, col in enumerate(ASK_PRICE_COLUMNS):
        data[col] = [(base_ask + i) / 100000.0] * n_samples  # Convert back to float

    # Simple bid prices (constant spread)
    for i, col in enumerate(BID_PRICE_COLUMNS):
        data[col] = [(base_bid - i) / 100000.0] * n_samples

    # Simple volumes (constant)
    for col in ASK_VOL_COLUMNS + BID_VOL_COLUMNS:
        data[col] = [1000000] * n_samples

    # Simple counts (constant)
    for col in ASK_COUNT_COLUMNS + BID_COUNT_COLUMNS:
        data[col] = [10] * n_samples

    return pl.DataFrame(data)


def create_edge_case_data() -> pl.DataFrame:
    """
    Create data with edge cases for testing robustness.
    """
    n_samples = 500

    data = {
        "ts_event": list(range(n_samples)),
        "ts_recv": list(range(n_samples)),
        "rtype": [10] * n_samples,
        "publisher_id": [1] * n_samples,
        "symbol": ["EDGE"] * n_samples,
    }

    # Price data with some extreme values
    base_price = 0.6600
    for i, col in enumerate(ASK_PRICE_COLUMNS):
        prices = [base_price + 0.0001 + i * 0.00001] * n_samples
        # Add some extreme outliers
        if i == 0:  # Only for first level
            prices[100] = base_price + 0.01  # Large price jump
            prices[200] = base_price - 0.01  # Large price drop
        data[col] = prices

    for i, col in enumerate(BID_PRICE_COLUMNS):
        prices = [base_price - 0.0001 - i * 0.00001] * n_samples
        if i == 0:
            prices[100] = base_price - 0.01
            prices[200] = base_price + 0.01
        data[col] = prices

    # Volume data with zeros and very large values
    for col in ASK_VOL_COLUMNS + BID_VOL_COLUMNS:
        volumes = [100000] * n_samples
        volumes[50] = 0  # Zero volume
        volumes[150] = 10000000  # Very large volume
        data[col] = volumes

    # Count data with edge cases
    for col in ASK_COUNT_COLUMNS + BID_COUNT_COLUMNS:
        counts = [5] * n_samples
        counts[25] = 0  # Zero count
        counts[75] = 100  # Very large count
        data[col] = counts

    return pl.DataFrame(data)
