#!/usr/bin/env python3
"""
Create sample data for testing label set builders.
"""

from pathlib import Path

import numpy as np
import polars as pl


def create_sample_data(n_samples: int = 10000, output_path: Path = None):
    """Create realistic sample market data."""
    np.random.seed(42)

    # Generate realistic price series with trends and volatility
    prices = []
    current_price = 1.2345  # EUR/USD-like

    for i in range(n_samples):
        # Regime switching every ~1000 samples
        regime = (i // 1000) % 4

        if regime == 0:  # Uptrend
            trend = 0.00002
            noise_scale = 0.0001
        elif regime == 1:  # Downtrend
            trend = -0.00001
            noise_scale = 0.0001
        elif regime == 2:  # High volatility sideways
            trend = 0.0
            noise_scale = 0.0003
        else:  # Low volatility sideways
            trend = 0.0
            noise_scale = 0.00005

        # Price evolution
        change = trend + np.random.normal(0, noise_scale)
        current_price += change
        current_price = max(current_price, 0.5)  # Floor price
        prices.append(current_price)

    # Create timestamps (microsecond precision)
    timestamps = np.arange(n_samples) * 1000000  # 1 second intervals in microseconds

    # Create DataFrame
    df = pl.DataFrame(
        {"ts_event": timestamps, "mid_price": prices, "symbol": ["EURUSD"] * n_samples}
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
        print(f"✅ Sample data created: {output_path}")
        print(f"📊 {n_samples:,} samples, price range: {min(prices):.4f} - {max(prices):.4f}")

    return df


if __name__ == "__main__":
    create_sample_data(n_samples=15000, output_path=Path("examples/sample_data.parquet"))
