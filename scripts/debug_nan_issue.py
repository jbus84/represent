#!/usr/bin/env python3
"""
Debug the NaN issue in MFE calculation.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))



def debug_mfe_calculation(dataset_file: Path):
    """Debug MFE calculation for a specific dataset."""
    print(f"🔍 Debugging MFE calculation for {dataset_file.name}")

    # Load data
    df = pl.read_parquet(dataset_file)
    print(f"   Loaded {len(df):,} rows")

    # Check mid_price column
    if "mid_price" not in df.columns:
        print("   ❌ No mid_price column")
        return

    mid_prices = df["mid_price"].to_numpy()
    print(f"   Mid prices shape: {mid_prices.shape}")
    print(f"   Mid prices dtype: {mid_prices.dtype}")
    print(f"   Mid prices range: [{np.nanmin(mid_prices):.6f}, {np.nanmax(mid_prices):.6f}]")

    # Check for NaN/inf in mid prices
    nan_count = np.sum(np.isnan(mid_prices))
    inf_count = np.sum(np.isinf(mid_prices))
    zero_count = np.sum(mid_prices == 0)

    print(f"   NaN count: {nan_count}")
    print(f"   Inf count: {inf_count}")
    print(f"   Zero count: {zero_count}")

    if nan_count > 0 or inf_count > 0:
        print(f"   ⚠️ Found {nan_count} NaN and {inf_count} inf values in mid_price")

    # Test MFE calculation manually on a small sample
    print("   Testing MFE calculation...")

    lookback_window = 200
    lookforward_horizon = 3000
    lookforward_offset = 1
    total_rows = len(mid_prices)

    # Try first valid sample
    stop_row = lookback_window
    if stop_row + lookforward_horizon + lookforward_offset < total_rows:
        # Calculate lookback mean
        lookback_prices = mid_prices[stop_row - lookback_window : stop_row]
        lookback_mean = np.mean(lookback_prices)

        # Get future prices
        future_start = stop_row + lookforward_offset
        future_end = future_start + lookforward_horizon
        future_prices = mid_prices[future_start:future_end]

        # Calculate returns
        returns = (future_prices - lookback_mean) / lookback_mean

        print(f"   Sample calculation at stop_row {stop_row}:")
        print(f"      Lookback mean: {lookback_mean:.6f}")
        print(f"      Future prices shape: {future_prices.shape}")
        print(f"      Returns shape: {returns.shape}")
        print(f"      Returns NaN count: {np.sum(np.isnan(returns))}")
        print(f"      Returns inf count: {np.sum(np.isinf(returns))}")

        if np.sum(np.isnan(returns)) > 0 or np.sum(np.isinf(returns)) > 0:
            print("      ❌ Found NaN/inf in returns calculation")
            print(f"      Lookback prices NaN: {np.sum(np.isnan(lookback_prices))}")
            print(f"      Future prices NaN: {np.sum(np.isnan(future_prices))}")
            print(f"      Lookback mean: {lookback_mean}")
            if lookback_mean == 0:
                print("      ⚠️ Lookback mean is zero - division by zero!")
        else:
            mfe_buy = np.max(returns)
            mfe_sell = -np.min(returns)
            print(f"      MFE buy: {mfe_buy:.6f}")
            print(f"      MFE sell: {mfe_sell:.6f}")
            print("      ✅ Sample calculation successful")
    else:
        print("   ❌ Dataset too small for MFE calculation")


def main():
    """Debug NaN issues in failing datasets."""
    print("🔍 DEBUGGING NAN ISSUES")
    print("=" * 25)

    input_dir = Path("/Users/danielfisher/data/databento/AUDUSD_classified_datasets")

    # Test the datasets that were failing
    failing_datasets = [
        "AUDUSD_M6AH5_dataset.parquet",
        "AUDUSD_M6AM5_dataset.parquet",
        "AUDUSD_M6AU5_dataset.parquet",
        "AUDUSD_M6AZ4_dataset.parquet",
    ]

    for dataset_name in failing_datasets:
        dataset_path = input_dir / dataset_name
        if dataset_path.exists():
            debug_mfe_calculation(dataset_path)
            print()
        else:
            print(f"❌ Dataset not found: {dataset_name}")

    # Also test a working one for comparison
    print("🔍 Testing working dataset for comparison:")
    working_dataset = input_dir / "AUDUSD_M6AM4_dataset.parquet"
    if working_dataset.exists():
        debug_mfe_calculation(working_dataset)


if __name__ == "__main__":
    main()
