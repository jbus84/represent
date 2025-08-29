#!/usr/bin/env python3
"""
Analyze dataset sizes to understand NaN issues in MFE calculation.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_datasets():
    """Analyze all symbol datasets to understand size constraints."""
    print("📊 DATASET SIZE ANALYSIS")
    print("=" * 30)

    input_dir = Path("/Users/danielfisher/data/databento/AUDUSD_classified_datasets")
    datasets = sorted(input_dir.glob("AUDUSD_*_dataset.parquet"))

    # MFE calculation requirements
    lookback_window = 200
    lookforward_horizon = 3000
    lookforward_offset = 1
    jump_size = 200

    print("MFE Requirements:")
    print(f"   Lookback window: {lookback_window}")
    print(f"   Lookforward horizon: {lookforward_horizon}")
    print(
        f"   Minimum rows needed: {lookback_window + lookforward_horizon + lookforward_offset} = {lookback_window + lookforward_horizon + lookforward_offset:,}"
    )
    print()

    for dataset_file in datasets:
        symbol = dataset_file.stem.split("_")[1]
        print(f"📊 {symbol}:")

        try:
            df = pl.read_parquet(dataset_file)
            total_rows = len(df)

            # Calculate if dataset is large enough for MFE
            min_required = lookback_window + lookforward_horizon + lookforward_offset
            valid_range = total_rows - min_required

            # Calculate potential MFE samples
            if valid_range > 0:
                potential_samples = (valid_range) // jump_size
                print(f"   Total rows: {total_rows:,}")
                print(f"   Valid range: {valid_range:,}")
                print(f"   Potential MFE samples: {potential_samples:,}")

                # Check if we have mid_price column
                if "mid_price" in df.columns:
                    mid_price_nulls = df["mid_price"].null_count()
                    print(f"   Mid price nulls: {mid_price_nulls}")

                    # Sample some mid prices to check for NaN
                    sample_prices = df["mid_price"].head(1000).to_list()
                    nan_count = sum(
                        1 for p in sample_prices if p is None or (isinstance(p, float) and p != p)
                    )  # NaN check
                    print(f"   Sample NaN count: {nan_count}/1000")

                    status = "✅ Should work" if potential_samples > 100 else "⚠️ Too few samples"
                else:
                    status = "❌ No mid_price column"
            else:
                print(f"   Total rows: {total_rows:,}")
                print(f"   Required: {min_required:,}")
                status = "❌ Too small for MFE"

            print(f"   Status: {status}")

        except Exception as e:
            print(f"   ❌ Error reading: {e}")

        print()


if __name__ == "__main__":
    analyze_datasets()
