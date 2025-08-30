#!/usr/bin/env python3
"""
Example: Using Cumulative Returns Target Generator

This example demonstrates how to use the CumulativeReturnsGenerator to create
regression targets that accumulate returns over a lookforward window.
"""

import numpy as np
import polars as pl

from represent import ModularDatasetBuilder, TargetGeneratorFactory


def main():
    """Run cumulative returns generation example."""
    print("🎯 Cumulative Returns Target Generator Example")
    print("=" * 50)

    # Create realistic price data (simulate FX market movements)
    np.random.seed(42)
    n_samples = 5000

    # Generate price series with volatility clustering
    returns = np.random.normal(0, 0.0005, n_samples - 1)  # Base volatility
    # Add volatility clustering (higher vol periods)
    vol_regime = np.random.choice([1.0, 2.5], size=n_samples - 1, p=[0.8, 0.2])
    returns *= vol_regime

    # Create price series starting at 1.0
    log_prices = np.concatenate([[0], np.cumsum(returns)])
    prices = np.exp(log_prices)

    print(f"📊 Generated {n_samples} price observations")
    print(f"   Price range: {prices.min():.4f} to {prices.max():.4f}")
    print(f"   Daily volatility: ~{np.std(returns) * 100 * np.sqrt(1440):.2f}% (scaled to daily)")

    # Create DataFrame
    df = pl.DataFrame(
        {
            "mid_price": prices,
            "ts_event": np.arange(n_samples),
            "symbol": ["EURUSD"] * n_samples,
        }
    )

    print("\n🔧 Creating Target Generators")

    # Create different cumulative returns generators
    generators = [
        # Short-term cumulative returns (500 ticks ≈ ~30 minutes)
        TargetGeneratorFactory.create(
            "cumulative_returns", lookforward_samples=500, target_name="cumret_short"
        ),
        # Medium-term cumulative returns (1500 ticks ≈ ~1.5 hours)
        TargetGeneratorFactory.create(
            "cumulative_returns", lookforward_samples=1500, target_name="cumret_medium"
        ),
        # Long-term cumulative returns (3000 ticks ≈ ~3 hours)
        TargetGeneratorFactory.create(
            "cumulative_returns", lookforward_samples=3000, target_name="cumret_long"
        ),
    ]

    # Create modular dataset builder
    builder = ModularDatasetBuilder(generators, verbose=True)

    print("\n🚀 Building Dataset with Multiple Cumulative Returns Targets")
    dataset = builder.build_dataset(df)

    print("\n📋 Final Dataset Summary:")
    print(f"   Columns: {list(dataset.columns)}")
    print(f"   Shape: {dataset.shape}")

    # Analyze the different cumulative returns targets
    for target_col in ["cumret_short", "cumret_medium", "cumret_long"]:
        if target_col in dataset.columns:
            data = dataset[target_col].to_numpy()
            valid_data = data[~np.isnan(data)]

            if len(valid_data) > 0:
                print(f"\n📈 {target_col.upper()} Statistics:")
                print(f"   Valid samples: {len(valid_data):,}")
                print(f"   Range: {valid_data.min():.2f} to {valid_data.max():.2f} bps")
                print(f"   Mean: {valid_data.mean():.2f} bps")
                print(f"   Std Dev: {valid_data.std():.2f} bps")
                skewness = np.mean(((valid_data - valid_data.mean()) / valid_data.std()) ** 3)
                print(f"   Skewness: {skewness:.2f}")

    # Show correlation between different horizons
    short_data = dataset["cumret_short"].to_numpy()
    medium_data = dataset["cumret_medium"].to_numpy()
    long_data = dataset["cumret_long"].to_numpy()

    # Find common valid indices
    valid_mask = (~np.isnan(short_data)) & (~np.isnan(medium_data)) & (~np.isnan(long_data))

    if np.sum(valid_mask) > 100:  # Need sufficient data for correlation
        short_valid = short_data[valid_mask]
        medium_valid = medium_data[valid_mask]
        long_valid = long_data[valid_mask]

        corr_short_medium = np.corrcoef(short_valid, medium_valid)[0, 1]
        corr_short_long = np.corrcoef(short_valid, long_valid)[0, 1]
        corr_medium_long = np.corrcoef(medium_valid, long_valid)[0, 1]

        print("\n🔗 Cross-Horizon Correlations:")
        print(f"   Short ↔ Medium: {corr_short_medium:.3f}")
        print(f"   Short ↔ Long:   {corr_short_long:.3f}")
        print(f"   Medium ↔ Long:  {corr_medium_long:.3f}")

    print("\n✅ Example completed successfully!")
    print("\n💡 Usage Notes:")
    print("   - Cumulative returns measure total price movement over N future samples")
    print("   - Values are in basis points (bps), where 100 bps = 1%")
    print("   - Longer horizons generally have higher volatility")
    print("   - This can be used for regression models predicting future returns")
    print("   - Combine with classification targets for multi-task learning")


if __name__ == "__main__":
    main()
