#!/usr/bin/env python3
"""
Example: Using Volatility-Scaled Returns Target Generator

This example demonstrates how to use the VolatilityScaledReturnsGenerator to create
regression targets with adaptive stop-loss and take-profit barriers based on realized volatility.

This approach is commonly used in FX trading for adaptive risk management.
"""

import numpy as np
import polars as pl

from represent import ModularDatasetBuilder, TargetGeneratorFactory


def main():
    """Run volatility-scaled returns generation example."""
    print("🎯 Volatility-Scaled Returns Target Generator Example")
    print("=" * 55)

    # Create realistic FX market data with regime-switching volatility
    np.random.seed(42)
    n_samples = 8000

    print(f"📊 Generating {n_samples} samples with regime-switching volatility")

    # Generate price series with different volatility regimes
    prices = []
    current_price = 1.2500  # EUR/USD-like starting price

    regime_lengths = [2000, 2500, 2000, 1500]  # Different regime durations
    regime_vols = [0.0003, 0.0015, 0.0008, 0.0020]  # Different volatility levels

    sample_count = 0
    for regime_len, regime_vol in zip(regime_lengths, regime_vols, strict=False):
        print(f"   Regime {len(prices) // 1000 + 1}: {regime_len} samples, σ={regime_vol:.4f}")

        for _ in range(min(regime_len, n_samples - sample_count)):
            # Add some autocorrelation for more realistic price dynamics
            base_change = np.random.normal(0, regime_vol)
            if len(prices) > 0:
                # Add slight mean reversion
                prev_change = np.log(
                    prices[-1] / prices[max(-10, -len(prices))]
                    if len(prices) >= 10
                    else current_price
                )
                base_change -= 0.1 * prev_change

            current_price = current_price * np.exp(base_change)
            prices.append(current_price)
            sample_count += 1

            if sample_count >= n_samples:
                break
        if sample_count >= n_samples:
            break

    print(f"   Final price range: {min(prices):.4f} to {max(prices):.4f}")

    # Create DataFrame
    df = pl.DataFrame(
        {
            "mid_price": prices,
            "ts_event": np.arange(n_samples),
            "symbol": ["EURUSD"] * n_samples,
        }
    )

    print("\n🔧 Creating Volatility-Scaled Returns Generators")

    # Create generators with different configurations
    generators = [
        # Conservative: 1.5x volatility, longer horizon
        TargetGeneratorFactory.create(
            "volatility_scaled_returns",
            volatility_window=500,
            vol_multiplier=1.5,
            horizon_ticks=2000,
            target_name="vol_scaled_conservative",
        ),
        # Balanced: 2.0x volatility (default), medium horizon
        TargetGeneratorFactory.create(
            "volatility_scaled_returns",
            volatility_window=500,
            vol_multiplier=2.0,
            horizon_ticks=1500,
            target_name="vol_scaled_balanced",
        ),
        # Aggressive: 3.0x volatility, shorter horizon
        TargetGeneratorFactory.create(
            "volatility_scaled_returns",
            volatility_window=300,
            vol_multiplier=3.0,
            horizon_ticks=1000,
            target_name="vol_scaled_aggressive",
        ),
    ]

    # Create modular dataset builder
    builder = ModularDatasetBuilder(generators, verbose=True)

    print("\n🚀 Building Dataset with Volatility-Scaled Returns")
    dataset = builder.build_dataset(df)

    print("\n📋 Final Dataset Summary:")
    print(f"   Columns: {list(dataset.columns)}")
    print(f"   Shape: {dataset.shape}")

    # Analyze the different volatility-scaled targets
    target_configs = [
        ("vol_scaled_conservative", "1.5x vol, 2000 ticks"),
        ("vol_scaled_balanced", "2.0x vol, 1500 ticks"),
        ("vol_scaled_aggressive", "3.0x vol, 1000 ticks"),
    ]

    for target_col, description in target_configs:
        if target_col in dataset.columns:
            data = dataset[target_col].to_numpy()
            valid_data = data[~np.isnan(data)]

            if len(valid_data) > 0:
                print(f"\n📈 {target_col.upper()} ({description}):")
                print(f"   Valid samples: {len(valid_data):,}")
                print(f"   Range: {valid_data.min():.2f} to {valid_data.max():.2f} bps")
                print(f"   Mean: {valid_data.mean():.2f} bps")
                print(f"   Std Dev: {valid_data.std():.2f} bps")

                # Calculate barrier hit statistics
                barrier_hits_positive = np.sum(valid_data > 100)  # Large positive returns
                barrier_hits_negative = np.sum(valid_data < -100)  # Large negative returns
                print(
                    f"   Barrier hits: {barrier_hits_positive} positive, {barrier_hits_negative} negative"
                )

                # Calculate return distribution
                percentiles = [10, 25, 50, 75, 90]
                pct_values = np.percentile(valid_data, percentiles)
                pct_str = ", ".join([f"P{p}:{v:.1f}" for p, v in zip(percentiles, pct_values, strict=False)])
                print(f"   Percentiles: {pct_str}")

    # Compare different configurations
    print("\n🔗 Cross-Configuration Analysis:")

    # Extract valid data for each configuration
    config_data = {}
    for target_col, _ in target_configs:
        if target_col in dataset.columns:
            data = dataset[target_col].to_numpy()
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 100:  # Need sufficient data for analysis
                config_data[target_col] = valid_data

    if len(config_data) >= 2:
        # Calculate correlations between different configurations
        config_names = list(config_data.keys())
        for i in range(len(config_names)):
            for j in range(i + 1, len(config_names)):
                name1, name2 = config_names[i], config_names[j]
                data1, data2 = config_data[name1], config_data[name2]

                # Find overlapping valid indices
                min_len = min(len(data1), len(data2))
                if min_len > 100:
                    corr = np.corrcoef(data1[:min_len], data2[:min_len])[0, 1]
                    print(f"   {name1} ↔ {name2}: correlation = {corr:.3f}")

        # Volatility comparison
        print("\n📊 Volatility Comparison:")
        for name, data in config_data.items():
            vol = np.std(data)
            sharpe = np.mean(data) / vol if vol > 0 else 0
            print(f"   {name}: volatility = {vol:.2f} bps, Sharpe-like ratio = {sharpe:.3f}")

    print("\n✅ Example completed successfully!")
    print("\n💡 Key Insights:")
    print("   - Volatility-scaled returns adapt to changing market conditions")
    print("   - Higher vol multipliers create wider barriers, allowing larger moves")
    print("   - Shorter horizons capture more immediate price reactions")
    print("   - This approach provides adaptive risk management for trading strategies")
    print("   - Different configurations can be used for different trading styles:")
    print("     • Conservative: Tight barriers, longer evaluation")
    print("     • Balanced: Standard barriers, medium evaluation")
    print("     • Aggressive: Wide barriers, quick evaluation")


if __name__ == "__main__":
    main()
