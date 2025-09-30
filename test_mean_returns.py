#!/usr/bin/env python3
"""Test the new mean log returns feature."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import polars as pl

from represent.target_generators.factory import TargetGeneratorFactory

print("=" * 80)
print("Testing Enhanced log_return_horizons with Mean Returns")
print("=" * 80)

# Create synthetic test data
n_samples = 10000
np.random.seed(42)

# Trending price with noise
trend = np.linspace(1.0, 1.01, n_samples)
noise = np.random.normal(0, 0.0001, n_samples).cumsum()
prices = trend + noise

test_df = pl.DataFrame({
    "row_idx": range(n_samples),
    "timestamp": range(n_samples),
    "symbol": ["TEST"] * n_samples,
    "mid_price": prices,
})

print(f"\nTest data: {len(test_df)} samples")
print(f"Price range: {prices.min():.6f} to {prices.max():.6f}")

# Test 1: Default behavior (with mean returns)
print("\n" + "=" * 80)
print("TEST 1: With Mean Returns (default)")
print("=" * 80)

generator_with_mean = TargetGeneratorFactory.create(
    "log_return_horizons",
    horizons=[50, 100, 250],
    include_mean_returns=True  # Explicit, but this is default
)

result_with_mean = generator_with_mean.generate_targets(test_df)

print("\nTarget columns generated:")
for col in result_with_mean.columns:
    if "log_return" in col:
        values = result_with_mean[col].to_numpy()
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(f"  {col}:")
            print(f"    Mean: {np.mean(valid_values):.4f} bps")
            print(f"    Std:  {np.std(valid_values):.4f} bps")
            print(f"    Valid: {len(valid_values):,} / {len(values):,}")

info = generator_with_mean.get_target_info()
print("\nGenerator info:")
print(f"  Description: {info['description']}")
print(f"  Total targets: {len(info['target_names'])}")
print(f"  Target names: {info['target_names']}")

# Test 2: Without mean returns (legacy behavior)
print("\n" + "=" * 80)
print("TEST 2: Without Mean Returns (legacy)")
print("=" * 80)

generator_without_mean = TargetGeneratorFactory.create(
    "log_return_horizons",
    horizons=[50, 100, 250],
    include_mean_returns=False
)

result_without_mean = generator_without_mean.generate_targets(test_df)

print("\nTarget columns generated:")
for col in result_without_mean.columns:
    if "log_return" in col:
        values = result_without_mean[col].to_numpy()
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(f"  {col}:")
            print(f"    Mean: {np.mean(valid_values):.4f} bps")
            print(f"    Std:  {np.std(valid_values):.4f} bps")

info = generator_without_mean.get_target_info()
print("\nGenerator info:")
print(f"  Description: {info['description']}")
print(f"  Total targets: {len(info['target_names'])}")

# Test 3: Compare endpoint vs mean returns
print("\n" + "=" * 80)
print("TEST 3: Endpoint vs Mean Returns Comparison")
print("=" * 80)

for horizon in [50, 100, 250]:
    endpoint_col = f"log_return_{horizon}t"
    mean_col = f"log_return_mean_{horizon}t"

    endpoint_vals = result_with_mean[endpoint_col].to_numpy()
    mean_vals = result_with_mean[mean_col].to_numpy()

    # Filter to valid samples
    valid_mask = ~(np.isnan(endpoint_vals) | np.isnan(mean_vals))
    endpoint_valid = endpoint_vals[valid_mask]
    mean_valid = mean_vals[valid_mask]

    correlation = np.corrcoef(endpoint_valid, mean_valid)[0, 1]

    print(f"\nHorizon {horizon} ticks:")
    print("  Endpoint return:")
    print(f"    Mean: {np.mean(endpoint_valid):.4f} bps")
    print(f"    Std:  {np.std(endpoint_valid):.4f} bps")
    print("  Mean return (path average):")
    print(f"    Mean: {np.mean(mean_valid):.4f} bps")
    print(f"    Std:  {np.std(mean_valid):.4f} bps")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  → Mean return is {'smoother' if np.std(mean_valid) < np.std(endpoint_valid) else 'similar volatility'}")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("""
1. ENDPOINT RETURNS (log_return_Nt):
   - Measures: Entry price to exit price (tick i to tick i+N)
   - Sensitive to: Exit tick noise, single-tick jumps
   - Use for: Final P&L prediction

2. MEAN RETURNS (log_return_mean_Nt):
   - Measures: Average of all tick-to-tick returns over horizon
   - Captures: Overall drift/trend, path information
   - Smoother: Less sensitive to exit tick noise
   - Use for: Trend prediction, shorter horizons

3. BOTH ARE USEFUL:
   - Endpoint: "Where will price end up?"
   - Mean: "What's the average drift/trend?"
   - Models can learn from both signals
   - Correlation typically 0.7-0.9 (related but different information)

4. RECOMMENDATION:
   - Use BOTH (default: include_mean_returns=True)
   - Total targets: 2x number of horizons
   - Example: horizons=[50,100,250] → 6 targets (3 endpoint + 3 mean)
""")

print("=" * 80)
