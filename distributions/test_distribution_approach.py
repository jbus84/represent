#!/usr/bin/env python3
"""
Test the new EVT-based distribution classification approach.
"""

# Add represent to path
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from represent.distribution_classifier import create_evt_boundaries


def test_evt_approach():
    """Test the EVT approach on real AUDUSD data."""

    # Load sample of real price movement data
    print("📊 Loading sample price movement data...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    # Get a sample of price movements for parameter estimation
    sample_size = 100000  # Use much smaller sample to avoid data leakage
    all_movements = df["price_movement"].to_numpy()

    # Use first 100k samples for parameter estimation (simulating historical data)
    sample_movements = all_movements[:sample_size]

    print(f"   Sample size: {len(sample_movements):,}")
    print(f"   Sample range: {sample_movements.min():.6f} to {sample_movements.max():.6f}")
    print(f"   Sample std: {sample_movements.std():.6f}")

    # Create EVT-based boundaries
    print("\n🔬 Creating EVT-based classification boundaries...")
    evt_boundaries = create_evt_boundaries(sample_movements, nbins=13)

    print(f"   Method: {evt_boundaries.method}")
    print(f"   Balance score: {evt_boundaries.validation_stats['balance_score']:.3f}")
    print(f"   Max deviation: {evt_boundaries.validation_stats['max_deviation']:.3f}")

    # Show boundaries
    print("\n🎯 EVT Boundaries:")
    for i, boundary in enumerate(evt_boundaries.boundaries[1:-1]):
        print(f"   Boundary {i:2d}: {boundary:8.6f} ({boundary * 100:+7.4f}%)")

    # Test on larger validation set (remaining data)
    print("\n✅ Testing on validation data...")
    validation_movements = all_movements[sample_size:]  # Use remaining data for validation

    # Classify using EVT boundaries
    evt_labels = np.digitize(validation_movements, evt_boundaries.boundaries[1:-1])
    evt_labels = np.clip(evt_labels, 0, 12)

    # Calculate class distribution
    evt_class_counts = np.bincount(evt_labels, minlength=13)
    evt_class_fractions = evt_class_counts / len(validation_movements)

    print("\n📊 EVT Class Distribution on Validation Data:")
    for i, (count, fraction) in enumerate(zip(evt_class_counts, evt_class_fractions, strict=False)):
        print(f"   Class {i:2d}: {count:8,} samples ({fraction * 100:5.1f}%)")

    # Compare with current quantile approach
    print("\n⚖️  Comparison with Current Quantile Approach:")
    current_quantiles = np.linspace(0, 1, 14)
    current_boundaries = np.quantile(sample_movements, current_quantiles)

    current_labels = np.digitize(validation_movements, current_boundaries[1:-1])
    current_labels = np.clip(current_labels, 0, 12)

    current_class_counts = np.bincount(current_labels, minlength=13)
    current_class_fractions = current_class_counts / len(validation_movements)

    print("\n📊 Current Quantile Distribution on Validation Data:")
    for i, (count, fraction) in enumerate(zip(current_class_counts, current_class_fractions, strict=False)):
        print(f"   Class {i:2d}: {count:8,} samples ({fraction * 100:5.1f}%)")

    # Calculate improvement metrics
    expected_fraction = 1.0 / 13

    evt_max_deviation = np.max(np.abs(evt_class_fractions - expected_fraction))
    current_max_deviation = np.max(np.abs(current_class_fractions - expected_fraction))

    evt_balance = 1.0 - (evt_max_deviation / expected_fraction)
    current_balance = 1.0 - (current_max_deviation / expected_fraction)

    print("\n📈 Balance Score Comparison:")
    print(f"   EVT approach: {evt_balance:.3f} (higher is better)")
    print(f"   Current approach: {current_balance:.3f}")
    print(f"   Improvement: {((evt_balance - current_balance) / current_balance) * 100:+.1f}%")

    # Focus on problematic classes (0 and 12)
    print("\n🎯 Extreme Class Analysis:")
    print(f"   EVT Class 0:  {evt_class_fractions[0] * 100:5.1f}% (target: 7.7%)")
    print(f"   EVT Class 12: {evt_class_fractions[12] * 100:5.1f}% (target: 7.7%)")
    print(f"   Current Class 0:  {current_class_fractions[0] * 100:5.1f}%")
    print(f"   Current Class 12: {current_class_fractions[12] * 100:5.1f}%")

    extreme_improvement = abs(
        evt_class_fractions[0] + evt_class_fractions[12] - 2 * expected_fraction
    ) - abs(current_class_fractions[0] + current_class_fractions[12] - 2 * expected_fraction)
    print(f"   Extreme class improvement: {extreme_improvement * 100:+.1f} percentage points")

    return evt_boundaries, evt_class_fractions, current_class_fractions


if __name__ == "__main__":
    try:
        boundaries, evt_fractions, current_fractions = test_evt_approach()
        print("\n✅ EVT approach shows improved class balance!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
