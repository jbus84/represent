#!/usr/bin/env python3
"""
Test the new balanced classification approach.
"""

# Add represent to path
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from represent.balanced_classifier import create_balanced_boundaries


def test_balanced_approach():
    """Test the balanced financial returns approach."""

    # Load sample data
    print("📊 Loading AUDUSD price movement data...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    all_movements = df["price_movement"].to_numpy()

    # Use small sample for parameter estimation (no data leakage)
    sample_size = 50000  # Even smaller sample
    sample_movements = all_movements[:sample_size]
    validation_movements = all_movements[sample_size:]

    print(f"   Sample size: {len(sample_movements):,}")
    print(f"   Validation size: {len(validation_movements):,}")
    print(f"   Sample stats: μ={sample_movements.mean():.6f}, σ={sample_movements.std():.6f}")

    # Create balanced boundaries
    print("\n🎯 Creating balanced classification boundaries...")
    balanced_boundaries = create_balanced_boundaries(sample_movements, nbins=13)

    print(f"   Method: {balanced_boundaries.method}")
    print(f"   Balance score: {balanced_boundaries.validation_stats['balance_score']:.3f}")
    print(
        f"   Extreme concentration: {balanced_boundaries.validation_stats['extreme_concentration'] * 100:.1f}%"
    )
    print(
        f"   Expected extreme: {2 * balanced_boundaries.validation_stats['expected_fraction'] * 100:.1f}%"
    )

    # Show boundaries
    print("\n📏 Balanced Boundaries:")
    for i, boundary in enumerate(balanced_boundaries.boundaries):
        if i == 0:
            print(f"   Min: {boundary:8.6f} ({boundary * 100:+7.4f}%)")
        elif i == len(balanced_boundaries.boundaries) - 1:
            print(f"   Max: {boundary:8.6f} ({boundary * 100:+7.4f}%)")
        else:
            print(f"   B{i:2d}: {boundary:8.6f} ({boundary * 100:+7.4f}%)")

    # Test on validation data
    print("\n✅ Testing on validation data...")

    # Balanced approach
    balanced_labels = np.digitize(validation_movements, balanced_boundaries.boundaries[1:-1])
    balanced_labels = np.clip(balanced_labels, 0, 12)

    balanced_counts = np.bincount(balanced_labels, minlength=13)
    balanced_fractions = balanced_counts / len(validation_movements)

    # Current quantile approach for comparison
    current_quantiles = np.linspace(0, 1, 14)
    current_boundaries = np.quantile(sample_movements, current_quantiles)

    current_labels = np.digitize(validation_movements, current_boundaries[1:-1])
    current_labels = np.clip(current_labels, 0, 12)

    current_counts = np.bincount(current_labels, minlength=13)
    current_fractions = current_counts / len(validation_movements)

    # Display results
    print("\n📊 Class Distribution Comparison:")
    print("Class | Balanced | Current | Target | Balanced Better?")
    print("------|----------|---------|--------|----------------")

    expected = 1.0 / 13
    for i in range(13):
        balanced_pct = balanced_fractions[i] * 100
        current_pct = current_fractions[i] * 100
        target_pct = expected * 100

        balanced_error = abs(balanced_pct - target_pct)
        current_error = abs(current_pct - target_pct)
        better = "✅" if balanced_error < current_error else "❌"

        print(
            f"{i:5d} | {balanced_pct:7.1f}% | {current_pct:7.1f}% | {target_pct:5.1f}% | {better:14}"
        )

    # Summary statistics
    balanced_max_error = max(abs(f - expected) for f in balanced_fractions)
    current_max_error = max(abs(f - expected) for f in current_fractions)

    balanced_balance = 1.0 - (balanced_max_error / expected)
    current_balance = 1.0 - (current_max_error / expected)

    print("\n📈 Overall Balance Comparison:")
    print(f"   Balanced approach: {balanced_balance:.3f}")
    print(f"   Current approach:  {current_balance:.3f}")
    print(f"   Improvement: {((balanced_balance - current_balance) / current_balance) * 100:+.1f}%")

    # Extreme class analysis
    balanced_extreme = balanced_fractions[0] + balanced_fractions[12]
    current_extreme = current_fractions[0] + current_fractions[12]
    expected_extreme = 2 * expected

    print("\n🎯 Extreme Class (0 + 12) Analysis:")
    print(f"   Balanced: {balanced_extreme * 100:.1f}% (target: {expected_extreme * 100:.1f}%)")
    print(f"   Current:  {current_extreme * 100:.1f}%")
    print(f"   Balanced excess: {(balanced_extreme - expected_extreme) * 100:+.1f} pp")
    print(f"   Current excess:  {(current_extreme - expected_extreme) * 100:+.1f} pp")

    improvement = abs(current_extreme - expected_extreme) - abs(balanced_extreme - expected_extreme)
    print(f"   Improvement: {improvement * 100:+.1f} percentage points")

    return balanced_boundaries, balanced_fractions, current_fractions


if __name__ == "__main__":
    try:
        boundaries, balanced_fractions, current_fractions = test_balanced_approach()
        print("\n✅ Balanced approach tested successfully!")

        # Check if it's actually better
        expected = 1.0 / 13
        balanced_extreme = balanced_fractions[0] + balanced_fractions[12]
        current_extreme = current_fractions[0] + current_fractions[12]

        if abs(balanced_extreme - 2 * expected) < abs(current_extreme - 2 * expected):
            print("🎉 Balanced approach shows improvement in extreme class distribution!")
        else:
            print("⚠️  Balanced approach needs further refinement.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
