#!/usr/bin/env python3
"""
Test the targeted balance classification approach.
"""

# Add represent to path
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from represent.targeted_classifier import create_targeted_boundaries


def test_targeted_approach():
    """Test the targeted balance approach."""

    # Load data
    print("📊 Loading AUDUSD price movement data...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    all_movements = df["price_movement"].to_numpy()

    # Use smaller sample for parameter optimization (avoids data leakage)
    sample_size = 50000  # Small sample for boundary optimization
    sample_movements = all_movements[:sample_size]
    validation_movements = all_movements[sample_size:]  # Large validation set

    print(f"   Sample size: {len(sample_movements):,}")
    print(f"   Validation size: {len(validation_movements):,}")
    print(f"   Sample stats: μ={sample_movements.mean():.6f}, σ={sample_movements.std():.6f}")

    # Create targeted boundaries
    print("\n🎯 Creating targeted balanced boundaries...")
    targeted_boundaries = create_targeted_boundaries(
        sample_movements,
        nbins=13,
        target_balance=0.8,  # 80% balance target
        max_iterations=100,  # More iterations for better optimization
    )

    print(f"   Method: {targeted_boundaries.method}")
    print(f"   Sample balance score: {targeted_boundaries.validation_stats['balance_score']:.3f}")
    print(
        f"   Well balanced classes: {targeted_boundaries.validation_stats['well_balanced_classes']}/13"
    )
    print(f"   Imbalanced classes: {targeted_boundaries.validation_stats['imbalanced_classes']}/13")

    # Show boundaries
    print("\n📏 Targeted Boundaries:")
    for i, boundary in enumerate(targeted_boundaries.boundaries):
        if i == 0:
            print(f"   Min: {boundary:8.6f} ({boundary * 100:+7.4f}%)")
        elif i == len(targeted_boundaries.boundaries) - 1:
            print(f"   Max: {boundary:8.6f} ({boundary * 100:+7.4f}%)")
        else:
            print(f"   B{i:2d}: {boundary:8.6f} ({boundary * 100:+7.4f}%)")

    # Test on validation data (the real test)
    print("\n✅ Testing on validation data...")

    # Targeted approach
    targeted_labels = np.digitize(validation_movements, targeted_boundaries.boundaries[1:-1])
    targeted_labels = np.clip(targeted_labels, 0, 12)

    targeted_counts = np.bincount(targeted_labels, minlength=13)
    targeted_fractions = targeted_counts / len(validation_movements)

    # Current quantile approach for comparison
    current_quantiles = np.linspace(0, 1, 14)
    current_boundaries = np.quantile(sample_movements, current_quantiles)

    current_labels = np.digitize(validation_movements, current_boundaries[1:-1])
    current_labels = np.clip(current_labels, 0, 12)

    current_counts = np.bincount(current_labels, minlength=13)
    current_fractions = current_counts / len(validation_movements)

    # Calculate metrics
    expected = 1.0 / 13

    # Targeted metrics
    targeted_deviations = np.abs(targeted_fractions - expected)
    targeted_max_deviation = np.max(targeted_deviations)
    targeted_balance = 1.0 - (targeted_max_deviation / expected)
    targeted_extreme = targeted_fractions[0] + targeted_fractions[12]

    # Current metrics
    current_deviations = np.abs(current_fractions - expected)
    current_max_deviation = np.max(current_deviations)
    current_balance = 1.0 - (current_max_deviation / expected)
    current_extreme = current_fractions[0] + current_fractions[12]

    # Display detailed comparison
    print("\n📊 Detailed Class Distribution Comparison:")
    print("Class | Targeted | Current | Target | Targeted Better? | Deviation Improvement")
    print("------|----------|---------|--------|------------------|----------------------")

    improvements = 0
    total_improvement = 0

    for i in range(13):
        targeted_pct = targeted_fractions[i] * 100
        current_pct = current_fractions[i] * 100
        target_pct = expected * 100

        targeted_dev = targeted_deviations[i] * 100
        current_dev = current_deviations[i] * 100
        dev_improvement = current_dev - targeted_dev

        better = "✅" if targeted_dev < current_dev else "❌"
        if targeted_dev < current_dev:
            improvements += 1
            total_improvement += dev_improvement

        print(
            f"{i:5d} | {targeted_pct:7.1f}% | {current_pct:7.1f}% | {target_pct:5.1f}% | {better:14} | {dev_improvement:+18.1f} pp"
        )

    # Summary statistics
    print("\n📈 Overall Performance Summary:")
    print(f"   Classes improved: {improvements}/13 ({improvements / 13 * 100:.0f}%)")
    print(f"   Total deviation reduction: {total_improvement:.1f} percentage points")
    print("   ")
    print("   Balance Scores:")
    print(f"      Targeted: {targeted_balance:.3f}")
    print(f"      Current:  {current_balance:.3f}")
    print(
        f"      Improvement: {((targeted_balance - current_balance) / abs(current_balance)) * 100:+.1f}%"
    )
    print("   ")
    print("   Extreme Class (0 + 12) Analysis:")
    expected_extreme = 2 * expected
    print(f"      Targeted: {targeted_extreme * 100:.1f}% (target: {expected_extreme * 100:.1f}%)")
    print(f"      Current:  {current_extreme * 100:.1f}%")

    targeted_extreme_error = abs(targeted_extreme - expected_extreme)
    current_extreme_error = abs(current_extreme - expected_extreme)
    extreme_improvement = current_extreme_error - targeted_extreme_error

    print(f"      Targeted error: {targeted_extreme_error * 100:.1f} pp")
    print(f"      Current error:  {current_extreme_error * 100:.1f} pp")
    print(f"      Improvement: {extreme_improvement * 100:+.1f} pp")

    # Success criteria
    success_criteria = [
        improvements >= 8,  # At least 8/13 classes improved
        targeted_balance > current_balance,  # Better overall balance
        extreme_improvement > 0,  # Better extreme class balance
    ]

    success_count = sum(success_criteria)
    print(f"\n🎯 Success Criteria ({success_count}/3 met):")
    print(f"   ✅ Majority classes improved: {improvements >= 8}")
    print(f"   ✅ Better overall balance: {targeted_balance > current_balance}")
    print(f"   ✅ Better extreme balance: {extreme_improvement > 0}")

    if success_count >= 2:
        print("\n🎉 Targeted approach shows significant improvement!")
        return targeted_boundaries, targeted_fractions, current_fractions
    else:
        print("\n⚠️  Targeted approach needs further refinement.")
        return None, targeted_fractions, current_fractions


if __name__ == "__main__":
    try:
        result = test_targeted_approach()
        if result[0] is not None:
            print("\n✅ Targeted optimization successful!")
        else:
            print("\n❌ More work needed on the approach.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
