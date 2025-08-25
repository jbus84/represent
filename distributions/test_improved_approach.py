#!/usr/bin/env python3
"""
Test the improved classification approaches.
"""

# Add represent to path
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from represent.improved_classifier import (
    create_optimized_boundaries,
    create_tail_adjusted_boundaries,
)


def test_improved_approaches():
    """Test both tail-adjusted and optimized approaches."""

    # Load data
    print("📊 Loading AUDUSD price movement data...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    all_movements = df["price_movement"].to_numpy()

    # Use sample for parameter estimation
    sample_size = 100000
    sample_movements = all_movements[:sample_size]
    validation_movements = all_movements[sample_size:]

    print(f"   Sample size: {len(sample_movements):,}")
    print(f"   Validation size: {len(validation_movements):,}")

    # Test different approaches
    approaches = []

    # 1. Current quantile approach (baseline)
    print("\n📊 Testing current quantile approach...")
    quantiles = np.linspace(0, 1, 14)
    current_boundaries = np.quantile(sample_movements, quantiles)
    approaches.append(("Current Quantile", current_boundaries))

    # 2. Tail-adjusted approach
    print("📊 Testing tail-adjusted approach...")
    tail_adjusted = create_tail_adjusted_boundaries(sample_movements, nbins=13, tail_adjustment=0.8)
    approaches.append(("Tail Adjusted", tail_adjusted.boundaries))

    # 3. Optimized quantile approach
    print("📊 Testing optimized quantile approach...")
    optimized = create_optimized_boundaries(sample_movements, nbins=13)
    approaches.append(("Optimized", optimized.boundaries))

    # Evaluate all approaches on validation data
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    results = {}

    for name, boundaries in approaches:
        print(f"\n🔍 {name} Approach:")

        # Classify validation data
        labels = np.digitize(validation_movements, boundaries[1:-1])
        labels = np.clip(labels, 0, 12)

        # Calculate metrics
        class_counts = np.bincount(labels, minlength=13)
        class_fractions = class_counts / len(validation_movements)

        expected_fraction = 1.0 / 13
        deviations = np.abs(class_fractions - expected_fraction)
        max_deviation = np.max(deviations)
        balance_score = 1.0 - (max_deviation / expected_fraction)

        extreme_concentration = class_fractions[0] + class_fractions[12]
        expected_extreme = 2 * expected_fraction
        extreme_excess = extreme_concentration - expected_extreme

        results[name] = {
            "class_fractions": class_fractions,
            "balance_score": balance_score,
            "extreme_concentration": extreme_concentration,
            "extreme_excess": extreme_excess,
            "max_deviation": max_deviation,
        }

        print(f"   Balance Score: {balance_score:.3f}")
        print(f"   Max Deviation: {max_deviation:.3f}")
        print(
            f"   Extreme Concentration: {extreme_concentration * 100:.1f}% (target: {expected_extreme * 100:.1f}%)"
        )
        print(f"   Extreme Excess: {extreme_excess * 100:+.1f} percentage points")

        # Show most problematic classes
        worst_classes = np.argsort(deviations)[-3:]  # 3 worst classes
        print("   Most imbalanced classes:")
        for cls in reversed(worst_classes):
            actual_pct = class_fractions[cls] * 100
            target_pct = expected_fraction * 100
            deviation_pct = deviations[cls] * 100
            print(
                f"      Class {cls}: {actual_pct:.1f}% (target: {target_pct:.1f}%, deviation: +{deviation_pct:.1f} pp)"
            )

    # Comparison table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print("\nClass Distribution Comparison:")
    print("Class | Current | Tail Adj | Optimized | Target")
    print("------|---------|----------|-----------|-------")

    for i in range(13):
        current_pct = results["Current Quantile"]["class_fractions"][i] * 100
        tail_pct = results["Tail Adjusted"]["class_fractions"][i] * 100
        opt_pct = results["Optimized"]["class_fractions"][i] * 100
        target_pct = expected_fraction * 100

        print(
            f"{i:5d} | {current_pct:7.1f}% | {tail_pct:8.1f}% | {opt_pct:9.1f}% | {target_pct:5.1f}%"
        )

    print("\nOverall Metrics:")
    print("Metric                 | Current | Tail Adj | Optimized")
    print("-----------------------|---------|----------|----------")

    for metric in ["balance_score", "extreme_concentration", "extreme_excess"]:
        current_val = results["Current Quantile"][metric]
        tail_val = results["Tail Adjusted"][metric]
        opt_val = results["Optimized"][metric]

        if metric == "extreme_concentration":
            print(
                f"{metric:22} | {current_val * 100:6.1f}% | {tail_val * 100:7.1f}% | {opt_val * 100:8.1f}%"
            )
        elif metric == "extreme_excess":
            print(
                f"{metric:22} | {current_val * 100:+6.1f} pp | {tail_val * 100:+6.1f} pp | {opt_val * 100:+7.1f} pp"
            )
        else:
            print(f"{metric:22} | {current_val:7.3f} | {tail_val:8.3f} | {opt_val:9.3f}")

    # Find best approach
    current_extreme_error = abs(results["Current Quantile"]["extreme_excess"])
    tail_extreme_error = abs(results["Tail Adjusted"]["extreme_excess"])
    opt_extreme_error = abs(results["Optimized"]["extreme_excess"])

    if tail_extreme_error < current_extreme_error or opt_extreme_error < current_extreme_error:
        print("\n✅ Improved approaches show better extreme class balance!")

        if tail_extreme_error < opt_extreme_error:
            print("🏆 Tail Adjusted approach is best:")
            print(
                f"   Extreme class improvement: {(current_extreme_error - tail_extreme_error) * 100:.1f} pp"
            )
            best_approach = "Tail Adjusted"
            best_boundaries = tail_adjusted.boundaries
        else:
            print("🏆 Optimized approach is best:")
            print(
                f"   Extreme class improvement: {(current_extreme_error - opt_extreme_error) * 100:.1f} pp"
            )
            best_approach = "Optimized"
            best_boundaries = optimized.boundaries

        return best_approach, best_boundaries, results
    else:
        print("\n⚠️  Improved approaches need further refinement.")
        return "Current Quantile", current_boundaries, results


if __name__ == "__main__":
    try:
        best_approach, best_boundaries, results = test_improved_approaches()
        print(f"\n🎯 Best approach: {best_approach}")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
