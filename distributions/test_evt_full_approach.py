#!/usr/bin/env python3
"""
Test the full EVT (Student's t + GPD) approach.
"""

# Add represent to path
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from represent import GlobalThresholdCalculator, GlobalThresholdConfig


def test_full_evt_approach():
    """Test the full EVT approach with Student's t + GPD."""

    print("🎯 TESTING FULL EVT APPROACH: Student's t + GPD")
    print("=" * 60)

    # Load data
    print("📊 Loading AUDUSD price movement data...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    sample_movements = df["price_movement"].to_numpy()[:100000]  # Sample for parameters
    validation_movements = df["price_movement"].to_numpy()[100000:]  # Validation

    print(f"   Sample size: {len(sample_movements):,}")
    print(f"   Validation size: {len(validation_movements):,}")
    print(f"   Sample stats: μ={sample_movements.mean():.6f}, σ={sample_movements.std():.6f}")

    # Create configurations for comparison
    configs = [
        ("Quantile (Baseline)", GlobalThresholdConfig(use_heavy_tailed=False)),
        ("EVT (Student's t + GPD)", GlobalThresholdConfig(use_heavy_tailed=True)),
    ]

    results = {}

    for name, config in configs:
        print(f"\n🔬 Testing {name}:")

        calculator = GlobalThresholdCalculator(config, verbose=True)

        if config.use_heavy_tailed:
            # Test the new EVT approach
            boundaries = calculator._calculate_heavy_tailed_boundaries(sample_movements)
        else:
            # Traditional quantiles
            quantiles = np.linspace(0, 1, 14)
            boundaries = np.quantile(sample_movements, quantiles)

        # Test on validation data
        labels = np.digitize(validation_movements, boundaries[1:-1])
        labels = np.clip(labels, 0, 12)

        class_counts = np.bincount(labels, minlength=13)
        class_fractions = class_counts / len(validation_movements)

        # Calculate metrics
        expected_fraction = 1.0 / 13
        deviations = np.abs(class_fractions - expected_fraction)
        max_deviation = np.max(deviations)
        balance_score = 1.0 - (max_deviation / expected_fraction)

        extreme_concentration = class_fractions[0] + class_fractions[12]
        extreme_excess = extreme_concentration - (2 * expected_fraction)

        results[name] = {
            "class_fractions": class_fractions,
            "balance_score": balance_score,
            "extreme_concentration": extreme_concentration,
            "extreme_excess": extreme_excess,
            "max_deviation": max_deviation,
        }

        print(f"   Balance Score: {balance_score:.3f}")
        print(f"   Max Deviation: {max_deviation:.3f}")
        print(f"   Extreme Concentration: {extreme_concentration * 100:.1f}%")
        print(f"   Extreme Excess: {extreme_excess * 100:+.1f} pp")

    # Detailed comparison
    print("\n" + "=" * 60)
    print("DETAILED COMPARISON: EVT vs Quantile")
    print("=" * 60)

    quantile_results = results["Quantile (Baseline)"]
    evt_results = results["EVT (Student's t + GPD)"]

    print("\nClass Distribution Comparison:")
    print("Class | Quantile | EVT (t+GPD) | Target | EVT Better?")
    print("------|----------|-------------|--------|------------")

    improvements = 0
    total_improvement = 0

    for i in range(13):
        q_pct = quantile_results["class_fractions"][i] * 100
        evt_pct = evt_results["class_fractions"][i] * 100
        target_pct = (1.0 / 13) * 100

        q_error = abs(q_pct - target_pct)
        evt_error = abs(evt_pct - target_pct)
        improvement = q_error - evt_error

        if improvement > 0:
            improvements += 1
            total_improvement += improvement
            status = "✅"
        else:
            status = "❌"

        print(
            f"{i:5d} | {q_pct:7.1f}% | {evt_pct:10.1f}% | {target_pct:5.1f}% | {improvement:+6.1f}pp {status}"
        )

    # Summary metrics
    print("\n📈 FULL EVT PERFORMANCE SUMMARY:")
    print(f"   Classes improved: {improvements}/13 ({improvements / 13 * 100:.0f}%)")
    print(f"   Total deviation reduction: {total_improvement:.1f} pp")
    print("   ")
    print("   Balance Score:")
    print(f"      Quantile: {quantile_results['balance_score']:7.3f}")
    print(f"      EVT:      {evt_results['balance_score']:7.3f}")
    balance_improvement = (
        (evt_results["balance_score"] - quantile_results["balance_score"])
        / abs(quantile_results["balance_score"])
    ) * 100
    print(f"      Improvement: {balance_improvement:+6.1f}%")
    print("   ")
    print("   Extreme Class Balance (0 + 12):")
    print(f"      Quantile excess: {quantile_results['extreme_excess'] * 100:+5.1f} pp")
    print(f"      EVT excess:      {evt_results['extreme_excess'] * 100:+5.1f} pp")
    extreme_improvement = quantile_results["extreme_excess"] - evt_results["extreme_excess"]
    print(f"      Improvement: {extreme_improvement * 100:+5.1f} pp")

    # Success evaluation
    success_criteria = [
        improvements >= 8,  # Majority improved
        evt_results["balance_score"] > quantile_results["balance_score"],  # Better balance
        abs(evt_results["extreme_excess"])
        < abs(quantile_results["extreme_excess"]),  # Better extremes
    ]

    success_count = sum(success_criteria)
    print(f"\n🎯 Success Criteria ({success_count}/3 met):")
    print(
        f"   {'✅' if success_criteria[0] else '❌'} Majority classes improved: {improvements >= 8}"
    )
    print(
        f"   {'✅' if success_criteria[1] else '❌'} Better balance score: {evt_results['balance_score'] > quantile_results['balance_score']}"
    )
    print(
        f"   {'✅' if success_criteria[2] else '❌'} Better extreme balance: {abs(evt_results['extreme_excess']) < abs(quantile_results['extreme_excess'])}"
    )

    if success_count >= 2:
        print("\n🎉 Full EVT approach shows significant improvement!")
        print("   The combination of Student's t + GPD provides superior modeling")
        print("   of financial returns compared to simple quantiles.")
        return True, evt_results, quantile_results
    else:
        print("\n⚠️  Full EVT approach needs refinement or has implementation issues.")
        return False, evt_results, quantile_results


if __name__ == "__main__":
    try:
        success, evt_results, quantile_results = test_full_evt_approach()

        if success:
            print("\n✅ Full EVT (Student's t + GPD) approach successful!")
            print("🔧 You were right - the combination provides better modeling!")
        else:
            print("\n❌ Implementation needs debugging or refinement.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
