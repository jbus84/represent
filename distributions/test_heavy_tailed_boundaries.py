#!/usr/bin/env python3
"""
Test the new heavy-tailed boundary generation in the global threshold calculator.
"""

# Add represent to path
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from represent import GlobalThresholdCalculator, GlobalThresholdConfig


def test_heavy_tailed_boundaries():
    """Test the heavy-tailed boundary approach integrated into GlobalThresholdCalculator."""

    # Load sample data
    print("📊 Loading AUDUSD price movement data for testing...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    # Get a sample to simulate what the calculator would see
    sample_movements = df["price_movement"].to_numpy()[:100000]  # First 100k as "training"
    validation_movements = df["price_movement"].to_numpy()[100000:]  # Rest for validation

    print(f"   Sample size: {len(sample_movements):,}")
    print(f"   Validation size: {len(validation_movements):,}")

    # Create calculator config
    config = GlobalThresholdConfig(
        currency="AUDUSD",
        nbins=13,
        lookback_rows=5000,
        lookforward_input=5000,
        lookforward_offset=500,
    )

    calculator = GlobalThresholdCalculator(config, verbose=True)

    # Test the new heavy-tailed boundary method directly
    print("\n🔬 Testing heavy-tailed boundary generation...")
    heavy_tailed_boundaries = calculator._calculate_heavy_tailed_boundaries(sample_movements)

    # Also generate quantile boundaries for comparison
    quantiles = np.linspace(0, 1, 14)
    quantile_boundaries = np.quantile(sample_movements, quantiles)

    print("\n📏 Boundary Comparison:")
    print(f"{'Boundary':>8} | {'Heavy-Tailed':>12} | {'Quantile':>12} | {'Difference':>10}")
    print("-" * 50)

    for i in range(len(heavy_tailed_boundaries)):
        ht_val = heavy_tailed_boundaries[i]
        q_val = quantile_boundaries[i]
        diff_pct = ((ht_val - q_val) / abs(q_val) * 100) if q_val != 0 else 0

        if i == 0:
            print(f"{'Min':>8} | {ht_val:>12.6f} | {q_val:>12.6f} | {diff_pct:>+8.1f}%")
        elif i == len(heavy_tailed_boundaries) - 1:
            print(f"{'Max':>8} | {ht_val:>12.6f} | {q_val:>12.6f} | {diff_pct:>+8.1f}%")
        else:
            print(f"{'B' + str(i):>8} | {ht_val:>12.6f} | {q_val:>12.6f} | {diff_pct:>+8.1f}%")

    # Test classification performance on validation data
    print("\n✅ Testing classification performance on validation data...")

    # Heavy-tailed classification
    ht_labels = np.digitize(validation_movements, heavy_tailed_boundaries[1:-1])
    ht_labels = np.clip(ht_labels, 0, 12)
    ht_counts = np.bincount(ht_labels, minlength=13)
    ht_fractions = ht_counts / len(validation_movements)

    # Quantile classification
    q_labels = np.digitize(validation_movements, quantile_boundaries[1:-1])
    q_labels = np.clip(q_labels, 0, 12)
    q_counts = np.bincount(q_labels, minlength=13)
    q_fractions = q_counts / len(validation_movements)

    # Analysis
    expected = 1.0 / 13

    print("\n📊 Class Distribution Comparison on Validation Data:")
    print(
        f"{'Class':>5} | {'Heavy-Tailed':>12} | {'Quantile':>12} | {'Target':>8} | {'HT Better?':>10}"
    )
    print("-" * 65)

    ht_improvements = 0
    total_ht_improvement = 0

    for i in range(13):
        ht_pct = ht_fractions[i] * 100
        q_pct = q_fractions[i] * 100
        target_pct = expected * 100

        ht_error = abs(ht_fractions[i] - expected)
        q_error = abs(q_fractions[i] - expected)

        better = "✅" if ht_error < q_error else "❌"
        if ht_error < q_error:
            ht_improvements += 1
            total_ht_improvement += (q_error - ht_error) * 100

        print(f"{i:5d} | {ht_pct:11.1f}% | {q_pct:11.1f}% | {target_pct:7.1f}% | {better:>8}")

    # Summary metrics
    ht_deviations = np.abs(ht_fractions - expected)
    q_deviations = np.abs(q_fractions - expected)

    ht_max_dev = np.max(ht_deviations)
    q_max_dev = np.max(q_deviations)

    ht_balance = 1.0 - (ht_max_dev / expected)
    q_balance = 1.0 - (q_max_dev / expected)

    # Extreme class analysis
    ht_extreme = ht_fractions[0] + ht_fractions[12]
    q_extreme = q_fractions[0] + q_fractions[12]
    expected_extreme = 2 * expected

    ht_extreme_error = abs(ht_extreme - expected_extreme)
    q_extreme_error = abs(q_extreme - expected_extreme)

    print("\n📈 Performance Summary:")
    print(
        f"   Classes improved with heavy-tailed: {ht_improvements}/13 ({ht_improvements / 13 * 100:.0f}%)"
    )
    print(f"   Total deviation improvement: {total_ht_improvement:.1f} percentage points")
    print("   ")
    print("   Balance Scores:")
    print(f"      Heavy-tailed: {ht_balance:.3f}")
    print(f"      Quantile:     {q_balance:.3f}")
    print(f"      Improvement:  {((ht_balance - q_balance) / abs(q_balance)) * 100:+.1f}%")
    print("   ")
    print("   Extreme Classes (0 + 12):")
    print(f"      Heavy-tailed: {ht_extreme * 100:.1f}% (target: {expected_extreme * 100:.1f}%)")
    print(f"      Quantile:     {q_extreme * 100:.1f}%")
    print(f"      HT error: {ht_extreme_error * 100:.1f} pp")
    print(f"      Q error:  {q_extreme_error * 100:.1f} pp")
    print(f"      Improvement: {(q_extreme_error - ht_extreme_error) * 100:+.1f} pp")

    # Success evaluation
    success_criteria = [
        ht_improvements >= 7,  # Majority of classes improved
        ht_extreme_error < q_extreme_error,  # Better extreme class balance
        ht_balance > q_balance,  # Better overall balance
    ]

    success_count = sum(success_criteria)
    print(f"\n🎯 Success Criteria ({success_count}/3 met):")
    print(
        f"   {'✅' if success_criteria[0] else '❌'} Majority classes improved: {ht_improvements >= 7}"
    )
    print(
        f"   {'✅' if success_criteria[1] else '❌'} Better extreme balance: {ht_extreme_error < q_extreme_error}"
    )
    print(
        f"   {'✅' if success_criteria[2] else '❌'} Better overall balance: {ht_balance > q_balance}"
    )

    if success_count >= 2:
        print("\n🎉 Heavy-tailed approach shows significant improvement!")
        print("   Ready to integrate into production pipeline")
        return True, heavy_tailed_boundaries, ht_fractions, q_fractions
    else:
        print("\n⚠️  Heavy-tailed approach needs refinement")
        return False, heavy_tailed_boundaries, ht_fractions, q_fractions


if __name__ == "__main__":
    try:
        success, boundaries, ht_fractions, q_fractions = test_heavy_tailed_boundaries()

        if success:
            print("\n✅ Heavy-tailed boundary generation successful!")
            print("🔧 The updated GlobalThresholdCalculator is ready for use.")
        else:
            print("\n❌ More refinement needed.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
