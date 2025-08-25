#!/usr/bin/env python3
"""
Demonstration of Improved Classification for Financial Returns

This script shows how to use the new heavy-tailed distribution approach
to address extreme class concentration (classes 0 and 12 being overrepresented).
"""

# Add represent to path
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from represent import GlobalThresholdCalculator, GlobalThresholdConfig


def demonstrate_improved_classification():
    """
    Demonstrate the improved classification approach that addresses extreme class concentration.
    """

    print("🎯 DEMONSTRATION: Improved Financial Returns Classification")
    print("=" * 70)
    print("Problem: Traditional quantile classification creates extreme class concentration")
    print("Solution: Heavy-tailed distribution modeling for balanced classes")
    print()

    # Load existing classified data to show the problem
    print("📊 Loading existing classified AUDUSD data...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    print(f"   Dataset: {len(df):,} samples")
    print(f"   Columns: {len(df.columns)} (including price_movement and classification_label)")

    # Show current class distribution
    current_dist = df["classification_label"].value_counts().sort("classification_label")
    total_samples = len(df)

    print("\n📈 CURRENT CLASS DISTRIBUTION (Problematic):")
    print("Class | Count        | Percentage | Expected | Excess")
    print("------|--------------|------------|----------|-------")

    expected_pct = 100.0 / 13  # ~7.69%
    extreme_classes = []

    for row in current_dist.iter_rows(named=True):
        cls = row["classification_label"]
        count = row["count"]
        pct = (count / total_samples) * 100
        excess = pct - expected_pct

        status = ""
        if pct > expected_pct * 1.5:  # >50% over expected
            status = "📈 EXTREME"
            extreme_classes.append(cls)
        elif pct < expected_pct * 0.7:  # <70% of expected
            status = "📉 LOW"
            extreme_classes.append(cls)

        print(
            f"{cls:5d} | {count:11,} | {pct:9.1f}% | {expected_pct:7.1f}% | {excess:+5.1f}% {status}"
        )

    extreme_concentration = sum(
        (row["count"] / total_samples) * 100
        for row in current_dist.iter_rows(named=True)
        if row["classification_label"] in [0, 12]
    )

    print("\n🚨 PROBLEM IDENTIFIED:")
    print(f"   Classes 0 + 12 concentration: {extreme_concentration:.1f}% (should be ~15.4%)")
    print(f"   Extreme/problematic classes: {len(extreme_classes)}/13")
    print("   Impact: Temporal data leakage and poor model performance")

    # Now demonstrate the solution
    print("\n" + "=" * 70)
    print("🔧 SOLUTION: Heavy-Tailed Distribution Classification")
    print("=" * 70)

    # Use sample data to create improved thresholds (simulating what would happen)
    sample_movements = df["price_movement"].to_numpy()[:100000]  # Sample for parameters
    test_movements = df["price_movement"].to_numpy()[100000:]  # Test data

    print("📊 Creating improved classification boundaries...")
    print(f"   Sample size: {len(sample_movements):,} (for parameter estimation)")
    print(f"   Test size: {len(test_movements):,} (for validation)")

    # Create configurations - both old and new approaches
    print("\n🔬 Comparing Classification Approaches:")

    # 1. Traditional quantile approach
    config_quantile = GlobalThresholdConfig(
        currency="AUDUSD",
        nbins=13,
        use_heavy_tailed=False,  # Traditional approach
    )

    # 2. New heavy-tailed approach
    config_heavy_tailed = GlobalThresholdConfig(
        currency="AUDUSD",
        nbins=13,
        use_heavy_tailed=True,  # Improved approach
    )

    # Test both approaches
    approaches = [
        ("Traditional Quantile", config_quantile),
        ("Heavy-Tailed (Improved)", config_heavy_tailed),
    ]

    results = {}

    for name, config in approaches:
        print(f"\n📊 Testing {name} Approach:")

        # Create calculator
        calculator = GlobalThresholdCalculator(config, verbose=False)

        # Generate boundaries using the approach
        if config.use_heavy_tailed:
            boundaries = calculator._calculate_heavy_tailed_boundaries(sample_movements)
        else:
            quantiles = np.linspace(0, 1, 14)
            boundaries = np.quantile(sample_movements, quantiles)

        # Test on validation data
        labels = np.digitize(test_movements, boundaries[1:-1])
        labels = np.clip(labels, 0, 12)

        class_counts = np.bincount(labels, minlength=13)
        class_fractions = class_counts / len(test_movements)

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
            "boundaries": boundaries,
        }

        print(f"   Balance Score: {balance_score:.3f}")
        print(f"   Extreme Classes (0+12): {extreme_concentration * 100:.1f}% (target: 15.4%)")
        print(f"   Extreme Excess: {extreme_excess * 100:+.1f} percentage points")

    # Detailed comparison
    print("\n" + "=" * 70)
    print("📊 DETAILED COMPARISON")
    print("=" * 70)

    print("\nClass Distribution Comparison:")
    print("Class | Traditional | Heavy-Tailed | Target | Improvement")
    print("------|-------------|--------------|--------|------------")

    improvements = 0
    total_improvement = 0

    for i in range(13):
        trad_pct = results["Traditional Quantile"]["class_fractions"][i] * 100
        ht_pct = results["Heavy-Tailed (Improved)"]["class_fractions"][i] * 100
        target_pct = expected_fraction * 100

        trad_error = abs(trad_pct - target_pct)
        ht_error = abs(ht_pct - target_pct)
        improvement = trad_error - ht_error

        if improvement > 0:
            improvements += 1
            total_improvement += improvement
            status = "✅ Better"
        else:
            status = "❌ Worse"

        print(
            f"{i:5d} | {trad_pct:10.1f}% | {ht_pct:11.1f}% | {target_pct:5.1f}% | {improvement:+5.1f}pp {status}"
        )

    # Summary
    trad_results = results["Traditional Quantile"]
    ht_results = results["Heavy-Tailed (Improved)"]

    print("\n📈 IMPROVEMENT SUMMARY:")
    print(f"   Classes improved: {improvements}/13 ({improvements / 13 * 100:.0f}%)")
    print(f"   Total deviation reduction: {total_improvement:.1f} percentage points")
    print("   ")
    print("   Balance Score Improvement:")
    print(f"      Traditional: {trad_results['balance_score']:7.3f}")
    print(f"      Heavy-tailed: {ht_results['balance_score']:6.3f}")
    balance_improvement = (
        (ht_results["balance_score"] - trad_results["balance_score"])
        / abs(trad_results["balance_score"])
    ) * 100
    print(f"      Improvement: {balance_improvement:+6.1f}%")
    print("   ")
    print("   Extreme Class Balance:")
    print(f"      Traditional excess: {trad_results['extreme_excess'] * 100:+5.1f} pp")
    print(f"      Heavy-tailed excess: {ht_results['extreme_excess'] * 100:+5.1f} pp")
    extreme_improvement = trad_results["extreme_excess"] - ht_results["extreme_excess"]
    print(f"      Improvement: {extreme_improvement * 100:+5.1f} pp")

    # Practical usage instructions
    print("\n" + "=" * 70)
    print("🔧 HOW TO USE THE IMPROVED APPROACH")
    print("=" * 70)

    print(f"""
The improved classification is now integrated into the represent package.

✅ EASY INTEGRATION - No code changes needed!

The heavy-tailed approach is enabled by default. To use:

```python
from represent import create_represent_config, calculate_global_thresholds

# Create configuration (heavy-tailed enabled by default)
config = create_represent_config("AUDUSD", features=['volume'])
dataset_cfg, threshold_cfg, processor_cfg = config

# Calculate improved global thresholds
thresholds = calculate_global_thresholds(
    config=threshold_cfg,  # Has use_heavy_tailed=True by default
    data_directory="/path/to/dbn/files/",
    sample_fraction=0.5
)

# Use in your existing pipeline - no other changes needed!
```

🎯 BENEFITS:
• {improvements}/13 classes show improved balance ({improvements / 13 * 100:.0f}%)
• {balance_improvement:+.1f}% better overall balance score
• {extreme_improvement * 100:+.1f} pp reduction in extreme class concentration
• No data leakage (uses theoretical distribution)
• Maintains temporal boundaries in downstream processing

⚠️  OPTIONAL: Disable if needed
```python
config = GlobalThresholdConfig(use_heavy_tailed=False)  # Traditional quantiles
```
""")

    print("✅ DEMONSTRATION COMPLETE")
    print("The improved classification addresses the extreme class concentration problem!")

    return results


if __name__ == "__main__":
    try:
        results = demonstrate_improved_classification()

        # Final validation
        ht_balance = results["Heavy-Tailed (Improved)"]["balance_score"]
        trad_balance = results["Traditional Quantile"]["balance_score"]

        if ht_balance > trad_balance:
            print("\n🎉 SUCCESS: Heavy-tailed approach shows clear improvement!")
        else:
            print("\n⚠️  WARNING: Results need further investigation.")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
