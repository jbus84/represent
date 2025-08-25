#!/usr/bin/env python3
"""
Test α-stable distribution approach for financial returns classification.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add represent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from represent.distributions.stable_classifier import create_stable_boundaries


def test_stable_distribution_approach():
    """Test α-stable distribution classification approach."""

    print("🎯 TESTING α-STABLE DISTRIBUTION APPROACH")
    print("=" * 55)
    print("Theory: Lévy-stable distributions for financial returns")
    print("Benefits: Heavy tails + asymmetry + scale invariance")
    print("Parameters: α (stability), β (skewness), γ (scale), δ (location)")
    print()

    # Load data
    print("📊 Loading AUDUSD price movement data...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    sample_movements = df["price_movement"].to_numpy()[:100000]
    validation_movements = df["price_movement"].to_numpy()[100000:]

    print(f"   Sample size: {len(sample_movements):,}")
    print(f"   Validation size: {len(validation_movements):,}")
    print(f"   Sample stats: μ={sample_movements.mean():.6f}, σ={sample_movements.std():.6f}")
    print(
        f"   Skewness: {((sample_movements - sample_movements.mean()) ** 3).mean() / sample_movements.std() ** 3:.3f}"
    )
    print(
        f"   Kurtosis: {((sample_movements - sample_movements.mean()) ** 4).mean() / sample_movements.std() ** 4:.3f}"
    )

    # Test different approaches
    approaches = []

    # 1. Traditional quantile approach
    print("\n🔬 Testing Traditional Quantile Approach...")
    quantiles = np.linspace(0, 1, 14)
    quantile_boundaries = np.quantile(sample_movements, quantiles)
    approaches.append(("Quantile", quantile_boundaries, None))

    # 2. α-stable distribution approach
    print("\n🔬 Testing α-Stable Distribution Approach...")
    stable_boundaries_obj = create_stable_boundaries(sample_movements, nbins=13)
    approaches.append(("α-Stable", stable_boundaries_obj.boundaries, stable_boundaries_obj))

    # Test all approaches on validation data
    results = {}

    for name, boundaries, extra_info in approaches:
        print(f"\n📊 Testing {name} on validation data...")

        # Classify validation data
        labels = np.digitize(validation_movements, boundaries[1:-1])
        labels = np.clip(labels, 0, 12)

        class_counts = np.bincount(labels, minlength=13)
        class_fractions = class_counts / len(validation_movements)

        # Calculate metrics
        expected = 1.0 / 13
        deviations = np.abs(class_fractions - expected)
        max_deviation = np.max(deviations)
        balance_score = 1.0 - (max_deviation / expected)

        extreme_concentration = class_fractions[0] + class_fractions[12]
        extreme_excess = extreme_concentration - (2 * expected)

        results[name] = {
            "class_fractions": class_fractions,
            "balance_score": balance_score,
            "extreme_concentration": extreme_concentration,
            "extreme_excess": extreme_excess,
            "boundaries": boundaries,
            "extra_info": extra_info,
        }

        print(f"   Balance Score: {balance_score:.3f}")
        print(f"   Extreme Classes (0+12): {extreme_concentration * 100:.1f}%")
        print(f"   Extreme Excess: {extreme_excess * 100:+.1f} pp")

        if extra_info and hasattr(extra_info, "alpha"):
            print(f"   α-stable params: α={extra_info.alpha:.3f}, β={extra_info.beta:.3f}")

    # Detailed comparison
    print("\n" + "=" * 55)
    print("DETAILED COMPARISON")
    print("=" * 55)

    quantile_results = results["Quantile"]
    stable_results = results["α-Stable"]

    print("\nClass Distribution Comparison:")
    print("Class | Quantile | α-Stable | Target | Stable Better?")
    print("------|----------|----------|--------|---------------")

    improvements = 0
    total_improvement = 0

    for i in range(13):
        q_pct = quantile_results["class_fractions"][i] * 100
        stable_pct = stable_results["class_fractions"][i] * 100
        target_pct = 100.0 / 13

        q_error = abs(q_pct - target_pct)
        stable_error = abs(stable_pct - target_pct)
        improvement = q_error - stable_error

        if improvement > 0:
            improvements += 1
            total_improvement += improvement
            status = "✅"
        else:
            status = "❌"

        print(
            f"{i:5d} | {q_pct:7.1f}% | {stable_pct:7.1f}% | {target_pct:5.1f}% | {improvement:+6.1f}pp {status}"
        )

    # Summary metrics
    balance_improvement = (
        (stable_results["balance_score"] - quantile_results["balance_score"])
        / abs(quantile_results["balance_score"])
    ) * 100
    extreme_improvement = quantile_results["extreme_excess"] - stable_results["extreme_excess"]

    print("\n📈 α-STABLE PERFORMANCE SUMMARY:")
    print(f"   Classes improved: {improvements}/13 ({improvements / 13 * 100:.0f}%)")
    print(f"   Total deviation reduction: {total_improvement:.1f} pp")
    print(f"   Balance score improvement: {balance_improvement:+.1f}%")
    print(f"   Extreme class improvement: {extreme_improvement * 100:+.1f} pp")

    # Success criteria
    success_criteria = [
        improvements >= 8,  # Majority improved
        stable_results["balance_score"] > quantile_results["balance_score"],  # Better balance
        abs(stable_results["extreme_excess"])
        < abs(quantile_results["extreme_excess"]),  # Better extremes
    ]

    success_count = sum(success_criteria)
    print(f"\n🎯 Success Criteria ({success_count}/3 met):")
    print(
        f"   {'✅' if success_criteria[0] else '❌'} Majority classes improved: {improvements >= 8}"
    )
    print(
        f"   {'✅' if success_criteria[1] else '❌'} Better balance score: {stable_results['balance_score'] > quantile_results['balance_score']}"
    )
    print(
        f"   {'✅' if success_criteria[2] else '❌'} Better extreme balance: {abs(stable_results['extreme_excess']) < abs(quantile_results['extreme_excess'])}"
    )

    if success_count >= 2:
        print("\n🎉 α-Stable distribution approach shows significant improvement!")
        print("   The heavy tails and asymmetry modeling provides superior classification.")
        return True, stable_results, quantile_results, stable_boundaries_obj
    else:
        print("\n⚠️  α-Stable approach needs refinement or has implementation issues.")
        return False, stable_results, quantile_results, stable_boundaries_obj


def analyze_stable_fit_quality(boundaries_obj, sample_data):
    """Analyze the quality of the α-stable distribution fit."""

    print("\n🔬 α-STABLE DISTRIBUTION FIT ANALYSIS:")
    print(f"   α (stability): {boundaries_obj.alpha:.3f} - ", end="")
    if boundaries_obj.alpha < 1.5:
        print("Very heavy tails")
    elif boundaries_obj.alpha < 1.8:
        print("Heavy tails (typical financial)")
    else:
        print("Moderate tails")

    print(f"   β (skewness): {boundaries_obj.beta:.3f} - ", end="")
    if abs(boundaries_obj.beta) < 0.1:
        print("Nearly symmetric")
    elif boundaries_obj.beta > 0:
        print("Positive skew (right tail heavier)")
    else:
        print("Negative skew (left tail heavier)")

    print(f"   γ (scale): {boundaries_obj.gamma:.6f}")
    print(f"   δ (location): {boundaries_obj.delta:.6f}")

    # Theoretical insights
    if boundaries_obj.alpha < 2.0:
        print("   📊 Infinite variance (α < 2.0) - typical of financial returns")
    if boundaries_obj.alpha <= 1.0:
        print("   📊 Infinite mean (α ≤ 1.0) - extreme heavy tails")

    return boundaries_obj


if __name__ == "__main__":
    try:
        success, stable_results, quantile_results, stable_obj = test_stable_distribution_approach()

        if success:
            analyze_stable_fit_quality(stable_obj, None)
            print("\n✅ α-Stable distribution approach successful!")
            print(
                "🔬 This is theoretically the most appropriate distribution for financial returns."
            )
        else:
            print("\n❌ Implementation may need debugging or parameter tuning.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
