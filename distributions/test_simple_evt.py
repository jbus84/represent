#!/usr/bin/env python3
"""
Test the simplified EVT-inspired approach.
"""

# Add represent to path
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from represent.simple_evt_classifier import test_evt_inspired_approach


def main():
    """Test the simplified EVT approach."""

    print("🎯 TESTING SIMPLIFIED EVT-INSPIRED APPROACH")
    print("=" * 55)
    print("Theory: Student's t + Power-law tail compression")
    print("Goal: Reduce classes 0/12 concentration while maintaining balance")
    print()

    # Load data
    print("📊 Loading AUDUSD data...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    sample_movements = df["price_movement"].to_numpy()[:100000]
    validation_movements = df["price_movement"].to_numpy()[100000:]

    print(f"   Sample: {len(sample_movements):,} for parameter fitting")
    print(f"   Validation: {len(validation_movements):,} for testing")

    # Test the approach
    results = test_evt_inspired_approach(sample_movements, validation_movements)

    # Compare results
    quantile_results = results["Quantile"]
    evt_results = results["EVT-Inspired"]

    print("\n" + "=" * 55)
    print("DETAILED COMPARISON")
    print("=" * 55)

    print("\nClass Distribution:")
    print("Class | Quantile | EVT-Insp | Target | Better?")
    print("------|----------|----------|--------|--------")

    improvements = 0

    for i in range(13):
        q_pct = quantile_results["class_fractions"][i] * 100
        evt_pct = evt_results["class_fractions"][i] * 100
        target_pct = 100.0 / 13

        q_error = abs(q_pct - target_pct)
        evt_error = abs(evt_pct - target_pct)

        if evt_error < q_error:
            improvements += 1
            status = "✅"
        else:
            status = "❌"

        print(f"{i:5d} | {q_pct:7.1f}% | {evt_pct:7.1f}% | {target_pct:5.1f}% | {status:6}")

    # Summary
    balance_improvement = (
        (evt_results["balance_score"] - quantile_results["balance_score"])
        / abs(quantile_results["balance_score"])
    ) * 100
    extreme_improvement = quantile_results["extreme_excess"] - evt_results["extreme_excess"]

    print("\n📈 SUMMARY:")
    print(f"   Classes improved: {improvements}/13 ({improvements / 13 * 100:.0f}%)")
    print(f"   Balance score improvement: {balance_improvement:+.1f}%")
    print(f"   Extreme class improvement: {extreme_improvement * 100:+.1f} pp")

    # Success criteria
    success_criteria = [
        improvements >= 8,
        evt_results["balance_score"] > quantile_results["balance_score"],
        abs(evt_results["extreme_excess"]) < abs(quantile_results["extreme_excess"]),
    ]

    success_count = sum(success_criteria)
    print(f"\n🎯 Success: {success_count}/3 criteria met")

    if success_count >= 2:
        print("✅ EVT-inspired approach shows clear improvement!")
        return True
    else:
        print("⚠️  Needs more refinement.")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 The simplified EVT approach successfully combines")
        print("   Student's t modeling with practical tail compression!")
    else:
        print("\n❌ More work needed on the approach.")
