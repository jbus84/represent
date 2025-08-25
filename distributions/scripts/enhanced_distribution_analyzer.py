#!/usr/bin/env python3
"""
Enhanced Distribution Analysis with Tail Prediction Focus

This provides comprehensive analysis of all distribution approaches with special focus on:
1. Tail prediction accuracy for classes 0 and 12 (where rewards are greatest)
2. Fixed EVT-Inspired approach with reduced extreme class concentration
3. Optimized Variance Gamma distribution for better tail modeling
4. New jump diffusion and mixture models
5. Detailed tail prediction metrics and performance analysis
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from comprehensive_distribution_tester import ComprehensiveDistributionTester, DistributionResults
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add represent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class EnhancedDistributionAnalyzer:
    """Enhanced analysis focused on tail prediction and correcting EVT issues."""

    def __init__(self, nbins: int = 13):
        self.nbins = nbins
        self.expected_fraction = 1.0 / nbins

    def fit_evt_inspired_fixed(self, data: np.ndarray):
        """
        FIXED EVT-Inspired approach with reduced extreme class concentration.

        Key fixes:
        1. Reduced tail compression from 75% to 50%
        2. Increased center preservation from 40% to 30%
        3. Better boundary spacing to prevent overcrowding
        4. Enhanced tail prediction while maintaining balance
        """

        # Fit Student's t-distribution
        df, loc, scale = stats.t.fit(data)
        df = max(2.1, min(30, df))

        # Generate quantiles with FIXED tail compression
        quantiles = np.linspace(0, 1, self.nbins + 1)
        boundaries = []

        # FIXED parameters - less aggressive compression
        tail_compression = 0.50  # REDUCED from 0.75 to 0.50
        center_preservation = 0.30  # REDUCED from 0.4 to 0.3 (more tail focus)

        for i, q in enumerate(quantiles):
            if i == 0:
                boundary = stats.t.ppf(0.001, df, loc=loc, scale=scale)
                if not np.isfinite(boundary):
                    boundary = data.min() - abs(data.min()) * 0.2
            elif i == len(quantiles) - 1:
                boundary = stats.t.ppf(0.999, df, loc=loc, scale=scale)
                if not np.isfinite(boundary):
                    boundary = data.max() + abs(data.max()) * 0.2
            else:
                distance_from_median = abs(q - 0.5)

                if distance_from_median > center_preservation:
                    # Apply REDUCED compression to tail
                    tail_strength = (distance_from_median - center_preservation) / (
                        0.5 - center_preservation
                    )
                    compression_factor = 1.0 - (1.0 - tail_compression) * tail_strength

                    if q < 0.5:
                        compressed_q = 0.5 - (0.5 - q) * compression_factor
                    else:
                        compressed_q = 0.5 + (q - 0.5) * compression_factor

                    boundary = stats.t.ppf(compressed_q, df, loc=loc, scale=scale)
                else:
                    boundary = stats.t.ppf(q, df, loc=loc, scale=scale)

                if not np.isfinite(boundary):
                    boundary = np.quantile(data, q)

            boundaries.append(boundary)

        boundaries = np.array(sorted(boundaries))

        # Ensure better boundary spacing to prevent overcrowding in extremes
        min_spacing = (boundaries[-1] - boundaries[0]) / (len(boundaries) * 500)  # Tighter spacing
        for i in range(1, len(boundaries)):
            if boundaries[i] - boundaries[i - 1] < min_spacing:
                boundaries[i] = boundaries[i - 1] + min_spacing

        params = {
            "df": df,
            "location": loc,
            "scale": scale,
            "tail_compression": tail_compression,
            "center_preservation": center_preservation,
            "method": "evt_inspired_fixed",
        }

        return boundaries, params

    def calculate_tail_prediction_metrics(self, results_dict, validation_data):
        """Calculate detailed metrics focused on tail prediction accuracy."""

        print("\n🎯 TAIL PREDICTION ANALYSIS - CLASSES 0 & 12 FOCUS")
        print("=" * 70)
        print("Analyzing accuracy for extreme classes where trading rewards are greatest")
        print()

        baseline_result = results_dict["Quantile (Baseline)"]
        baseline_labels = np.digitize(validation_data, baseline_result.boundaries[1:-1])
        baseline_labels = np.clip(baseline_labels, 0, self.nbins - 1)

        # Metrics for each approach
        tail_metrics = {}

        for name, result in results_dict.items():
            # Classify validation data
            labels = np.digitize(validation_data, result.boundaries[1:-1])
            labels = np.clip(labels, 0, self.nbins - 1)

            class_counts = np.bincount(labels, minlength=self.nbins)
            class_fractions = class_counts / len(validation_data)

            # Tail prediction metrics
            class_0_actual = class_fractions[0] * 100
            class_12_actual = class_fractions[12] * 100
            total_extreme = class_0_actual + class_12_actual

            target_each = self.expected_fraction * 100  # 7.69%
            target_total = 2 * target_each  # 15.38%

            # Tail accuracy metrics
            class_0_error = abs(class_0_actual - target_each)
            class_12_error = abs(class_12_actual - target_each)
            total_error = abs(total_extreme - target_total)

            # Tail coverage quality (how well we identify extreme movements)
            # Find the actual extreme movements in validation data
            data_sorted = np.sort(validation_data)
            true_bottom_7_7_pct = data_sorted[int(0.077 * len(data_sorted))]
            true_top_7_7_pct = data_sorted[int(0.923 * len(data_sorted))]

            # Check how well our boundaries capture true extremes
            boundary_0_1 = result.boundaries[1]  # Boundary between class 0 and 1
            boundary_11_12 = result.boundaries[12]  # Boundary between class 11 and 12

            # Coverage accuracy (closer boundaries to true extremes = better)
            left_boundary_accuracy = (
                abs(boundary_0_1 - true_bottom_7_7_pct) / abs(true_bottom_7_7_pct) * 100
            )
            right_boundary_accuracy = (
                abs(boundary_11_12 - true_top_7_7_pct) / abs(true_top_7_7_pct) * 100
            )

            # Composite tail score (lower is better)
            tail_score = (
                class_0_error
                + class_12_error
                + total_error
                + left_boundary_accuracy
                + right_boundary_accuracy
            ) / 5

            tail_metrics[name] = {
                "class_0_pct": class_0_actual,
                "class_12_pct": class_12_actual,
                "total_extreme": total_extreme,
                "class_0_error": class_0_error,
                "class_12_error": class_12_error,
                "total_error": total_error,
                "left_boundary_accuracy": left_boundary_accuracy,
                "right_boundary_accuracy": right_boundary_accuracy,
                "tail_score": tail_score,
                "result": result,
            }

        # Display tail prediction analysis
        print(
            f"{'Approach':>25} | {'Cls 0':>6} | {'Cls 12':>7} | {'Total':>6} | {'Tail Score':>10} | {'Quality':>8}"
        )
        print("-" * 85)

        # Sort by tail score (lower is better)
        sorted_tail = sorted(tail_metrics.items(), key=lambda x: x[1]["tail_score"])

        for name, metrics in sorted_tail:
            quality = (
                "🌟 BEST"
                if metrics == sorted_tail[0][1]
                else "✅ GOOD"
                if metrics["tail_score"] < 5.0
                else "⚠️  FAIR"
                if metrics["tail_score"] < 10.0
                else "❌ POOR"
            )

            print(
                f"{name:>25} | {metrics['class_0_pct']:5.1f}% | {metrics['class_12_pct']:6.1f}% | "
                f"{metrics['total_extreme']:5.1f}% | {metrics['tail_score']:9.1f} | {quality:>8}"
            )

        # Detailed boundary accuracy analysis
        print("\n📐 BOUNDARY ACCURACY ANALYSIS")
        print("-" * 50)
        print("True extreme thresholds:")
        print(f"   Bottom 7.7% threshold: {true_bottom_7_7_pct:.6f}")
        print(f"   Top 7.7% threshold: {true_top_7_7_pct:.6f}")
        print()

        print(
            f"{'Approach':>25} | {'Left Boundary':>13} | {'Right Boundary':>14} | {'Avg Error':>10}"
        )
        print("-" * 75)

        for name, metrics in sorted_tail[:5]:  # Top 5 only
            left_boundary = metrics["result"].boundaries[1]
            right_boundary = metrics["result"].boundaries[12]
            avg_error = (metrics["left_boundary_accuracy"] + metrics["right_boundary_accuracy"]) / 2

            print(
                f"{name:>25} | {left_boundary:12.6f} | {right_boundary:13.6f} | {avg_error:9.1f}%"
            )

        return tail_metrics, sorted_tail[0][0], sorted_tail[0][1]  # Best approach

    def analyze_reward_potential(self, tail_metrics):
        """Analyze trading reward potential based on tail prediction accuracy."""

        print("\n💰 TRADING REWARD POTENTIAL ANALYSIS")
        print("=" * 60)
        print("Extreme classes (0 & 12) typically offer highest trading rewards")
        print("Better tail prediction = better reward capture opportunity")
        print()

        # Assume reward potential is inversely related to tail prediction error
        # Lower tail score = higher reward potential

        reward_analysis = []
        for name, metrics in tail_metrics.items():
            if name == "Quantile (Baseline)":
                continue  # Skip baseline

            baseline_metrics = tail_metrics["Quantile (Baseline)"]

            # Calculate relative improvement over baseline
            tail_improvement = baseline_metrics["tail_score"] - metrics["tail_score"]
            improvement_pct = (tail_improvement / baseline_metrics["tail_score"]) * 100

            # Estimate reward potential (higher is better)
            if tail_improvement > 0:
                reward_potential = min(100, max(0, improvement_pct * 2))  # Cap at 100%
                quality_rating = (
                    "🌟 EXCELLENT"
                    if reward_potential > 50
                    else "✅ VERY GOOD"
                    if reward_potential > 25
                    else "⚠️  GOOD"
                    if reward_potential > 10
                    else "❌ MARGINAL"
                )
            else:
                reward_potential = 0
                quality_rating = "❌ NO IMPROVEMENT"

            reward_analysis.append(
                (name, improvement_pct, reward_potential, quality_rating, metrics)
            )

        # Sort by reward potential
        reward_analysis.sort(key=lambda x: x[2], reverse=True)

        print(
            f"{'Approach':>25} | {'Tail Improve':>12} | {'Reward Potential':>16} | {'Rating':>15}"
        )
        print("-" * 85)

        for name, improvement, potential, rating, _ in reward_analysis:
            print(f"{name:>25} | {improvement:+10.1f}% | {potential:14.1f}% | {rating:>15}")

        return reward_analysis

    def comprehensive_recommendation(self, results_dict, tail_metrics, reward_analysis):
        """Provide comprehensive recommendation focused on tail prediction and rewards."""

        print("\n🏆 COMPREHENSIVE RECOMMENDATION - TAIL PREDICTION FOCUS")
        print("=" * 70)

        # Get best approaches for different criteria
        best_tail = min(tail_metrics.items(), key=lambda x: x[1]["tail_score"])
        best_reward = max(reward_analysis, key=lambda x: x[2]) if reward_analysis else None
        best_balance = max(
            results_dict.items(),
            key=lambda x: x[1].balance_score if x[0] != "Quantile (Baseline)" else -1,
        )

        print(f"🎯 BEST FOR TAIL PREDICTION: {best_tail[0]}")
        print(f"   Tail Score: {best_tail[1]['tail_score']:.2f} (lower is better)")
        print(f"   Class 0: {best_tail[1]['class_0_pct']:.1f}% (target: 7.7%)")
        print(f"   Class 12: {best_tail[1]['class_12_pct']:.1f}% (target: 7.7%)")

        if best_reward:
            print(f"\n💰 BEST FOR REWARD POTENTIAL: {best_reward[0]}")
            print(f"   Reward Potential: {best_reward[2]:.1f}%")
            print(f"   Tail Improvement: {best_reward[1]:+.1f}%")

        print(f"\n⚖️  BEST FOR OVERALL BALANCE: {best_balance[0]}")
        print(f"   Balance Score: {best_balance[1].balance_score:.3f}")
        print(f"   Extreme Concentration: {best_balance[1].extreme_concentration * 100:.1f}%")

        # Final recommendation based on tail prediction priority
        if best_tail[1]["tail_score"] < 5.0:  # Good tail prediction
            recommended = best_tail[0]
            print(f"\n🌟 FINAL RECOMMENDATION: {recommended}")
            print(
                f"   Reason: Excellent tail prediction accuracy (score: {best_tail[1]['tail_score']:.2f})"
            )
            print("   Expected benefit: Superior identification of extreme price movements")
            print("   Trading impact: Higher probability of capturing extreme market events")
        elif best_reward and best_reward[2] > 25:  # Good reward potential
            recommended = best_reward[0]
            print(f"\n✅ FINAL RECOMMENDATION: {recommended}")
            print(
                f"   Reason: Strong reward potential ({best_reward[2]:.1f}%) with good tail prediction"
            )
            print("   Expected benefit: Balanced performance with reward focus")
        else:
            recommended = best_balance[0]
            print(f"\n⚠️  FINAL RECOMMENDATION: {recommended}")
            print("   Reason: Best overall balance, though tail prediction could be improved")
            print("   Expected benefit: Stable performance across all classes")

        return recommended


def main():
    """Run enhanced distribution analysis focused on tail prediction."""

    # Load data
    print("📊 Loading AUDUSD price movement data...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    # CORRECTED: Proper 70/30 temporal train/test split
    total_samples = len(df)
    train_end_idx = int(total_samples * 0.7)

    # Training data: First 70% for fitting distributions
    train_movements = df["price_movement"].to_numpy()[:train_end_idx]
    # Test data: Last 30% for validation (completely separate)
    test_movements = df["price_movement"].to_numpy()[train_end_idx:]

    # Use first 100K from TRAINING data only for analysis
    sample_movements = train_movements[:100000]
    validation_movements = test_movements  # Use proper test set for validation

    print(f"   Total dataset: {total_samples:,} samples")
    print(f"   Training data (70%): {len(train_movements):,} samples")
    print(f"   Test data (30%): {len(test_movements):,} samples")
    print(f"   Analysis sample (from training): {len(sample_movements):,} samples")
    print(f"   Validation sample (test data): {len(validation_movements):,} samples")

    # Run comprehensive testing first
    tester = ComprehensiveDistributionTester(nbins=13)
    results = tester.run_comprehensive_test(sample_movements, validation_movements)

    # Add FIXED EVT-inspired approach
    analyzer = EnhancedDistributionAnalyzer(nbins=13)
    print("\n🔧 Testing FIXED EVT-Inspired (Reduced Extreme Concentration)...")

    try:
        evt_boundaries, evt_params = analyzer.fit_evt_inspired_fixed(sample_movements)

        # Test on validation
        evt_labels = np.digitize(validation_movements, evt_boundaries[1:-1])
        evt_labels = np.clip(evt_labels, 0, 12)

        evt_class_counts = np.bincount(evt_labels, minlength=13)
        evt_class_fractions = evt_class_counts / len(validation_movements)

        expected = 1.0 / 13
        evt_deviations = np.abs(evt_class_fractions - expected)
        evt_max_deviation = np.max(evt_deviations)
        evt_balance_score = 1.0 - (evt_max_deviation / expected)

        evt_extreme_concentration = evt_class_fractions[0] + evt_class_fractions[12]
        evt_extreme_excess = evt_extreme_concentration - (2 * expected)

        evt_result = DistributionResults(
            name="EVT-Inspired FIXED",
            boundaries=evt_boundaries,
            class_fractions=evt_class_fractions,
            balance_score=evt_balance_score,
            extreme_concentration=evt_extreme_concentration,
            extreme_excess=evt_extreme_excess,
            parameters=evt_params,
            fit_quality="good" if evt_balance_score > 0 else "acceptable",
            success=True,
        )

        results["EVT-Inspired FIXED"] = evt_result

        print(f"   Balance Score: {evt_balance_score:.3f}")
        print(f"   Extreme Classes (0+12): {evt_extreme_concentration * 100:.1f}%")
        print(f"   Extreme Excess: {evt_extreme_excess * 100:+.1f} pp")

    except Exception as e:
        print(f"   ❌ FIXED EVT-Inspired test failed: {e}")

    # Enhanced analysis
    tail_metrics, best_tail_name, best_tail_metrics = analyzer.calculate_tail_prediction_metrics(
        results, validation_movements
    )
    reward_analysis = analyzer.analyze_reward_potential(tail_metrics)
    recommended = analyzer.comprehensive_recommendation(results, tail_metrics, reward_analysis)

    return results, recommended, tail_metrics


if __name__ == "__main__":
    try:
        results, recommended, tail_metrics = main()
        print(f"\n📋 Enhanced analysis complete. Recommended approach: {recommended}")
        print("🎯 Focus: Optimized for tail prediction and trading reward potential")
    except Exception as e:
        print(f"\n❌ Enhanced analysis failed: {e}")
        import traceback

        traceback.print_exc()
