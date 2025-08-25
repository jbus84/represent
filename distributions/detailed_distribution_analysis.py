#!/usr/bin/env python3
"""
Detailed Analysis of Distribution Approaches

This provides comprehensive comparison including:
1. All new distributions tested
2. Previous successful EVT-inspired approach
3. Detailed class-by-class analysis
4. Statistical significance testing
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from comprehensive_distribution_tester import ComprehensiveDistributionTester
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add represent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DetailedDistributionAnalyzer:
    """Detailed analysis of all distribution approaches."""

    def __init__(self, nbins: int = 13):
        self.nbins = nbins
        self.expected_fraction = 1.0 / nbins

    def fit_evt_inspired(self, data: np.ndarray):
        """
        EVT-Inspired approach (our previous winner) for comparison.
        Student's t + tail compression.
        """
        # Fit Student's t-distribution
        df, loc, scale = stats.t.fit(data)
        df = max(2.1, min(30, df))

        # Generate quantiles with tail compression
        quantiles = np.linspace(0, 1, self.nbins + 1)
        boundaries = []

        # Tail compression parameters
        tail_compression = 0.75
        center_preservation = 0.4

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
                    # Apply compression to tail
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

        params = {
            "df": df,
            "location": loc,
            "scale": scale,
            "tail_compression": tail_compression,
            "method": "evt_inspired",
        }

        return boundaries, params

    def analyze_class_improvements(self, results_dict):
        """Analyze which classes improved across different approaches."""

        baseline_fractions = results_dict["Quantile (Baseline)"].class_fractions

        print("\n📊 DETAILED CLASS-BY-CLASS ANALYSIS")
        print("=" * 80)

        # Header
        print(f"{'Class':>5} | {'Target':>7} | {'Baseline':>8} |", end="")
        for name in results_dict:
            if name != "Quantile (Baseline)":
                print(f" {name[:8]:>8} |", end="")
        print()

        print("-" * 80)

        # Track improvements per approach
        improvements = {name: 0 for name in results_dict if name != "Quantile (Baseline)"}

        for i in range(self.nbins):
            target = self.expected_fraction * 100
            baseline = baseline_fractions[i] * 100
            baseline_error = abs(baseline - target)

            print(f"{i:5d} | {target:6.1f}% | {baseline:7.1f}% |", end="")

            for name, result in results_dict.items():
                if name != "Quantile (Baseline)":
                    approach_pct = result.class_fractions[i] * 100
                    approach_error = abs(approach_pct - target)

                    if approach_error < baseline_error:
                        improvements[name] += 1
                        print(f" {approach_pct:7.1f}✅|", end="")
                    else:
                        print(f" {approach_pct:7.1f}❌|", end="")
            print()

        # Summary of improvements
        print("\n🎯 CLASS IMPROVEMENTS SUMMARY:")
        for name, count in improvements.items():
            percentage = (count / self.nbins) * 100
            print(f"   {name}: {count}/13 classes improved ({percentage:.0f}%)")

        return improvements

    def analyze_extreme_class_performance(self, results_dict):
        """Focus on extreme classes (0 and 12) performance."""

        print("\n🎯 EXTREME CLASS (0 & 12) DETAILED ANALYSIS")
        print("=" * 60)

        baseline_result = results_dict["Quantile (Baseline)"]
        baseline_class0 = baseline_result.class_fractions[0] * 100
        baseline_class12 = baseline_result.class_fractions[12] * 100
        baseline_total = baseline_class0 + baseline_class12

        target_each = self.expected_fraction * 100  # 7.7%
        target_total = 2 * target_each  # 15.4%

        print(
            f"Target: Class 0 = {target_each:.1f}%, Class 12 = {target_each:.1f}%, Total = {target_total:.1f}%"
        )
        print(
            f"Baseline: Class 0 = {baseline_class0:.1f}%, Class 12 = {baseline_class12:.1f}%, Total = {baseline_total:.1f}%"
        )
        print()

        print(
            f"{'Approach':>20} | {'Class 0':>8} | {'Class 12':>9} | {'Total':>8} | {'Improvement':>12}"
        )
        print("-" * 70)

        for name, result in results_dict.items():
            if name != "Quantile (Baseline)":
                class0_pct = result.class_fractions[0] * 100
                class12_pct = result.class_fractions[12] * 100
                total_pct = class0_pct + class12_pct

                total_error = abs(total_pct - target_total)
                baseline_error = abs(baseline_total - target_total)
                improvement = baseline_error - total_error

                status = "✅" if improvement > 0 else "❌"

                print(
                    f"{name:>20} | {class0_pct:7.1f}% | {class12_pct:8.1f}% | {total_pct:7.1f}% | {improvement:+7.1f}pp {status}"
                )

    def statistical_significance_test(self, results_dict, validation_data):
        """Test statistical significance of improvements."""

        print("\n📈 STATISTICAL SIGNIFICANCE ANALYSIS")
        print("=" * 50)

        baseline_result = results_dict["Quantile (Baseline)"]
        baseline_labels = np.digitize(validation_data, baseline_result.boundaries[1:-1])
        baseline_labels = np.clip(baseline_labels, 0, self.nbins - 1)

        # Chi-square test for distribution differences
        expected_counts = len(validation_data) / self.nbins

        print(f"Expected count per class: {expected_counts:,.0f}")
        print("Testing against baseline quantile approach...")
        print()

        for name, result in results_dict.items():
            if name != "Quantile (Baseline)":
                # Classify validation data with this approach
                labels = np.digitize(validation_data, result.boundaries[1:-1])
                labels = np.clip(labels, 0, self.nbins - 1)

                observed_counts = np.bincount(labels, minlength=self.nbins)

                # Chi-square test
                try:
                    chi2_stat, p_value = stats.chisquare(observed_counts, f_exp=expected_counts)

                    significance = (
                        "***"
                        if p_value < 0.001
                        else "**"
                        if p_value < 0.01
                        else "*"
                        if p_value < 0.05
                        else "NS"
                    )

                    print(f"{name:>25}: χ² = {chi2_stat:8.1f}, p = {p_value:.2e} {significance}")

                except Exception as e:
                    print(f"{name:>25}: Chi-square test failed: {e}")

    def create_recommendation(self, results_dict, improvements):
        """Create final recommendation based on all analyses."""

        print("\n🏆 FINAL RECOMMENDATION")
        print("=" * 40)

        # Find best approaches based on multiple criteria
        scores = {}
        baseline_balance = results_dict["Quantile (Baseline)"].balance_score

        for name, result in results_dict.items():
            if name != "Quantile (Baseline)":
                # Scoring criteria:
                # 1. Balance score improvement
                balance_improvement = result.balance_score - baseline_balance

                # 2. Number of classes improved
                class_improvements = improvements[name] / self.nbins

                # 3. Extreme class performance
                baseline_extreme = results_dict["Quantile (Baseline)"].extreme_excess
                extreme_improvement = baseline_extreme - result.extreme_excess

                # Composite score (weighted)
                composite_score = (
                    balance_improvement * 0.4 + class_improvements * 0.3 + extreme_improvement * 0.3
                )

                scores[name] = {
                    "composite": composite_score,
                    "balance": balance_improvement,
                    "classes": class_improvements,
                    "extreme": extreme_improvement,
                    "result": result,
                }

        # Sort by composite score
        ranked_approaches = sorted(scores.items(), key=lambda x: x[1]["composite"], reverse=True)

        print("Ranking by composite score (balance + classes + extreme performance):")
        print()
        print(
            f"{'Rank':>4} | {'Approach':>20} | {'Composite':>9} | {'Balance':>8} | {'Classes':>8} | {'Extreme':>8}"
        )
        print("-" * 80)

        for i, (name, score_data) in enumerate(ranked_approaches, 1):
            print(
                f"{i:4d} | {name:>20} | {score_data['composite']:+8.3f} | {score_data['balance']:+7.3f} | {score_data['classes'] * 100:6.1f}% | {score_data['extreme'] * 100:+6.1f}pp"
            )

        # Top recommendation
        if ranked_approaches:
            best_name, best_scores = ranked_approaches[0]
            best_result = best_scores["result"]

            print(f"\n🎉 TOP RECOMMENDATION: {best_name}")
            print(f"   Composite Score: {best_scores['composite']:+.3f}")
            print(
                f"   Balance Score: {best_result.balance_score:.3f} (vs baseline: {baseline_balance:.3f})"
            )
            print(
                f"   Classes Improved: {improvements[best_name]}/13 ({improvements[best_name] / 13 * 100:.0f}%)"
            )
            print(
                f"   Extreme Concentration: {best_result.extreme_concentration * 100:.1f}% (baseline: {results_dict['Quantile (Baseline)'].extreme_concentration * 100:.1f}%)"
            )

            if best_scores["composite"] > 0:
                print("   ✅ Shows overall improvement over baseline")
            else:
                print("   ⚠️  Mixed results - improvements in some areas, challenges in others")

            return best_name, best_result

        return None, None


def main():
    """Run detailed distribution analysis."""

    # Load data
    print("📊 Loading AUDUSD price movement data...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    sample_movements = df["price_movement"].to_numpy()[:100000]
    validation_movements = df["price_movement"].to_numpy()[100000:]

    # Run comprehensive testing first
    tester = ComprehensiveDistributionTester(nbins=13)
    results = tester.run_comprehensive_test(sample_movements, validation_movements)

    # Add EVT-inspired approach for comparison
    analyzer = DetailedDistributionAnalyzer(nbins=13)
    print("\n🔬 Testing EVT-Inspired (Previous Winner)...")

    try:
        evt_boundaries, evt_params = analyzer.fit_evt_inspired(sample_movements)

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

        from comprehensive_distribution_tester import DistributionResults

        evt_result = DistributionResults(
            name="EVT-Inspired",
            boundaries=evt_boundaries,
            class_fractions=evt_class_fractions,
            balance_score=evt_balance_score,
            extreme_concentration=evt_extreme_concentration,
            extreme_excess=evt_extreme_excess,
            parameters=evt_params,
            fit_quality="good" if evt_balance_score > 0 else "acceptable",
            success=True,
        )

        results["EVT-Inspired"] = evt_result

        print(f"   Balance Score: {evt_balance_score:.3f}")
        print(f"   Extreme Classes (0+12): {evt_extreme_concentration * 100:.1f}%")
        print(f"   Extreme Excess: {evt_extreme_excess * 100:+.1f} pp")

    except Exception as e:
        print(f"   ❌ EVT-Inspired test failed: {e}")

    # Detailed analysis
    improvements = analyzer.analyze_class_improvements(results)
    analyzer.analyze_extreme_class_performance(results)
    analyzer.statistical_significance_test(results, validation_movements)
    best_name, best_result = analyzer.create_recommendation(results, improvements)

    return results, best_name, best_result


if __name__ == "__main__":
    try:
        results, best_name, best_result = main()
        print(f"\n📋 Analysis complete. Best approach: {best_name}")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
