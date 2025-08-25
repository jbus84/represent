"""
Improved Stable Distribution Classifier

This uses α-stable principles but with practical implementations that work
for financial returns classification.
"""

import numpy as np
from scipy import stats


class PracticalStableClassifier:
    """
    Practical α-stable distribution classifier for financial returns.

    Uses α-stable theory but with robust numerical implementations.
    """

    def __init__(self, nbins: int = 13):
        self.nbins = nbins

    def estimate_stable_params(self, data: np.ndarray) -> dict:
        """Estimate α-stable parameters using robust methods."""

        # Calculate basic statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        skew_val = stats.skew(data)
        kurt_val = stats.kurtosis(data, fisher=True)

        # Estimate α (stability parameter) from kurtosis
        # Financial returns typically have α ∈ [1.2, 1.9]
        # Higher kurtosis → lower α (heavier tails)
        if kurt_val > 15:
            alpha = 1.2
        elif kurt_val > 8:
            alpha = 1.4
        elif kurt_val > 4:
            alpha = 1.6
        elif kurt_val > 2:
            alpha = 1.8
        else:
            alpha = 1.9

        # Estimate β (skewness parameter)
        # β ∈ [-1, 1], financial data usually |β| < 0.5
        beta = np.clip(skew_val * 0.2, -0.4, 0.4)

        # Estimate scale and location
        gamma = std_val * 0.7  # Scale parameter
        delta = mean_val  # Location parameter

        return {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "sample_stats": {
                "mean": mean_val,
                "std": std_val,
                "skew": skew_val,
                "kurtosis": kurt_val,
            },
        }

    def create_stable_boundaries(self, data: np.ndarray) -> np.ndarray:
        """
        Create boundaries using α-stable inspired approach.

        Instead of trying to compute exact α-stable quantiles (which is numerically
        challenging), this uses α-stable theory to modify Student's t boundaries.
        """

        # Estimate stable parameters
        params = self.estimate_stable_params(data)
        alpha = params["alpha"]
        beta = params["beta"]
        gamma = params["gamma"]
        delta = params["delta"]

        print(f"   Stable params: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.6f}, δ={delta:.6f}")

        # Convert α-stable parameters to equivalent t-distribution
        # Lower α → heavier tails → lower degrees of freedom
        df = max(2.1, 4.0 / (2.1 - alpha) if alpha < 2.0 else 30.0)

        # Generate base quantiles with t-distribution
        quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
        boundaries = stats.t.ppf(quantiles, df, loc=delta, scale=gamma)

        # Apply asymmetry correction based on β parameter
        if abs(beta) > 0.05:
            # Shift boundaries to introduce asymmetry
            # Positive β → heavier right tail, negative β → heavier left tail
            for i, q in enumerate(quantiles):
                if q < 0.5:
                    # Left side - adjust based on β
                    boundaries[i] *= 1.0 - beta * 0.2
                elif q > 0.5:
                    # Right side - adjust based on β
                    boundaries[i] *= 1.0 + beta * 0.2

        # Apply tail compression to reduce extreme class concentration
        # This is inspired by the finite-variance property of truncated stable distributions
        tail_compression = 0.8  # Compress extreme quantiles
        center_region = 0.3  # ±30% around median preserved

        for i, q in enumerate(quantiles):
            distance_from_median = abs(q - 0.5)

            if distance_from_median > center_region:
                # This is in the tail - apply compression
                tail_strength = (distance_from_median - center_region) / (0.5 - center_region)
                compression_factor = 1.0 - (1.0 - tail_compression) * tail_strength

                if q < 0.5:
                    # Compress left tail
                    compressed_q = 0.5 - (0.5 - q) * compression_factor
                else:
                    # Compress right tail
                    compressed_q = 0.5 + (q - 0.5) * compression_factor

                # Recalculate boundary with compressed quantile
                boundaries[i] = stats.t.ppf(compressed_q, df, loc=delta, scale=gamma)

        # Ensure monotonicity and finite values
        boundaries = np.array(sorted(boundaries))
        boundaries = boundaries[np.isfinite(boundaries)]

        # Extend if needed
        if len(boundaries) < self.nbins + 1:
            # Linear interpolation to get required number
            x = np.linspace(0, len(boundaries) - 1, self.nbins + 1)
            boundaries = np.interp(x, np.arange(len(boundaries)), boundaries)
        elif len(boundaries) > self.nbins + 1:
            # Subsample to get required number
            indices = np.linspace(0, len(boundaries) - 1, self.nbins + 1, dtype=int)
            boundaries = boundaries[indices]

        return boundaries, params


def test_practical_stable_approach():
    """Test the practical stable distribution approach."""

    print("🔬 Testing Practical α-Stable Approach")
    print("=" * 45)

    # Load data
    import polars as pl

    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    sample_movements = df["price_movement"].to_numpy()[:100000]
    validation_movements = df["price_movement"].to_numpy()[100000:]

    print(f"Sample: {len(sample_movements):,}, Validation: {len(validation_movements):,}")

    # Test approaches
    classifier = PracticalStableClassifier(nbins=13)

    # Generate boundaries
    stable_boundaries, params = classifier.create_stable_boundaries(sample_movements)

    # Compare with quantiles
    quantiles = np.linspace(0, 1, 14)
    quantile_boundaries = np.quantile(sample_movements, quantiles)

    approaches = [("Quantile", quantile_boundaries), ("Stable", stable_boundaries)]

    results = {}

    for name, boundaries in approaches:
        # Test on validation
        labels = np.digitize(validation_movements, boundaries[1:-1])
        labels = np.clip(labels, 0, 12)

        class_counts = np.bincount(labels, minlength=13)
        class_fractions = class_counts / len(validation_movements)

        expected = 1.0 / 13
        deviations = np.abs(class_fractions - expected)
        max_deviation = np.max(deviations)
        balance_score = 1.0 - (max_deviation / expected)

        extreme_concentration = class_fractions[0] + class_fractions[12]
        extreme_excess = extreme_concentration - (2 * expected)

        results[name] = {
            "class_fractions": class_fractions,
            "balance_score": balance_score,
            "extreme_excess": extreme_excess,
        }

        print(f"\n{name} Results:")
        print(f"   Balance Score: {balance_score:.3f}")
        print(f"   Extreme (0+12): {extreme_concentration * 100:.1f}%")
        print(f"   Extreme Excess: {extreme_excess * 100:+.1f} pp")

    # Compare
    quantile_results = results["Quantile"]
    stable_results = results["Stable"]

    print("\nComparison:")
    print("Class | Quantile | Stable | Target | Better?")
    print("------|----------|--------|--------|--------")

    improvements = 0
    for i in range(13):
        q_pct = quantile_results["class_fractions"][i] * 100
        s_pct = stable_results["class_fractions"][i] * 100
        target = 100.0 / 13

        better = abs(s_pct - target) < abs(q_pct - target)
        if better:
            improvements += 1

        print(
            f"{i:5d} | {q_pct:7.1f}% | {s_pct:5.1f}% | {target:5.1f}% | {'✅' if better else '❌'}"
        )

    print(f"\nSummary: {improvements}/13 classes improved")

    balance_improvement = (
        (stable_results["balance_score"] - quantile_results["balance_score"])
        / abs(quantile_results["balance_score"])
    ) * 100
    extreme_improvement = quantile_results["extreme_excess"] - stable_results["extreme_excess"]

    print(f"Balance improvement: {balance_improvement:+.1f}%")
    print(f"Extreme improvement: {extreme_improvement * 100:+.1f} pp")

    return improvements >= 8, stable_boundaries, params


if __name__ == "__main__":
    try:
        success, boundaries, params = test_practical_stable_approach()
        if success:
            print("\n✅ Practical α-stable approach successful!")
        else:
            print("\n⚠️  Needs refinement.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
