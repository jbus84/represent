"""
Simplified EVT-Inspired Classification

This uses the key insights from EVT (Extreme Value Theory) without the complexity:
1. Student's t-distribution for overall heavy-tailed behavior
2. Power-law tail compression to address extreme class concentration
3. Stable, predictable results while maintaining theoretical foundation
"""

import numpy as np
from scipy import stats


def calculate_evt_inspired_boundaries(price_movements: np.ndarray, nbins: int = 13) -> np.ndarray:
    """
    Calculate boundaries using EVT-inspired approach.

    This combines:
    1. Student's t-distribution fit (handles heavy tails)
    2. Power-law compression of extreme quantiles (reduces class 0/12 concentration)

    The key insight: Financial returns have heavy tails, but simple quantiles
    create too much concentration in extreme classes. We need to "compress" the
    tail boundaries while preserving the center.

    Args:
        price_movements: Array of price movements
        nbins: Number of classification bins

    Returns:
        Array of boundary values
    """

    # Step 1: Fit Student's t-distribution
    try:
        df, loc, scale = stats.t.fit(price_movements)
        df = max(2.1, min(30, df))  # Constrain to reasonable range

        print(f"   Student's t fit: df={df:.2f}, loc={loc:.6f}, scale={scale:.6f}")

        # Use t-distribution if meaningful heavy tails, else normal
        use_t_dist = df < 10

    except Exception:
        # Fallback to normal
        loc = np.mean(price_movements)
        scale = np.std(price_movements)
        use_t_dist = False
        print(f"   Normal fallback: loc={loc:.6f}, scale={scale:.6f}")

    # Step 2: Generate quantiles with EVT-inspired tail compression
    quantiles = np.linspace(0, 1, nbins + 1)
    boundaries = []

    # Tail compression parameters based on EVT theory
    # Financial returns typically show power-law behavior in tails
    tail_compression = 0.75  # Compress tail quantiles by this factor
    center_preservation = 0.4  # Preserve center quantiles (±40% around median)

    for i, q in enumerate(quantiles):
        if i == 0:
            # Minimum boundary - extend for coverage
            if use_t_dist:
                boundary = stats.t.ppf(0.001, df, loc=loc, scale=scale)
            else:
                boundary = stats.norm.ppf(0.001, loc=loc, scale=scale)

            if not np.isfinite(boundary):
                boundary = price_movements.min() - abs(price_movements.min()) * 0.2

        elif i == len(quantiles) - 1:
            # Maximum boundary - extend for coverage
            if use_t_dist:
                boundary = stats.t.ppf(0.999, df, loc=loc, scale=scale)
            else:
                boundary = stats.norm.ppf(0.999, loc=loc, scale=scale)

            if not np.isfinite(boundary):
                boundary = price_movements.max() + abs(price_movements.max()) * 0.2

        else:
            # Internal boundaries with tail compression

            # Determine if this quantile is in the tails or center
            distance_from_median = abs(q - 0.5)

            if distance_from_median > center_preservation:
                # This is in the tail - apply compression

                # Calculate compression factor (stronger for more extreme quantiles)
                tail_strength = (distance_from_median - center_preservation) / (
                    0.5 - center_preservation
                )
                compression_factor = 1.0 - (1.0 - tail_compression) * tail_strength

                # Apply compression by moving quantile toward center
                if q < 0.5:
                    # Lower tail
                    compressed_q = 0.5 - (0.5 - q) * compression_factor
                else:
                    # Upper tail
                    compressed_q = 0.5 + (q - 0.5) * compression_factor

                # Use compressed quantile for boundary
                if use_t_dist:
                    boundary = stats.t.ppf(compressed_q, df, loc=loc, scale=scale)
                else:
                    boundary = stats.norm.ppf(compressed_q, loc=loc, scale=scale)

            else:
                # This is in the center - use normal quantile
                if use_t_dist:
                    boundary = stats.t.ppf(q, df, loc=loc, scale=scale)
                else:
                    boundary = stats.norm.ppf(q, loc=loc, scale=scale)

            # Fallback for any numerical issues
            if not np.isfinite(boundary):
                boundary = np.quantile(price_movements, q)

        boundaries.append(boundary)

    # Step 3: Ensure monotonicity and proper spacing
    boundaries = np.array(sorted(boundaries))

    # Ensure minimum spacing
    min_spacing = (boundaries[-1] - boundaries[0]) / (len(boundaries) * 1000)
    for i in range(1, len(boundaries)):
        if boundaries[i] - boundaries[i - 1] < min_spacing:
            boundaries[i] = boundaries[i - 1] + min_spacing

    print(f"   Tail compression applied: {tail_compression:.2f}")
    print(f"   Center preservation: ±{center_preservation * 100:.0f}% around median")

    return boundaries


def test_evt_inspired_approach(price_movements: np.ndarray, validation_movements: np.ndarray):
    """Test the EVT-inspired approach against quantiles."""

    print("🔬 Testing EVT-Inspired Approach (Student's t + Tail Compression)")

    # Generate boundaries
    evt_boundaries = calculate_evt_inspired_boundaries(price_movements, nbins=13)

    # Generate quantile boundaries for comparison
    quantiles = np.linspace(0, 1, 14)
    quantile_boundaries = np.quantile(price_movements, quantiles)

    # Test both on validation data
    approaches = [("Quantile", quantile_boundaries), ("EVT-Inspired", evt_boundaries)]

    results = {}

    for name, boundaries in approaches:
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
            "extreme_excess": extreme_excess,
            "boundaries": boundaries,
        }

        print(f"\n{name} Results:")
        print(f"   Balance Score: {balance_score:.3f}")
        print(f"   Extreme Classes (0+12): {extreme_concentration * 100:.1f}%")
        print(f"   Extreme Excess: {extreme_excess * 100:+.1f} pp")

    return results
