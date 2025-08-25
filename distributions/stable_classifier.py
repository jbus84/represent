"""
Stable Distribution Classifier for Financial Returns

This module uses α-stable (Lévy-stable) distributions which are theoretically
the most appropriate for financial returns due to:

1. Heavy tails (power-law decay)
2. Potential asymmetry (skewness)
3. Scale invariance properties
4. Generalized Central Limit Theorem applicability

α-stable distributions are characterized by four parameters:
- α (stability): tail heaviness (0 < α ≤ 2)
- β (skewness): asymmetry (-1 ≤ β ≤ 1)
- γ (scale): similar to standard deviation
- δ (location): similar to mean
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class StableBoundaries:
    """Container for stable distribution classification boundaries."""

    boundaries: np.ndarray
    nbins: int
    alpha: float
    beta: float
    gamma: float
    delta: float
    method: str
    validation_stats: dict


class StableDistributionClassifier:
    """
    Classifier using α-stable distributions for financial returns.

    α-stable distributions are the natural choice for financial data because:
    - They arise from the Generalized CLT
    - They exhibit heavy tails observed in financial returns
    - They can model asymmetry (different left/right tail behavior)
    - They have scale invariance properties important in finance
    """

    def __init__(self, nbins: int = 13):
        """Initialize stable distribution classifier."""
        self.nbins = nbins

    def fit_stable_distribution(self, data: np.ndarray) -> tuple:
        """
        Fit α-stable distribution to financial returns data.

        Uses method of moments approximation for speed and reliability.

        Args:
            data: Array of price movements

        Returns:
            Tuple of (alpha, beta, gamma, delta) parameters
        """
        # Use method of moments for reliable, fast fitting
        return self._fit_stable_moments(data)

    def _fit_stable_moments(self, data: np.ndarray) -> tuple:
        """
        Fit stable distribution using method of moments.

        This uses empirical relationships between moments and α-stable parameters.
        """
        # Calculate sample statistics
        mean_val = np.mean(data)
        var_val = np.var(data)
        skew_val = stats.skew(data)
        kurt_val = stats.kurtosis(data, fisher=True)  # Excess kurtosis

        # Estimate α from kurtosis (financial data typically α ∈ [1.2, 1.8])
        # Higher kurtosis → lower α (heavier tails)
        if kurt_val > 10:
            alpha = 1.2
        elif kurt_val > 6:
            alpha = 1.4
        elif kurt_val > 3:
            alpha = 1.6
        else:
            alpha = 1.8

        # Estimate β from skewness
        beta = np.clip(skew_val * 0.3, -0.8, 0.8)  # Scale down skewness

        # Estimate γ (scale) from variance and α
        # For α-stable: var ~ γ^(2/α) (approximately)
        if alpha < 2.0:
            gamma = (var_val ** (alpha / 2)) * 0.5
        else:
            gamma = np.sqrt(var_val)

        # Estimate δ (location)
        delta = mean_val

        return alpha, beta, gamma, delta

    def _approximate_stable_params(self, data: np.ndarray) -> tuple:
        """Fallback stable parameter estimation."""
        # Conservative estimates based on typical financial data
        alpha = 1.5  # Heavy tails, typical for financial returns
        beta = 0.1  # Slight positive skew (common in returns)
        gamma = np.std(data) * 0.8  # Scale parameter
        delta = np.median(data)  # Location parameter

        return alpha, beta, gamma, delta

    def generate_stable_boundaries(self, data: np.ndarray) -> StableBoundaries:
        """
        Generate classification boundaries using α-stable distribution.

        Args:
            data: Sample of price movements

        Returns:
            StableBoundaries object with fitted parameters and boundaries
        """
        # Fit stable distribution
        alpha, beta, gamma, delta = self.fit_stable_distribution(data)

        print(f"   α-stable fit: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.6f}, δ={delta:.6f}")

        # Generate boundaries using stable distribution quantiles
        quantiles = np.linspace(0.001, 0.999, self.nbins + 1)  # Avoid extreme quantiles

        try:
            # Try to use fitted stable distribution for boundaries
            boundaries = self._stable_quantiles(quantiles, alpha, beta, gamma, delta)
        except Exception:
            # Fallback to data quantiles if stable quantile calculation fails
            print("   Falling back to empirical quantiles")
            boundaries = np.quantile(data, quantiles)

        # Validate boundaries
        validation_stats = self._validate_stable_boundaries(
            data, boundaries, alpha, beta, gamma, delta
        )

        return StableBoundaries(
            boundaries=boundaries,
            nbins=self.nbins,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            method="Alpha_Stable",
            validation_stats=validation_stats,
        )

    def _stable_quantiles(
        self, quantiles: np.ndarray, alpha: float, beta: float, gamma: float, delta: float
    ) -> np.ndarray:
        """
        Calculate quantiles from α-stable distribution.

        This uses various methods depending on available implementations.
        """
        try:
            # Method 1: Use scipy if available
            boundaries = stats.levy_stable.ppf(quantiles, alpha, beta, loc=delta, scale=gamma)

            if np.any(~np.isfinite(boundaries)):
                raise ValueError("Non-finite boundaries from scipy")

            return boundaries

        except (AttributeError, ValueError):
            pass

        # Method 2: Approximate using Student's t with adjusted parameters
        # α-stable → t-distribution approximation for practical use

        # Convert α to degrees of freedom (empirical relationship)
        # Lower α → heavier tails → lower df
        df = max(2.1, 2.0 / (2.1 - alpha))

        # Generate t-distribution quantiles
        t_quantiles = stats.t.ppf(quantiles, df, loc=delta, scale=gamma)

        # Apply asymmetry correction based on β
        if abs(beta) > 0.1:
            # Shift quantiles to introduce asymmetry
            shift_factor = beta * gamma * 0.2
            asymmetric_quantiles = t_quantiles + shift_factor * np.sign(quantiles - 0.5)
            return asymmetric_quantiles
        else:
            return t_quantiles

    def _validate_stable_boundaries(
        self,
        data: np.ndarray,
        boundaries: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
    ) -> dict:
        """Validate stable distribution boundaries."""
        # Classify data using boundaries
        labels = np.digitize(data, boundaries[1:-1])
        labels = np.clip(labels, 0, self.nbins - 1)

        class_counts = np.bincount(labels, minlength=self.nbins)
        class_fractions = class_counts / len(data)

        # Calculate balance metrics
        expected_fraction = 1.0 / self.nbins
        deviations = np.abs(class_fractions - expected_fraction)
        max_deviation = np.max(deviations)
        balance_score = 1.0 - (max_deviation / expected_fraction)

        # Extreme class analysis
        extreme_concentration = class_fractions[0] + class_fractions[self.nbins - 1]
        expected_extreme = 2 * expected_fraction
        extreme_excess = extreme_concentration - expected_extreme

        return {
            "class_fractions": class_fractions.tolist(),
            "balance_score": float(balance_score),
            "extreme_concentration": float(extreme_concentration),
            "extreme_excess": float(extreme_excess),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "delta": float(delta),
            "sample_size": len(data),
        }


def create_stable_boundaries(data: np.ndarray, nbins: int = 13) -> StableBoundaries:
    """
    Create classification boundaries using α-stable distribution.

    Args:
        data: Sample of price movements
        nbins: Number of classification bins

    Returns:
        StableBoundaries with fitted α-stable distribution

    Example:
        boundaries = create_stable_boundaries(sample_movements)
        labels = np.digitize(all_movements, boundaries.boundaries[1:-1])
    """
    classifier = StableDistributionClassifier(nbins=nbins)
    return classifier.generate_stable_boundaries(data)
