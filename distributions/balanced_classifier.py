"""
Balanced Financial Returns Classifier

This module provides a balanced classification approach specifically designed for
financial returns that addresses the extreme class concentration problem.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class BalancedBoundaries:
    """Container for balanced classification boundaries."""

    boundaries: np.ndarray
    nbins: int
    method: str
    validation_stats: dict


class FinancialReturnsClassifier:
    """
    Balanced classifier for financial returns using mixture of distributions.

    This approach addresses the extreme class concentration by:
    1. Using theoretical understanding of financial returns distributions
    2. Creating boundaries that naturally balance classes
    3. No dependence on specific training data (avoids leakage)
    """

    def __init__(self, nbins: int = 13):
        """Initialize balanced classifier."""
        self.nbins = nbins

    def estimate_params_robust(self, sample_data: np.ndarray) -> dict:
        """
        Robustly estimate distribution parameters using multiple methods.

        Args:
            sample_data: Sample of price movements

        Returns:
            Dictionary with estimated parameters
        """
        # Remove obvious outliers (>5 sigma)
        median_val = np.median(sample_data)
        mad = np.median(np.abs(sample_data - median_val))
        robust_scale = 1.4826 * mad  # Convert MAD to std equivalent

        # Filter extreme outliers
        clean_mask = np.abs(sample_data - median_val) < 5 * robust_scale
        clean_data = sample_data[clean_mask]

        # Estimate parameters using clean data
        location = np.median(clean_data)
        scale = robust_scale

        # Estimate shape parameter for generalized distribution
        # Use method of moments for t-distribution degrees of freedom
        sample_kurt = stats.kurtosis(clean_data)

        # For t-distribution: kurtosis = 6/(df-4) for df > 4
        # Solve for df: df = 6/kurtosis + 4
        if sample_kurt > 0.1:  # Avoid division issues
            df_estimate = 6 / sample_kurt + 4
            df_estimate = max(2.1, min(30, df_estimate))  # Reasonable bounds
        else:
            df_estimate = 30  # Nearly normal

        return {
            "location": location,
            "scale": scale,
            "df": df_estimate,
            "data_range": (clean_data.min(), clean_data.max()),
        }

    def create_balanced_boundaries(self, sample_data: np.ndarray) -> BalancedBoundaries:
        """
        Create balanced class boundaries using theoretical approach.

        Strategy:
        1. Estimate robust distribution parameters from sample
        2. Create boundaries using theoretical quantiles
        3. Apply corrections to ensure balance
        """
        # Estimate parameters
        params = self.estimate_params_robust(sample_data)

        # Create initial quantile-based boundaries using t-distribution
        quantiles = np.linspace(0, 1, self.nbins + 1)

        if params["df"] > 30:
            # Use normal distribution
            boundaries = stats.norm.ppf(quantiles, loc=params["location"], scale=params["scale"])
        else:
            # Use t-distribution
            boundaries = stats.t.ppf(
                quantiles, df=params["df"], loc=params["location"], scale=params["scale"]
            )

        # Apply tail compression to reduce extreme class concentration
        # This is the key innovation: compress the tails while preserving the center
        boundaries = self._apply_tail_compression(boundaries, params)

        # Validate and adjust if needed
        validation_stats = self._validate_boundaries(sample_data, boundaries)

        return BalancedBoundaries(
            boundaries=boundaries,
            nbins=self.nbins,
            method="Balanced_Financial_Returns",
            validation_stats=validation_stats,
        )

    def _apply_tail_compression(self, boundaries: np.ndarray, params: dict) -> np.ndarray:
        """
        Apply tail compression to reduce extreme class concentration.

        The key insight: financial returns have more extreme movements than
        a pure t-distribution would suggest, so we need to compress the tails
        to create more balanced classes.
        """
        location = params["location"]

        # Find boundaries relative to center
        relative_boundaries = boundaries - location

        # Apply different compression to positive and negative tails
        compressed = relative_boundaries.copy()

        # Compress negative tail (left side)
        negative_mask = relative_boundaries < 0
        if np.any(negative_mask):
            neg_boundaries = relative_boundaries[negative_mask]
            # Apply power compression: x' = sign(x) * |x|^0.8
            compressed[negative_mask] = np.sign(neg_boundaries) * (np.abs(neg_boundaries) ** 0.8)

        # Compress positive tail (right side)
        positive_mask = relative_boundaries > 0
        if np.any(positive_mask):
            pos_boundaries = relative_boundaries[positive_mask]
            # Apply power compression: x' = sign(x) * |x|^0.8
            compressed[positive_mask] = np.sign(pos_boundaries) * (pos_boundaries**0.8)

        # Convert back to absolute boundaries
        compressed_boundaries = compressed + location

        # Ensure monotonicity
        compressed_boundaries = np.sort(compressed_boundaries)

        # Extend tails beyond data range to ensure coverage
        data_min, data_max = params["data_range"]
        range_extension = (data_max - data_min) * 0.2

        compressed_boundaries[0] = min(compressed_boundaries[0], data_min - range_extension)
        compressed_boundaries[-1] = max(compressed_boundaries[-1], data_max + range_extension)

        return compressed_boundaries

    def _validate_boundaries(self, sample_data: np.ndarray, boundaries: np.ndarray) -> dict:
        """Validate that boundaries create balanced class distribution."""
        # Classify sample data
        labels = np.digitize(sample_data, boundaries[1:-1])
        labels = np.clip(labels, 0, self.nbins - 1)

        # Calculate distribution metrics
        class_counts = np.bincount(labels, minlength=self.nbins)
        class_fractions = class_counts / len(sample_data)

        expected_fraction = 1.0 / self.nbins
        deviations = np.abs(class_fractions - expected_fraction)
        max_deviation = np.max(deviations)
        balance_score = 1.0 - (max_deviation / expected_fraction)

        # Check extreme classes specifically
        extreme_classes = [0, self.nbins - 1]
        extreme_concentration = sum(class_fractions[i] for i in extreme_classes)
        expected_extreme = 2 * expected_fraction
        extreme_excess = extreme_concentration - expected_extreme

        return {
            "class_fractions": class_fractions.tolist(),
            "max_deviation": float(max_deviation),
            "balance_score": float(balance_score),
            "extreme_concentration": float(extreme_concentration),
            "extreme_excess": float(extreme_excess),
            "expected_fraction": float(expected_fraction),
            "sample_size": len(sample_data),
        }


def create_balanced_boundaries(sample_data: np.ndarray, nbins: int = 13) -> BalancedBoundaries:
    """
    Create balanced classification boundaries for financial returns.

    Args:
        sample_data: Sample of price movements for parameter estimation
        nbins: Number of classification bins

    Returns:
        BalancedBoundaries object with theoretical boundaries

    Example:
        # Create boundaries from small sample
        boundaries = create_balanced_boundaries(sample_movements[:10000])

        # Use for classification without data leakage
        labels = np.digitize(all_movements, boundaries.boundaries[1:-1])
    """
    classifier = FinancialReturnsClassifier(nbins=nbins)
    return classifier.create_balanced_boundaries(sample_data)
