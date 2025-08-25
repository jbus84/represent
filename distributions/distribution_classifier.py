"""
Distribution-Based Classification using Generalized Pareto Distribution (GPD)

This module provides statistical distribution-based class boundary generation
that doesn't rely on specific training data, avoiding data leakage issues.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class DistributionBoundaries:
    """Container for distribution-based classification boundaries."""

    boundaries: np.ndarray
    nbins: int
    distribution_params: dict
    method: str
    validation_stats: dict


class EVTClassifier:
    """
    Extreme Value Theory (EVT) based classifier using Generalized Pareto Distribution.

    This approach models the center and tails separately:
    1. Center (~90%): Normal/Student-t distribution
    2. Tails (~10%): GPD for extreme movements

    Benefits:
    - No data leakage (doesn't use actual training data)
    - Realistic modeling of financial returns
    - Balanced class distribution
    - Theoretical foundation
    """

    def __init__(
        self, nbins: int = 13, center_fraction: float = 0.8, tail_threshold_percentile: float = 10.0
    ):
        """
        Initialize EVT classifier.

        Args:
            nbins: Number of classification bins
            center_fraction: Fraction of data considered "center" (non-extreme)
            tail_threshold_percentile: Percentile threshold for defining tails
        """
        self.nbins = nbins
        self.center_fraction = center_fraction
        self.tail_threshold_percentile = tail_threshold_percentile

    def estimate_distribution_params(self, price_movements: np.ndarray) -> dict:
        """
        Estimate parameters for center and tail distributions.

        Args:
            price_movements: Array of price movements (percentage changes)

        Returns:
            Dictionary with fitted distribution parameters
        """
        # Remove extreme outliers (beyond 3 sigma) before fitting
        sigma_est = np.std(price_movements)
        mask = np.abs(price_movements) < 3 * sigma_est
        clean_movements = price_movements[mask]

        # Fit center distribution (Student-t for heavy tails but not extreme)
        center_threshold = np.percentile(np.abs(clean_movements), 100 * self.center_fraction)
        center_data = clean_movements[np.abs(clean_movements) <= center_threshold]

        # Fit Student-t distribution to center
        try:
            t_params = stats.t.fit(center_data)
            center_df, center_loc, center_scale = t_params
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Fallback to normal distribution
            center_loc = np.mean(center_data)
            center_scale = np.std(center_data)
            center_df = np.inf  # Normal distribution limit

        # Fit GPD to positive and negative tails
        positive_tail = clean_movements[clean_movements > center_threshold]
        negative_tail = -clean_movements[clean_movements < -center_threshold]

        # Fit GPD to positive tail (excesses above threshold)
        pos_excesses = positive_tail - center_threshold
        try:
            pos_gpd_params = stats.genpareto.fit(pos_excesses, floc=0)
            pos_shape, pos_loc, pos_scale = pos_gpd_params
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Fallback exponential parameters
            pos_shape = 0.1
            pos_scale = np.mean(pos_excesses) if len(pos_excesses) > 0 else center_scale
            pos_loc = 0

        # Fit GPD to negative tail (excesses above threshold)
        neg_excesses = negative_tail - center_threshold
        try:
            neg_gpd_params = stats.genpareto.fit(neg_excesses, floc=0)
            neg_shape, neg_loc, neg_scale = neg_gpd_params
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Fallback exponential parameters
            neg_shape = 0.1
            neg_scale = np.mean(neg_excesses) if len(neg_excesses) > 0 else center_scale
            neg_loc = 0

        return {
            "center_threshold": center_threshold,
            "center_params": {"df": center_df, "loc": center_loc, "scale": center_scale},
            "pos_tail_params": {"c": pos_shape, "loc": pos_loc, "scale": pos_scale},
            "neg_tail_params": {"c": neg_shape, "loc": neg_loc, "scale": neg_scale},
            "tail_fraction": 1 - self.center_fraction,
        }

    def generate_theoretical_boundaries(self, params: dict) -> np.ndarray:
        """
        Generate class boundaries based on theoretical distribution.

        Args:
            params: Distribution parameters from estimate_distribution_params

        Returns:
            Array of class boundaries
        """
        center_threshold = params["center_threshold"]
        center_params = params["center_params"]
        pos_tail_params = params["pos_tail_params"]
        neg_tail_params = params["neg_tail_params"]

        # Calculate what fraction goes to each region
        center_fraction = self.center_fraction

        # Number of bins for each region (approximately proportional)
        center_bins = int(self.nbins * center_fraction)
        tail_bins_each = (self.nbins - center_bins) // 2

        # Ensure we use all bins
        if center_bins + 2 * tail_bins_each < self.nbins:
            center_bins += self.nbins - (center_bins + 2 * tail_bins_each)

        boundaries = []

        # Negative tail boundaries (most negative to -center_threshold)
        if tail_bins_each > 0:
            # Generate quantiles for negative tail
            neg_quantiles = np.linspace(0, 1, tail_bins_each + 1)[:-1]  # Exclude 1.0

            for q in neg_quantiles:
                # GPD quantile function for excesses
                excess_quantile = stats.genpareto.ppf(q, **neg_tail_params)
                # Convert back to original scale (negative)
                boundary = -(center_threshold + excess_quantile)
                boundaries.append(boundary)

        # Center region boundaries
        if center_bins > 0:
            center_quantiles = np.linspace(0, 1, center_bins + 1)[:-1]  # Exclude 1.0

            for q in center_quantiles:
                if center_params["df"] == np.inf:
                    # Normal distribution
                    boundary = stats.norm.ppf(
                        q, loc=center_params["loc"], scale=center_params["scale"]
                    )
                else:
                    # Student-t distribution
                    boundary = stats.t.ppf(
                        q,
                        center_params["df"],
                        loc=center_params["loc"],
                        scale=center_params["scale"],
                    )

                # Constrain to center region
                boundary = max(boundary, -center_threshold)
                boundary = min(boundary, center_threshold)
                boundaries.append(boundary)

        # Positive tail boundaries (center_threshold to most positive)
        if tail_bins_each > 0:
            pos_quantiles = np.linspace(0, 1, tail_bins_each + 1)[:-1]  # Exclude 1.0

            for q in pos_quantiles:
                # GPD quantile function for excesses
                excess_quantile = stats.genpareto.ppf(q, **pos_tail_params)
                # Convert back to original scale (positive)
                boundary = center_threshold + excess_quantile
                boundaries.append(boundary)

        # Sort boundaries and add extremes
        boundaries = np.array(sorted(boundaries))

        # Ensure we have the right number of boundaries (nbins-1 internal + 2 extremes)
        if len(boundaries) > self.nbins - 1:
            # Remove middle boundaries to get exactly nbins-1
            boundaries = boundaries[:: len(boundaries) // (self.nbins - 1)][: self.nbins - 1]
        elif len(boundaries) < self.nbins - 1:
            # Add linearly spaced boundaries to fill gaps
            min_bound, max_bound = boundaries[0], boundaries[-1]
            full_boundaries = np.linspace(min_bound, max_bound, self.nbins + 1)
            boundaries = full_boundaries[1:-1]  # Remove extremes

        return boundaries

    def create_distribution_boundaries(
        self, sample_movements: np.ndarray
    ) -> DistributionBoundaries:
        """
        Create classification boundaries using EVT approach.

        Args:
            sample_movements: Sample of price movements for parameter estimation

        Returns:
            DistributionBoundaries with theoretical boundaries
        """
        # Estimate distribution parameters from sample
        params = self.estimate_distribution_params(sample_movements)

        # Generate theoretical boundaries
        boundaries = self.generate_theoretical_boundaries(params)

        # Add extreme boundaries
        min_extreme = min(sample_movements.min(), boundaries[0]) - abs(boundaries[0]) * 0.1
        max_extreme = max(sample_movements.max(), boundaries[-1]) + abs(boundaries[-1]) * 0.1

        full_boundaries = np.array([min_extreme] + list(boundaries) + [max_extreme])

        # Validate boundaries create balanced distribution
        validation_stats = self._validate_boundaries(sample_movements, full_boundaries)

        return DistributionBoundaries(
            boundaries=full_boundaries,
            nbins=self.nbins,
            distribution_params=params,
            method="EVT_GPD",
            validation_stats=validation_stats,
        )

    def _validate_boundaries(self, sample_data: np.ndarray, boundaries: np.ndarray) -> dict:
        """Validate that boundaries create reasonable class distribution."""
        # Classify sample data using boundaries
        labels = np.digitize(sample_data, boundaries[1:-1])
        labels = np.clip(labels, 0, self.nbins - 1)

        # Calculate class distribution
        class_counts = np.bincount(labels, minlength=self.nbins)
        class_fractions = class_counts / len(sample_data)

        # Calculate balance metrics
        expected_fraction = 1.0 / self.nbins
        max_deviation = np.max(np.abs(class_fractions - expected_fraction))
        balance_score = 1.0 - (max_deviation / expected_fraction)  # 1.0 = perfect balance

        return {
            "class_fractions": class_fractions.tolist(),
            "max_deviation": float(max_deviation),
            "balance_score": float(balance_score),
            "expected_fraction": float(expected_fraction),
            "sample_size": len(sample_data),
        }


def create_evt_boundaries(
    sample_movements: np.ndarray, nbins: int = 13, **kwargs
) -> DistributionBoundaries:
    """
    Convenience function to create EVT-based classification boundaries.

    Args:
        sample_movements: Sample of price movements for parameter estimation
        nbins: Number of classification bins
        **kwargs: Additional parameters for EVTClassifier

    Returns:
        DistributionBoundaries object

    Example:
        # Use small sample to estimate parameters, create theoretical boundaries
        boundaries = create_evt_boundaries(sample_movements, nbins=13)

        # Use boundaries for classification (no data leakage)
        labels = np.digitize(all_movements, boundaries.boundaries[1:-1])
    """
    classifier = EVTClassifier(nbins=nbins, **kwargs)
    return classifier.create_distribution_boundaries(sample_movements)
