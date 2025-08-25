"""
Improved Classification for Financial Returns

This module provides an improved approach to address extreme class concentration
while maintaining reasonable distributional properties.
"""

import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ImprovedBoundaries:
    """Container for improved classification boundaries."""

    boundaries: np.ndarray
    nbins: int
    method: str
    validation_stats: dict
    adjustment_factor: float


class TailAdjustedClassifier:
    """
    Tail-adjusted classifier that reduces extreme class concentration.

    Key insight: The extreme class concentration happens because financial returns
    have heavier tails than the quantile approach assumes. This classifier
    adjusts the tail boundaries to create more balanced classes.
    """

    def __init__(self, nbins: int = 13, tail_adjustment: float = 0.85):
        """
        Initialize tail-adjusted classifier.

        Args:
            nbins: Number of classification bins
            tail_adjustment: Factor to adjust tail boundaries (0.5-0.95)
                           Lower values = less extreme boundaries = more balanced
        """
        self.nbins = nbins
        self.tail_adjustment = tail_adjustment

    def create_tail_adjusted_boundaries(self, sample_data: np.ndarray) -> ImprovedBoundaries:
        """
        Create boundaries with tail adjustment to reduce extreme class concentration.

        Strategy:
        1. Start with quantile boundaries (baseline)
        2. Identify tail boundaries (typically classes 0, 1, 11, 12)
        3. Adjust tail boundaries toward center to reduce concentration
        4. Preserve center boundaries for normal classification
        """
        # Start with standard quantile boundaries
        quantiles = np.linspace(0, 1, self.nbins + 1)
        base_boundaries = np.quantile(sample_data, quantiles)

        # Calculate adjustment
        adjusted_boundaries = self._adjust_tail_boundaries(base_boundaries, sample_data)

        # Validate
        validation_stats = self._validate_boundaries(sample_data, adjusted_boundaries)

        return ImprovedBoundaries(
            boundaries=adjusted_boundaries,
            nbins=self.nbins,
            method="Tail_Adjusted",
            validation_stats=validation_stats,
            adjustment_factor=self.tail_adjustment,
        )

    def _adjust_tail_boundaries(
        self, base_boundaries: np.ndarray, sample_data: np.ndarray
    ) -> np.ndarray:
        """
        Adjust tail boundaries to reduce extreme class concentration.

        The approach:
        - Move extreme tail boundaries closer to center
        - Keep center boundaries relatively unchanged
        - Ensure monotonicity
        """
        adjusted = base_boundaries.copy()

        # Calculate data statistics for reference
        data_mean = np.mean(sample_data)

        # Define which boundaries to adjust (outer tails)
        n_tail_boundaries = 3  # Adjust outermost 3 boundaries on each side

        # Adjust left tail (negative extreme)
        for i in range(1, n_tail_boundaries + 1):
            if i < len(adjusted) - 1:  # Don't adjust extreme boundaries
                # Move boundary toward center by tail_adjustment factor
                center_distance = adjusted[i] - data_mean
                adjusted[i] = data_mean + center_distance * self.tail_adjustment

        # Adjust right tail (positive extreme)
        for i in range(len(adjusted) - n_tail_boundaries - 1, len(adjusted) - 1):
            if i > 0:  # Don't adjust extreme boundaries
                # Move boundary toward center by tail_adjustment factor
                center_distance = adjusted[i] - data_mean
                adjusted[i] = data_mean + center_distance * self.tail_adjustment

        # Ensure monotonicity and reasonable spacing
        adjusted = self._ensure_monotonic(adjusted)

        # Extend extremes to ensure full coverage
        data_range = sample_data.max() - sample_data.min()
        extension = data_range * 0.1
        adjusted[0] = sample_data.min() - extension
        adjusted[-1] = sample_data.max() + extension

        return adjusted

    def _ensure_monotonic(self, boundaries: np.ndarray) -> np.ndarray:
        """Ensure boundaries are monotonically increasing."""
        # Sort first
        boundaries = np.sort(boundaries)

        # Ensure minimum spacing between boundaries
        min_spacing = (boundaries[-1] - boundaries[0]) / (len(boundaries) * 100)

        for i in range(1, len(boundaries)):
            if boundaries[i] - boundaries[i - 1] < min_spacing:
                boundaries[i] = boundaries[i - 1] + min_spacing

        return boundaries

    def _validate_boundaries(self, sample_data: np.ndarray, boundaries: np.ndarray) -> dict:
        """Validate boundary performance."""
        # Classify sample data
        labels = np.digitize(sample_data, boundaries[1:-1])
        labels = np.clip(labels, 0, self.nbins - 1)

        # Calculate metrics
        class_counts = np.bincount(labels, minlength=self.nbins)
        class_fractions = class_counts / len(sample_data)

        expected_fraction = 1.0 / self.nbins
        deviations = np.abs(class_fractions - expected_fraction)
        max_deviation = np.max(deviations)
        balance_score = 1.0 - (max_deviation / expected_fraction)

        # Extreme class metrics
        extreme_fractions = class_fractions[[0, self.nbins - 1]]
        extreme_concentration = np.sum(extreme_fractions)
        expected_extreme = 2 * expected_fraction
        extreme_excess = extreme_concentration - expected_extreme

        # Per-class analysis
        class_analysis = []
        for i in range(self.nbins):
            deviation = deviations[i]
            status = "balanced" if deviation < expected_fraction * 0.3 else "imbalanced"
            class_analysis.append(
                {
                    "class": i,
                    "fraction": float(class_fractions[i]),
                    "count": int(class_counts[i]),
                    "deviation": float(deviation),
                    "status": status,
                }
            )

        return {
            "class_fractions": class_fractions.tolist(),
            "class_counts": class_counts.tolist(),
            "max_deviation": float(max_deviation),
            "balance_score": float(balance_score),
            "extreme_concentration": float(extreme_concentration),
            "extreme_excess": float(extreme_excess),
            "expected_fraction": float(expected_fraction),
            "class_analysis": class_analysis,
            "sample_size": len(sample_data),
        }


class OptimizedQuantileClassifier:
    """
    Optimized quantile classifier that iteratively improves boundary placement.

    This approach starts with quantiles but then optimizes the boundaries to
    minimize class imbalance through iterative adjustment.
    """

    def __init__(self, nbins: int = 13, max_iterations: int = 10):
        self.nbins = nbins
        self.max_iterations = max_iterations

    def create_optimized_boundaries(self, sample_data: np.ndarray) -> ImprovedBoundaries:
        """Create optimized boundaries through iterative improvement."""

        # Start with quantile boundaries
        current_boundaries = np.quantile(sample_data, np.linspace(0, 1, self.nbins + 1))

        best_boundaries = current_boundaries.copy()
        best_balance_score = -float("inf")

        # Iteratively improve boundaries
        for _iteration in range(self.max_iterations):
            # Try small adjustments to boundaries
            test_boundaries = self._adjust_boundaries_iteratively(current_boundaries, sample_data)

            # Evaluate balance
            validation = self._validate_boundaries(sample_data, test_boundaries)
            balance_score = validation["balance_score"]

            # Keep if better
            if balance_score > best_balance_score:
                best_balance_score = balance_score
                best_boundaries = test_boundaries.copy()
                current_boundaries = test_boundaries.copy()
            else:
                # If no improvement, try smaller adjustments
                break

        final_validation = self._validate_boundaries(sample_data, best_boundaries)

        return ImprovedBoundaries(
            boundaries=best_boundaries,
            nbins=self.nbins,
            method="Optimized_Quantile",
            validation_stats=final_validation,
            adjustment_factor=1.0,
        )

    def _adjust_boundaries_iteratively(
        self, boundaries: np.ndarray, sample_data: np.ndarray
    ) -> np.ndarray:
        """Make small adjustments to improve balance."""
        adjusted = boundaries.copy()

        # Focus on the most problematic boundaries (extremes)
        # Small movements toward better balance
        adjustment_factor = 0.05  # 5% adjustment

        data_range = sample_data.max() - sample_data.min()
        adjustment_size = data_range * adjustment_factor

        # Adjust boundaries that create the most imbalanced classes
        labels = np.digitize(sample_data, boundaries[1:-1])
        labels = np.clip(labels, 0, self.nbins - 1)
        class_counts = np.bincount(labels, minlength=self.nbins)

        expected_count = len(sample_data) / self.nbins

        # Find most over-represented classes and adjust their boundaries
        for i in range(self.nbins):
            if class_counts[i] > expected_count * 1.5:  # Over-represented
                # Adjust boundaries to reduce this class size
                if i > 0:  # Has left boundary
                    adjusted[i] -= adjustment_size
                if i < self.nbins - 1:  # Has right boundary
                    adjusted[i + 1] += adjustment_size

        # Ensure monotonicity
        adjusted = np.sort(adjusted)

        return adjusted

    def _validate_boundaries(self, sample_data: np.ndarray, boundaries: np.ndarray) -> dict:
        """Validate boundary performance (same as TailAdjustedClassifier)."""
        labels = np.digitize(sample_data, boundaries[1:-1])
        labels = np.clip(labels, 0, self.nbins - 1)

        class_counts = np.bincount(labels, minlength=self.nbins)
        class_fractions = class_counts / len(sample_data)

        expected_fraction = 1.0 / self.nbins
        deviations = np.abs(class_fractions - expected_fraction)
        max_deviation = np.max(deviations)
        balance_score = 1.0 - (max_deviation / expected_fraction)

        extreme_concentration = class_fractions[0] + class_fractions[self.nbins - 1]
        expected_extreme = 2 * expected_fraction
        extreme_excess = extreme_concentration - expected_extreme

        return {
            "class_fractions": class_fractions.tolist(),
            "class_counts": class_counts.tolist(),
            "max_deviation": float(max_deviation),
            "balance_score": float(balance_score),
            "extreme_concentration": float(extreme_concentration),
            "extreme_excess": float(extreme_excess),
            "expected_fraction": float(expected_fraction),
            "sample_size": len(sample_data),
        }


def create_tail_adjusted_boundaries(
    sample_data: np.ndarray, nbins: int = 13, tail_adjustment: float = 0.85
) -> ImprovedBoundaries:
    """
    Create tail-adjusted boundaries to reduce extreme class concentration.

    Args:
        sample_data: Sample of price movements
        nbins: Number of classification bins
        tail_adjustment: Tail adjustment factor (0.5-0.95)

    Returns:
        ImprovedBoundaries with adjusted boundaries
    """
    classifier = TailAdjustedClassifier(nbins=nbins, tail_adjustment=tail_adjustment)
    return classifier.create_tail_adjusted_boundaries(sample_data)


def create_optimized_boundaries(sample_data: np.ndarray, nbins: int = 13) -> ImprovedBoundaries:
    """
    Create optimized boundaries through iterative improvement.

    Args:
        sample_data: Sample of price movements
        nbins: Number of classification bins

    Returns:
        ImprovedBoundaries with optimized boundaries
    """
    classifier = OptimizedQuantileClassifier(nbins=nbins)
    return classifier.create_optimized_boundaries(sample_data)
