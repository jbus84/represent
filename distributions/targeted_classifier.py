"""
Targeted Classification for Financial Returns

This module directly addresses the extreme class concentration problem
by analyzing the actual data distribution and creating boundaries that
specifically target balanced classes.
"""

import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class TargetedBoundaries:
    """Container for targeted classification boundaries."""

    boundaries: np.ndarray
    nbins: int
    method: str
    validation_stats: dict
    target_fractions: np.ndarray


class TargetedBalanceClassifier:
    """
    Classifier that directly targets balanced class distribution.

    Key insight: Instead of using theoretical distributions, use iterative
    boundary optimization to directly achieve the target class balance.
    This avoids the data leakage issue by not using future data, while
    still achieving better balance.
    """

    def __init__(self, nbins: int = 13, target_balance: float = 0.9, max_iterations: int = 50):
        """
        Initialize targeted classifier.

        Args:
            nbins: Number of classification bins
            target_balance: Target balance score (0.9 = 90% balanced)
            max_iterations: Maximum optimization iterations
        """
        self.nbins = nbins
        self.target_balance = target_balance
        self.max_iterations = max_iterations
        self.expected_fraction = 1.0 / nbins

    def create_targeted_boundaries(self, sample_data: np.ndarray) -> TargetedBoundaries:
        """
        Create boundaries optimized for target class balance.

        Strategy:
        1. Start with quantile boundaries
        2. Iteratively adjust boundaries to reduce imbalance
        3. Focus specifically on problematic extreme classes
        4. Use gradient descent style optimization
        """

        # Initialize with quantile boundaries
        current_boundaries = np.quantile(sample_data, np.linspace(0, 1, self.nbins + 1))
        best_boundaries = current_boundaries.copy()
        best_balance_score = self._calculate_balance_score(sample_data, current_boundaries)

        print(f"   Initial balance score: {best_balance_score:.3f}")

        # Iterative optimization
        learning_rate = 0.1
        iteration_count = 0
        for _iteration in range(self.max_iterations):
            iteration_count = _iteration + 1
            # Calculate gradients for each boundary
            gradients = self._calculate_gradients(sample_data, current_boundaries, learning_rate)

            # Apply gradients to boundaries
            new_boundaries = current_boundaries + gradients

            # Ensure monotonicity
            new_boundaries = self._ensure_monotonic(new_boundaries)

            # Evaluate new boundaries
            balance_score = self._calculate_balance_score(sample_data, new_boundaries)

            # Accept if better
            if balance_score > best_balance_score:
                best_balance_score = balance_score
                best_boundaries = new_boundaries.copy()
                current_boundaries = new_boundaries.copy()
            else:
                # Reduce learning rate and continue
                learning_rate *= 0.9
                if learning_rate < 0.01:
                    break

        print(
            f"   Final balance score: {best_balance_score:.3f} (after {iteration_count} iterations)"
        )

        # Create final validation
        validation_stats = self._validate_boundaries(sample_data, best_boundaries)

        target_fractions = np.full(self.nbins, self.expected_fraction)

        return TargetedBoundaries(
            boundaries=best_boundaries,
            nbins=self.nbins,
            method="Targeted_Balance",
            validation_stats=validation_stats,
            target_fractions=target_fractions,
        )

    def _calculate_gradients(
        self, sample_data: np.ndarray, boundaries: np.ndarray, lr: float
    ) -> np.ndarray:
        """Calculate gradients for boundary adjustment."""
        gradients = np.zeros_like(boundaries)

        # Current classification
        labels = np.digitize(sample_data, boundaries[1:-1])
        labels = np.clip(labels, 0, self.nbins - 1)
        class_counts = np.bincount(labels, minlength=self.nbins)
        class_fractions = class_counts / len(sample_data)

        # Calculate imbalance for each class
        imbalances = class_fractions - self.expected_fraction

        # For each internal boundary, calculate how to adjust it
        for b_idx in range(1, len(boundaries) - 1):  # Skip extremes
            class_left = b_idx - 1  # Class to the left of this boundary
            class_right = b_idx  # Class to the right of this boundary

            # If left class is over-represented and right is under-represented
            if class_left < self.nbins and class_right < self.nbins:
                left_imbalance = imbalances[class_left]
                right_imbalance = imbalances[class_right]

                # Move boundary to balance these classes
                gradient = (left_imbalance - right_imbalance) * lr

                # Scale gradient by local data density for stability
                data_range = boundaries[-1] - boundaries[0]
                gradient *= data_range * 0.01  # Small steps

                gradients[b_idx] = gradient

        return gradients

    def _calculate_balance_score(self, sample_data: np.ndarray, boundaries: np.ndarray) -> float:
        """Calculate balance score for given boundaries."""
        labels = np.digitize(sample_data, boundaries[1:-1])
        labels = np.clip(labels, 0, self.nbins - 1)

        class_counts = np.bincount(labels, minlength=self.nbins)
        class_fractions = class_counts / len(sample_data)

        deviations = np.abs(class_fractions - self.expected_fraction)
        max_deviation = np.max(deviations)
        balance_score = 1.0 - (max_deviation / self.expected_fraction)

        return balance_score

    def _ensure_monotonic(self, boundaries: np.ndarray) -> np.ndarray:
        """Ensure boundaries remain monotonically increasing."""
        # Sort to ensure monotonicity
        boundaries = np.sort(boundaries)

        # Ensure minimum spacing
        data_range = boundaries[-1] - boundaries[0]
        min_spacing = data_range / (len(boundaries) * 1000)  # Very small minimum

        for i in range(1, len(boundaries)):
            if boundaries[i] - boundaries[i - 1] < min_spacing:
                boundaries[i] = boundaries[i - 1] + min_spacing

        return boundaries

    def _validate_boundaries(self, sample_data: np.ndarray, boundaries: np.ndarray) -> dict:
        """Comprehensive validation of boundaries."""
        labels = np.digitize(sample_data, boundaries[1:-1])
        labels = np.clip(labels, 0, self.nbins - 1)

        class_counts = np.bincount(labels, minlength=self.nbins)
        class_fractions = class_counts / len(sample_data)

        deviations = np.abs(class_fractions - self.expected_fraction)
        max_deviation = np.max(deviations)
        balance_score = 1.0 - (max_deviation / self.expected_fraction)

        # Extreme class analysis
        extreme_classes = [0, self.nbins - 1]
        extreme_concentration = sum(class_fractions[i] for i in extreme_classes)
        expected_extreme = 2 * self.expected_fraction
        extreme_excess = extreme_concentration - expected_extreme

        # Per-class analysis
        class_analysis = []
        for i in range(self.nbins):
            deviation = deviations[i]
            deviation_pct = (deviation / self.expected_fraction) * 100

            if deviation < self.expected_fraction * 0.2:
                status = "well_balanced"
            elif deviation < self.expected_fraction * 0.5:
                status = "acceptable"
            else:
                status = "imbalanced"

            class_analysis.append(
                {
                    "class": i,
                    "fraction": float(class_fractions[i]),
                    "count": int(class_counts[i]),
                    "deviation": float(deviation),
                    "deviation_pct": float(deviation_pct),
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
            "expected_fraction": float(self.expected_fraction),
            "class_analysis": class_analysis,
            "sample_size": len(sample_data),
            "well_balanced_classes": len(
                [c for c in class_analysis if c["status"] == "well_balanced"]
            ),
            "imbalanced_classes": len([c for c in class_analysis if c["status"] == "imbalanced"]),
        }


def create_targeted_boundaries(
    sample_data: np.ndarray, nbins: int = 13, target_balance: float = 0.9, max_iterations: int = 50
) -> TargetedBoundaries:
    """
    Create boundaries optimized for balanced class distribution.

    Args:
        sample_data: Sample of price movements for optimization
        nbins: Number of classification bins
        target_balance: Target balance score
        max_iterations: Maximum optimization iterations

    Returns:
        TargetedBoundaries with optimized boundaries

    Example:
        # Create balanced boundaries from sample
        boundaries = create_targeted_boundaries(sample_movements[:50000])

        # Use for classification (no data leakage)
        labels = np.digitize(all_movements, boundaries.boundaries[1:-1])
    """
    classifier = TargetedBalanceClassifier(
        nbins=nbins, target_balance=target_balance, max_iterations=max_iterations
    )
    return classifier.create_targeted_boundaries(sample_data)
