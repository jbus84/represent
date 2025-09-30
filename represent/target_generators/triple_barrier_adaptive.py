"""
Triple Barrier Method Target Generator

This module implements the Triple Barrier Method for financial time series labeling,
a popular technique in quantitative finance for creating structured trading labels.

The method uses three barriers:
1. Upper barrier: Profit target (positive return threshold)
2. Lower barrier: Stop loss (negative return threshold)
3. Time barrier: Maximum holding period (lookforward window)

Directional labels based on which barrier is hit first:
- +1: Upper barrier hit first → Long signal (upward price movement detected)
- -1: Lower barrier hit first → Short signal (downward price movement detected)
-  0: Time barrier hit first → No signal (timeout, no significant directional move)

Returns calculation:
- Long positions (+1): profit from upward moves (exit - entry) / entry
- Short positions (-1): profit from downward moves (entry - exit) / entry

Key Features:
- Symmetric or asymmetric profit/loss barriers
- Transaction cost integration for realistic labeling
- Bayesian-optimizable parameters (barriers, lookforward, transaction costs)
- Memory-efficient implementation for large datasets

References:
- "Advances in Financial Machine Learning" by Marcos López de Prado
- Triple barrier method for structured financial labels
"""

import warnings
from typing import Any

import numpy as np
import polars as pl

from .base import TargetGenerator


class TripleBarrierGeneratorAdaptive(TargetGenerator):
    """
    Triple Barrier Method target generator for structured financial labeling.

    Creates labels based on which of three barriers is reached first:
    - Upper barrier: Profit target (+barrier_width)
    - Lower barrier: Stop loss (-barrier_width)
    - Time barrier: Maximum holding period (lookforward_window)

    This method is particularly effective for creating balanced datasets with
    clear risk/reward structures and realistic transaction cost integration.
    """

    @property
    def required_columns(self) -> list[str]:
        """Return list of required DataFrame columns."""
        return ["mid_price"]

    @property
    def target_type(self) -> str:
        """Return the type of targets generated."""
        return "classification"

    def __init__(
        self,
        lookforward_window: int = 2000,  # FIXED: 2K ticks lookforward window
        lookback_window: int = 2000,  # FIXED: 2K ticks lookback window
        barrier_width: float = 1.,  # FIXED: 1 sigma barriers for better signal quality
        transaction_cost: float = 0.0001,  # 1 pip transaction cost
        target_name: str = "adaptive_triple_barrier_label",
    ):
        """
        Initialize Triple Barrier generator.

        Args:
            lookforward_window: Maximum holding period in ticks (time barrier)
            barrier_width: Default width for both barriers
            transaction_cost: Transaction cost as fraction (e.g., 0.0001 = 1 pip)
        """
        self.lookforward_window = lookforward_window
        self.lookback_window = lookback_window
        self.barrier_width = barrier_width
        self.transaction_cost = transaction_cost
        self.target_name = target_name


    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate triple barrier labels for the input DataFrame."""

        prices = df["mid_price"].to_numpy()

        if len(prices) < self.lookforward_window + self.lookback_window:
            warnings.warn(
                f"Insufficient data for triple barrier labeling: {len(prices)} samples. "
                f"Need at least {self.lookforward_window + self.lookback_window}. "
                f"Returning neutral labels.",
                stacklevel=2
            )
            labels = np.zeros(len(prices), dtype=np.int32)
            label_metadata = np.zeros(len(prices), dtype=np.float32)
            # Create empty plotting metadata for insufficient data case
            volatilities = np.zeros(len(prices), dtype=np.float32)
            upper_barriers = np.zeros(len(prices), dtype=np.float32)
            lower_barriers = np.zeros(len(prices), dtype=np.float32)
            exit_prices = np.zeros(len(prices), dtype=np.float32)
            exit_indices = np.zeros(len(prices), dtype=np.int32)
        else:
            labels, label_metadata, volatilities, upper_barriers, lower_barriers, exit_prices, exit_indices = self._compute_triple_barrier_labels_with_metadata(prices)

        # Create base DataFrame with metadata
        result_df = self._create_base_target_df(df, symbol)

        # Add target column
        result_df = result_df.with_columns(pl.Series(self.target_name, labels))

        # Add ALL metadata columns for complete plotting information
        result_df = result_df.with_columns([
            pl.Series(f"{self.target_name}_return", label_metadata),  # Expected return
            pl.Series(f"{self.target_name}_barrier_width", np.full(len(labels), self.barrier_width)),  # Parameter
            pl.Series(f"{self.target_name}_volatility", volatilities),  # Actual volatility used
            pl.Series(f"{self.target_name}_upper_barrier", upper_barriers),  # Upper barrier level
            pl.Series(f"{self.target_name}_lower_barrier", lower_barriers),  # Lower barrier level
            pl.Series(f"{self.target_name}_exit_price", exit_prices),  # Exit price
            pl.Series(f"{self.target_name}_exit_index", exit_indices),  # Exit index (relative to entry)
        ])

        return result_df

    def _compute_triple_barrier_labels_with_metadata(self, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute triple barrier labels WITH full plotting metadata.

        Returns:
            Tuple of (labels, returns, volatilities, upper_barriers, lower_barriers, exit_prices, exit_indices)
        """
        n_samples = len(prices)
        labels = np.zeros(n_samples, dtype=np.int32)
        metadata = np.zeros(n_samples, dtype=np.float32)

        # NEW: Store all plotting information
        volatilities = np.zeros(n_samples, dtype=np.float32)
        upper_barriers = np.zeros(n_samples, dtype=np.float32)
        lower_barriers = np.zeros(n_samples, dtype=np.float32)
        exit_prices = np.zeros(n_samples, dtype=np.float32)
        exit_indices = np.zeros(n_samples, dtype=np.int32)

        # Process each position
        for i in range(self.lookback_window, n_samples - self.lookforward_window):
            entry_price = prices[i]

            # Calculate and STORE volatility
            volatility = np.std(prices[i-self.lookback_window:i+1])
            volatilities[i] = volatility

            # Calculate and STORE barrier levels
            upper_threshold = entry_price + (volatility * self.barrier_width)
            lower_threshold = entry_price - (volatility * self.barrier_width)
            upper_barriers[i] = upper_threshold
            lower_barriers[i] = lower_threshold

            # Look ahead for barrier hits
            future_prices = prices[i+1:i+1+self.lookforward_window]

            # Find first barrier hit and STORE exit information
            upper_hits = np.where(future_prices >= upper_threshold)[0]
            lower_hits = np.where(future_prices <= lower_threshold)[0]

            if len(upper_hits) > 0 and len(lower_hits) > 0:
                # Both barriers hit - use the first one
                upper_time = upper_hits[0]
                lower_time = lower_hits[0]

                if upper_time < lower_time:
                    # Upper barrier hit first → LONG signal
                    labels[i] = 1
                    exit_idx = upper_time
                    exit_price = future_prices[upper_time]
                    realized_return = (exit_price - entry_price) / entry_price - self.transaction_cost
                elif lower_time < upper_time:
                    # Lower barrier hit first → SHORT signal
                    labels[i] = -1
                    exit_idx = lower_time
                    exit_price = future_prices[lower_time]
                    realized_return = (entry_price - exit_price) / entry_price - self.transaction_cost
                else:
                    # Simultaneous hit (rare) - treat as timeout
                    labels[i] = 0
                    exit_idx = len(future_prices) - 1
                    exit_price = future_prices[-1]
                    realized_return = abs(exit_price - entry_price) / entry_price - self.transaction_cost

            elif len(upper_hits) > 0:
                # Only upper barrier hit → LONG signal
                labels[i] = 1
                exit_idx = upper_hits[0]
                exit_price = future_prices[upper_hits[0]]
                realized_return = (exit_price - entry_price) / entry_price - self.transaction_cost

            elif len(lower_hits) > 0:
                # Only lower barrier hit → SHORT signal
                labels[i] = -1
                exit_idx = lower_hits[0]
                exit_price = future_prices[lower_hits[0]]
                realized_return = (entry_price - exit_price) / entry_price - self.transaction_cost

            else:
                # Time barrier hit (timeout) - no directional move
                labels[i] = 0
                exit_idx = len(future_prices) - 1
                exit_price = future_prices[-1]
                realized_return = (exit_price - entry_price) / entry_price - self.transaction_cost

            # Store exit information
            exit_prices[i] = exit_price
            exit_indices[i] = exit_idx  # Relative to entry point
            metadata[i] = realized_return

        return labels, metadata, volatilities, upper_barriers, lower_barriers, exit_prices, exit_indices

    def _compute_triple_barrier_labels(self, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute triple barrier labels for price series.

        Returns:
            Tuple of (labels, metadata) where:
            - labels: Array of barrier labels (+1, 0, -1)
            - metadata: Array of realized returns or probabilities
        """
        n_samples = len(prices)
        labels = np.zeros(n_samples, dtype=np.int32)
        metadata = np.zeros(n_samples, dtype=np.float32)

        # Process each position
        for i in range(self.lookback_window, n_samples - self.lookforward_window):
            entry_price = prices[i]

            threshold = np.std(prices[i-self.lookback_window:i+1])


            # Adjust barriers for current volatility
            # FIXED: Use absolute barriers, not percentage-based
            upper_threshold = entry_price + (threshold * self.barrier_width)
            lower_threshold = entry_price - (threshold * self.barrier_width)

            # Look ahead for barrier hits
            future_prices = prices[i+1:i+1+self.lookforward_window]

            # Find first barrier hit
            upper_hits = np.where(future_prices >= upper_threshold)[0]
            lower_hits = np.where(future_prices <= lower_threshold)[0]

            if len(upper_hits) > 0 and len(lower_hits) > 0:
                # Both barriers hit - use the first one
                upper_time = upper_hits[0]
                lower_time = lower_hits[0]

                if upper_time < lower_time:
                    # Upper barrier hit first → price moved UP → LONG signal
                    labels[i] = 1  # Go long (upward move detected)
                    exit_price = future_prices[upper_time]
                    # Long position return: profit from upward moves
                    realized_return = (exit_price - entry_price) / entry_price - self.transaction_cost
                elif lower_time < upper_time:
                    # Lower barrier hit first → price moved DOWN → SHORT signal
                    labels[i] = -1  # Go short (downward move detected)
                    exit_price = future_prices[lower_time]
                    # Short position return: profit from downward moves
                    realized_return = (entry_price - exit_price) / entry_price - self.transaction_cost
                else:
                    # Simultaneous hit (rare) - treat as time barrier
                    labels[i] = 0
                    # No directional signal - use final price change
                    final_price = future_prices[-1]
                    realized_return = abs(final_price - entry_price) / entry_price - self.transaction_cost

            elif len(upper_hits) > 0:
                # Only upper barrier hit → price moved UP → LONG signal
                labels[i] = 1  # Go long (upward move detected)
                exit_price = future_prices[upper_hits[0]]
                # Long position return: profit from upward moves
                realized_return = (exit_price - entry_price) / entry_price - self.transaction_cost

            elif len(lower_hits) > 0:
                # Only lower barrier hit → price moved DOWN → SHORT signal
                labels[i] = -1  # Go short (downward move detected)
                exit_price = future_prices[lower_hits[0]]
                # Short position return: profit from downward moves
                realized_return = (entry_price - exit_price) / entry_price - self.transaction_cost

            else:
                # Time barrier hit (timeout) - no significant directional move
                labels[i] = 0
                exit_price = future_prices[-1]
                # Timeout: calculate actual return based on final price change minus transaction cost
                realized_return = (exit_price - entry_price) / entry_price - self.transaction_cost

            metadata[i] = realized_return

        return labels, metadata


    def get_target_info(self) -> dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "classification",
            "description": f"Triple barrier method with {self.lookforward_window} tick window, "
                          f"±{self.barrier_width:.1%} barriers, {self.transaction_cost*10000:.1f}bp costs",
            "parameters": {
                "lookforward_window": self.lookforward_window,
                "lookback_window": self.lookback_window,
                "barrier_width": self.barrier_width,
                "transaction_cost": self.transaction_cost,
            }
        }

    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return (f"TripleBarrierGenerator(window={self.lookforward_window}, "
                f"barriers=±{self.barrier_width:.1%}, tc={self.transaction_cost*10000:.1f}bp)")
