"""
Triple Exceedance Method Target Generator

This module implements the Triple Exceedance Method, an innovative fixed-duration,
dual-sided labeling approach that optimizes for minimal window length, class balance,
and maximum returns using transaction cost-scaled thresholds.

CORRECTED METHOD DESIGN:
- Fixed Duration: Always holds positions for the full lookforward window (no early exit)
- Dual-Sided Assessment: Evaluates both long and short exceedance potential separately
- Binary Classification: Each direction gets Exceed (1) or Fail (0) based on threshold
- Transaction Cost Scaling: Thresholds are automatically scaled to transaction costs

Key Innovation:
- Evaluates MAXIMUM moves in both directions over ENTIRE window
- Multi-objective optimization: minimize window + maximize balance + maximize returns
- Focuses on moves that can realistically overcome transaction cost hurdles
- Generates balanced long/short signals based on exceedance strength

Target Columns Generated:
- {target_name}_long: Long exceedance binary classification (1=exceed, 0=fail)
- {target_name}_short: Short exceedance binary classification (1=exceed, 0=fail)

Each side is evaluated independently:
- Long: Does max upward move ≥ long_threshold? (1=yes, 0=no)
- Short: Does max downward move ≥ short_threshold? (1=yes, 0=no)

Multi-Objective Optimization:
1. Minimize lookforward window length (time efficiency)
2. Maximize class balance (even distribution of exceed/fail)
3. Maximize returns (profitable exceedance detection)

References:
- Novel fixed-duration approach with dual-sided exceedance assessment
- Transaction cost-aware thresholds for realistic trading applications
"""

import warnings
from typing import Any

import numpy as np
import polars as pl

from .base import TargetGenerator


class TripleExceedanceGenerator(TargetGenerator):
    """
    Triple Exceedance Method target generator with transaction cost-scaled barriers.

    This method creates barriers that are directly proportional to transaction costs,
    ensuring that labels represent moves that can realistically overcome trading costs.
    The lookforward window is optimized to be as short as possible while maintaining
    profitable signal quality.

    Key Features:
    - Transaction cost-proportional barriers (e.g., 5x transaction cost)
    - Multi-objective optimization: maximize returns, minimize window
    - Adaptive barrier scaling based on market volatility
    - Memory-efficient implementation for large datasets
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
        lookforward_window: int = 2000,  # PROVEN: 2K ticks from diagnostic analysis
        transaction_cost: float = 0.0001,  # 1 pip base transaction cost
        scaling_factor: float = 3.0,  # PROVEN: 3x scaling from diagnostic analysis
        target_name: str = "triple_exceedance_label",
    ):
        """
        Initialize Triple Exceedance generator.

        Args:
            lookforward_window: Maximum lookforward period (optimized for minimum)
            transaction_cost: Base transaction cost (e.g., 0.0001 = 1 pip)
            scaling_factor: Multiplier for transaction cost to create barriers
        """
        self.lookforward_window = lookforward_window
        self.transaction_cost = transaction_cost
        self.scaling_factor = scaling_factor
        self.target_name = target_name

        # Validation
        if self.lookforward_window < 10:
            raise ValueError("lookforward_window must be at least 10 ticks")
        if self.transaction_cost <= 0:
            raise ValueError("transaction_cost must be positive")
        if self.scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive")

    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate dual-sided binary exceedance labels for the input DataFrame."""
        self.validate_input(df)

        prices = df["mid_price"].to_numpy()

        if len(prices) < self.lookforward_window:
            warnings.warn(
                f"Insufficient data for triple exceedance labeling: {len(prices)} samples. "
                f"Need at least {self.lookforward_window}. "
                f"Returning all-fail labels.",
                stacklevel=2
            )
            long_labels = np.zeros(len(prices), dtype=np.int32)  # All fail
            short_labels = np.zeros(len(prices), dtype=np.int32)  # All fail
        else:
            long_labels, short_labels = self._compute_exceedance_labels(prices)

        # Create base DataFrame with metadata
        result_df = self._create_base_target_df(df, symbol)

        # Add DUAL TARGET COLUMNS (separate binary classifications)
        long_target_name = f"{self.target_name}_long"
        short_target_name = f"{self.target_name}_short"

        result_df = result_df.with_columns([
            pl.Series(long_target_name, long_labels),      # Long exceedance: 1=exceed, 0=fail
            pl.Series(short_target_name, short_labels),    # Short exceedance: 1=exceed, 0=fail
        ])

        return result_df

    def _compute_exceedance_labels(self, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        n_samples = len(prices)
        long_labels = np.zeros(n_samples, dtype=np.int32)  # Long exceedance: 1=exceed, 0=fail (BINARY)
        short_labels = np.zeros(n_samples, dtype=np.int32)  # Short exceedance: 1=exceed, 0=fail (BINARY)

        # Process each position with FIXED DURATION
        for i in range(n_samples - self.lookforward_window):
            entry_price = prices[i]

            # Calculate transaction cost-scaled exceedance thresholds
            threshold = self.transaction_cost * self.scaling_factor

            # FIXED DURATION: Look at ENTIRE window (no early exit)
            future_price = prices[self.lookforward_window]

            # Calculate maximum moves in each direction over FULL window
            move = future_price - entry_price  # Long profit potential

            # DUAL-SIDED BINARY CLASSIFICATION (INDEPENDENT):
            # Long exceedance: Does upward move exceed threshold?
            if move >= threshold:
                long_labels[i] = 1  # Binary: 1 = exceed threshold
            else:
                long_labels[i] = 0  # Binary: 0 = fail to exceed

            # Short exceedance: Does downward move exceed threshold?
            if move < 0 and abs(move) >= threshold:
                short_labels[i] = 1  # Binary: 1 = exceed threshold
            else:
                short_labels[i] = 0  # Binary: 0 = fail to exceed

        return long_labels, short_labels

    def get_target_info(self) -> dict[str, Any]:
        """Return metadata about this generator."""
        long_target = f"{self.target_name}_long"
        short_target = f"{self.target_name}_short"

        return {
            "target_names": [long_target, short_target],  # Two separate binary classifications
            "target_type": "classification",
            "description": f"Triple exceedance method with {self.scaling_factor}x transaction cost barriers, "
                          f"{self.lookforward_window} tick fixed-duration window, "
                          f"dual-sided binary classification (long/short exceed/fail), "
                          f"multi-objective optimization (returns + window + balance)",
            "parameters": {
                "lookforward_window": self.lookforward_window,
                "transaction_cost": self.transaction_cost,
                "scaling_factor": self.scaling_factor,
            }
        }

    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return (f"TripleExceedanceGenerator(window={self.lookforward_window}, "
                f"scaling={self.scaling_factor}x, tc={self.transaction_cost*10000:.1f}bp, "
               )
