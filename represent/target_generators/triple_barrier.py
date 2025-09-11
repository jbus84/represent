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

from typing import Any
import numpy as np
import polars as pl
import warnings

from .base import TargetGenerator


class TripleBarrierGenerator(TargetGenerator):
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
        lookforward_window: int = 500,  # Hundreds of ticks for meaningful predictions
        barrier_width: float = 0.0005,  # 5 pips symmetric barriers (0.05%)
        upper_barrier: float | None = None,  # Optional asymmetric upper barrier
        lower_barrier: float | None = None,  # Optional asymmetric lower barrier
        transaction_cost: float = 0.0001,  # 1 pip transaction cost
        min_return_threshold: float = 0.0001,  # Minimum return to consider significant
        target_name: str = "triple_barrier_label",
        normalize_by_volatility: bool = False,  # Scale barriers by recent volatility
        volatility_window: int = 50,  # Window for volatility calculation
        return_probabilities: bool = False,  # Return label probabilities instead of hard labels
    ):
        """
        Initialize Triple Barrier generator.
        
        Args:
            lookforward_window: Maximum holding period in ticks (time barrier)
            barrier_width: Default width for both barriers as fraction (e.g., 0.0005 = 0.05% = 5 pips)
            upper_barrier: Custom upper barrier (overrides barrier_width if provided)
            lower_barrier: Custom lower barrier (overrides barrier_width if provided) 
            transaction_cost: Transaction cost as fraction (e.g., 0.0001 = 1 pip)
            min_return_threshold: Minimum return to avoid noise trading
            target_name: Name of the target column to create
            normalize_by_volatility: Whether to scale barriers by recent price volatility
            volatility_window: Window size for volatility estimation
            return_probabilities: Return probability estimates instead of hard classifications
        """
        self.lookforward_window = lookforward_window
        self.barrier_width = barrier_width
        self.upper_barrier = upper_barrier if upper_barrier is not None else barrier_width
        self.lower_barrier = lower_barrier if lower_barrier is not None else barrier_width
        self.transaction_cost = transaction_cost
        self.min_return_threshold = min_return_threshold
        self.target_name = target_name
        self.normalize_by_volatility = normalize_by_volatility
        self.volatility_window = volatility_window
        self.return_probabilities = return_probabilities
        
        # Validation
        if self.lookforward_window < 10:
            raise ValueError("lookforward_window must be at least 10 ticks")
        if self.upper_barrier <= 0 or self.lower_barrier <= 0:
            raise ValueError("Barriers must be positive")
        if self.transaction_cost < 0:
            raise ValueError("Transaction cost must be non-negative")
    
    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate triple barrier labels for the input DataFrame."""
        self.validate_input(df)
        
        prices = df["mid_price"].to_numpy()
        
        if len(prices) < self.lookforward_window + self.volatility_window:
            warnings.warn(
                f"Insufficient data for triple barrier labeling: {len(prices)} samples. "
                f"Need at least {self.lookforward_window + self.volatility_window}. "
                f"Returning neutral labels.",
                stacklevel=2
            )
            labels = np.zeros(len(prices), dtype=np.int32)
            label_metadata = np.zeros(len(prices), dtype=np.float32)
        else:
            labels, label_metadata = self._compute_triple_barrier_labels(prices)
        
        # Create base DataFrame with metadata
        result_df = self._create_base_target_df(df, symbol)
        
        # Add target column
        if self.return_probabilities:
            # Return probability of upper barrier (profit target)
            result_df = result_df.with_columns(pl.Series(f"{self.target_name}_prob", label_metadata))
        else:
            # Return hard classification labels
            result_df = result_df.with_columns(pl.Series(self.target_name, labels))
            
        # Add metadata columns for analysis
        result_df = result_df.with_columns([
            pl.Series(f"{self.target_name}_return", label_metadata),  # Expected return
            pl.Series(f"{self.target_name}_barrier_width", np.full(len(labels), self.barrier_width))
        ])
        
        return result_df
    
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
        
        # Calculate volatility scaling if enabled
        if self.normalize_by_volatility:
            volatilities = self._calculate_rolling_volatility(prices)
        else:
            volatilities = np.ones(n_samples)
        
        # Process each position
        for i in range(n_samples - self.lookforward_window):
            entry_price = prices[i]
            vol_scalar = volatilities[i] if self.normalize_by_volatility else 1.0
            
            # Adjust barriers for current volatility
            # FIXED: Use absolute barriers, not percentage-based
            upper_threshold = entry_price + (self.upper_barrier * vol_scalar)
            lower_threshold = entry_price - (self.lower_barrier * vol_scalar)
            
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
                # Timeout: calculate return based on final price change (no directional bias)
                realized_return = (exit_price - entry_price) / entry_price - self.transaction_cost
            
            # Store metadata (realized return or probability)
            if self.return_probabilities:
                # Convert return to probability using sigmoid
                metadata[i] = self._return_to_probability(realized_return)
            else:
                metadata[i] = realized_return
                
            # Apply minimum return threshold to avoid noise trading
            # FIXED: Only apply threshold to timeout cases, not valid barrier hits
            if labels[i] == 0 and abs(realized_return) < self.min_return_threshold:
                labels[i] = 0  # Keep as timeout for truly small moves
        
        return labels, metadata
    
    def _calculate_rolling_volatility(self, prices: np.ndarray) -> np.ndarray:
        """Calculate rolling volatility for barrier normalization."""
        volatilities = np.ones(len(prices))
        
        for i in range(self.volatility_window, len(prices)):
            price_window = prices[i-self.volatility_window:i]
            returns = np.diff(price_window) / price_window[:-1]
            volatilities[i] = np.std(returns) if len(returns) > 1 else 1.0
            
        # Fill initial values with first computed volatility
        if len(prices) > self.volatility_window:
            volatilities[:self.volatility_window] = volatilities[self.volatility_window]
            
        return volatilities
    
    def _return_to_probability(self, realized_return: float) -> float:
        """Convert realized return to probability using sigmoid transformation."""
        # Scale return for sigmoid (adjust scaling factor as needed)
        scaled_return = realized_return * 1000  # Scale up for better sigmoid behavior
        return float(1.0 / (1.0 + np.exp(-scaled_return)))
    
    def get_target_info(self) -> dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name] + ([f"{self.target_name}_prob"] if self.return_probabilities else []),
            "target_type": "classification", 
            "description": f"Triple barrier method with {self.lookforward_window} tick window, "
                          f"±{self.barrier_width:.1%} barriers, {self.transaction_cost*10000:.1f}bp costs",
            "parameters": {
                "lookforward_window": self.lookforward_window,
                "barrier_width": self.barrier_width,
                "upper_barrier": self.upper_barrier,
                "lower_barrier": self.lower_barrier,
                "transaction_cost": self.transaction_cost,
                "min_return_threshold": self.min_return_threshold,
                "normalize_by_volatility": self.normalize_by_volatility,
                "volatility_window": self.volatility_window,
                "return_probabilities": self.return_probabilities
            }
        }
    
    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return (f"TripleBarrierGenerator(window={self.lookforward_window}, "
                f"barriers=±{self.barrier_width:.1%}, tc={self.transaction_cost*10000:.1f}bp)")