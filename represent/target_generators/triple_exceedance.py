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

from typing import Any
import numpy as np
import polars as pl
import warnings

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
        lookforward_window: int = 200,  # Optimized for minimum effective length
        transaction_cost: float = 0.0001,  # 1 pip base transaction cost
        scaling_factor: float = 5.0,  # Barriers = transaction_cost × scaling_factor
        upper_scaling: float | None = None,  # Optional asymmetric upper scaling
        lower_scaling: float | None = None,  # Optional asymmetric lower scaling
        min_exceedance_threshold: float = 0.5,  # Minimum exceedance ratio to trigger
        target_name: str = "triple_exceedance_label",
        adaptive_scaling: bool = True,  # Scale barriers by recent volatility
        volatility_window: int = 50,  # Window for volatility-based scaling
        window_penalty_weight: float = 0.1,  # Weight for window length in optimization
        balance_weight: float = 0.2,  # Weight for class balance in optimization
        target_balance_ratio: float = 0.33,  # Target ratio for each class (0.33 = balanced)
        return_exceedance_ratio: bool = False,  # Return exceedance ratios instead of labels
    ):
        """
        Initialize Triple Exceedance generator.
        
        Args:
            lookforward_window: Maximum lookforward period (optimized for minimum)
            transaction_cost: Base transaction cost (e.g., 0.0001 = 1 pip)
            scaling_factor: Multiplier for transaction cost to create barriers
            upper_scaling: Custom upper barrier scaling (overrides scaling_factor)
            lower_scaling: Custom lower barrier scaling (overrides scaling_factor)
            min_exceedance_threshold: Minimum ratio of barrier that must be exceeded
            target_name: Name of the target column to create
            adaptive_scaling: Whether to adapt scaling to market volatility
            volatility_window: Window for volatility estimation
            window_penalty_weight: Penalty weight for longer windows in optimization
            balance_weight: Weight for class balance objective in multi-objective optimization
            target_balance_ratio: Target ratio for each class (0.33 = perfectly balanced)
            return_exceedance_ratio: Return continuous exceedance ratios instead of discrete labels
        """
        self.lookforward_window = lookforward_window
        self.transaction_cost = transaction_cost
        self.scaling_factor = scaling_factor
        self.upper_scaling = upper_scaling if upper_scaling is not None else scaling_factor
        self.lower_scaling = lower_scaling if lower_scaling is not None else scaling_factor
        self.min_exceedance_threshold = min_exceedance_threshold
        self.target_name = target_name
        self.adaptive_scaling = adaptive_scaling
        self.volatility_window = volatility_window
        self.window_penalty_weight = window_penalty_weight
        self.balance_weight = balance_weight
        self.target_balance_ratio = target_balance_ratio
        self.return_exceedance_ratio = return_exceedance_ratio
        
        # Validation
        if self.lookforward_window < 10:
            raise ValueError("lookforward_window must be at least 10 ticks")
        if self.transaction_cost <= 0:
            raise ValueError("transaction_cost must be positive")
        if self.scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive")
        if not 0 <= self.min_exceedance_threshold <= 1:
            raise ValueError("min_exceedance_threshold must be between 0 and 1")
    
    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate dual-sided binary exceedance labels for the input DataFrame."""
        self.validate_input(df)
        
        prices = df["mid_price"].to_numpy()
        
        if len(prices) < self.lookforward_window + self.volatility_window:
            warnings.warn(
                f"Insufficient data for triple exceedance labeling: {len(prices)} samples. "
                f"Need at least {self.lookforward_window + self.volatility_window}. "
                f"Returning all-fail labels.",
                stacklevel=2
            )
            long_labels = np.zeros(len(prices), dtype=np.int32)  # All fail
            short_labels = np.zeros(len(prices), dtype=np.int32)  # All fail
            exceedance_ratios = np.zeros(len(prices), dtype=np.float32)
        else:
            long_labels, short_labels, exceedance_ratios = self._compute_exceedance_labels(prices)
        
        # Create base DataFrame with metadata
        result_df = self._create_base_target_df(df, symbol)
        
        # Add DUAL TARGET COLUMNS (separate binary classifications)
        long_target_name = f"{self.target_name}_long"
        short_target_name = f"{self.target_name}_short"
        
        result_df = result_df.with_columns([
            pl.Series(long_target_name, long_labels),      # Long exceedance: 1=exceed, 0=fail
            pl.Series(short_target_name, short_labels),    # Short exceedance: 1=exceed, 0=fail
        ])
        
        # Add metadata columns for analysis and optimization
        result_df = result_df.with_columns([
            pl.Series(f"{self.target_name}_exceedance_ratio", exceedance_ratios),  # Combined strength
            pl.Series(f"{self.target_name}_long_threshold", 
                     np.full(len(long_labels), self.transaction_cost * self.upper_scaling)),
            pl.Series(f"{self.target_name}_short_threshold", 
                     np.full(len(short_labels), self.transaction_cost * self.lower_scaling))
        ])
        
        return result_df
    
    def _compute_exceedance_labels(self, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute exceedance labels using fixed-duration, dual-sided approach.
        
        CORRECTED LOGIC:
        - Fixed duration: Always hold for full lookforward window (no early exit)
        - Dual-sided: Generate separate long and short exceedance assessments
        - Binary classification: Exceed (1) or Fail (0) for each direction independently
        - Multi-objective: Optimize for minimal window + class balance + returns
        
        Returns:
            Tuple of (long_labels, short_labels, exceedance_ratios) where:
            - long_labels: Binary array for long exceedance (1=exceed, 0=fail)
            - short_labels: Binary array for short exceedance (1=exceed, 0=fail) 
            - exceedance_ratios: Combined exceedance strength for optimization
        """
        n_samples = len(prices)
        long_labels = np.zeros(n_samples, dtype=np.int32)  # Long exceedance: 1=exceed, 0=fail
        short_labels = np.zeros(n_samples, dtype=np.int32)  # Short exceedance: 1=exceed, 0=fail
        exceedance_ratios = np.zeros(n_samples, dtype=np.float32)  # For optimization
        
        # Calculate volatility scaling if enabled
        if self.adaptive_scaling:
            volatilities = self._calculate_rolling_volatility(prices)
        else:
            volatilities = np.ones(n_samples)
        
        # Process each position with FIXED DURATION
        for i in range(n_samples - self.lookforward_window):
            entry_price = prices[i]
            vol_scalar = volatilities[i] if self.adaptive_scaling else 1.0
            
            # Calculate transaction cost-scaled exceedance thresholds
            long_threshold = self.transaction_cost * self.upper_scaling * vol_scalar
            short_threshold = self.transaction_cost * self.lower_scaling * vol_scalar
            
            # FIXED DURATION: Look at ENTIRE window (no early exit)
            future_prices = prices[i+1:i+1+self.lookforward_window]
            
            if len(future_prices) == 0:
                continue
                
            # Calculate maximum moves in each direction over FULL window
            max_upward_move = np.max(future_prices) - entry_price  # Long profit potential  
            max_downward_move = entry_price - np.min(future_prices)  # Short profit potential
            
            # DUAL-SIDED BINARY CLASSIFICATION (INDEPENDENT):
            # Long exceedance: Does upward move exceed threshold?
            long_exceeds = max_upward_move >= long_threshold
            long_labels[i] = 1 if long_exceeds else 0
            
            # Short exceedance: Does downward move exceed threshold?
            short_exceeds = max_downward_move >= short_threshold  
            short_labels[i] = 1 if short_exceeds else 0
            
            # Calculate exceedance ratios for optimization (combined metric)
            long_exceedance_ratio = max_upward_move / long_threshold if long_threshold > 0 else 0
            short_exceedance_ratio = max_downward_move / short_threshold if short_threshold > 0 else 0
            
            # Store combined exceedance strength for multi-objective optimization
            if long_exceeds or short_exceeds:
                # At least one side succeeded - store the stronger ratio
                exceedance_ratios[i] = max(long_exceedance_ratio, short_exceedance_ratio)
            else:
                # Both failed - store max potential (scaled down)
                exceedance_ratios[i] = max(long_exceedance_ratio, short_exceedance_ratio) * 0.5
        
        return long_labels, short_labels, exceedance_ratios
    
    def _calculate_rolling_volatility(self, prices: np.ndarray) -> np.ndarray:
        """Calculate rolling volatility for adaptive barrier scaling."""
        volatilities = np.ones(len(prices))
        
        for i in range(self.volatility_window, len(prices)):
            price_window = prices[i-self.volatility_window:i]
            returns = np.diff(price_window) / price_window[:-1]
            volatilities[i] = max(np.std(returns), 0.0001) if len(returns) > 1 else 1.0
            
        # Fill initial values with first computed volatility
        if len(prices) > self.volatility_window:
            volatilities[:self.volatility_window] = volatilities[self.volatility_window]
            
        # Normalize volatilities (mean = 1.0)
        mean_vol = np.mean(volatilities)
        if mean_vol > 0:
            volatilities = volatilities / mean_vol
            
        return volatilities
    
    def calculate_fitness_score(self, prices: np.ndarray) -> float:
        """
        Calculate multi-objective fitness score: maximize returns, minimize window length, maximize class balance.
        
        This method optimizes for three objectives:
        1. Maximize expected returns (profitability)
        2. Minimize lookforward window length (time efficiency) 
        3. Maximize class balance (even distribution of labels)
        """
        # Generate labels for the price series
        labels, exceedance_ratios = self._compute_exceedance_labels(prices)
        
        # Objective 1: Calculate returns (maximize)
        returns = []
        for i in range(len(labels) - 1):
            if labels[i] != 0:  # Non-neutral label
                # Calculate forward return
                entry_price = prices[i]
                exit_price = prices[i+1] if i+1 < len(prices) else entry_price
                
                if labels[i] == 1:  # Long position
                    ret = (exit_price - entry_price) / entry_price - self.transaction_cost
                else:  # Short position
                    ret = (entry_price - exit_price) / entry_price - self.transaction_cost
                    
                returns.append(ret)
        
        if len(returns) == 0:
            return -1000.0  # Heavy penalty for no trades
        
        # Return component (scaled up for optimization)
        total_return = sum(returns)
        return_score = total_return * 1000  # Scale up for optimization
        
        # Objective 2: Window length penalty (minimize window)
        max_reasonable_window = 500  # Reference maximum
        window_penalty = (self.lookforward_window / max_reasonable_window) * self.window_penalty_weight
        
        # Objective 3: Class balance score (maximize balance)
        label_counts = {}
        for label in [-1, 0, 1]:
            label_counts[label] = np.sum(labels == label)
        
        total_labels = len(labels)
        if total_labels == 0:
            balance_score = -100  # Penalty
        else:
            # Calculate how close each class is to the target ratio
            target_count = total_labels * self.target_balance_ratio
            balance_deviations = []
            
            for label in [-1, 0, 1]:
                actual_count = label_counts[label]
                deviation = abs(actual_count - target_count) / target_count if target_count > 0 else 1
                balance_deviations.append(deviation)
            
            # Balance score: lower deviations = better balance
            avg_deviation = np.mean(balance_deviations)
            balance_score = max(0, 1 - avg_deviation)  # Score from 0 to 1, higher is better
            balance_score *= self.balance_weight
        
        # Combine all objectives
        # Maximize: return_score + balance_score
        # Minimize: window_penalty
        multi_objective_score = return_score + balance_score - window_penalty
        
        return float(multi_objective_score)
    
    def calculate_class_balance_metrics(self, labels: np.ndarray) -> dict:
        """
        Calculate detailed class balance metrics for analysis.
        
        Returns:
            Dict with balance metrics including entropy, deviation, etc.
        """
        label_counts = {}
        for label in [-1, 0, 1]:
            label_counts[label] = np.sum(labels == label)
        
        total_labels = len(labels)
        if total_labels == 0:
            return {"balance_score": 0, "entropy": 0, "max_deviation": 1}
        
        # Calculate proportions
        proportions = {label: count/total_labels for label, count in label_counts.items()}
        
        # Calculate entropy (higher = more balanced)
        entropy = 0
        for prop in proportions.values():
            if prop > 0:
                entropy -= prop * np.log2(prop)
        max_entropy = np.log2(3)  # Maximum entropy for 3 classes
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate deviation from target balance
        target_prop = self.target_balance_ratio
        deviations = [abs(proportions[label] - target_prop) for label in [-1, 0, 1]]
        max_deviation = max(deviations)
        avg_deviation = np.mean(deviations)
        
        # Balance score (0-1, higher is better)
        balance_score = max(0, 1 - avg_deviation / target_prop) if target_prop > 0 else 0
        
        return {
            "proportions": proportions,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "max_deviation": max_deviation,
            "avg_deviation": avg_deviation,
            "balance_score": balance_score,
            "label_counts": label_counts
        }
    
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
                          f"adaptive_scaling={self.adaptive_scaling}, "
                          f"multi-objective optimization (returns + window + balance)",
            "parameters": {
                "lookforward_window": self.lookforward_window,
                "transaction_cost": self.transaction_cost,
                "scaling_factor": self.scaling_factor,
                "upper_scaling": self.upper_scaling,
                "lower_scaling": self.lower_scaling,
                "min_exceedance_threshold": self.min_exceedance_threshold,
                "adaptive_scaling": self.adaptive_scaling,
                "volatility_window": self.volatility_window,
                "window_penalty_weight": self.window_penalty_weight,
                "balance_weight": self.balance_weight,
                "target_balance_ratio": self.target_balance_ratio,
                "return_exceedance_ratio": self.return_exceedance_ratio
            }
        }
    
    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return (f"TripleExceedanceGenerator(window={self.lookforward_window}, "
                f"scaling={self.scaling_factor}x, tc={self.transaction_cost*10000:.1f}bp, "
                f"adaptive={self.adaptive_scaling})")