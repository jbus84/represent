"""
Classification Target Generators

This module provides target generators for classification tasks.
"""

from typing import Dict, Any, List
import numpy as np
import polars as pl

from .base import TargetGenerator
from ..global_threshold_calculator import GlobalThresholds


class QuantileClassificationGenerator(TargetGenerator):
    """
    Generates uniform distribution classification targets using quantiles.
    
    This generator uses the first half of the data to calculate quantile boundaries,
    then applies those boundaries to the entire dataset to ensure uniform distribution.
    """
    
    def __init__(
        self,
        nbins: int = 13,
        lookforward_window: int = 5000,
        lookback_window: int = 5000,
        target_name: str = "classification_label"
    ):
        """
        Initialize quantile classification generator.
        
        Args:
            nbins: Number of classification bins
            lookforward_window: Size of lookforward window for price movement calculation
            lookback_window: Size of lookback window for baseline calculation
            target_name: Name of the target column to create
        """
        self.nbins = nbins
        self.lookforward_window = lookforward_window
        self.lookback_window = lookback_window
        self.target_name = target_name
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Generate quantile-based classification targets."""
        self.validate_input(df)
        
        # Calculate price movements
        price_movements = self._calculate_price_movements(df)
        
        # Use first half of data to define quantile boundaries
        first_half_size = len(price_movements) // 2
        if first_half_size < self.nbins * 10:  # Minimum samples per bin
            # Use all data if insufficient for reliable quantiles
            training_movements = price_movements
        else:
            training_movements = price_movements[:first_half_size]
        
        # Filter out NaN values for quantile calculation
        valid_training_movements = training_movements[~np.isnan(training_movements)]
        
        if len(valid_training_movements) == 0:
            # No valid movements - create dummy labels
            labels = np.zeros(len(price_movements), dtype=int)
            return {self.target_name: labels}
        
        # Calculate quantile boundaries for uniform distribution
        quantile_boundaries = np.quantile(
            valid_training_movements,
            np.linspace(0, 1, self.nbins + 1)
        )
        
        # Apply classification to all data
        labels = np.digitize(price_movements, quantile_boundaries[1:-1])
        labels = np.clip(labels, 0, self.nbins - 1)
        
        return {self.target_name: labels}
    
    def _calculate_price_movements(self, df: pl.DataFrame) -> np.ndarray:
        """Calculate price movements for classification."""
        mid_prices = df["mid_price"].to_numpy()
        price_movements = np.full(len(mid_prices), np.nan)
        
        for i in range(self.lookback_window, len(mid_prices) - self.lookforward_window):
            # Baseline: mean of lookback window
            baseline_start = i - self.lookback_window
            baseline_end = i
            baseline_mean = np.mean(mid_prices[baseline_start:baseline_end])
            
            # Target: mean of lookforward window
            target_start = i + 1
            target_end = i + 1 + self.lookforward_window
            target_mean = np.mean(mid_prices[target_start:target_end])
            
            # Price movement in basis points
            if baseline_mean > 0:
                price_movements[i] = ((target_mean - baseline_mean) / baseline_mean) * 10000
        
        return price_movements
    
    def get_target_info(self) -> Dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "classification",
            "description": f"Quantile-based classification with {self.nbins} uniform bins",
            "parameters": {
                "nbins": self.nbins,
                "lookforward_window": self.lookforward_window,
                "lookback_window": self.lookback_window,
            }
        }
    
    @property
    def target_type(self) -> str:
        return "classification"
    
    @property
    def required_columns(self) -> List[str]:
        return ["mid_price"]


class GlobalThresholdClassificationGenerator(TargetGenerator):
    """
    Uses pre-computed global thresholds for consistent classification across symbols.
    
    This generator ensures consistent classification boundaries across all symbols
    and datasets by using globally calculated thresholds.
    """
    
    def __init__(
        self,
        global_thresholds: GlobalThresholds,
        lookforward_window: int = 5000,
        lookback_window: int = 5000,
        target_name: str = "classification_label"
    ):
        """
        Initialize global threshold classification generator.
        
        Args:
            global_thresholds: Pre-computed global threshold boundaries
            lookforward_window: Size of lookforward window for price movement calculation
            lookback_window: Size of lookback window for baseline calculation
            target_name: Name of the target column to create
        """
        self.global_thresholds = global_thresholds
        self.lookforward_window = lookforward_window
        self.lookback_window = lookback_window
        self.target_name = target_name
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Generate classification targets using global thresholds."""
        self.validate_input(df)
        
        # Calculate price movements
        price_movements = self._calculate_price_movements(df)
        
        # Apply global thresholds
        labels = np.digitize(price_movements, self.global_thresholds.quantile_boundaries[1:-1])
        labels = np.clip(labels, 0, self.global_thresholds.nbins - 1)
        
        return {self.target_name: labels}
    
    def _calculate_price_movements(self, df: pl.DataFrame) -> np.ndarray:
        """Calculate price movements for classification."""
        mid_prices = df["mid_price"].to_numpy()
        price_movements = np.full(len(mid_prices), np.nan)
        
        for i in range(self.lookback_window, len(mid_prices) - self.lookforward_window):
            # Baseline: mean of lookback window
            baseline_start = i - self.lookback_window
            baseline_end = i
            baseline_mean = np.mean(mid_prices[baseline_start:baseline_end])
            
            # Target: mean of lookforward window
            target_start = i + 1
            target_end = i + 1 + self.lookforward_window
            target_mean = np.mean(mid_prices[target_start:target_end])
            
            # Price movement in basis points
            if baseline_mean > 0:
                price_movements[i] = ((target_mean - baseline_mean) / baseline_mean) * 10000
        
        return price_movements
    
    def get_target_info(self) -> Dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "classification",
            "description": f"Global threshold classification with {self.global_thresholds.nbins} bins",
            "parameters": {
                "nbins": self.global_thresholds.nbins,
                "lookforward_window": self.lookforward_window,
                "lookback_window": self.lookback_window,
                "global_thresholds": True,
            }
        }
    
    @property
    def target_type(self) -> str:
        return "classification"
    
    @property
    def required_columns(self) -> List[str]:
        return ["mid_price"]