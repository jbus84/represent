"""
TStrends-based Target Generators

This module provides target generators based on the tstrends library approaches
for trend labeling and classification. Each approach is wrapped as a modular
target generator that can be combined with other target types.

References:
- tstrends library: https://github.com/agpenas/tstrends
- Paper approaches: Cumulative Trend Labelling, Oracle labelling, etc.
"""

from typing import Dict, Any, List
import numpy as np
import polars as pl

try:
    from tstrends.trend_labelling import (
        BinaryCTL, 
        TernaryCTL, 
        OracleBinaryTrendLabeller, 
        OracleTernaryTrendLabeller
    )
    from tstrends.label_tuning import RemainingValueTuner
    TSTRENDS_AVAILABLE = True
except ImportError:
    TSTRENDS_AVAILABLE = False

from .base import TargetGenerator


class BinaryCTLGenerator(TargetGenerator):
    """
    Binary Cumulative Trend Labelling (CTL) target generator.
    
    This generator uses the Binary CTL approach from tstrends to create
    binary trend labels based on cumulative price movements.
    
    Reference: tstrends.trend_labelling.BinaryCTL
    """
    
    def __init__(
        self,
        omega: float = 0.02,
        target_name: str = "binary_ctl_label"
    ):
        """
        Initialize Binary CTL generator.
        
        Args:
            omega: Threshold parameter for trend detection
            target_name: Name of the target column to create
        """
        if not TSTRENDS_AVAILABLE:
            raise ImportError(
                "tstrends library is required for TStrends-based generators. "
                "Install with: uv add git+https://github.com/agpenas/tstrends.git"
            )
        
        self.omega = omega
        self.target_name = target_name
        self.labeller = BinaryCTL(omega=omega)
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Generate binary CTL targets."""
        self.validate_input(df)
        
        # Extract price series - tstrends expects list of lists with native Python types
        prices = df["mid_price"].to_numpy()
        
        # Ensure we have valid numeric data
        prices = prices[~np.isnan(prices)]
        if len(prices) == 0:
            return {self.target_name: np.array([], dtype=np.int32)}
        
        # Convert to list of native Python floats
        price_series_list = [float(p) for p in prices.tolist()]
        price_series_list = [price_series_list]  # Wrap in list for tstrends API
        
        # Generate labels using Binary CTL
        labels_list = self.labeller.get_labels(price_series_list)
        
        # Extract labels for our single series
        labels = labels_list[0] if len(labels_list) > 0 else np.zeros(len(prices), dtype=np.int32)
        
        # Ensure labels are integers and same length as input
        if len(labels) != len(prices):
            # Pad or truncate to match input length
            if len(labels) < len(prices):
                labels = np.pad(labels, (0, len(prices) - len(labels)), mode='constant', constant_values=0)
            else:
                labels = labels[:len(prices)]
        
        labels = labels.astype(np.int32)
        
        return {self.target_name: labels}
    
    def get_target_info(self) -> Dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "classification",
            "description": f"Binary Cumulative Trend Labelling with omega={self.omega}",
            "parameters": {
                "omega": self.omega,
                "approach": "Binary CTL",
                "library": "tstrends"
            }
        }
    
    @property
    def target_type(self) -> str:
        return "classification"
    
    @property
    def required_columns(self) -> List[str]:
        return ["mid_price"]


class TernaryCTLGenerator(TargetGenerator):
    """
    Ternary Cumulative Trend Labelling (CTL) target generator.
    
    This generator uses the Ternary CTL approach from tstrends to create
    three-class trend labels (up, down, sideways).
    
    Reference: tstrends.trend_labelling.TernaryCTL
    """
    
    def __init__(
        self,
        marginal_change_thres: float = 0.02,
        window_size: int = 10,
        target_name: str = "ternary_ctl_label"
    ):
        """
        Initialize Ternary CTL generator.
        
        Args:
            marginal_change_thres: Marginal change threshold
            window_size: Window size for trend detection
            target_name: Name of the target column to create
        """
        if not TSTRENDS_AVAILABLE:
            raise ImportError(
                "tstrends library is required for TStrends-based generators. "
                "Install with: uv add git+https://github.com/agpenas/tstrends.git"
            )
        
        self.marginal_change_thres = marginal_change_thres
        self.window_size = window_size
        self.target_name = target_name
        self.labeller = TernaryCTL(marginal_change_thres=marginal_change_thres, window_size=window_size)
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Generate ternary CTL targets."""
        self.validate_input(df)
        
        # Extract price series
        prices = df["mid_price"].to_numpy()
        
        # Generate labels using Ternary CTL
        labels = self.labeller.get_labels(prices)
        
        # Ensure labels are integers
        labels = labels.astype(np.int32)
        
        return {self.target_name: labels}
    
    def get_target_info(self) -> Dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "classification",
            "description": f"Ternary Cumulative Trend Labelling with threshold={self.marginal_change_thres}",
            "parameters": {
                "marginal_change_thres": self.marginal_change_thres,
                "window_size": self.window_size,
                "approach": "Ternary CTL",
                "library": "tstrends"
            }
        }
    
    @property
    def target_type(self) -> str:
        return "classification"
    
    @property
    def required_columns(self) -> List[str]:
        return ["mid_price"]


class OracleBinaryTrendGenerator(TargetGenerator):
    """
    Oracle Binary Trend Labelling target generator.
    
    This generator uses the Oracle Binary approach from tstrends which
    provides optimal binary trend labels based on future price knowledge.
    
    Reference: tstrends.trend_labelling.OracleBinaryTrendLabeller
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        target_name: str = "oracle_binary_label"
    ):
        """
        Initialize Oracle Binary generator.
        
        Args:
            transaction_cost: Transaction cost for oracle optimization
            target_name: Name of the target column to create
        """
        if not TSTRENDS_AVAILABLE:
            raise ImportError(
                "tstrends library is required for TStrends-based generators. "
                "Install with: uv add git+https://github.com/agpenas/tstrends.git"
            )
        
        self.transaction_cost = transaction_cost
        self.target_name = target_name
        self.labeller = OracleBinaryTrendLabeller(transaction_cost=transaction_cost)
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Generate oracle binary targets."""
        self.validate_input(df)
        
        # Extract price series
        prices = df["mid_price"].to_numpy()
        
        # Generate labels using Oracle Binary
        labels = self.labeller.get_labels(prices)
        
        # Ensure labels are integers
        labels = labels.astype(np.int32)
        
        return {self.target_name: labels}
    
    def get_target_info(self) -> Dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "classification",
            "description": "Oracle Binary Trend Labelling (optimal binary labels)",
            "parameters": {
                "approach": "Oracle Binary",
                "library": "tstrends",
                "optimal": True
            }
        }
    
    @property
    def target_type(self) -> str:
        return "classification"
    
    @property
    def required_columns(self) -> List[str]:
        return ["mid_price"]


class OracleTernaryTrendGenerator(TargetGenerator):
    """
    Oracle Ternary Trend Labelling target generator.
    
    This generator uses the Oracle Ternary approach from tstrends which
    provides optimal three-class trend labels based on future price knowledge.
    
    Reference: tstrends.trend_labelling.OracleTernaryTrendLabeller
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        neutral_reward_factor: float = 0.5,
        target_name: str = "oracle_ternary_label"
    ):
        """
        Initialize Oracle Ternary generator.
        
        Args:
            transaction_cost: Transaction cost for oracle optimization
            neutral_reward_factor: Neutral reward factor for ternary classification
            target_name: Name of the target column to create
        """
        if not TSTRENDS_AVAILABLE:
            raise ImportError(
                "tstrends library is required for TStrends-based generators. "
                "Install with: uv add git+https://github.com/agpenas/tstrends.git"
            )
        
        self.transaction_cost = transaction_cost
        self.neutral_reward_factor = neutral_reward_factor
        self.target_name = target_name
        self.labeller = OracleTernaryTrendLabeller(
            transaction_cost=transaction_cost,
            neutral_reward_factor=neutral_reward_factor
        )
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Generate oracle ternary targets."""
        self.validate_input(df)
        
        # Extract price series
        prices = df["mid_price"].to_numpy()
        
        # Generate labels using Oracle Ternary
        labels = self.labeller.get_labels(prices)
        
        # Ensure labels are integers
        labels = labels.astype(np.int32)
        
        return {self.target_name: labels}
    
    def get_target_info(self) -> Dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "classification",
            "description": "Oracle Ternary Trend Labelling (optimal ternary labels)",
            "parameters": {
                "approach": "Oracle Ternary",
                "library": "tstrends",
                "optimal": True
            }
        }
    
    @property
    def target_type(self) -> str:
        return "classification"
    
    @property
    def required_columns(self) -> List[str]:
        return ["mid_price"]


class TunedTrendGenerator(TargetGenerator):
    """
    Tuned Trend Labelling target generator.
    
    This generator uses the RemainingValueTuner from tstrends to optimize
    trend labelling parameters and generate tuned trend labels.
    
    Reference: tstrends.label_tuning.RemainingValueTuner
    """
    
    def __init__(
        self,
        base_labeller_type: str = "binary_ctl",
        omega: float = 0.02,
        target_name: str = "tuned_trend_label"
    ):
        """
        Initialize Tuned Trend generator.
        
        Args:
            base_labeller_type: Type of base labeller ("binary_ctl" or "ternary_ctl")
            omega: Initial threshold parameter
            target_name: Name of the target column to create
        """
        if not TSTRENDS_AVAILABLE:
            raise ImportError(
                "tstrends library is required for TStrends-based generators. "
                "Install with: uv add git+https://github.com/agpenas/tstrends.git"
            )
        
        self.base_labeller_type = base_labeller_type
        self.omega = omega
        self.target_name = target_name
        
        # Create base labeller
        if base_labeller_type == "binary_ctl":
            self.base_labeller = BinaryCTL(omega=omega)
        elif base_labeller_type == "ternary_ctl":
            self.base_labeller = TernaryCTL(omega=omega)
        else:
            raise ValueError(f"Unsupported base labeller type: {base_labeller_type}")
        
        self.tuner = RemainingValueTuner()
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Generate tuned trend targets."""
        self.validate_input(df)
        
        # Extract price series
        prices = df["mid_price"].to_numpy()
        
        # Use tuner to optimize and generate labels
        try:
            # The tuner might optimize parameters and return labels
            tuned_labels = self.tuner.tune(prices, self.base_labeller)
            
            # If tuner returns optimized labeller, get labels
            if hasattr(tuned_labels, 'get_labels'):
                labels = tuned_labels.get_labels(prices)
            else:
                # If tuner returns labels directly
                labels = tuned_labels
            
        except Exception:
            # Fallback to base labeller if tuning fails
            labels = self.base_labeller.get_labels(prices)
        
        # Ensure labels are integers
        labels = labels.astype(np.int32)
        
        return {self.target_name: labels}
    
    def get_target_info(self) -> Dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "classification",
            "description": f"Tuned {self.base_labeller_type} trend labelling",
            "parameters": {
                "base_labeller": self.base_labeller_type,
                "omega": self.omega,
                "approach": "Tuned Trend Labelling",
                "library": "tstrends"
            }
        }
    
    @property
    def target_type(self) -> str:
        return "classification"
    
    @property
    def required_columns(self) -> List[str]:
        return ["mid_price"]