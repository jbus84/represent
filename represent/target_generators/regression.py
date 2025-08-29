"""
Regression Target Generators

This module provides target generators for regression tasks.
"""

from typing import Dict, Any, List
import numpy as np
import polars as pl

from .base import TargetGenerator


class DirectionalMFEGenerator(TargetGenerator):
    """
    Generates buy-side and sell-side MFE (Maximum Favorable Excursion) regression targets.
    
    This generator calculates the maximum favorable price movement for both long and short
    positions over a configurable lookforward horizon.
    """
    
    def __init__(
        self,
        lookforward_horizon: int = 3000,
        lookback_window: int = 200,
        expected_fee_pips: float = 0.7,
        winsorize_percentiles: tuple[float, float] = (1.0, 99.0),
        target_names: tuple[str, str] = ("mfe_buy_bps", "mfe_sell_bps")
    ):
        """
        Initialize directional MFE generator.
        
        Args:
            lookforward_horizon: Forward horizon for MFE calculation (in ticks)
            lookback_window: Lookback window for noise smoothing (in ticks)
            expected_fee_pips: Expected trading fee in pips
            winsorize_percentiles: Percentiles for winsorization (lower, upper)
            target_names: Names for buy and sell target columns
        """
        self.lookforward_horizon = lookforward_horizon
        self.lookback_window = lookback_window
        self.expected_fee_pips = expected_fee_pips
        self.winsorize_percentiles = winsorize_percentiles
        self.target_names = target_names
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Generate directional MFE regression targets."""
        self.validate_input(df)
        
        mid_prices = df["mid_price"].to_numpy()
        mfe_buy, mfe_sell = self._calculate_directional_mfe(mid_prices)
        
        # Apply winsorization
        mfe_buy = self._winsorize(mfe_buy, self.winsorize_percentiles)
        mfe_sell = self._winsorize(mfe_sell, self.winsorize_percentiles)
        
        return {
            self.target_names[0]: mfe_buy,
            self.target_names[1]: mfe_sell
        }
    
    def _calculate_directional_mfe(self, mid_prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate directional MFE targets for both buy and sell sides."""
        n = len(mid_prices)
        mfe_buy = np.full(n, np.nan)
        mfe_sell = np.full(n, np.nan)
        
        for i in range(self.lookback_window, n - self.lookforward_horizon):
            # Smoothed entry price using lookback window
            entry_start = i - self.lookback_window
            entry_price = np.mean(mid_prices[entry_start:i])
            
            # Future prices over lookforward horizon
            future_start = i + 1
            future_end = i + 1 + self.lookforward_horizon
            future_prices = mid_prices[future_start:future_end]
            
            # Buy-side MFE: maximum favorable excursion for long position
            max_favorable_buy = np.max(future_prices)
            mfe_buy_raw = ((max_favorable_buy - entry_price) / entry_price) * 10000  # BPS
            mfe_buy[i] = mfe_buy_raw - self.expected_fee_pips * 10  # Subtract fees (pips to BPS)
            
            # Sell-side MFE: maximum favorable excursion for short position
            min_favorable_sell = np.min(future_prices)
            mfe_sell_raw = ((entry_price - min_favorable_sell) / entry_price) * 10000  # BPS
            mfe_sell[i] = mfe_sell_raw - self.expected_fee_pips * 10  # Subtract fees (pips to BPS)
        
        return mfe_buy, mfe_sell
    
    def _winsorize(self, data: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
        """Apply winsorization to handle outliers."""
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data
        
        valid_data = data[valid_mask]
        lower_bound = np.percentile(valid_data, percentiles[0])
        upper_bound = np.percentile(valid_data, percentiles[1])
        
        winsorized = data.copy()
        winsorized[valid_mask] = np.clip(valid_data, lower_bound, upper_bound)
        
        return winsorized
    
    def get_target_info(self) -> Dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": list(self.target_names),
            "target_type": "regression",
            "description": f"Directional MFE targets over {self.lookforward_horizon} tick horizon",
            "parameters": {
                "lookforward_horizon": self.lookforward_horizon,
                "lookback_window": self.lookback_window,
                "expected_fee_pips": self.expected_fee_pips,
                "winsorize_percentiles": self.winsorize_percentiles,
            }
        }
    
    @property
    def target_type(self) -> str:
        return "regression"
    
    @property
    def required_columns(self) -> List[str]:
        return ["mid_price"]


class PriceMovementGenerator(TargetGenerator):
    """
    Generates simple price movement regression targets.
    
    This generator calculates the percentage price change over a lookforward window.
    """
    
    def __init__(
        self,
        lookforward_window: int = 5000,
        lookback_window: int = 5000,
        target_name: str = "price_movement_bps"
    ):
        """
        Initialize price movement generator.
        
        Args:
            lookforward_window: Size of lookforward window
            lookback_window: Size of lookback window for baseline
            target_name: Name of the target column to create
        """
        self.lookforward_window = lookforward_window
        self.lookback_window = lookback_window
        self.target_name = target_name
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Generate price movement regression targets."""
        self.validate_input(df)
        
        mid_prices = df["mid_price"].to_numpy()
        price_movements = self._calculate_price_movements(mid_prices)
        
        return {self.target_name: price_movements}
    
    def _calculate_price_movements(self, mid_prices: np.ndarray) -> np.ndarray:
        """Calculate price movements in basis points."""
        n = len(mid_prices)
        price_movements = np.full(n, np.nan)
        
        for i in range(self.lookback_window, n - self.lookforward_window):
            # Baseline: mean of lookback window
            baseline_start = i - self.lookback_window
            baseline_mean = np.mean(mid_prices[baseline_start:i])
            
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
            "target_type": "regression",
            "description": f"Price movement over {self.lookforward_window} tick window",
            "parameters": {
                "lookforward_window": self.lookforward_window,
                "lookback_window": self.lookback_window,
            }
        }
    
    @property
    def target_type(self) -> str:
        return "regression"
    
    @property
    def required_columns(self) -> List[str]:
        return ["mid_price"]


class VolatilityGenerator(TargetGenerator):
    """
    Generates volatility-based regression targets.
    
    This generator calculates rolling volatility over a lookforward window.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        target_name: str = "volatility_target"
    ):
        """
        Initialize volatility generator.
        
        Args:
            window_size: Size of rolling window for volatility calculation
            target_name: Name of the target column to create
        """
        self.window_size = window_size
        self.target_name = target_name
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility regression targets."""
        self.validate_input(df)
        
        mid_prices = df["mid_price"].to_numpy()
        volatility = self._calculate_rolling_volatility(mid_prices)
        
        return {self.target_name: volatility}
    
    def _calculate_rolling_volatility(self, mid_prices: np.ndarray) -> np.ndarray:
        """Calculate rolling volatility."""
        n = len(mid_prices)
        volatility = np.full(n, np.nan)
        
        # Calculate log returns
        log_returns = np.diff(np.log(mid_prices))
        
        for i in range(self.window_size, n):
            # Get returns for the window
            window_returns = log_returns[i-self.window_size:i]
            
            # Calculate volatility (standard deviation of returns)
            volatility[i] = np.std(window_returns) * 10000  # Convert to basis points
        
        return volatility
    
    def get_target_info(self) -> Dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "regression", 
            "description": f"Rolling volatility over {self.window_size} tick window",
            "parameters": {
                "window_size": self.window_size,
            }
        }
    
    @property
    def target_type(self) -> str:
        return "regression"
    
    @property
    def required_columns(self) -> List[str]:
        return ["mid_price"]