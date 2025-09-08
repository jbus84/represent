"""
Regression Target Generators

This module provides target generators for regression tasks.
"""

from typing import Any

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
        winsorize_percentiles: tuple[float, float] = (0.1, 99.9),
        target_names: tuple[str, str] = ("mfe_buy_bps", "mfe_sell_bps"),
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

    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate directional MFE regression targets."""
        self.validate_input(df)

        # Create base target DataFrame with keys
        target_df = self._create_base_target_df(df, symbol)

        mid_prices = df["mid_price"].to_numpy()
        mfe_buy, mfe_sell = self._calculate_directional_mfe(mid_prices)

        # Apply winsorization
        mfe_buy = self._winsorize(mfe_buy, self.winsorize_percentiles)
        mfe_sell = self._winsorize(mfe_sell, self.winsorize_percentiles)

        # Add target columns
        target_df = target_df.with_columns([
            pl.Series(self.target_names[0], mfe_buy),
            pl.Series(self.target_names[1], mfe_sell)
        ])

        return target_df

    def _calculate_directional_mfe(self, mid_prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate directional MFE targets for both buy and sell sides."""
        n = len(mid_prices)
        mfe_buy = np.full(n, np.nan)
        mfe_sell = np.full(n, np.nan)

        for i in range(self.lookback_window, n - self.lookforward_horizon):
            # Entry price at current time i (actual entry point for MFE calculation)
            entry_price = mid_prices[i]

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

    def get_target_info(self) -> dict[str, Any]:
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
            },
        }

    @property
    def target_type(self) -> str:
        return "regression"

    @property
    def required_columns(self) -> list[str]:
        return ["mid_price"]


class LogReturnHorizonsGenerator(TargetGenerator):
    """
    Generates log return targets across multiple horizon windows.

    This generator calculates log returns over multiple lookforward horizons
    ranging from 1k to 5k ticks, providing a comprehensive view of price
    movements across different time scales.
    """

    def __init__(
        self,
        horizons: list[int] | None = None,
        lookback_window: int = 1000,
        target_prefix: str = "log_return",
    ):
        """
        Initialize log return horizons generator.

        Args:
            horizons: List of horizon windows in ticks (defaults to [1000, 2000, 3000, 4000, 5000])
            lookback_window: Size of lookback window for baseline
            target_prefix: Prefix for target column names
        """
        self.horizons = horizons or [1000, 2000, 3000, 4000, 5000]
        self.lookback_window = lookback_window
        self.target_prefix = target_prefix

        # Generate target names for each horizon
        self.target_names = [f"{target_prefix}_{horizon}t" for horizon in self.horizons]

    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate log return targets for all horizons."""
        self.validate_input(df)

        # Create base target DataFrame with keys
        target_df = self._create_base_target_df(df, symbol)

        mid_prices = df["mid_price"].to_numpy()

        # Calculate log returns for all horizons
        target_columns = []
        for i, horizon in enumerate(self.horizons):
            target_name = self.target_names[i]
            log_returns = self._calculate_log_returns(mid_prices, horizon)
            target_columns.append(pl.Series(target_name, log_returns))

        # Add all target columns
        target_df = target_df.with_columns(target_columns)

        return target_df

    def _calculate_log_returns(self, mid_prices: np.ndarray, horizon: int) -> np.ndarray:
        """Calculate log returns for a specific horizon in basis points."""
        n = len(mid_prices)
        log_returns = np.full(n, np.nan)

        max_lookforward = max(self.horizons)

        for i in range(self.lookback_window, n - max_lookforward):
            # Current price (end of lookback window)
            current_price = mid_prices[i]

            # Future price at specific horizon
            future_price = mid_prices[i + horizon]

            # Calculate log return in basis points
            if current_price > 0 and future_price > 0:
                log_return = np.log(future_price / current_price)
                log_returns[i] = log_return * 10000  # Convert to basis points

        return log_returns

    def get_target_info(self) -> dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": self.target_names,
            "target_type": "regression",
            "description": f"Log returns across {len(self.horizons)} horizon windows ({min(self.horizons)}-{max(self.horizons)} ticks)",
            "parameters": {
                "horizons": self.horizons,
                "lookback_window": self.lookback_window,
            },
        }

    @property
    def target_type(self) -> str:
        return "regression"

    @property
    def required_columns(self) -> list[str]:
        return ["mid_price"]


class PriceMovementGenerator(TargetGenerator):
    """
    Generates simple price movement regression targets.

    This generator calculates the percentage price change over a lookforward window.
    DEPRECATED: Use LogReturnHorizonsGenerator for log return based calculations.
    """

    def __init__(
        self,
        lookforward_window: int = 5000,
        lookback_window: int = 5000,
        target_name: str = "price_movement_bps",
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

    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate price movement regression targets."""
        self.validate_input(df)

        # Create base target DataFrame with keys
        target_df = self._create_base_target_df(df, symbol)

        mid_prices = df["mid_price"].to_numpy()
        price_movements = self._calculate_price_movements(mid_prices)

        # Add target column
        target_df = target_df.with_columns(pl.Series(self.target_name, price_movements))

        return target_df

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

    def get_target_info(self) -> dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "regression",
            "description": f"Price movement over {self.lookforward_window} tick window",
            "parameters": {
                "lookforward_window": self.lookforward_window,
                "lookback_window": self.lookback_window,
            },
        }

    @property
    def target_type(self) -> str:
        return "regression"

    @property
    def required_columns(self) -> list[str]:
        return ["mid_price"]


class VolatilityGenerator(TargetGenerator):
    """
    Generates volatility-based regression targets.

    This generator calculates rolling volatility over a lookforward window.
    """

    def __init__(self, window_size: int = 1000, target_name: str = "volatility_target"):
        """
        Initialize volatility generator.

        Args:
            window_size: Size of rolling window for volatility calculation
            target_name: Name of the target column to create
        """
        self.window_size = window_size
        self.target_name = target_name

    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate volatility regression targets."""
        self.validate_input(df)

        # Create base target DataFrame with keys
        target_df = self._create_base_target_df(df, symbol)

        mid_prices = df["mid_price"].to_numpy()
        volatility = self._calculate_rolling_volatility(mid_prices)

        # Add target column
        target_df = target_df.with_columns(pl.Series(self.target_name, volatility))

        return target_df

    def _calculate_rolling_volatility(self, mid_prices: np.ndarray) -> np.ndarray:
        """Calculate rolling volatility."""
        n = len(mid_prices)
        volatility = np.full(n, np.nan)

        # Calculate log returns
        log_returns = np.diff(np.log(mid_prices))

        for i in range(self.window_size, n):
            # Get returns for the window
            window_returns = log_returns[i - self.window_size : i]

            # Calculate volatility (standard deviation of returns)
            volatility[i] = np.std(window_returns) * 10000  # Convert to basis points

        return volatility

    def get_target_info(self) -> dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "regression",
            "description": f"Rolling volatility over {self.window_size} tick window",
            "parameters": {
                "window_size": self.window_size,
            },
        }

    @property
    def target_type(self) -> str:
        return "regression"

    @property
    def required_columns(self) -> list[str]:
        return ["mid_price"]


class CumulativeReturnsGenerator(TargetGenerator):
    """
    Generates cumulative returns regression targets.

    This generator calculates the cumulative sum of returns over a lookforward window.
    """

    def __init__(
        self, lookforward_samples: int = 3000, target_name: str = "cumulative_returns_bps"
    ):
        """
        Initialize cumulative returns generator.

        Args:
            lookforward_samples: Number of samples to accumulate returns over
            target_name: Name of the target column to create
        """
        self.lookforward_samples = lookforward_samples
        self.target_name = target_name

    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate cumulative returns regression targets."""
        self.validate_input(df)

        # Create base target DataFrame with keys
        target_df = self._create_base_target_df(df, symbol)

        mid_prices = df["mid_price"].to_numpy()
        cumulative_returns = self._calculate_cumulative_returns(mid_prices)

        # Add target column
        target_df = target_df.with_columns(pl.Series(self.target_name, cumulative_returns))

        return target_df

    def _calculate_cumulative_returns(self, mid_prices: np.ndarray) -> np.ndarray:
        """Calculate cumulative returns over lookforward window."""
        n = len(mid_prices)
        cumulative_returns = np.full(n, np.nan)

        # Skip calculation if we don't have enough data points
        if n < 2:
            return cumulative_returns

        # Calculate log returns, handling NaN values
        with np.errstate(divide="ignore", invalid="ignore"):
            log_returns = np.diff(np.log(mid_prices))

        # For each valid starting position
        for i in range(0, n - self.lookforward_samples - 1):
            # Get the next N returns starting from position i
            future_returns = log_returns[i : i + self.lookforward_samples]

            # Skip if we don't have enough future data or any non-finite values
            if len(future_returns) < self.lookforward_samples or not np.all(
                np.isfinite(future_returns)
            ):
                continue

            # Calculate cumulative sum of returns
            cumsum_returns = np.sum(future_returns)

            # Convert to basis points
            cumulative_returns[i] = cumsum_returns * 10000

        return cumulative_returns

    def get_target_info(self) -> dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "regression",
            "description": f"Cumulative returns over next {self.lookforward_samples} samples",
            "parameters": {
                "lookforward_samples": self.lookforward_samples,
            },
        }

    @property
    def target_type(self) -> str:
        return "regression"

    @property
    def required_columns(self) -> list[str]:
        return ["mid_price"]


class VolatilityScaledReturnsGenerator(TargetGenerator):
    """
    Generates volatility-scaled cumulative returns regression targets.

    This generator computes short-term realized volatility and sets dynamic
    stop-loss/take-profit barriers as multiples of volatility. It then calculates
    the realized PnL over a fixed horizon given those adaptive barriers.

    This approach is commonly used in FX trading for adaptive risk management.
    """

    def __init__(
        self,
        volatility_window: int = 500,
        vol_multiplier: float = 2.0,  # Default multiplier
        horizon_ticks: int = 2000,  # Default horizon
        min_barrier_bps: float = 3.0,  # Reasonable minimum to avoid tiny barriers
        target_name: str = "vol_scaled_returns_bps",
    ):
        """
        Initialize volatility-scaled returns generator.

        Args:
            volatility_window: Window size for computing realized volatility (in ticks)
            vol_multiplier: Multiplier for volatility to set stop/target levels
            horizon_ticks: Fixed tick count for evaluation horizon
            min_barrier_bps: Minimum barrier size in basis points to avoid noise
            target_name: Name of the target column to create
        """
        self.volatility_window = volatility_window
        self.vol_multiplier = vol_multiplier
        self.horizon_ticks = horizon_ticks
        self.min_barrier_bps = min_barrier_bps
        self.target_name = target_name

    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate volatility-scaled returns regression targets."""
        self.validate_input(df)

        # Create base target DataFrame with keys
        target_df = self._create_base_target_df(df, symbol)

        mid_prices = df["mid_price"].to_numpy()
        vol_scaled_returns = self._calculate_vol_scaled_returns(mid_prices)

        # Add target column
        target_df = target_df.with_columns(pl.Series(self.target_name, vol_scaled_returns))

        return target_df

    def _calculate_vol_scaled_returns(self, mid_prices: np.ndarray) -> np.ndarray:
        """Calculate volatility-scaled returns with adaptive barriers."""
        n = len(mid_prices)
        vol_scaled_returns = np.full(n, np.nan)

        # Skip calculation if we don't have enough data points
        if n < self.volatility_window + self.horizon_ticks + 1:
            return vol_scaled_returns

        # Calculate log returns for volatility estimation
        with np.errstate(divide="ignore", invalid="ignore"):
            log_returns = np.diff(np.log(mid_prices))

        # For each valid starting position
        for i in range(self.volatility_window, n - self.horizon_ticks - 1):
            entry_price = mid_prices[i]
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            # Compute short-term realized volatility using lookback window
            vol_start = i - self.volatility_window
            vol_returns = log_returns[vol_start:i]

            # Skip if volatility window contains NaN values
            if not np.all(np.isfinite(vol_returns)) or len(vol_returns) == 0:
                continue

            # Calculate realized volatility (standard deviation of returns)
            realized_vol = np.std(vol_returns)

            if not np.isfinite(realized_vol) or realized_vol <= 0:
                continue

            # Set dynamic stop-loss and take-profit levels based on volatility
            vol_threshold = self.vol_multiplier * realized_vol

            # Apply minimum barrier only if volatility-based threshold is very small
            min_threshold = self.min_barrier_bps / 10000.0
            if vol_threshold < min_threshold:
                # In very low volatility periods, use minimum threshold
                vol_threshold = min_threshold

            # Convert volatility threshold to price levels
            stop_loss_price = entry_price * np.exp(-vol_threshold)
            take_profit_price = entry_price * np.exp(vol_threshold)

            # Evaluate over the fixed horizon
            horizon_start = i + 1
            horizon_end = i + 1 + self.horizon_ticks
            future_prices = mid_prices[horizon_start:horizon_end]

            # Skip if horizon contains NaN values
            if not np.all(np.isfinite(future_prices)) or len(future_prices) == 0:
                continue

            # Track PnL evolution with dynamic barriers
            pnl_bps = self._calculate_barrier_pnl(
                entry_price, future_prices, stop_loss_price, take_profit_price
            )

            vol_scaled_returns[i] = pnl_bps

        return vol_scaled_returns

    def _calculate_barrier_pnl(
        self,
        entry_price: float,
        future_prices: np.ndarray,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> float:
        """
        Calculate realized PnL given dynamic stop-loss and take-profit barriers.

        The PnL is calculated as:
        1. If stop-loss is hit first: return the actual PnL at the breaching price
        2. If take-profit is hit first: return the actual PnL at the breaching price
        3. If neither is hit: return the final horizon PnL
        """

        # Check each price tick for barrier breaches
        for price in future_prices:
            # Check stop-loss breach (price falls below stop)
            if price <= stop_loss_price:
                # Return actual PnL at the breaching price in basis points
                return ((price - entry_price) / entry_price) * 10000

            # Check take-profit breach (price rises above target)
            if price >= take_profit_price:
                # Return actual PnL at the breaching price in basis points
                return ((price - entry_price) / entry_price) * 10000

        # Neither barrier hit - return final horizon PnL
        final_price = future_prices[-1]
        return ((final_price - entry_price) / entry_price) * 10000

    def get_target_info(self) -> dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "regression",
            "description": f"Volatility-scaled returns with {self.vol_multiplier}x vol barriers over {self.horizon_ticks} ticks",
            "parameters": {
                "volatility_window": self.volatility_window,
                "vol_multiplier": self.vol_multiplier,
                "horizon_ticks": self.horizon_ticks,
                "min_barrier_bps": self.min_barrier_bps,
            },
        }

    @property
    def target_type(self) -> str:
        return "regression"

    @property
    def required_columns(self) -> list[str]:
        return ["mid_price"]


class RemainingValueTunerGenerator(TargetGenerator):
    """
    Generates regression targets using remaining value tuning approach.

    This generator transforms discrete trend labels into continuous values that represent
    the remaining potential of each trend. For each time point, it calculates the difference
    between the current price and the maximum/minimum price reached by the end of the trend.

    Key features:
    - Uptrends: positive values indicating remaining upside potential
    - Downtrends: negative values indicating remaining downside potential
    - Neutral trends: values close to zero
    - Optional monotonicity enforcement to prevent trend reversals
    """

    def __init__(
        self,
        lookback_rows: int = 5000,
        lookforward_input: int = 3000,
        lookforward_offset: int = 500,
        trend_threshold_bps: float = 20.0,  # Minimum trend magnitude in bps
        neutral_factor: float = 0.5,  # Factor for neutral zone around zero movement
        enforce_monotonicity: bool = True,
        target_name: str = "remaining_value_tuned_bps",
    ):
        """
        Initialize remaining value tuner generator.

        Args:
            lookback_rows: Historical window for trend context
            lookforward_input: Forward window for trend evaluation
            lookforward_offset: Offset before forward window starts
            trend_threshold_bps: Minimum trend magnitude to classify as up/down trend
            neutral_factor: Factor for determining neutral zone size
            enforce_monotonicity: If True, prevent trend reversals in intervals
            target_name: Name of the target column to create
        """
        self.lookback_rows = lookback_rows
        self.lookforward_input = lookforward_input
        self.lookforward_offset = lookforward_offset
        self.trend_threshold_bps = trend_threshold_bps
        self.neutral_factor = neutral_factor
        self.enforce_monotonicity = enforce_monotonicity
        self.target_name = target_name

    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate remaining value tuned regression targets."""
        self.validate_input(df)

        # Create base target DataFrame with keys
        target_df = self._create_base_target_df(df, symbol)

        mid_prices = df["mid_price"].to_numpy()
        remaining_values = self._calculate_remaining_value_tuned(mid_prices)

        # Add target column
        target_df = target_df.with_columns(pl.Series(self.target_name, remaining_values))

        return target_df

    def _calculate_remaining_value_tuned(self, mid_prices: np.ndarray) -> np.ndarray:
        """Calculate remaining value tuned targets."""
        n = len(mid_prices)
        remaining_values = np.full(n, np.nan)

        # Skip calculation if we don't have enough data
        if n < self.lookback_rows + self.lookforward_input + self.lookforward_offset + 1:
            return remaining_values

        # Calculate for each valid position
        for i in range(self.lookback_rows, n - self.lookforward_input - self.lookforward_offset):
            # Get current price
            current_price = mid_prices[i]
            if not np.isfinite(current_price) or current_price <= 0:
                continue

            # Define forward price window
            forward_start = i + 1 + self.lookforward_offset
            forward_end = forward_start + self.lookforward_input
            future_prices = mid_prices[forward_start:forward_end]

            # Skip if future window contains invalid data
            if not np.all(np.isfinite(future_prices)) or len(future_prices) == 0:
                continue

            # Calculate trend remaining value
            remaining_value_bps = self._calculate_trend_remaining_value(
                current_price, future_prices
            )

            remaining_values[i] = remaining_value_bps

        # Apply monotonicity enforcement if requested
        if self.enforce_monotonicity:
            remaining_values = self._enforce_monotonicity(remaining_values)

        return remaining_values

    def _calculate_trend_remaining_value(
        self, current_price: float, future_prices: np.ndarray
    ) -> float:
        """
        Calculate the remaining value for a trend starting at current price.

        This represents how much potential movement remains from the current point
        to the maximum/minimum that will be achieved in the future window.
        """

        # Find maximum and minimum prices in the future window
        max_price = np.max(future_prices)
        min_price = np.min(future_prices)
        final_price = future_prices[-1]

        # Calculate potential moves in basis points
        max_upside_bps = ((max_price - current_price) / current_price) * 10000
        max_downside_bps = ((min_price - current_price) / current_price) * 10000
        final_move_bps = ((final_price - current_price) / current_price) * 10000

        # Determine trend classification based on dominant movement
        upside_magnitude = abs(max_upside_bps)
        downside_magnitude = abs(max_downside_bps)
        threshold = self.trend_threshold_bps

        # Calculate remaining value based on trend type
        if upside_magnitude > threshold and upside_magnitude > downside_magnitude:
            # Uptrend: return remaining upside potential
            # This is the total upside that will be achieved from this point
            return max_upside_bps

        elif downside_magnitude > threshold and downside_magnitude > upside_magnitude:
            # Downtrend: return remaining downside (negative values)
            # This is the total downside that will be achieved from this point
            return max_downside_bps

        else:
            # Neutral/sideways: return small value based on final outcome
            neutral_threshold = threshold * self.neutral_factor
            if abs(final_move_bps) < neutral_threshold:
                return final_move_bps * 0.2  # Small value for neutral trends
            else:
                # Weak trend - return partial final movement
                return final_move_bps * 0.6

    def _enforce_monotonicity(self, remaining_values: np.ndarray) -> np.ndarray:
        """
        Enforce monotonicity within trend intervals to prevent reversals.

        This ensures that within each trend interval, the remaining values
        decrease monotonically as we approach the trend's peak/trough.
        """
        # Create a copy to modify
        monotonic_values = remaining_values.copy()
        valid_mask = ~np.isnan(remaining_values)

        if not np.any(valid_mask):
            return monotonic_values

        # Apply simple smoothing to reduce noise while maintaining trend direction
        valid_indices = np.where(valid_mask)[0]
        valid_values = remaining_values[valid_mask]

        # Use a rolling median to smooth transitions while preserving trend characteristics
        window_size = min(30, len(valid_values) // 20)
        if window_size > 1:
            from scipy.ndimage import median_filter

            try:
                # Smooth using median filter to preserve trend direction
                smoothed_values = median_filter(valid_values, size=window_size, mode="nearest")
                for i, orig_idx in enumerate(valid_indices):
                    monotonic_values[orig_idx] = smoothed_values[i]
            except ImportError:
                # Fallback if scipy not available - use simple rolling average
                for i in range(len(valid_values)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(valid_values), i + window_size // 2 + 1)
                    window_values = valid_values[start_idx:end_idx]
                    monotonic_values[valid_indices[i]] = np.mean(window_values)

        return monotonic_values

    def get_target_info(self) -> dict[str, Any]:
        """Return metadata about this generator."""
        return {
            "target_names": [self.target_name],
            "target_type": "regression",
            "description": f"Remaining value tuned labels with {self.trend_threshold_bps} bps threshold, monotonic: {self.enforce_monotonicity}",
            "parameters": {
                "lookback_rows": self.lookback_rows,
                "lookforward_input": self.lookforward_input,
                "lookforward_offset": self.lookforward_offset,
                "trend_threshold_bps": self.trend_threshold_bps,
                "neutral_factor": self.neutral_factor,
                "enforce_monotonicity": self.enforce_monotonicity,
            },
        }

    @property
    def target_type(self) -> str:
        return "regression"

    @property
    def required_columns(self) -> list[str]:
        """Return list of required columns."""
        return ["mid_price"]
