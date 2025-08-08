"""
High-performance market depth processing pipeline.
Optimized for <10ms array generation and zero-copy operations.
"""

import numpy as np
import polars as pl
from typing import Optional, Union

from .constants import (
    ASK_PRICE_COLUMNS,
    BID_PRICE_COLUMNS,
    ASK_VOL_COLUMNS,
    BID_VOL_COLUMNS,
    ASK_COUNT_COLUMNS,
    BID_COUNT_COLUMNS,
    ASK_ANCHOR_COLUMN,
    BID_ANCHOR_COLUMN,
    PRICE_LEVELS,
    TIME_BINS,
    VOLUME_DTYPE,
    FEATURE_TYPES,
    DEFAULT_FEATURES,
    FEATURE_INDEX_MAP,
    MAX_FEATURES,
    FeatureType,
    get_output_shape,
)
from .config import create_represent_config
from .data_structures import PriceLookupTable, VolumeGrid, OutputBuffer


class MarketDepthProcessor:
    """
    Ultra-high-performance market depth processor.
    Designed to meet <10ms array generation requirements.
    Now supports multiple feature types: volume, variance, and trade_counts.
    """

    def __init__(
        self, 
        features: Optional[Union[list[str], list[FeatureType]]] = None,
        currency: str = "AUDUSD"
    ):
        """Initialize processor with pre-allocated structures.

        Args:
            features: List of features to extract. Can be strings or FeatureType enums.
                     Options: 'volume', 'variance', 'trade_counts' or FeatureType enum values
                     Defaults to ['volume'] for backward compatibility.
            currency: Currency pair for configuration (used for micro_pip_size and ticks_per_bin)
        """
        # Load RepresentConfig for this currency
        self.config = create_represent_config(currency)
        
        # Pre-compute values for performance
        self.micro_pip_multiplier = 1.0 / self.config.micro_pip_size
        # Validate and set features
        if features is None:
            self.features: list[str] = DEFAULT_FEATURES.copy()
        else:
            # Convert FeatureType enums to strings if needed
            self.features = []
            for feature in features:
                if isinstance(feature, FeatureType):
                    self.features.append(feature.value)
                else:
                    # Assume it's a string if not FeatureType
                    self.features.append(str(feature))

        # Validate feature types
        invalid_features = set(self.features) - set(FEATURE_TYPES)
        if invalid_features:
            raise ValueError(
                f"Invalid features: {invalid_features}. Valid options: {FEATURE_TYPES}"
            )

        if len(self.features) == 0:
            raise ValueError("At least one feature must be specified")

        if len(self.features) > MAX_FEATURES:
            raise ValueError(f"Too many features: {len(self.features)}. Maximum: {MAX_FEATURES}")

        # Sort features by index for consistent ordering
        self.features = sorted(self.features, key=lambda f: FEATURE_INDEX_MAP[f])
        self.output_shape = get_output_shape(self.features)

        # Pre-allocate all data structures to avoid runtime allocations
        # One grid per feature type
        self._ask_grids = {feature: VolumeGrid() for feature in self.features}
        self._bid_grids = {feature: VolumeGrid() for feature in self.features}
        # One output buffer per feature to avoid sharing
        self._output_buffers = {feature: OutputBuffer() for feature in self.features}

        # Pre-allocate temporary arrays for processing (per feature)
        self._temp_ask_volumes: dict[str, np.ndarray] = {}
        self._temp_bid_volumes: dict[str, np.ndarray] = {}
        for feature in self.features:
            self._temp_ask_volumes[feature] = np.empty(
                (PRICE_LEVELS, TIME_BINS), dtype=VOLUME_DTYPE
            )
            self._temp_bid_volumes[feature] = np.empty(
                (PRICE_LEVELS, TIME_BINS), dtype=VOLUME_DTYPE
            )

        # Cache for price lookup table (will be created when needed)
        self._price_lookup: Optional[PriceLookupTable] = None
        self._cached_mid_price: float = 0.0

        # Pre-compile Polars expressions for performance
        self._price_conversion_expressions: Optional[list[pl.Expr]] = None
        self._time_bin_expression: Optional[pl.Expr] = None
        self._compiled_expressions_ready = False

    def _prepare_expressions(self) -> None:
        """Pre-compile Polars expressions for optimal performance."""
        if self._compiled_expressions_ready:
            return

        # Pre-compile price conversion expressions
        ask_exprs = [
            (pl.col(col) * self.micro_pip_multiplier).round().cast(pl.Int64).alias(col)
            for col in ASK_PRICE_COLUMNS
        ]
        bid_exprs = [
            (pl.col(col) * self.micro_pip_multiplier).round().cast(pl.Int64).alias(col)
            for col in BID_PRICE_COLUMNS
        ]
        self._price_conversion_expressions = ask_exprs + bid_exprs

        # Pre-compile time bin expression using config values
        expected_samples = self.config.samples * 2  # Use 2x config samples as the "standard" size
        self._time_bin_expression = (pl.int_range(0, expected_samples) // self.config.ticks_per_bin).alias("tick_bin")

        self._compiled_expressions_ready = True

    def _convert_prices_to_micro_pips(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert prices to integer micro-pip format with vectorized operations."""
        self._prepare_expressions()
        return df.with_columns(self._price_conversion_expressions)

    def _add_time_bins(self, df: pl.DataFrame, input_length: Optional[int] = None) -> pl.DataFrame:
        """Add time bin column using pre-compiled expression, adapting to input size."""
        if input_length is None:
            input_length = len(df)

        # For standard expected size, use pre-compiled expression
        expected_samples = self.config.samples * 2  # Same as used in _prepare_expressions
        if input_length == expected_samples:
            return df.with_columns(self._time_bin_expression)

        # For other sizes, create dynamic time bins
        ticks_per_bin = max(1, input_length // TIME_BINS)  # Ensure at least 1 tick per bin
        time_bin_expr = (pl.int_range(0, input_length) // ticks_per_bin).alias("tick_bin")
        return df.with_columns(time_bin_expr)

    def _calculate_mid_price(self, df: pl.DataFrame) -> float:
        """Calculate mid price from last row with minimal overhead."""
        last_row = df.tail(1)
        ask_price = last_row[ASK_ANCHOR_COLUMN][0]
        bid_price = last_row[BID_ANCHOR_COLUMN][0]
        return float((ask_price + bid_price) / 2)

    def _get_or_create_price_lookup(self, mid_price: float) -> PriceLookupTable:
        """Get cached price lookup table or create new one if mid price changed."""
        if self._price_lookup is None or abs(mid_price - self._cached_mid_price) > 0.5:
            self._price_lookup = PriceLookupTable(mid_price)
            self._cached_mid_price = mid_price
        return self._price_lookup

    def _process_side_data_vectorized(
        self,
        df: pl.DataFrame,
        price_columns: list[str],
        data_columns_map: dict[str, list[str]],
        lookup_table: PriceLookupTable,
        grids: dict[str, VolumeGrid],
    ) -> None:
        """Process ask or bid side data with full vectorization for multiple features.

        Args:
            df: Input DataFrame
            price_columns: List of price column names
            data_columns_map: Map of feature name to column names (e.g., {'volume': vol_cols, 'trade_counts': count_cols})
            lookup_table: Price lookup table
            grids: Map of feature name to VolumeGrid
        """
        # Clear all grids first
        for grid in grids.values():
            grid.clear()

        # Group prices by time bins using lazy evaluation (shared across features)
        grouped_prices = (
            df.lazy()
            .select([*price_columns, "tick_bin"])
            .group_by("tick_bin")
            .agg([pl.col(col).mean().floor().cast(pl.Int64) for col in price_columns])
            .sort("tick_bin")
            .collect()
        )

        # Convert prices to numpy arrays for vectorized processing
        prices_array = grouped_prices.select(price_columns).to_numpy()  # Shape: (time_bins, 10)

        # Vectorized price-to-index conversion (shared across features)
        prices_flat = prices_array.flatten()
        indices_flat = lookup_table.vectorized_lookup(prices_flat)
        indices_array = indices_flat.reshape(prices_array.shape)

        # Create coordinate arrays for valid mappings (shared)
        valid_mask = indices_array >= 0

        if valid_mask.any():
            # Get coordinates of valid entries (shared)
            time_indices, _ = np.where(valid_mask)
            y_coords = indices_array[valid_mask]
            x_coords = time_indices

            # Process each feature type separately
            for feature, data_columns in data_columns_map.items():
                if feature not in self.features:
                    continue

                grid = grids[feature]

                # Group data by time bins for this feature
                if feature == FeatureType.VOLUME.value:
                    # Use median for volume data
                    grouped_data = (
                        df.lazy()
                        .select([*data_columns, "tick_bin"])
                        .group_by("tick_bin")
                        .agg([pl.col(col).median() for col in data_columns])
                        .sort("tick_bin")
                        .collect()
                    )
                elif feature == FeatureType.TRADE_COUNTS.value:
                    # Use sum for trade counts
                    grouped_data = (
                        df.lazy()
                        .select([*data_columns, "tick_bin"])
                        .group_by("tick_bin")
                        .agg([pl.col(col).sum() for col in data_columns])
                        .sort("tick_bin")
                        .collect()
                    )
                elif feature == FeatureType.VARIANCE.value:
                    # For variance, calculate variance of volume data per time bin
                    # This follows the notebook implementation: .var() on volume columns
                    grouped_data = (
                        df.lazy()
                        .select([*data_columns, "tick_bin"])
                        .group_by("tick_bin")
                        .agg([pl.col(col).var() for col in data_columns])
                        .sort("tick_bin")
                        .collect()
                    )
                else:
                    continue

                # Convert to numpy for all features
                data_array = grouped_data.select(data_columns).to_numpy()  # Shape: (time_bins, 10)

                # Apply valid mask and set volumes in grid
                data_values = data_array[valid_mask]
                grid.set_volumes(y_coords, x_coords, data_values)

    def process(self, df: pl.DataFrame) -> np.ndarray:
        """
        Main processing pipeline optimized for <10ms execution.
        Now supports multiple feature extraction.

        Args:
            df: Input DataFrame with market data

        Returns:
            Feature array with shape determined by features:
            - Single feature: (402, 500)
            - Multiple features: (N, 402, 500) where N is number of features
        """
        # Validate input size - allow flexible sizes for conversion workflows
        input_length = len(df)
        if input_length < 500:  # Minimum for meaningful time bins
            raise ValueError(f"Input must have at least 500 samples, got {input_length}")

        # Step 1: Convert prices to micro-pips (vectorized)
        df_processed = self._convert_prices_to_micro_pips(df)

        # Step 2: Add time bins (pre-compiled expression) - adapt to input size
        df_processed = self._add_time_bins(df_processed, input_length)

        # Step 3: Calculate mid price and get lookup table
        mid_price = self._calculate_mid_price(df_processed)
        lookup_table = self._get_or_create_price_lookup(mid_price)

        # Step 4: Prepare data column mappings for each feature
        ask_data_columns: dict[str, list[str]] = {}
        bid_data_columns: dict[str, list[str]] = {}

        for feature in self.features:
            if feature == FeatureType.VOLUME.value:
                ask_data_columns[feature] = ASK_VOL_COLUMNS
                bid_data_columns[feature] = BID_VOL_COLUMNS
            elif feature == FeatureType.TRADE_COUNTS.value:
                ask_data_columns[feature] = ASK_COUNT_COLUMNS
                bid_data_columns[feature] = BID_COUNT_COLUMNS
            elif feature == FeatureType.VARIANCE.value:
                # Variance uses volume columns to calculate variance per time bin
                ask_data_columns[feature] = ASK_VOL_COLUMNS
                bid_data_columns[feature] = BID_VOL_COLUMNS

        # Step 5: Process ask side (vectorized, all features)
        self._process_side_data_vectorized(
            df_processed, ASK_PRICE_COLUMNS, ask_data_columns, lookup_table, self._ask_grids
        )

        # Step 6: Process bid side (vectorized, all features)
        self._process_side_data_vectorized(
            df_processed, BID_PRICE_COLUMNS, bid_data_columns, lookup_table, self._bid_grids
        )

        # Step 7: Calculate cumulative volumes and generate output for each feature
        if len(self.features) == 1:
            # Single feature: return 2D array (402, 500)
            feature = self.features[0]
            ask_cumulative = self._ask_grids[feature].get_cumulative_volume(reverse=False)
            bid_cumulative = self._bid_grids[feature].get_cumulative_volume(reverse=True)
            result = self._output_buffers[feature].compute_normalized_difference(
                ask_cumulative, bid_cumulative
            )
            return result.copy()
        else:
            # Multiple features: return 3D array (N, 402, 500)
            feature_arrays: list[np.ndarray] = []

            for feature in self.features:
                ask_cumulative = self._ask_grids[feature].get_cumulative_volume(reverse=False)
                bid_cumulative = self._bid_grids[feature].get_cumulative_volume(reverse=True)
                feature_result = self._output_buffers[feature].compute_normalized_difference(
                    ask_cumulative, bid_cumulative
                )
                feature_arrays.append(feature_result.copy())  # Copy to avoid buffer reuse issues

            # Stack features along first dimension
            stacked_result: np.ndarray = np.stack(feature_arrays, axis=0)
            return stacked_result


# Factory function for easy instantiation
def create_processor(
    features: Optional[Union[list[str], list[FeatureType]]] = None,
    currency: str = "AUDUSD"
) -> MarketDepthProcessor:
    """Create a new market depth processor instance.

    Args:
        features: List of features to extract. Can be strings or FeatureType enums.
                 Options: 'volume', 'variance', 'trade_counts' or FeatureType enum values
                 Defaults to ['volume'] for backward compatibility.
        currency: Currency pair for configuration (used for micro_pip_size and ticks_per_bin)
    """
    return MarketDepthProcessor(features=features, currency=currency)


# Main API function that matches the reference implementation interface
def process_market_data(
    df: pl.DataFrame, 
    features: Optional[Union[list[str], list[FeatureType]]] = None,
    currency: str = "AUDUSD"
) -> np.ndarray:
    """
    Process market data and return normalized depth representation.

    This function provides the same interface as the reference implementation
    but with significant performance optimizations and extended feature support.

    Args:
        df: Polars DataFrame with market data
        features: List of features to extract. Can be strings or FeatureType enums.
                 Options: 'volume', 'variance', 'trade_counts' or FeatureType enum values
                 Defaults to ['volume'] for backward compatibility.
        currency: Currency pair for configuration (used for micro_pip_size and ticks_per_bin)

    Returns:
        numpy array with normalized market depth:
        - Single feature: shape (402, 500)
        - Multiple features: shape (N, 402, 500) where N is number of features
    """
    processor = create_processor(features=features, currency=currency)
    return processor.process(df)
