"""
High-performance market depth processing pipeline.
Optimized for <10ms array generation and zero-copy operations.
"""

from typing import cast

import numpy as np
import polars as pl

from .configs import MarketDepthProcessorConfig
from .constants import (
    ASK_ANCHOR_COLUMN,
    ASK_COUNT_COLUMNS,
    ASK_PRICE_COLUMNS,
    ASK_VOL_COLUMNS,
    BID_ANCHOR_COLUMN,
    BID_COUNT_COLUMNS,
    BID_PRICE_COLUMNS,
    BID_VOL_COLUMNS,
    FEATURE_INDEX_MAP,
    FEATURE_TYPES,
    MAX_FEATURES,
    VOLUME_DTYPE,
    FeatureType,
    get_output_shape,
)
from .data_structures import OutputBuffer, VolumeGrid


class MarketDepthProcessor:
    """
    Ultra-high-performance market depth processor.
    Designed to meet <10ms array generation requirements.
    Now supports multiple feature types: volume, variance, and trade_counts.
    """

    def __init__(
        self,
        config: MarketDepthProcessorConfig | None = None,
        features: list[str] | list[FeatureType] | None = None,
        # Legacy support
        legacy_config=None,
    ):
        """Initialize processor with pre-allocated structures.

        Args:
            config: MarketDepthProcessorConfig with focused configuration (new preferred way)
            features: List of features to extract (overrides config.features if provided)
            legacy_config: Legacy RepresentConfig for backward compatibility
        """
        # Handle legacy usage - legacy_config should now be a tuple from create_represent_config
        if config is None and legacy_config is not None:
            if isinstance(legacy_config, tuple) and len(legacy_config) == 3:
                # New style: tuple of (DatasetBuilderConfig, GlobalThresholdConfig, MarketDepthProcessorConfig)
                config = legacy_config[2]  # Use MarketDepthProcessorConfig
            else:
                # Very old style - create default config
                config = MarketDepthProcessorConfig()

        # Default config if none provided
        if config is None:
            config = MarketDepthProcessorConfig()

        self.config = config

        # Pre-compute values for performance
        self.micro_pip_multiplier = 1.0 / self.config.micro_pip_size
        # Validate and set features (parameters override config)
        if features is None:
            # Use features from config
            self.features: list[str] = self.config.features.copy()
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

        # Cache ladder geometry
        self.price_range = self.config.price_range
        self.grid_price_levels = cast(int, self.config.price_levels)
        self.collapse_stride = cast(int, self.config.collapse_stride_value)
        self.output_price_levels = cast(int, self.config.effective_price_levels)

        bin_groups = cast(tuple[tuple[int, ...], ...] | None, self.config.bin_groups)

        self._collapse_groups: list[np.ndarray] | None

        if bin_groups is not None:
            self._collapse_groups = [
                np.asarray(group, dtype=np.int32) for group in bin_groups
            ]
            self.collapse_stride = 1
        elif self.config.target_price_levels is not None:
            target_levels = cast(int, self.config.target_price_levels)
            target_range = (target_levels - 2) // 2
            indices = np.arange(self.price_range)
            self._collapse_groups = [
                np.asarray(group, dtype=np.int32) for group in np.array_split(indices, target_range)
            ]
            self.collapse_stride = 1
        else:
            self._collapse_groups = None

        # Calculate time_bins from config
        time_bins = self.config.samples // self.config.ticks_per_bin
        self.output_shape = get_output_shape(
            self.features, time_bins=time_bins, price_levels=self.output_price_levels
        )

        # Pre-allocate all data structures to avoid runtime allocations
        # One grid per feature type
        self._ask_grids = {
            feature: VolumeGrid(time_bins=time_bins, price_levels=self.grid_price_levels)
            for feature in self.features
        }
        self._bid_grids = {
            feature: VolumeGrid(time_bins=time_bins, price_levels=self.grid_price_levels)
            for feature in self.features
        }
        # One output buffer per feature to avoid sharing
        self._output_buffers = {
            feature: OutputBuffer(
                time_bins=time_bins, price_levels=self.output_price_levels
            )
            for feature in self.features
        }

        # Pre-allocate temporary arrays for processing (per feature)
        self._temp_ask_volumes: dict[str, np.ndarray] = {}
        self._temp_bid_volumes: dict[str, np.ndarray] = {}
        for feature in self.features:
            self._temp_ask_volumes[feature] = np.empty(
                (self.grid_price_levels, time_bins), dtype=VOLUME_DTYPE
            )
            self._temp_bid_volumes[feature] = np.empty(
                (self.grid_price_levels, time_bins), dtype=VOLUME_DTYPE
            )

        # Pre-compile Polars expressions for performance
        self._price_conversion_expressions: list[pl.Expr] | None = None
        self._time_bin_expression: pl.Expr | None = None
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
        self._time_bin_expression = (
            pl.int_range(0, expected_samples) // self.config.ticks_per_bin
        ).alias("tick_bin")

        self._compiled_expressions_ready = True

    def _convert_prices_to_micro_pips(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert prices to integer micro-pip format with vectorized operations."""
        self._prepare_expressions()
        return df.with_columns(self._price_conversion_expressions)

    def _add_time_bins(self, df: pl.DataFrame, input_length: int | None = None) -> pl.DataFrame:
        """Add time bin column using pre-compiled expression, adapting to input size."""
        if input_length is None:
            input_length = len(df)

        # For standard expected size, use pre-compiled expression
        expected_samples = self.config.samples * 2  # Same as used in _prepare_expressions
        if input_length == expected_samples:
            return df.with_columns(self._time_bin_expression)

        # For other sizes, create dynamic time bins
        time_bins = getattr(
            self.config, "time_bins", self.config.samples // self.config.ticks_per_bin
        )
        ticks_per_bin = max(1, input_length // time_bins)  # Ensure at least 1 tick per bin
        time_bin_expr = (pl.int_range(0, input_length) // ticks_per_bin).alias("tick_bin")
        return df.with_columns(time_bin_expr)

    def _compute_mid_prices_per_bin(self, df: pl.DataFrame, expected_bins: int) -> np.ndarray:
        """Compute per-bin mid prices in micro-pip units."""

        mid_expr = (
            (pl.col(ASK_ANCHOR_COLUMN) + pl.col(BID_ANCHOR_COLUMN)) / 2
        ).round().cast(pl.Int64).alias("mid_price")

        grouped = (
            df.lazy()
                .select(["tick_bin", mid_expr])
                .group_by("tick_bin")
                .agg(pl.col("mid_price").mean().round().cast(pl.Int64))
                .sort("tick_bin")
                .collect()
        )

        mid_prices = grouped["mid_price"].to_numpy().astype(np.int64)

        if len(mid_prices) == 0:
            return np.zeros(expected_bins, dtype=np.int64)

        if len(mid_prices) < expected_bins:
            padding = np.full(expected_bins - len(mid_prices), mid_prices[-1], dtype=np.int64)
            mid_prices = np.concatenate([mid_prices, padding])
        elif len(mid_prices) > expected_bins:
            mid_prices = mid_prices[:expected_bins]

        return mid_prices

    def _process_side_data_vectorized(
        self,
        df: pl.DataFrame,
        price_columns: list[str],
        data_columns_map: dict[str, list[str]],
        mid_prices: np.ndarray,
        grids: dict[str, VolumeGrid],
    ) -> None:
        """Process ask or bid side data with full vectorization for multiple features.

        Args:
            df: Input DataFrame
            price_columns: List of price column names
            data_columns_map: Map of feature name to column names (e.g., {'volume': vol_cols, 'trade_counts': count_cols})
            mid_prices: Array of mid prices per time bin in micro-pip units
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

        if prices_array.size == 0:
            return

        bins_in_frame = prices_array.shape[0]
        if len(mid_prices) < bins_in_frame:
            # Pad missing bins with the last known mid price to maintain alignment
            fill_value = mid_prices[-1] if len(mid_prices) else 0
            padding = np.full(bins_in_frame - len(mid_prices), fill_value, dtype=np.int64)
            effective_mid_prices = np.concatenate([mid_prices, padding])
        else:
            effective_mid_prices = mid_prices[:bins_in_frame]

        offsets = prices_array - effective_mid_prices[:, None]
        offsets_int = offsets.astype(np.int64)

        indices_array = np.full(prices_array.shape, -1, dtype=np.int32)
        valid_price_mask = prices_array > 0

        bid_mask = (offsets_int < 0) & (offsets_int >= -self.price_range)
        ask_mask = (offsets_int > 0) & (offsets_int <= self.price_range)

        bid_mask &= valid_price_mask
        ask_mask &= valid_price_mask

        if bid_mask.any():
            indices_array[bid_mask] = (
                np.abs(offsets_int[bid_mask]).astype(np.int32) - 1
            )
        if ask_mask.any():
            indices_array[ask_mask] = (
                self.price_range + 1 + offsets_int[ask_mask]
            ).astype(np.int32)

        valid_mask = indices_array >= 0

        if not valid_mask.any():
            return

        time_indices, _ = np.where(valid_mask)
        y_coords = indices_array[valid_mask]
        x_coords = time_indices

        for feature, data_columns in data_columns_map.items():
            if feature not in self.features:
                continue

            grid = grids[feature]

            if feature == FeatureType.VOLUME.value:
                grouped_data = (
                    df.lazy()
                    .select([*data_columns, "tick_bin"])
                    .group_by("tick_bin")
                    .agg([pl.col(col).median() for col in data_columns])
                    .sort("tick_bin")
                    .collect()
                )
            elif feature == FeatureType.TRADE_COUNTS.value:
                grouped_data = (
                    df.lazy()
                    .select([*data_columns, "tick_bin"])
                    .group_by("tick_bin")
                    .agg([pl.col(col).sum() for col in data_columns])
                    .sort("tick_bin")
                    .collect()
                )
            elif feature == FeatureType.VARIANCE.value:
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

            data_array = grouped_data.select(data_columns).to_numpy()
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

        # Step 3: Calculate per-bin mid prices for centering
        time_bins_resolved = cast(int, self.config.time_bins)
        mid_prices = self._compute_mid_prices_per_bin(df_processed, time_bins_resolved)

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
            df_processed, ASK_PRICE_COLUMNS, ask_data_columns, mid_prices, self._ask_grids
        )

        # Step 6: Process bid side (vectorized, all features)
        self._process_side_data_vectorized(
            df_processed, BID_PRICE_COLUMNS, bid_data_columns, mid_prices, self._bid_grids
        )

        # Step 7: Calculate cumulative volumes and generate output for each feature
        if len(self.features) == 1:
            # Single feature: return 2D array (402, 500)
            feature = self.features[0]
            ask_cumulative = self._ask_grids[feature].get_cumulative_volume(
                reverse=False, stride=self.collapse_stride, groups=self._collapse_groups
            )
            bid_cumulative = self._bid_grids[feature].get_cumulative_volume(
                reverse=True, stride=self.collapse_stride, groups=self._collapse_groups
            )
            result = self._output_buffers[feature].compute_normalized_difference(
                ask_cumulative, bid_cumulative
            )
            return result.copy()
        else:
            # Multiple features: return 3D array (N, 402, 500)
            feature_arrays: list[np.ndarray] = []

            for feature in self.features:
                ask_cumulative = self._ask_grids[feature].get_cumulative_volume(
                    reverse=False, stride=self.collapse_stride, groups=self._collapse_groups
                )
                bid_cumulative = self._bid_grids[feature].get_cumulative_volume(
                    reverse=True, stride=self.collapse_stride, groups=self._collapse_groups
                )
                feature_result = self._output_buffers[feature].compute_normalized_difference(
                    ask_cumulative, bid_cumulative
                )
                feature_arrays.append(feature_result.copy())  # Copy to avoid buffer reuse issues

            # Stack features along first dimension
            stacked_result: np.ndarray = np.stack(feature_arrays, axis=0)
            return stacked_result


# Factory function for easy instantiation with backward compatibility
def create_processor(
    config: MarketDepthProcessorConfig | None = None,
    features: list[str] | list[FeatureType] | None = None,
    # Legacy support
    legacy_config=None,
) -> MarketDepthProcessor:
    """Create a new market depth processor instance.

    Args:
        config: MarketDepthProcessorConfig with focused configuration (preferred)
        features: List of features to extract (overrides config.features if provided)
        legacy_config: Legacy RepresentConfig for backward compatibility
    """
    if config is None and legacy_config is not None:
        # Legacy usage
        return MarketDepthProcessor(legacy_config=legacy_config, features=features)
    else:
        # New focused config usage
        return MarketDepthProcessor(config=config, features=features)


# Main API function with backward compatibility
def process_market_data(
    df: pl.DataFrame,
    config: MarketDepthProcessorConfig | None = None,
    features: list[str] | list[FeatureType] | None = None,
    # Legacy support
    legacy_config=None,
) -> np.ndarray:
    """
    Process market data and return normalized depth representation.

    Args:
        df: Polars DataFrame with market data
        config: MarketDepthProcessorConfig with focused configuration (preferred)
        features: List of features to extract (overrides config.features if provided)
        legacy_config: Legacy RepresentConfig for backward compatibility

    Returns:
        numpy array with normalized market depth:
        - Single feature: shape (402, time_bins)
        - Multiple features: shape (N, 402, time_bins) where N is number of features
    """
    if config is None and legacy_config is not None:
        # Legacy usage
        processor = create_processor(legacy_config=legacy_config, features=features)
    else:
        # New focused config usage
        processor = create_processor(config=config, features=features)
    return processor.process(df)
