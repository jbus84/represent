"""
Focused Configuration Models for Each Core Module

This replaces the monolithic RepresentConfig with three focused Pydantic models,
each containing only the parameters needed by its respective module.
"""

from typing import cast

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class DatasetBuilderConfig(BaseModel):
    """
    Configuration for DatasetBuilder module.

    Focused on creating comprehensive symbol datasets from multiple DBN files
    using the symbol-split-merge architecture.
    """

    # Core parameters
    currency: str = Field(
        default="AUDUSD", description="Currency pair identifier (e.g., 'AUDUSD', 'EURUSD')"
    )

    # Price movement calculation parameters
    lookback_rows: int = Field(
        default=5000, gt=0, description="Number of historical rows for baseline calculation"
    )
    lookforward_input: int = Field(
        default=5000, gt=0, description="Size of lookforward window for classification"
    )
    lookforward_offset: int = Field(
        default=500, ge=0, description="Offset before starting lookforward window"
    )

    @field_validator("currency")
    @classmethod
    def validate_currency_pair(cls, v: str) -> str:
        if len(v) != 6 or not v.isalpha():
            raise ValueError(f"Currency pair must be 6 alphabetic characters, got: {v}")
        return v.upper()

    @computed_field
    def min_required_samples(self) -> int:
        """Minimum samples needed for processing this configuration."""
        return self.lookback_rows + self.lookforward_input + self.lookforward_offset


class GlobalThresholdConfig(BaseModel):
    """
    Configuration for GlobalThresholdCalculator module.

    Focused on calculating consistent classification thresholds across
    multiple files for uniform distribution.
    """

    # Core parameters
    currency: str = Field(
        default="AUDUSD", description="Currency pair identifier (e.g., 'AUDUSD', 'EURUSD')"
    )
    nbins: int = Field(default=13, ge=3, le=20, description="Number of classification bins")

    # Price movement calculation parameters (same as DatasetBuilder)
    lookback_rows: int = Field(
        default=5000, gt=0, description="Number of historical rows for baseline calculation"
    )
    lookforward_input: int = Field(
        default=5000, gt=0, description="Size of lookforward window for classification"
    )
    lookforward_offset: int = Field(
        default=500, ge=0, description="Offset before starting lookforward window"
    )

    # Performance parameters
    max_samples_per_file: int = Field(
        default=10000,
        gt=0,
        description="Maximum samples to extract per file for performance optimization",
    )
    sample_fraction: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Fraction of files to use for threshold calculation",
    )
    jump_size: int = Field(
        default=100,
        gt=0,
        description="Step size for sampling positions in price movement calculation (performance optimization)",
    )
    use_heavy_tailed: bool = Field(
        default=True,
        description="Use heavy-tailed distribution (Student's t) for better class balance instead of quantiles",
    )

    @field_validator("currency")
    @classmethod
    def validate_currency_pair(cls, v: str) -> str:
        if len(v) != 6 or not v.isalpha():
            raise ValueError(f"Currency pair must be 6 alphabetic characters, got: {v}")
        return v.upper()

    @field_validator("nbins")
    @classmethod
    def validate_nbins(cls, v: int) -> int:
        if v not in [3, 5, 7, 9, 13]:
            raise ValueError(f"Unsupported nbins value: {v}. Supported values: 3, 5, 7, 9, 13")
        return v


class PriceBinSpec(BaseModel):
    """Specification describing variable-width pip bins."""

    limit_pips: float | None = Field(
        default=None,
        gt=0.0,
        description="Upper bound in pips for this bin size; None means extend to the end",
    )
    bin_size_pips: float = Field(
        gt=0.0,
        description="Size of each bin within this range, measured in pips",
    )


class MarketDepthProcessorConfig(BaseModel):
    """Configuration for MarketDepthProcessor module."""

    # Feature parameters
    features: list[str] = Field(
        default=["volume"],
        description="Features to extract: ['volume', 'variance', 'trade_counts']",
    )

    # Ladder geometry
    price_range: int = Field(
        default=200,
        gt=0,
        description="Half-width of the captured ladder in micro-pip steps (total rows = 2*price_range + 2)",
    )
    collapse_stride: int | None = Field(
        default=None,
        ge=1,
        description="Optional stride to collapse adjacent price levels after indexing",
    )
    target_price_levels: int | None = Field(
        default=None,
        ge=4,
        description="Optional explicit number of post-collapse price levels",
    )
    pip_size: float = Field(
        default=0.0001,
        gt=0.0,
        description="Nominal pip size used when interpreting bin_spec",
    )
    bin_spec: list[PriceBinSpec] | None = Field(
        default=None,
        description="Variable-width pip bins ordered from the mid outward",
    )

    # Tensor dimension parameters
    samples: int = Field(
        default=50000,
        ge=1000,
        description="Number of samples to process (affects tensor time dimension)",
    )
    ticks_per_bin: int = Field(default=100, gt=0, description="Number of ticks per time bin")

    # Price precision parameters
    micro_pip_size: float = Field(
        default=0.00001, gt=0.0, description="Micro pip size for price precision"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: list[str]) -> list[str]:
        valid_features = {"volume", "variance", "trade_counts"}
        invalid = set(v) - valid_features
        if invalid:
            raise ValueError(f"Invalid features: {invalid}. Valid: {valid_features}")
        return v

    @model_validator(mode="after")
    def validate_price_range_alignment(self) -> "MarketDepthProcessorConfig":
        if self.collapse_stride is not None and self.collapse_stride < 1:
            raise ValueError("collapse_stride must be greater than zero when provided")

        if self.collapse_stride is not None and self.target_price_levels is not None:
            raise ValueError("Provide either collapse_stride or target_price_levels, not both")

        if self.bin_spec is not None:
            if self.collapse_stride is not None or self.target_price_levels is not None:
                raise ValueError("bin_spec cannot be combined with collapse_stride or target_price_levels")
            if not self.bin_spec:
                raise ValueError("bin_spec must contain at least one entry")

        if self.target_price_levels is not None:
            if (self.target_price_levels - 2) % 2 != 0:
                raise ValueError("target_price_levels must satisfy (levels - 2) being divisible by 2")

            target_range = (self.target_price_levels - 2) // 2
            if target_range <= 0:
                raise ValueError("target_price_levels must be greater than 2")

            if target_range > self.price_range:
                raise ValueError("target_price_levels cannot exceed captured price_range resolution")

            if self.price_range % target_range != 0:
                raise ValueError(
                    "price_range must be divisible by half of target_price_levels minus one"
                )

        if self.collapse_stride is not None and self.price_range % self.collapse_stride != 0:
            raise ValueError(
                "price_range must be divisible by collapse_stride so pooled bins stay symmetric"
            )

        if self.bin_spec is not None:
            bin_groups = self._compute_bin_groups()
            object.__setattr__(self, "_bin_groups", tuple(bin_groups))

        return self

    @computed_field(return_type=int)
    def time_bins(self) -> int:
        """Auto-computed time bins based on samples and ticks_per_bin."""
        return self.samples // self.ticks_per_bin

    @computed_field(return_type=int)
    def price_levels(self) -> int:
        """Full-resolution price ladder size used during indexing."""
        return self.price_range * 2 + 2

    @computed_field(return_type=int)
    def collapse_stride_value(self) -> int:
        """Resolved collapse stride after validation."""
        if self.bin_spec is not None or self.target_price_levels is not None:
            return 1
        return self.collapse_stride or 1

    @computed_field(return_type=int)
    def effective_price_levels(self) -> int:
        """Price ladder after optional collapse stride is applied."""
        if self.bin_spec is not None:
            groups = cast(tuple[tuple[int, ...], ...], getattr(self, "_bin_groups", ()))
            return len(groups) * 2 + 2
        if self.target_price_levels is not None:
            return self.target_price_levels

        stride = cast(int, self.collapse_stride_value)
        effective_range = self.price_range // stride
        return effective_range * 2 + 2

    @computed_field(return_type=tuple[tuple[int, ...], ...] | None)
    def bin_groups(self) -> tuple[tuple[int, ...], ...] | None:
        """Immutable representation of per-side bin groupings."""
        return getattr(self, "_bin_groups", None)

    def _compute_bin_groups(self) -> list[tuple[int, ...]]:
        """Compute bid/ask aggregation groups from bin_spec."""
        if self.bin_spec is None:
            return []

        ratio = self.pip_size / self.micro_pip_size
        if ratio <= 0:
            raise ValueError("pip_size must be greater than micro_pip_size")

        if abs(round(ratio) - ratio) > 1e-6:
            raise ValueError("pip_size must be a multiple of micro_pip_size")
        steps_per_pip = int(round(ratio))

        total_steps = self.price_range
        groups: list[tuple[int, ...]] = []
        consumed = 0
        previous_limit_steps = 0

        for spec in self.bin_spec:
            limit_steps = total_steps
            if spec.limit_pips is not None:
                limit_steps = min(total_steps, int(round(spec.limit_pips * steps_per_pip)))
                if limit_steps <= previous_limit_steps:
                    raise ValueError("bin_spec limits must be strictly increasing")

            bin_steps = int(round(spec.bin_size_pips * steps_per_pip))
            if bin_steps <= 0:
                raise ValueError("bin_size_pips produces zero-width bins")

            segment_end = limit_steps
            while consumed < min(segment_end, total_steps):
                next_consumed = min(consumed + bin_steps, min(segment_end, total_steps))
                if next_consumed == consumed:
                    break
                groups.append(tuple(range(consumed, next_consumed)))
                consumed = next_consumed

            previous_limit_steps = limit_steps

            if consumed >= total_steps:
                break

        if consumed < total_steps:
            raise ValueError("bin_spec does not cover entire price_range")

        return groups

    @computed_field(return_type=tuple[int, ...])
    def output_shape(self) -> tuple[int, ...]:
        """Auto-computed output tensor shape based on features."""
        time_bins = self.samples // self.ticks_per_bin
        price_levels = cast(int, self.effective_price_levels)
        if len(self.features) == 1:
            return (price_levels, time_bins)  # 2D tensor for single feature
        else:
            return (len(self.features), price_levels, time_bins)  # 3D tensor for multiple features


# Convenience factory functions for common configurations


def create_dataset_builder_config(
    currency: str = "AUDUSD",
    lookback_rows: int = 5000,
    lookforward_input: int = 5000,
    lookforward_offset: int = 500,
) -> DatasetBuilderConfig:
    """Create a DatasetBuilderConfig with common parameters."""
    return DatasetBuilderConfig(
        currency=currency,
        lookback_rows=lookback_rows,
        lookforward_input=lookforward_input,
        lookforward_offset=lookforward_offset,
    )


def create_threshold_config(
    currency: str = "AUDUSD",
    nbins: int = 13,
    lookback_rows: int = 5000,
    lookforward_input: int = 5000,
    lookforward_offset: int = 500,
    max_samples_per_file: int = 10000,
    sample_fraction: float = 0.5,
    jump_size: int = 100,
) -> GlobalThresholdConfig:
    """Create a GlobalThresholdConfig with common parameters."""
    return GlobalThresholdConfig(
        currency=currency,
        nbins=nbins,
        lookback_rows=lookback_rows,
        lookforward_input=lookforward_input,
        lookforward_offset=lookforward_offset,
        max_samples_per_file=max_samples_per_file,
        sample_fraction=sample_fraction,
        jump_size=jump_size,
    )


def create_processor_config(
    features: list[str] | None = None,
    samples: int = 50000,
    ticks_per_bin: int = 100,
    micro_pip_size: float = 0.00001,
    price_range: int = 200,
    collapse_stride: int | None = None,
    target_price_levels: int | None = None,
    pip_size: float = 0.0001,
    bin_spec: list[PriceBinSpec] | None = None,
) -> MarketDepthProcessorConfig:
    """Create a MarketDepthProcessorConfig with common parameters."""
    if features is None:
        features = ["volume"]

    return MarketDepthProcessorConfig(
        features=features,
        samples=samples,
        ticks_per_bin=ticks_per_bin,
        micro_pip_size=micro_pip_size,
        price_range=price_range,
        collapse_stride=collapse_stride,
        target_price_levels=target_price_levels,
        pip_size=pip_size,
        bin_spec=bin_spec,
    )


def create_compatible_configs(
    currency: str = "AUDUSD",
    features: list[str] | None = None,
    lookback_rows: int = 5000,
    lookforward_input: int = 5000,
    lookforward_offset: int = 500,
    nbins: int = 13,
    samples: int = 50000,
    ticks_per_bin: int = 100,
    micro_pip_size: float = 0.00001,
    jump_size: int = 100,
    price_range: int = 200,
    collapse_stride: int | None = None,
    target_price_levels: int | None = None,
    pip_size: float = 0.0001,
    bin_spec: list[PriceBinSpec] | None = None,
) -> tuple[DatasetBuilderConfig, GlobalThresholdConfig, MarketDepthProcessorConfig]:
    """
    Create compatible configurations for all three modules.

    This ensures the same currency and price movement parameters are used
    across DatasetBuilder and GlobalThresholdCalculator.
    """
    if features is None:
        features = ["volume"]

    # Apply currency-specific optimizations for MarketDepthProcessorConfig only
    # (DatasetBuilder optimizations are handled in create_represent_config)
    if currency.upper() in ["USDJPY", "EURJPY", "GBPJPY"]:
        micro_pip_size = 0.001
        if nbins == 13:  # Only change default nbins
            nbins = 9  # Fewer bins for JPY pairs

    dataset_config = DatasetBuilderConfig(
        currency=currency,
        lookback_rows=lookback_rows,
        lookforward_input=lookforward_input,
        lookforward_offset=lookforward_offset,
    )

    threshold_config = GlobalThresholdConfig(
        currency=currency,
        nbins=nbins,
        lookback_rows=lookback_rows,
        lookforward_input=lookforward_input,
        lookforward_offset=lookforward_offset,
        jump_size=jump_size,
    )

    processor_config = MarketDepthProcessorConfig(
        features=features,
        samples=samples,
        ticks_per_bin=ticks_per_bin,
        micro_pip_size=micro_pip_size,
        price_range=price_range,
        collapse_stride=collapse_stride,
        target_price_levels=target_price_levels,
        pip_size=pip_size,
        bin_spec=bin_spec,
    )

    return dataset_config, threshold_config, processor_config


# Legacy compatibility functions
def create_represent_config(
    currency: str = "AUDUSD",
    nbins: int = 13,
    samples: int = 25000,
    features: list[str] | None = None,
    **kwargs,
) -> tuple[DatasetBuilderConfig, GlobalThresholdConfig, MarketDepthProcessorConfig]:
    """
    Create compatible configurations for all three modules (legacy compatibility).

    This replaces the old RepresentConfig factory function.

    Args:
        currency: Currency pair identifier
        nbins: Number of classification bins
        samples: Number of samples to process
        features: Features to extract
        **kwargs: Additional configuration parameters

    Returns:
        Tuple of (DatasetBuilderConfig, GlobalThresholdConfig, MarketDepthProcessorConfig)
    """
    if features is None:
        features = ["volume"]

    # Apply currency-specific optimizations - only if not explicitly provided
    if "lookforward_input" in kwargs:
        lookforward_input = kwargs["lookforward_input"]  # Use explicit value
    else:
        # Apply currency-specific defaults
        if currency.upper() == "GBPUSD":
            lookforward_input = 3000  # Shorter for volatility
        else:
            lookforward_input = 5000  # Default

    if "micro_pip_size" in kwargs:
        micro_pip_size = kwargs["micro_pip_size"]  # Use explicit value
    else:
        # Apply currency-specific defaults
        if currency.upper() in ["USDJPY", "EURJPY", "GBPJPY"]:
            micro_pip_size = 0.001
        else:
            micro_pip_size = 0.00001  # Default

    # Only set nbins if not explicitly provided
    if "nbins" not in kwargs and nbins == 13:  # 13 is the default
        if currency.upper() in ["USDJPY", "EURJPY", "GBPJPY"]:
            nbins = 9  # Fewer bins for JPY pairs

    # Remove parameters we've already handled from kwargs
    filtered_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["lookforward_input", "micro_pip_size", "nbins"]
    }

    return create_compatible_configs(
        currency=currency,
        features=features,
        nbins=nbins,
        samples=samples,
        micro_pip_size=micro_pip_size,
        lookforward_input=lookforward_input,
        **filtered_kwargs,
    )


def list_available_currencies() -> list[str]:
    """
    List all available currency configurations.

    Returns:
        List of available currency pair identifiers
    """
    return ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY", "EURJPY"]
