"""
New Configuration Architecture Design

This demonstrates focused Pydantic models for each of the three core modules,
replacing the monolithic RepresentConfig.
"""

from pydantic import BaseModel, Field, computed_field, field_validator


class PriceMovementConfig(BaseModel):
    """
    Configuration for price movement calculation.
    Used by both DatasetBuilder and GlobalThresholdCalculator.
    """
    currency: str = Field(
        default="AUDUSD",
        description="Currency pair identifier (e.g., 'AUDUSD', 'EURUSD')"
    )
    lookback_rows: int = Field(
        default=5000,
        gt=0,
        description="Number of historical rows for baseline calculation"
    )
    lookforward_input: int = Field(
        default=5000,
        gt=0,
        description="Size of lookforward window for classification"
    )
    lookforward_offset: int = Field(
        default=500,
        ge=0,
        description="Offset before starting lookforward window"
    )

    @field_validator("currency")
    @classmethod
    def validate_currency_pair(cls, v: str) -> str:
        if len(v) != 6 or not v.isalpha():
            raise ValueError(f"Currency pair must be 6 alphabetic characters, got: {v}")
        return v.upper()


class DatasetBuilderConfig(BaseModel):
    """
    Configuration for DatasetBuilder module.
    Focused on dataset creation from multiple DBN files.
    """
    price_movement: PriceMovementConfig = Field(
        default_factory=PriceMovementConfig,
        description="Price movement calculation parameters"
    )

    @property
    def currency(self) -> str:
        """Convenience property to access currency."""
        return self.price_movement.currency


class GlobalThresholdConfig(BaseModel):
    """
    Configuration for GlobalThresholdCalculator module.
    Focused on consistent classification thresholds across files.
    """
    price_movement: PriceMovementConfig = Field(
        default_factory=PriceMovementConfig,
        description="Price movement calculation parameters"
    )
    nbins: int = Field(
        default=13,
        ge=3, le=20,
        description="Number of classification bins"
    )
    max_samples_per_file: int = Field(
        default=10000,
        gt=0,
        description="Maximum samples to extract per file for performance optimization"
    )
    sample_fraction: float = Field(
        default=0.5,
        gt=0.0, le=1.0,
        description="Fraction of files to use for threshold calculation"
    )

    @field_validator("nbins")
    @classmethod
    def validate_nbins(cls, v: int) -> int:
        if v not in [3, 5, 7, 9, 13]:
            raise ValueError(f"Unsupported nbins value: {v}. Supported values: 3, 5, 7, 9, 13")
        return v

    @property
    def currency(self) -> str:
        """Convenience property to access currency."""
        return self.price_movement.currency


class MarketDepthProcessorConfig(BaseModel):
    """
    Configuration for MarketDepthProcessor module.
    Focused on converting market data to tensors.
    """
    features: list[str] = Field(
        default=["volume"],
        description="Features to extract: ['volume', 'variance', 'trade_counts']"
    )
    samples: int = Field(
        default=50000,
        ge=25000,
        description="Number of samples to process (affects tensor dimensions)"
    )
    ticks_per_bin: int = Field(
        default=100,
        gt=0,
        description="Number of ticks per time bin"
    )
    micro_pip_size: float = Field(
        default=0.00001,
        gt=0.0,
        description="Micro pip size for price precision"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: list[str]) -> list[str]:
        valid_features = {"volume", "variance", "trade_counts"}
        invalid = set(v) - valid_features
        if invalid:
            raise ValueError(f"Invalid features: {invalid}. Valid: {valid_features}")
        return v

    @computed_field
    def time_bins(self) -> int:
        """Auto-computed time bins based on samples and ticks_per_bin."""
        return self.samples // self.ticks_per_bin


# Example usage:

# Dataset Builder
dataset_config = DatasetBuilderConfig(
    price_movement=PriceMovementConfig(
        currency="AUDUSD",
        lookback_rows=5000,
        lookforward_input=5000,
        lookforward_offset=500
    )
)

# Global Threshold Calculator
threshold_config = GlobalThresholdConfig(
    price_movement=PriceMovementConfig(
        currency="AUDUSD",
        lookback_rows=5000,
        lookforward_input=5000,
        lookforward_offset=500
    ),
    nbins=13,
    max_samples_per_file=10000
)

# Market Depth Processor
processor_config = MarketDepthProcessorConfig(
    features=["volume", "variance"],
    samples=50000,
    ticks_per_bin=100,
    micro_pip_size=0.00001
)

print("✅ New focused configuration architecture designed!")
print(f"   Dataset Builder currency: {dataset_config.currency}")
print(f"   Threshold Config bins: {threshold_config.nbins}")
print(f"   Processor features: {processor_config.features}")
print(f"   Processor time bins: {processor_config.time_bins}")
