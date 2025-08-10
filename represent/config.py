"""
Configuration models for market depth processing.
Supports currency-specific configurations with default values and dynamic generation.
"""

from typing import List, Optional, Tuple
from pathlib import Path
import logging

from pydantic import BaseModel, Field, field_validator, computed_field

logger = logging.getLogger(__name__)


class RepresentConfig(BaseModel):
    """
    Simplified flat configuration structure for the represent package.
    
    This replaces the complex nested CurrencyConfig structure with a flat,
    user-friendly interface while maintaining all functionality.
    """
    
    # Core configuration
    currency: str = Field(
        default="AUDUSD", 
        description="Currency pair identifier (e.g., 'AUDUSD', 'EURUSD')"
    )
    nbins: int = Field(
        default=13, 
        ge=3, le=20, 
        description="Number of classification bins"
    )
    samples: int = Field(
        default=25000, 
        gt=0, 
        description="Number of samples to process"
    )
    features: List[str] = Field(
        default=["volume"], 
        description="Features to extract: ['volume', 'variance', 'trade_counts']"
    )
    
    # Timing parameters - now configurable!
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
    
    # Processing parameters
    batch_size: int = Field(
        default=1000, 
        gt=0, 
        description="Number of samples per processing batch"
    )
    ticks_per_bin: int = Field(
        default=100, 
        gt=0, 
        description="Number of ticks per time bin"
    )
    jump_size: int = Field(
        default=100,
        gt=0,
        description="Step size for sampling positions in price movement calculation (performance optimization)"
    )
    
    # Sampling configuration
    sampling_mode: str = Field(
        default="consecutive", 
        description="Sampling mode: 'consecutive' or 'random'"
    )
    coverage_percentage: float = Field(
        default=1.0, 
        ge=0.0, le=1.0, 
        description="Percentage of dataset to process"
    )
    
    # Advanced parameters
    micro_pip_size: float = Field(
        default=0.00001, 
        gt=0.0, 
        description="Micro pip size for price precision"
    )
    true_pip_size: float = Field(
        default=0.0001, 
        gt=0.0, 
        description="True pip size for the currency pair"
    )
    
    # Performance parameters
    max_samples_per_file: int = Field(
        default=10000,
        gt=0,
        description="Maximum samples to extract per file for performance optimization"
    )
    target_samples: int = Field(
        default=1000,
        gt=0,
        description="Target number of samples for analysis operations"
    )
    
    @field_validator("currency")
    @classmethod
    def validate_currency_pair(cls, v: str) -> str:
        if len(v) != 6 or not v.isalpha():
            raise ValueError(f"Currency pair must be 6 alphabetic characters, got: {v}")
        return v.upper()
    
    @field_validator("features")
    @classmethod
    def validate_features(cls, v: List[str]) -> List[str]:
        valid_features = {"volume", "variance", "trade_counts"}
        invalid = set(v) - valid_features
        if invalid:
            raise ValueError(f"Invalid features: {invalid}. Valid: {valid_features}")
        return v
    
    @field_validator("sampling_mode")
    @classmethod
    def validate_sampling_mode(cls, v: str) -> str:
        if v not in ["consecutive", "random", "stratified_random"]:
            raise ValueError(f"Invalid sampling_mode: {v}")
        return v
    
    @field_validator("nbins")
    @classmethod
    def validate_nbins(cls, v: int) -> int:
        if v not in [3, 5, 7, 9, 13]:
            raise ValueError(f"Unsupported nbins value: {v}. Supported values: 3, 5, 7, 9, 13")
        return v
    
    @computed_field
    def time_bins(self) -> int:
        """Auto-computed time bins based on samples and ticks_per_bin."""
        return self.samples // self.ticks_per_bin
    
    min_symbol_samples: int = Field(
        default_factory=lambda: 100,
        gt=0,
        description="Minimum samples required per symbol for processing"
    )
    
    @computed_field
    def output_shape(self) -> Tuple[int, int]:
        """Auto-computed output shape (PRICE_LEVELS, time_bins)."""
        time_bins_value = self.samples // self.ticks_per_bin
        return (402, time_bins_value)  # 402 = 200 bid + 200 ask + 2 mid
    


def create_represent_config(
    currency: str = "AUDUSD",
    nbins: int = 13,
    samples: int = 25000,
    features: Optional[List[str]] = None,
    **kwargs
) -> RepresentConfig:
    """
    Create a simplified RepresentConfig with sensible defaults.
    
    Args:
        currency: Currency pair identifier
        nbins: Number of classification bins
        samples: Number of samples to process
        features: Features to extract
        **kwargs: Additional configuration parameters
    
    Returns:
        RepresentConfig with optimized defaults
    """
    if features is None:
        features = ["volume"]
    
    # Apply currency-specific optimizations
    config_data = {
        "currency": currency,
        "nbins": nbins,
        "samples": samples,
        "features": features,
        **kwargs
    }
    
    # Currency-specific adjustments (only if not explicitly provided)
    if currency.upper() == "GBPUSD":
        config_data.setdefault("lookforward_input", 3000)  # Shorter for volatility
    elif currency.upper() in ["USDJPY", "EURJPY", "GBPJPY"]:
        config_data.setdefault("true_pip_size", 0.01)
        config_data.setdefault("micro_pip_size", 0.001)
        # Only set nbins if not explicitly provided
        if "nbins" not in kwargs and nbins == 13:  # 13 is the default
            config_data["nbins"] = 9  # Fewer bins for JPY pairs
    
    return RepresentConfig(**config_data)


def list_available_currencies(config_dir: Optional[Path] = None) -> list[str]:
    """
    List all available currency configurations.
    
    Note: Static config files have been replaced with dynamic config generation.
    This function now returns a predefined list of supported currencies.

    Args:
        config_dir: Deprecated - config files no longer used

    Returns:
        List of available currency pair identifiers
    """
    if config_dir is not None:
        logger.warning("config_dir parameter is deprecated. Static config files have been replaced with dynamic generation.")

    # Return predefined list of supported currencies
    return ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY", "EURJPY"]
