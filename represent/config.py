"""
Configuration models for market depth processing.
Supports currency-specific configurations with default values and dynamic generation.
"""

from typing import Dict, Optional, Union
from pathlib import Path
import json
import logging

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class SamplingConfig(BaseModel):
    """Configuration for data sampling strategies."""

    sampling_mode: str = Field(
        default="consecutive", description="Sampling mode: 'consecutive' or 'random'"
    )
    coverage_percentage: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Percentage of dataset to process (0.0-1.0)"
    )
    end_tick_strategy: str = Field(
        default="uniform_random", description="Strategy for selecting end ticks in random mode"
    )
    min_tick_spacing: int = Field(
        default=100, ge=1, description="Minimum spacing between sampled end ticks"
    )
    seed: Optional[int] = Field(default=42, description="Random seed for reproducible sampling")
    max_samples: Optional[int] = Field(
        default=None, description="Maximum number of samples to process (None = no limit)"
    )

    @field_validator("sampling_mode")
    @classmethod
    def validate_sampling_mode(cls, v: str) -> str:
        if v not in ["consecutive", "random", "stratified_random"]:
            raise ValueError(f"Invalid sampling_mode: {v}")
        return v

    @field_validator("end_tick_strategy")
    @classmethod
    def validate_end_tick_strategy(cls, v: str) -> str:
        if v not in ["uniform_random", "weighted_random", "temporal_distribution"]:
            raise ValueError(f"Invalid end_tick_strategy: {v}")
        return v


class ClassificationConfig(BaseModel):
    """Configuration for price movement classification."""

    micro_pip_size: float = Field(
        default=0.00001, gt=0.0, description="Micro pip size for price precision"
    )
    true_pip_size: float = Field(
        default=0.0001, gt=0.0, description="True pip size for the currency pair"
    )
    ticks_per_bin: int = Field(default=100, gt=0, description="Number of ticks per time bin")
    lookforward_offset: int = Field(
        default=500, ge=0, description="Offset before starting lookforward window"
    )
    lookforward_input: int = Field(
        default=5000, gt=0, description="Size of lookforward window for classification"
    )
    lookback_rows: int = Field(
        default=5000, gt=0, description="Number of historical rows for baseline calculation"
    )
    nbins: int = Field(default=13, ge=3, le=20, description="Number of classification bins")
    bin_thresholds: Dict[int, Dict[int, Dict[int | str, Dict[str, float]]]] = Field(
        default_factory=lambda: {
            13: {
                100: {
                    5000: {
                        "bin_1": 1.41,
                        "bin_2": 2.61,
                        "bin_3": 3.75,
                        "bin_4": 4.75,
                        "bin_5": 6.53,
                        "bin_6": 10.13,
                    },
                    3000: {
                        "bin_1": 0.5,
                        "bin_2": 1.7,
                        "bin_3": 3.0,
                        "bin_4": 4.3,
                        "bin_5": 6.0,
                        "bin_6": 8.45,
                    },
                }
            },
            9: {
                10: {5000: {"bin_1": 0.31, "bin_2": 0.91, "bin_3": 1.6, "bin_4": 2.55}},
                100: {5000: {"bin_1": 0.51, "bin_2": 2.25, "bin_3": 4.0, "bin_4": 6.35}},
            },
            7: {
                10: {5000: {"bin_1": 0.3, "bin_2": 0.9, "bin_3": 1.7}},
                100: {5000: {"bin_1": 0.7, "bin_2": 2.7, "bin_3": 5.5}},
            },
            5: {
                10: {5000: {"bin_1": 0.5, "bin_2": 1.5}},
                100: {5000: {"bin_1": 1.0, "bin_2": 3.0}},
            },
            3: {10: {5000: {"bin_1": 0.75}}, 100: {5000: {"bin_1": 1.5}}},
        },
        description="Hierarchical threshold configuration by bins/ticks/lookforward",
    )

    @field_validator("nbins")
    @classmethod
    def validate_nbins(cls, v: int) -> int:
        if v not in [3, 5, 7, 9, 13]:
            raise ValueError(f"Unsupported nbins value: {v}. Supported values: 3, 5, 7, 9, 13")
        return v

    @model_validator(mode="after")
    def validate_bin_thresholds_completeness(self) -> "ClassificationConfig":
        """Validate that bin thresholds contain complete threshold sets for the specified nbins."""
        required_bins = self._get_required_bins_for_nbins(self.nbins)

        # Check if we have thresholds for this nbins value
        if self.nbins not in self.bin_thresholds:
            raise ValueError(f"No bin thresholds defined for nbins={self.nbins}")

        # Check that at least one configuration exists
        bins_config = self.bin_thresholds[self.nbins]
        if not bins_config:
            raise ValueError(f"Empty bin thresholds for nbins={self.nbins}")

        # Validate that each ticks_per_bin configuration has complete threshold sets
        for ticks_per_bin, ticks_config in bins_config.items():
            if not ticks_config:
                raise ValueError(
                    f"Empty configuration for nbins={self.nbins}, ticks_per_bin={ticks_per_bin}"
                )

            for lookforward, thresholds in ticks_config.items():
                if not thresholds:
                    raise ValueError(
                        f"Invalid thresholds format for nbins={self.nbins}, ticks_per_bin={ticks_per_bin}, lookforward={lookforward}"
                    )

                # Check that all required bins are present
                missing_bins = required_bins - set(thresholds.keys())
                if missing_bins:
                    raise ValueError(
                        f"Missing bin thresholds {missing_bins} for nbins={self.nbins}, ticks_per_bin={ticks_per_bin}, lookforward={lookforward}"
                    )

                # Check that all values are numeric
                for bin_name, value in thresholds.items():
                    if value <= 0:
                        raise ValueError(
                            f"Invalid threshold value {value} for {bin_name} in nbins={self.nbins}, ticks_per_bin={ticks_per_bin}, lookforward={lookforward}"
                        )

        return self

    def _get_required_bins_for_nbins(self, nbins: int) -> set[str]:
        """Get the required bin names for a given nbins value."""
        if nbins == 13:
            return {"bin_1", "bin_2", "bin_3", "bin_4", "bin_5", "bin_6"}
        elif nbins == 9:
            return {"bin_1", "bin_2", "bin_3", "bin_4"}
        elif nbins == 7:
            return {"bin_1", "bin_2", "bin_3"}
        elif nbins == 5:
            return {"bin_1", "bin_2"}
        elif nbins == 3:
            return {"bin_1"}
        else:
            raise ValueError(f"Unsupported nbins value: {nbins}")

    def get_thresholds(
        self, ticks_per_bin: Optional[int] = None, lookforward_input: Optional[int] = None
    ) -> Dict[str, float]:
        """Get threshold values for the current configuration.

        This method assumes validation has passed, so thresholds are guaranteed to exist.
        """
        ticks = ticks_per_bin or self.ticks_per_bin
        lookforward = lookforward_input or self.lookforward_input

        # Get the configuration for this nbins - validation guarantees it exists
        bins_config = self.bin_thresholds[self.nbins]

        # Try exact match for ticks_per_bin
        if ticks in bins_config:
            ticks_config = bins_config[ticks]

            # Try exact match for lookforward_input
            if lookforward in ticks_config:
                return ticks_config[lookforward]

            # Use first available lookforward configuration for this ticks_per_bin
            return next(iter(ticks_config.values()))

        # Use first available ticks_per_bin configuration
        first_ticks_config = next(iter(bins_config.values()))
        return next(iter(first_ticks_config.values()))


class CurrencyConfig(BaseModel):
    """Currency-specific configuration container."""

    currency_pair: str = Field(description="Currency pair identifier (e.g., 'AUDUSD', 'EURUSD')")
    classification: ClassificationConfig = Field(
        default_factory=ClassificationConfig,
        description="Classification configuration for this currency",
    )
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig, description="Sampling configuration for this currency"
    )
    description: Optional[str] = Field(
        default=None, description="Human-readable description of this configuration"
    )

    @field_validator("currency_pair")
    @classmethod
    def validate_currency_pair(cls, v: str) -> str:
        if len(v) != 6 or not v.isalpha():
            raise ValueError(f"Currency pair must be 6 alphabetic characters, got: {v}")
        return v.upper()


def load_currency_config(currency: str, config_dir: Optional[Path] = None) -> CurrencyConfig:
    """
    Load currency-specific configuration.
    
    Note: Static config files have been replaced with dynamic config generation.
    This function now returns optimized default configurations.
    
    Args:
        currency: Currency pair identifier (e.g., 'AUDUSD')
        config_dir: Deprecated - config files no longer used
    Returns:
        CurrencyConfig for the specified currency
    Raises:
        ValueError: If currency is not supported
    """
    if config_dir is not None:
        logger.warning("config_dir parameter is deprecated. Static config files have been replaced with dynamic generation.")
    
    # Return optimized default configuration
    return get_default_currency_config(currency)


def load_config_from_file(config_path: Union[str, Path]) -> CurrencyConfig:
    """
    Load configuration from a specific file path (JSON only).
    
    Note: YAML support has been removed. Use dynamic config generation instead.

    Args:
        config_path: Path to configuration file (.json)

    Returns:
        CurrencyConfig loaded from file

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix.lower() == ".json":
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return CurrencyConfig(**config_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}. Only JSON is supported.")



def get_default_currency_config(currency: str) -> CurrencyConfig:
    """
    Get default configuration for a currency pair.

    Args:
        currency: Currency pair identifier

    Returns:
        Default CurrencyConfig with currency-specific adjustments
    """
    currency = currency.upper()

    # Currency-specific adjustments
    if currency == "AUDUSD":
        # AUDUSD-specific optimizations from empirical analysis
        classification = ClassificationConfig(
            true_pip_size=0.0001, nbins=13, lookforward_input=5000, ticks_per_bin=100
        )
        sampling = SamplingConfig(coverage_percentage=0.8)
        description = "Optimized configuration for AUD/USD major currency pair"

    elif currency == "EURUSD":
        # EURUSD optimizations
        classification = ClassificationConfig(
            true_pip_size=0.0001, nbins=13, lookforward_input=5000, ticks_per_bin=100
        )
        sampling = SamplingConfig(coverage_percentage=0.9)
        description = "Optimized configuration for EUR/USD most liquid currency pair"

    elif currency == "GBPUSD":
        # GBPUSD optimizations (more volatile)
        classification = ClassificationConfig(
            true_pip_size=0.0001,
            nbins=13,
            lookforward_input=3000,  # Shorter window for volatility
            ticks_per_bin=100,
        )
        sampling = SamplingConfig(coverage_percentage=0.7)
        description = "Optimized configuration for GBP/USD with volatility adjustments"

    elif currency in ["USDJPY", "EURJPY", "GBPJPY"]:
        # JPY pairs (different pip size)
        classification = ClassificationConfig(
            true_pip_size=0.01,  # JPY pairs use different pip size
            micro_pip_size=0.001,
            nbins=9,  # Fewer bins for different dynamics
            lookforward_input=5000,
        )
        sampling = SamplingConfig(coverage_percentage=0.6)
        description = f"Optimized configuration for {currency} with JPY-specific pip sizing"

    else:
        # Generic configuration for other pairs
        classification = ClassificationConfig()
        sampling = SamplingConfig()
        description = f"Default configuration for {currency}"

    return CurrencyConfig(
        currency_pair=currency,
        classification=classification,
        sampling=sampling,
        description=description,
    )


def save_currency_config(config: CurrencyConfig, config_dir: Optional[Path] = None) -> Path:
    """
    Save currency configuration to file.

    Args:
        config: CurrencyConfig to save
        config_dir: Directory to save configuration files

    Returns:
        Path to saved configuration file
    """
    if config_dir is None:
        config_dir = Path(__file__).parent / "configs"

    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / f"{config.currency_pair.lower()}.json"

    with open(config_file, "w") as f:
        json.dump(config.model_dump(), f, indent=2)

    return config_file


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
