"""
Target Generator Factory

This module provides a factory pattern for creating target generators from
configuration and registering new generator types.
"""

from typing import Any

from .base import TargetGenerator


class TargetGeneratorFactory:
    """
    Factory for creating target generators from configuration.

    This factory allows registration of new generator types and creation
    of generators by name with parameters.
    """

    _GENERATORS: dict[str, type[TargetGenerator]] = {}

    @classmethod
    def register(cls, name: str, generator_class: type[TargetGenerator]) -> None:
        """
        Register a new target generator type.

        Args:
            name: Unique name for the generator type
            generator_class: Class that implements TargetGenerator interface

        Raises:
            ValueError: If name is already registered or class doesn't implement interface
        """
        if name in cls._GENERATORS:
            raise ValueError(f"Generator type '{name}' is already registered")

        if not issubclass(generator_class, TargetGenerator):
            raise ValueError("Generator class must implement TargetGenerator interface")

        cls._GENERATORS[name] = generator_class

    @classmethod
    def create(cls, generator_type: str, **kwargs: Any) -> TargetGenerator:
        """
        Create a target generator by name.

        Args:
            generator_type: Name of registered generator type
            **kwargs: Parameters to pass to generator constructor

        Returns:
            Configured TargetGenerator instance

        Raises:
            ValueError: If generator type is not registered
        """
        if generator_type not in cls._GENERATORS:
            available = sorted(cls._GENERATORS.keys())
            raise ValueError(
                f"Unknown generator type: '{generator_type}'. Available types: {available}"
            )

        generator_class = cls._GENERATORS[generator_type]
        return generator_class(**kwargs)

    @classmethod
    def list_available(cls) -> dict[str, str]:
        """
        List all available generator types.

        Returns:
            Dict mapping generator names to class names
        """
        return {name: cls.__name__ for name, cls in cls._GENERATORS.items()}

    @classmethod
    def get_generator_info(cls, generator_type: str) -> dict[str, Any]:
        """
        Get information about a specific generator type.

        Args:
            generator_type: Name of registered generator type

        Returns:
            Dict with generator metadata

        Raises:
            ValueError: If generator type is not registered
        """
        if generator_type not in cls._GENERATORS:
            raise ValueError(f"Unknown generator type: '{generator_type}'")

        generator_class = cls._GENERATORS[generator_type]

        # Create a temporary instance to get info (if possible)
        try:
            # Try to create with no args (for generators with all defaults)
            temp_instance = generator_class()
            info = temp_instance.get_target_info()
        except Exception:
            # If that fails, just return basic class info
            info = {
                "class_name": generator_class.__name__,
                "target_type": "unknown",
                "description": generator_class.__doc__ or "No description available",
            }

        return info


# Auto-register built-in generators when module is imported
def _register_builtin_generators():
    """Register all built-in generator types."""
    try:
        from .classification import (
            GlobalThresholdClassificationGenerator,
            QuantileClassificationGenerator,
        )

        TargetGeneratorFactory.register("quantile_classification", QuantileClassificationGenerator)
        TargetGeneratorFactory.register(
            "global_threshold_classification", GlobalThresholdClassificationGenerator
        )

    except ImportError:
        pass  # Classification generators not available

    try:
        from .regression import (
            CumulativeReturnsGenerator,
            DirectionalMFEGenerator,
            LogReturnHorizonsGenerator,
            PriceMovementGenerator,
            RemainingValueTunerGenerator,
            VolatilityGenerator,
            VolatilityScaledReturnsGenerator,
        )

        TargetGeneratorFactory.register("directional_mfe", DirectionalMFEGenerator)
        TargetGeneratorFactory.register("log_return_horizons", LogReturnHorizonsGenerator)
        TargetGeneratorFactory.register("price_movement", PriceMovementGenerator)
        TargetGeneratorFactory.register("volatility", VolatilityGenerator)
        TargetGeneratorFactory.register("cumulative_returns", CumulativeReturnsGenerator)
        TargetGeneratorFactory.register(
            "volatility_scaled_returns", VolatilityScaledReturnsGenerator
        )
        TargetGeneratorFactory.register("remaining_value_tuner", RemainingValueTunerGenerator)

    except ImportError:
        pass  # Regression generators not available

    try:
        from .tstrends_labeling import (
            BinaryCTLGenerator,
            OracleBinaryTrendGenerator,
            OracleTernaryTrendGenerator,
            TernaryCTLGenerator,
            TunedTrendGenerator,
        )

        TargetGeneratorFactory.register("binary_ctl", BinaryCTLGenerator)
        TargetGeneratorFactory.register("ternary_ctl", TernaryCTLGenerator)
        TargetGeneratorFactory.register("oracle_binary", OracleBinaryTrendGenerator)
        TargetGeneratorFactory.register("oracle_ternary", OracleTernaryTrendGenerator)
        TargetGeneratorFactory.register("tuned_trend", TunedTrendGenerator)

    except ImportError:
        pass  # TStrends generators not available


# Register built-in generators
_register_builtin_generators()
