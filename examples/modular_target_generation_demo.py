#!/usr/bin/env python3
"""
Modular Target Generation Demo

This demo shows how to use the new modular target generation system to create
datasets with multiple target types (classification and regression) using
pluggable target generators.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent import (
    DirectionalMFEGenerator,
    ModularDatasetBuilder,
    PriceMovementGenerator,
    QuantileClassificationGenerator,
    TargetGeneratorFactory,
    VolatilityGenerator,
    create_modular_builder,
)


def demo_single_target_generation():
    """Demo: Single target generation."""
    print("🎯 DEMO 1: Single Target Generation")
    print("=" * 50)

    # Create sample market data
    sample_data = create_sample_market_data()

    # Classification only
    print("\n📊 Classification Target Only:")
    classification_gen = QuantileClassificationGenerator(nbins=13)
    builder = ModularDatasetBuilder([classification_gen])

    dataset = builder.build_dataset(sample_data)
    print(f"   Result columns: {dataset.columns}")
    print(f"   Classification labels: {sorted(dataset['classification_label'].unique().to_list())}")

    # Regression only
    print("\n📈 Regression Target Only:")
    mfe_gen = DirectionalMFEGenerator(lookforward_horizon=1000)
    builder = ModularDatasetBuilder([mfe_gen])

    dataset = builder.build_dataset(sample_data)
    print(f"   Result columns: {dataset.columns}")
    print(
        f"   MFE buy range: [{dataset['mfe_buy_bps'].min():.1f}, {dataset['mfe_buy_bps'].max():.1f}] BPS"
    )


def demo_multi_target_generation():
    """Demo: Multiple target generation."""
    print("\n\n🎯 DEMO 2: Multi-Target Generation")
    print("=" * 50)

    # Create sample market data
    sample_data = create_sample_market_data()

    # Combine multiple target generators
    generators = [
        QuantileClassificationGenerator(nbins=13, target_name="price_direction_13class"),
        DirectionalMFEGenerator(lookforward_horizon=1000, target_names=("mfe_buy", "mfe_sell")),
        PriceMovementGenerator(lookforward_window=500, target_name="price_movement"),
        VolatilityGenerator(window_size=200, target_name="volatility"),
    ]

    builder = ModularDatasetBuilder(generators)
    dataset = builder.build_dataset(sample_data)

    print("\n📊 Multi-target dataset created:")
    print(f"   Total columns: {len(dataset.columns)}")
    print(
        f"   Target columns: {[col for col in dataset.columns if col not in sample_data.columns]}"
    )

    # Show target statistics
    print("\n📈 Target Statistics:")
    print(f"   Classification classes: {len(dataset['price_direction_13class'].unique())}")
    print(f"   MFE buy valid values: {dataset['mfe_buy'].count():,}")
    print(f"   Price movement valid values: {dataset['price_movement'].count():,}")
    print(f"   Volatility valid values: {dataset['volatility'].count():,}")


def demo_factory_pattern():
    """Demo: Using factory pattern for configuration."""
    print("\n\n🎯 DEMO 3: Factory Pattern Configuration")
    print("=" * 50)

    # Create sample market data
    sample_data = create_sample_market_data()

    # Create generators using factory
    print("\n🏭 Creating generators using factory:")

    # List available generator types
    available = TargetGeneratorFactory.list_available()
    print(f"   Available generators: {list(available.keys())}")

    # Create generators by name
    generators = [
        TargetGeneratorFactory.create("quantile_classification", nbins=13),
        TargetGeneratorFactory.create("directional_mfe", lookforward_horizon=1000),
        TargetGeneratorFactory.create("volatility", window_size=300),
    ]

    builder = ModularDatasetBuilder(generators)
    dataset = builder.build_dataset(sample_data)

    print("\n📊 Factory-created dataset:")
    print(f"   Columns: {dataset.columns}")

    # Show builder info
    builder_info = builder.get_builder_info()
    print("\n🔍 Builder Information:")
    print(f"   Total generators: {builder_info['total_generators']}")
    print(f"   Classification targets: {len(builder_info['classification_generators'])}")
    print(f"   Regression targets: {len(builder_info['regression_generators'])}")


def demo_configuration_based_creation():
    """Demo: Configuration-based builder creation."""
    print("\n\n🎯 DEMO 4: Configuration-Based Creation")
    print("=" * 50)

    # Create sample market data
    sample_data = create_sample_market_data()

    # Define configuration
    generator_configs = [
        {"type": "quantile_classification", "nbins": 13, "target_name": "direction_13class"},
        {
            "type": "directional_mfe",
            "lookforward_horizon": 1500,
            "target_names": ("mfe_long", "mfe_short"),
        },
        {"type": "price_movement", "lookforward_window": 800, "target_name": "price_change_bps"},
    ]

    # Create builder from configuration
    builder = create_modular_builder(generator_configs)
    dataset = builder.build_dataset(sample_data)

    print("\n📊 Configuration-based dataset:")
    print(f"   Total rows: {len(dataset):,}")
    print(
        f"   Target columns: {[col for col in dataset.columns if col not in sample_data.columns]}"
    )


def demo_custom_target_generator():
    """Demo: Custom target generator."""
    print("\n\n🎯 DEMO 5: Custom Target Generator")
    print("=" * 50)

    from represent.target_generators.base import TargetGenerator

    class MomentumGenerator(TargetGenerator):
        """Custom momentum-based target generator."""

        def __init__(self, momentum_window: int = 500):
            self.momentum_window = momentum_window

        def generate_targets(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
            """Generate momentum score targets."""
            mid_prices = df["mid_price"].to_numpy()
            momentum = np.full(len(mid_prices), np.nan)

            for i in range(self.momentum_window, len(mid_prices)):
                # Simple momentum: current price vs average of past window
                past_avg = np.mean(mid_prices[i - self.momentum_window : i])
                current_price = mid_prices[i]
                momentum[i] = ((current_price - past_avg) / past_avg) * 10000  # BPS

            return {"momentum_score": momentum}

        def get_target_info(self) -> dict[str, any]:
            return {
                "target_names": ["momentum_score"],
                "target_type": "regression",
                "description": f"Momentum score over {self.momentum_window} tick window",
                "parameters": {"momentum_window": self.momentum_window},
            }

        @property
        def target_type(self) -> str:
            return "regression"

        @property
        def required_columns(self) -> list[str]:
            return ["mid_price"]

    # Register custom generator
    TargetGeneratorFactory.register("momentum", MomentumGenerator)

    # Create sample market data
    sample_data = create_sample_market_data()

    # Use custom generator
    custom_gen = MomentumGenerator(momentum_window=300)
    builder = ModularDatasetBuilder([custom_gen])
    dataset = builder.build_dataset(sample_data)

    print("\n📊 Custom momentum target:")
    print(f"   Momentum column: {'momentum_score' in dataset.columns}")
    print(f"   Valid momentum values: {dataset['momentum_score'].count():,}")

    # Also test factory creation
    factory_gen = TargetGeneratorFactory.create("momentum", momentum_window=400)
    print(f"   Factory-created generator: {factory_gen.__class__.__name__}")


def create_sample_market_data(n_rows: int = 10000) -> pl.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)  # For reproducible results

    # Generate realistic-looking price data
    base_price = 0.6500  # AUDUSD-like price
    price_changes = np.random.normal(0, 0.0001, n_rows)  # Small random changes
    prices = base_price + np.cumsum(price_changes)

    # Ensure prices stay positive
    prices = np.maximum(prices, 0.5000)

    # Create timestamps
    timestamps = np.arange(n_rows) * 1000  # Millisecond timestamps

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "mid_price": prices,
            "volume": np.random.exponential(1000, n_rows),  # Volume data
        }
    )


def main():
    """Run all demos."""
    print("🚀 MODULAR TARGET GENERATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo shows how to use the new pluggable target generation system")
    print("to create datasets with multiple target types.")

    try:
        demo_single_target_generation()
        demo_multi_target_generation()
        demo_factory_pattern()
        demo_configuration_based_creation()
        demo_custom_target_generator()

        print("\n\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\n💡 Key Benefits:")
        print("   ✅ Pluggable target generators")
        print("   ✅ Mix classification and regression targets")
        print("   ✅ Easy to add custom labeling logic")
        print("   ✅ Configuration-driven target creation")
        print("   ✅ Factory pattern for generator management")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
