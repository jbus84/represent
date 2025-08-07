#!/usr/bin/env python3
"""
New Configuration System Demo

Shows the simplified RepresentConfig system with configurable
lookback/lookforward parameters (no more hardcoded 2000!)
"""

import sys
from pathlib import Path

# Add represent to path for examples
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent import RepresentConfig, create_represent_config


def demo_basic_config():
    """Demo basic RepresentConfig usage."""
    print("🎯 Demo 1: Basic Configuration")
    print("=" * 50)
    
    # Create default config
    config = RepresentConfig()
    
    print("Default RepresentConfig:")
    print(f"  Currency: {config.currency}")
    print(f"  NBins: {config.nbins}")
    print(f"  Samples: {config.samples:,}")
    print(f"  Features: {config.features}")
    print(f"  Lookback Rows: {config.lookback_rows:,}")
    print(f"  Lookforward Input: {config.lookforward_input:,}")
    print(f"  Batch Size: {config.batch_size:,}")
    

def demo_custom_config():
    """Demo custom configuration with no hardcoded values."""
    print("\n🎯 Demo 2: Custom Configuration (No More Hardcoded Values!)")
    print("=" * 60)
    
    # The key fix - fully configurable timing parameters!
    custom_config = RepresentConfig(
        currency="GBPUSD",
        nbins=9,
        samples=50000,
        features=["volume", "variance"],
        lookback_rows=3000,      # ✅ Any value you want!
        lookforward_input=4000,  # ✅ Any value you want!
        batch_size=1500,         # ✅ Any value you want!
        ticks_per_bin=50
    )
    
    print("Custom RepresentConfig:")
    print(f"  Currency: {custom_config.currency}")
    print(f"  NBins: {custom_config.nbins}")
    print(f"  Samples: {custom_config.samples:,}")
    print(f"  Features: {custom_config.features}")
    print(f"  Lookback Rows: {custom_config.lookback_rows:,} ✅ Configurable!")
    print(f"  Lookforward Input: {custom_config.lookforward_input:,} ✅ Configurable!")
    print(f"  Batch Size: {custom_config.batch_size:,} ✅ Configurable!")
    print(f"  Auto-computed time bins: {custom_config.time_bins}")


def demo_currency_optimizations():
    """Demo currency-specific optimizations."""
    print("\n🎯 Demo 3: Currency-Specific Optimizations")
    print("=" * 50)
    
    currencies = ["AUDUSD", "GBPUSD", "USDJPY"]
    
    for currency in currencies:
        config = create_represent_config(currency=currency)
        print(f"\n{currency}:")
        print(f"  Lookforward: {config.lookforward_input:,}")
        print(f"  True Pip Size: {config.true_pip_size}")
        print(f"  NBins: {config.nbins}")


def demo_flexible_parameters():
    """Demo the key improvement - flexible timing parameters."""
    print("\n🎯 Demo 4: Flexible Timing Parameters (The Key Fix!)")
    print("=" * 55)
    
    # Show different combinations that were impossible before
    test_configs = [
        {"lookback_rows": 1000, "lookforward_input": 2000, "batch_size": 500},
        {"lookback_rows": 5000, "lookforward_input": 3000, "batch_size": 2000},
        {"lookback_rows": 10000, "lookforward_input": 1000, "batch_size": 3000},
    ]
    
    for i, config_params in enumerate(test_configs, 1):
        config = RepresentConfig(**config_params)
        print(f"Config {i}:")
        print(f"  Lookback: {config.lookback_rows:,}")
        print(f"  Lookforward: {config.lookforward_input:,}")
        print(f"  Batch Size: {config.batch_size:,}")
        print("  ✅ All parameters fully configurable!")


def demo_validation():
    """Demo configuration validation."""
    print("\n🎯 Demo 5: Built-in Validation")
    print("=" * 40)
    
    try:
        # This will fail with a helpful error message
        RepresentConfig(currency="INVALID")
    except ValueError as e:
        print(f"❌ Validation caught error: {e}")
    
    try:
        # This will also fail with a helpful error message
        RepresentConfig(features=["invalid_feature"])
    except ValueError as e:
        print(f"❌ Validation caught error: {e}")
    
    print("✅ Built-in validation prevents configuration errors!")


if __name__ == "__main__":
    print("🚀 New RepresentConfig System Demo")
    print("=" * 60)
    
    demo_basic_config()
    demo_custom_config()
    demo_currency_optimizations()
    demo_flexible_parameters()
    demo_validation()
    
    print("\n🎉 Configuration is now fully flexible!")
    print("   • No more hardcoded lookback/lookforward values")
    print("   • Simple flat structure (no nested configs)")
    print("   • Currency-specific optimizations")
    print("   • Built-in validation")
    print("   • Auto-computed derived values")