#!/usr/bin/env python3
"""
Demonstration of the new simplified RepresentConfig system.

This example shows how the new flat configuration structure makes it easy
to specify lookback and lookforward parameters without complex nested structures.
"""

from represent import RepresentConfig, create_represent_config

def main():
    print("🎯 RepresentConfig Demonstration")
    print("=" * 50)
    
    # 1. Create default configuration
    print("\n1. Default Configuration:")
    config = RepresentConfig()
    print(f"   Currency: {config.currency}")
    print(f"   NBins: {config.nbins}")
    print(f"   Samples: {config.samples:,}")
    print(f"   Features: {config.features}")
    print(f"   Lookback Rows: {config.lookback_rows:,}")
    print(f"   Lookforward Input: {config.lookforward_input:,}")
    print(f"   Batch Size: {config.batch_size:,}")
    
    # 2. Create customized configuration with different lookback/lookforward
    print("\n2. Customized Configuration (No more hardcoded 2000!):")
    custom_config = RepresentConfig(
        currency="GBPUSD",
        nbins=9,
        samples=50000,
        features=["volume", "variance"],
        lookback_rows=3000,    # Fully configurable!
        lookforward_input=4000, # Fully configurable!  
        batch_size=1500,       # No more hardcoded 2000!
    )
    print(f"   Currency: {custom_config.currency}")
    print(f"   NBins: {custom_config.nbins}")
    print(f"   Samples: {custom_config.samples:,}")
    print(f"   Features: {custom_config.features}")
    print(f"   Lookback Rows: {custom_config.lookback_rows:,}")
    print(f"   Lookforward Input: {custom_config.lookforward_input:,}")
    print(f"   Batch Size: {custom_config.batch_size:,}")
    
    # 3. Show convenience function with currency-specific optimizations
    print("\n3. Currency-Specific Optimizations:")
    currencies = ["AUDUSD", "GBPUSD", "USDJPY"]
    
    for currency in currencies:
        config = create_represent_config(currency=currency)
        print(f"   {currency}:")
        print(f"     Lookforward: {config.lookforward_input:,}")
        print(f"     True Pip Size: {config.true_pip_size}")
        print(f"     NBins: {config.nbins}")
    
    # 4. Show configuration flexibility  
    print("\n4. Configuration Simplicity:")
    print("   ✅ Single flat configuration class")
    print("   ✅ No complex nested structures")
    print(f"   ✅ Direct access to lookback: {custom_config.lookback_rows}")
    print(f"   ✅ Direct access to lookforward: {custom_config.lookforward_input}")
    
    # 5. Show the key improvement - flexible timing parameters
    print("\n5. Flexible Timing Parameters (The Key Fix!):")
    test_configs = [
        {"lookback_rows": 1000, "lookforward_input": 2000},
        {"lookback_rows": 2500, "lookforward_input": 3500},
        {"lookback_rows": 8000, "lookforward_input": 1000},
    ]
    
    for i, config_params in enumerate(test_configs, 1):
        config = RepresentConfig(**config_params)
        print(f"   Config {i}: lookback={config.lookback_rows:,}, lookforward={config.lookforward_input:,} ✅")
    
    print("\n🎉 Configuration is now fully flexible - no more hardcoded values!")

if __name__ == "__main__":
    main()