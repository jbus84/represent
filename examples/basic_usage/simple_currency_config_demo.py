"""
Simple currency configuration demonstration.

Shows how to use the new RepresentConfig system with different currency
configurations and parameters.
"""

from represent import RepresentConfig


def demo_currency_configs():
    """Demonstrate currency-specific configurations."""
    
    print("💱 Currency Configuration Demo")
    print("="*35)
    
    # Default AUDUSD configuration
    print("\n1. Default AUDUSD Configuration:")
    audusd_config = RepresentConfig()
    print(f"   Currency: {audusd_config.currency}")
    print(f"   Lookback rows: {audusd_config.lookback_rows}")
    print(f"   Lookforward input: {audusd_config.lookforward_input}")
    print(f"   Batch size: {audusd_config.batch_size}")
    print(f"   Features: {audusd_config.features}")
    print(f"   NBins: {audusd_config.nbins}")
    
    # Custom GBPUSD configuration
    print("\n2. Custom GBPUSD Configuration:")
    gbpusd_config = RepresentConfig(
        currency="GBPUSD",
        lookback_rows=4000,      # Custom value
        lookforward_input=3000,  # Custom value
        batch_size=1500,         # Custom value
        features=["volume", "variance"]  # Multiple features
    )
    print(f"   Currency: {gbpusd_config.currency}")
    print(f"   Lookback rows: {gbpusd_config.lookback_rows}")
    print(f"   Lookforward input: {gbpusd_config.lookforward_input}")
    print(f"   Batch size: {gbpusd_config.batch_size}")
    print(f"   Features: {gbpusd_config.features}")
    print(f"   NBins: {gbpusd_config.nbins}")
    
    # High-frequency trading configuration
    print("\n3. High-Frequency Trading Configuration:")
    hft_config = RepresentConfig(
        currency="EURJPY",
        lookback_rows=1000,      # Shorter for speed
        lookforward_input=500,   # Shorter prediction window
        batch_size=500,          # Smaller batches
        features=["volume"],     # Single feature for speed
        nbins=7                  # Fewer classes for speed
    )
    print(f"   Currency: {hft_config.currency}")
    print(f"   Lookback rows: {hft_config.lookback_rows}")
    print(f"   Lookforward input: {hft_config.lookforward_input}")
    print(f"   Batch size: {hft_config.batch_size}")
    print(f"   Features: {hft_config.features}")
    print(f"   NBins: {hft_config.nbins}")
    
    # Research configuration with all features
    print("\n4. Research Configuration (All Features):")
    research_config = RepresentConfig(
        currency="AUDUSD",
        lookback_rows=8000,      # Long context
        lookforward_input=6000,  # Long prediction window
        batch_size=2000,         # Large batches
        features=["volume", "variance", "trade_counts"],  # All features
        nbins=13                 # Detailed analysis (supported values: 3, 5, 7, 9, 13)
    )
    print(f"   Currency: {research_config.currency}")
    print(f"   Lookback rows: {research_config.lookback_rows}")
    print(f"   Lookforward input: {research_config.lookforward_input}")
    print(f"   Batch size: {research_config.batch_size}")
    print(f"   Features: {research_config.features}")
    print(f"   NBins: {research_config.nbins}")
    
    print("\n✅ All configurations created successfully!")
    print("\n💡 Key benefits of new config system:")
    print("   - No hardcoded values (lookback/lookforward fully configurable)")
    print("   - Simple, flat structure (no complex nesting)")
    print("   - Currency-specific optimizations")
    print("   - Flexible feature combinations")
    print("   - Validation and type safety")


if __name__ == "__main__":
    demo_currency_configs()