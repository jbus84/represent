#!/usr/bin/env python3
"""
Test Windowed Enhanced Output Calculation

Test that the updated enhanced output calculation uses the same windowing 
strategy as optimization and produces more consistent results.
"""

import numpy as np
import polars as pl
from pathlib import Path

def test_windowed_calculation():
    """Test the windowed enhanced output calculation."""
    print("🧪 Testing Windowed Enhanced Output Calculation")
    print("=" * 50)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    prices = df["mid_price"].to_numpy()[:100000]  # Use 100K samples for testing
    
    print(f"Test data: {len(prices):,} samples")
    print(f"Price range: {prices.min():.6f} to {prices.max():.6f}")
    print()
    
    # Import the function we just updated
    import sys
    sys.path.append('/Users/danielfisher/repositories/represent/examples')
    from symbol_optimization_runner import calculate_additional_metrics
    
    # Test with Binary CTL optimized parameters
    binary_params = {"omega": 0.0}  # From optimization results
    print("🎯 Testing Binary CTL with windowed calculation:")
    
    try:
        metrics = calculate_additional_metrics(prices, "binary_ctl", binary_params)
        
        if "error" in metrics:
            print(f"   ❌ Error: {metrics['error']}")
        else:
            print(f"   ⚖️  Class balance: {metrics['class_balance_score']:.1f}% ({metrics['num_classes']} classes)")
            print(f"   📊 Label distribution: {metrics['label_distribution']}")
            print(f"   🪟 Windows: {metrics['valid_windows']}/{metrics['total_windows']} valid")
            print(f"   🔄 Trades: {metrics['num_trades']:,}/window ({metrics['trading_frequency']:.2f}% frequency)")
            print(f"   💰 Mean return/trade: {metrics['mean_return_per_trade']:.6f}")
            print(f"   📈 Total PnL: {metrics['total_pnl']:.6f}")
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test with Ternary CTL optimized parameters
    ternary_params = {"marginal_change_thres": 0.0446, "window_size": 501}
    print("🎯 Testing Ternary CTL with windowed calculation:")
    
    try:
        metrics = calculate_additional_metrics(prices, "ternary_ctl", ternary_params)
        
        if "error" in metrics:
            print(f"   ❌ Error: {metrics['error']}")
        else:
            print(f"   ⚖️  Class balance: {metrics['class_balance_score']:.1f}% ({metrics['num_classes']} classes)")
            print(f"   📊 Label distribution: {metrics['label_distribution']}")
            print(f"   🪟 Windows: {metrics['valid_windows']}/{metrics['total_windows']} valid")
            print(f"   🔄 Trades: {metrics['num_trades']:,}/window ({metrics['trading_frequency']:.2f}% frequency)")
            print(f"   💰 Mean return/trade: {metrics['mean_return_per_trade']:.6f}")
            print(f"   📈 Total PnL: {metrics['total_pnl']:.6f}")
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("💡 Summary:")
    print("- Windowed calculation should now better match optimization results")
    print("- Class balance should be more representative across sampled windows")
    print("- Mean return per trade should align with optimization findings")


if __name__ == "__main__":
    test_windowed_calculation()