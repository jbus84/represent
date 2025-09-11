#!/usr/bin/env python3
"""
Test Triple Barrier Parameter Type Conversion Fix

Test that the updated parameter conversion handles float-to-int conversion
properly for triple barrier methods.
"""

import numpy as np
import polars as pl
from pathlib import Path

def test_triple_barrier_fix():
    """Test the parameter conversion fix for triple barrier methods."""
    print("🧪 Testing Triple Barrier Parameter Type Conversion Fix")
    print("=" * 60)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    prices = df["mid_price"].to_numpy()[:50000]  # Use 50K samples for testing
    
    print(f"Test data: {len(prices):,} samples")
    print()
    
    # Import the functions we just updated
    import sys
    sys.path.append('/Users/danielfisher/repositories/represent/examples')
    from symbol_optimization_runner import calculate_additional_metrics, convert_params_for_generator
    
    # Test Triple Barrier with float parameters (as they come from optimization)
    triple_barrier_params = {
        'lookforward_window': 1768.0,  # Float from optimization
        'barrier_width': 0.0030348335902746558,
        'min_return_threshold': 5.9698897438064404e-05,
        'volatility_window': 144.9810076690062,  # Float from optimization
        'normalize_by_volatility': 0.07958187305619836,  # Float from optimization (should be bool)
    }
    
    print("🎯 Testing Triple Barrier parameter conversion:")
    print(f"   Original params: {triple_barrier_params}")
    
    try:
        converted_params = convert_params_for_generator("triple_barrier", triple_barrier_params)
        print(f"   Converted params: {converted_params}")
        print(f"   lookforward_window type: {type(converted_params['lookforward_window'])}")
        print(f"   volatility_window type: {type(converted_params['volatility_window'])}")
        print(f"   normalize_by_volatility type: {type(converted_params['normalize_by_volatility'])}")
        
        # Test that the parameters work with the generator
        metrics = calculate_additional_metrics(prices, "triple_barrier", triple_barrier_params)
        
        if "error" in metrics:
            print(f"   ❌ Metrics calculation failed: {metrics['error']}")
        else:
            print(f"   ✅ Metrics calculation succeeded!")
            print(f"   ⚖️  Class balance: {metrics['class_balance_score']:.1f}% ({metrics['num_classes']} classes)")
            print(f"   📊 Label distribution: {metrics['label_distribution']}")
            print(f"   💰 Mean return/trade: {metrics['mean_return_per_trade']:.6f}")
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test Triple Exceedance with float parameters
    triple_exceedance_params = {
        'lookforward_window': 1496.0,  # Float from optimization
        'scaling_factor': 19.302155196862056,
        'min_exceedance_threshold': 0.8642264086386979,
        'volatility_window': 499.39867501976823,  # Float from optimization
        'window_penalty_weight': 0.34978580535424886,
        'balance_weight': 0.5459550636056117,
        'target_balance_ratio': 0.32038938648717,
        'adaptive_scaling': 0.4008657445169629,  # Float from optimization (should be bool)
    }
    
    print("🎯 Testing Triple Exceedance parameter conversion:")
    print(f"   Original lookforward_window: {triple_exceedance_params['lookforward_window']} ({type(triple_exceedance_params['lookforward_window'])})")
    print(f"   Original adaptive_scaling: {triple_exceedance_params['adaptive_scaling']} ({type(triple_exceedance_params['adaptive_scaling'])})")
    
    try:
        converted_params = convert_params_for_generator("triple_exceedance", triple_exceedance_params)
        print(f"   Converted lookforward_window: {converted_params['lookforward_window']} ({type(converted_params['lookforward_window'])})")
        print(f"   Converted adaptive_scaling: {converted_params['adaptive_scaling']} ({type(converted_params['adaptive_scaling'])})")
        
        # Test that the parameters work with the generator
        metrics = calculate_additional_metrics(prices, "triple_exceedance", triple_exceedance_params)
        
        if "error" in metrics:
            print(f"   ❌ Metrics calculation failed: {metrics['error']}")
        else:
            print(f"   ✅ Metrics calculation succeeded!")
            print(f"   ⚖️  Class balance: {metrics['class_balance_score']:.1f}% ({metrics['num_classes']} classes)")
            print(f"   💰 Mean return/trade: {metrics['mean_return_per_trade']:.6f}")
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("💡 Summary:")
    print("- Parameter type conversion should now handle float-to-int and float-to-bool properly")
    print("- Triple barrier methods should no longer throw 'float cannot be interpreted as integer' errors")


if __name__ == "__main__":
    test_triple_barrier_fix()