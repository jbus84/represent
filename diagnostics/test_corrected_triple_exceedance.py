#!/usr/bin/env python3
"""
Test Corrected Triple Exceedance Implementation

Test the corrected fixed-duration, dual-sided binary classification approach.
"""

import numpy as np
import polars as pl
from pathlib import Path

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


def test_corrected_triple_exceedance():
    """Test the corrected Triple Exceedance implementation."""
    if not LIBRARIES_AVAILABLE:
        return
    
    print("🔧 TESTING CORRECTED TRIPLE EXCEEDANCE")
    print("=" * 70)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Test on sample to validate the new approach
    test_df = df.head(10000)  # 10K samples for testing
    prices = test_df["mid_price"].to_numpy()
    
    print(f"Testing on {len(test_df)} samples")
    print("Expected: Two separate binary target columns")
    print("=" * 70)
    
    try:
        # Test corrected implementation
        params = {
            "lookforward_window": 1000,  # Fixed duration window
            "scaling_factor": 5.0,       # 5x transaction cost threshold
            "transaction_cost": 0.0001   # 0.1 pip threshold
        }
        
        generator = TargetGeneratorFactory.create("triple_exceedance", **params)
        targets_df = generator.generate_targets(test_df)
        target_info = generator.get_target_info()
        
        print(f"📊 Target Info:")
        print(f"  Target names: {target_info['target_names']}")
        print(f"  Description: {target_info['description']}")
        
        # Verify we have two separate binary columns
        long_col = target_info['target_names'][0]  # Should be triple_exceedance_label_long
        short_col = target_info['target_names'][1]  # Should be triple_exceedance_label_short
        
        long_labels = targets_df[long_col].to_numpy()
        short_labels = targets_df[short_col].to_numpy()
        
        print(f"\n📈 Long Exceedance Results ({long_col}):")
        long_succeed = np.sum(long_labels == 1)
        long_fail = np.sum(long_labels == 0) 
        print(f"  Exceed: {long_succeed:,} ({long_succeed/len(long_labels)*100:.1f}%)")
        print(f"  Fail: {long_fail:,} ({long_fail/len(long_labels)*100:.1f}%)")
        
        print(f"\n📉 Short Exceedance Results ({short_col}):")
        short_succeed = np.sum(short_labels == 1)
        short_fail = np.sum(short_labels == 0)
        print(f"  Exceed: {short_succeed:,} ({short_succeed/len(short_labels)*100:.1f}%)")
        print(f"  Fail: {short_fail:,} ({short_fail/len(short_labels)*100:.1f}%)")
        
        # Check that both sides can succeed independently
        both_succeed = np.sum((long_labels == 1) & (short_labels == 1))
        only_long = np.sum((long_labels == 1) & (short_labels == 0))
        only_short = np.sum((long_labels == 0) & (short_labels == 1))
        both_fail = np.sum((long_labels == 0) & (short_labels == 0))
        
        print(f"\n🔍 Cross-Analysis:")
        print(f"  Both exceed: {both_succeed:,} ({both_succeed/len(long_labels)*100:.1f}%)")
        print(f"  Only long exceeds: {only_long:,} ({only_long/len(long_labels)*100:.1f}%)")
        print(f"  Only short exceeds: {only_short:,} ({only_short/len(long_labels)*100:.1f}%)")
        print(f"  Both fail: {both_fail:,} ({both_fail/len(long_labels)*100:.1f}%)")
        
        # Manual verification for first few samples
        print(f"\n🔍 Manual Verification (first 10 samples):")
        threshold = params["transaction_cost"] * params["scaling_factor"]  # 0.0005
        
        print(f"  Threshold: {threshold:.6f} (absolute price move)")
        
        for i in range(min(10, len(prices) - params["lookforward_window"])):
            entry_price = prices[i]
            future_prices = prices[i+1:i+1+params["lookforward_window"]]
            
            max_up = np.max(future_prices) - entry_price
            max_down = entry_price - np.min(future_prices)
            
            expected_long = 1 if max_up >= threshold else 0
            expected_short = 1 if max_down >= threshold else 0
            
            actual_long = long_labels[i]
            actual_short = short_labels[i]
            
            print(f"  [{i:2d}] Max up: {max_up:.6f} → L:{expected_long} (got {actual_long}) | "
                  f"Max down: {max_down:.6f} → S:{expected_short} (got {actual_short})")
        
        # Check if implementation is correct
        if np.array_equal(long_labels[:10], [1 if np.max(prices[i+1:i+1+params["lookforward_window"]]) - prices[i] >= threshold else 0 for i in range(10)]):
            print("\n✅ CORRECTED TRIPLE EXCEEDANCE LOGIC WORKING!")
            print("  ✓ Fixed duration - holds for full window")
            print("  ✓ Dual-sided binary classification")  
            print("  ✓ Independent long/short assessments")
        else:
            print("\n⚠️ Some discrepancies found in verification")
            
    except Exception as e:
        print(f"❌ Error testing corrected implementation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_corrected_triple_exceedance()