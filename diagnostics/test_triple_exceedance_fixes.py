#!/usr/bin/env python3
"""
Test Triple Exceedance Fixes

Quick test to verify the Triple Exceedance logic fixes are working correctly.
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


def test_triple_exceedance_fixes():
    """Test the Triple Exceedance fixes."""
    if not LIBRARIES_AVAILABLE:
        return
    
    print("🔧 TESTING TRIPLE EXCEEDANCE FIXES")
    print("=" * 60)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Test on first 10K samples
    test_df = df.head(10000)
    prices = test_df["mid_price"].to_numpy()
    
    # Test both configurations
    configs = [
        {
            "name": "Triple Exceedance (Short - Fixed)",
            "params": {"lookforward_window": 1000, "scaling_factor": 5.0, "transaction_cost": 0.0001}
        },
        {
            "name": "Triple Exceedance (Long - Fixed)", 
            "params": {"lookforward_window": 5000, "scaling_factor": 10.0, "transaction_cost": 0.0001}
        }
    ]
    
    for config in configs:
        print(f"\n📊 Testing {config['name']}")
        print("-" * 50)
        
        try:
            # Generate labels with fixed implementation
            generator = TargetGeneratorFactory.create("triple_exceedance", **config["params"])
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Analyze results
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            
            print(f"Label Distribution:")
            for label_val, pct in zip(unique_labels, percentages):
                if label_val == -1:
                    print(f"  Loss: {pct:.1f}%")
                elif label_val == 0:
                    print(f"  Timeout: {pct:.1f}%")
                elif label_val == 1:
                    print(f"  Profit: {pct:.1f}%")
            
            # Manual verification on first 100 samples
            lookforward = config["params"]["lookforward_window"]
            scaling_factor = config["params"]["scaling_factor"]
            transaction_cost = config["params"]["transaction_cost"]
            barrier_size = transaction_cost * scaling_factor
            
            manual_hits = {"profit": 0, "loss": 0, "timeout": 0}
            discrepancies = 0
            
            debug_samples = min(100, len(prices) - lookforward)
            
            for i in range(debug_samples):
                entry_price = prices[i]
                upper_barrier = entry_price + barrier_size  # Now absolute
                lower_barrier = entry_price - barrier_size  # Now absolute
                
                future_prices = prices[i+1:i+1+lookforward]
                
                if len(future_prices) == 0:
                    manual_result = 0
                else:
                    hit_upper = np.any(future_prices >= upper_barrier)
                    hit_lower = np.any(future_prices <= lower_barrier)
                    
                    if hit_upper and hit_lower:
                        upper_hit_idx = np.argmax(future_prices >= upper_barrier)
                        lower_hit_idx = np.argmax(future_prices <= lower_barrier)
                        manual_result = 1 if upper_hit_idx < lower_hit_idx else -1
                    elif hit_upper:
                        manual_result = 1
                    elif hit_lower:
                        manual_result = -1
                    else:
                        manual_result = 0
                
                # Count manual results
                if manual_result == 1:
                    manual_hits["profit"] += 1
                elif manual_result == -1:
                    manual_hits["loss"] += 1
                else:
                    manual_hits["timeout"] += 1
                
                # Compare with generated
                if manual_result != labels[i]:
                    discrepancies += 1
            
            print(f"Manual Verification (first {debug_samples} samples):")
            print(f"  Manual - Profit: {manual_hits['profit']}, Loss: {manual_hits['loss']}, Timeout: {manual_hits['timeout']}")
            
            generated_sample = labels[:debug_samples]
            gen_profit = np.sum(generated_sample == 1)
            gen_loss = np.sum(generated_sample == -1)
            gen_timeout = np.sum(generated_sample == 0)
            
            print(f"  Generated - Profit: {gen_profit}, Loss: {gen_loss}, Timeout: {gen_timeout}")
            print(f"  Discrepancies: {discrepancies}")
            
            if discrepancies == 0:
                print("✅ TRIPLE EXCEEDANCE LOGIC FIXED!")
            else:
                print(f"⚠️ Still {discrepancies} discrepancies found")
                
        except Exception as e:
            print(f"❌ Error testing {config['name']}: {e}")


if __name__ == "__main__":
    test_triple_exceedance_fixes()