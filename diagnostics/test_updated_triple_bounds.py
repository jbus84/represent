#!/usr/bin/env python3
"""
Test Updated Triple Methods Bounds

Validate that the new micro-volatility optimized bounds generate
more reasonable results than the degenerate solutions we found.
"""

import numpy as np
import polars as pl
from pathlib import Path

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    from tstrends.returns_estimation import ReturnsEstimatorWithFees
    from tstrends.returns_estimation.fees_config import FeesConfig
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


def test_new_bounds():
    """Test the new micro-volatility optimized bounds."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("🧪 Testing Updated Triple Methods Bounds")
    print("=" * 60)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(30000)  # 30K samples
    prices = test_df["mid_price"].to_numpy()
    
    print(f"Test data: {len(test_df):,} samples")
    print(f"Price range: {prices.min():.6f} to {prices.max():.6f}")
    print()
    
    # Test configurations with new bounds
    test_configs = [
        {
            "method": "triple_barrier",
            "name": "NEW Triple Barrier Bounds",
            "params": {
                "lookforward_window": 500,         # Mid-range from new bounds (200-2000)
                "barrier_width": 0.00002,         # Mid-range from new bounds (0.000005-0.0001)
                "min_return_threshold": 1e-7,     # Mid-range from new bounds (1e-8 to 1e-6)
                "volatility_window": 100,         # Mid-range from new bounds (50-200)
                "normalize_by_volatility": False,
            }
        },
        {
            "method": "triple_exceedance", 
            "name": "NEW Triple Exceedance Bounds",
            "params": {
                "lookforward_window": 300,         # Mid-range from new bounds (100-1000)
                "scaling_factor": 2.5,            # Mid-range from new bounds (1.1-5.0)
                "min_exceedance_threshold": 0.2,  # Mid-range from new bounds (0.05-0.5)
                "volatility_window": 100,         # Mid-range from new bounds (20-200)
                "window_penalty_weight": 0.1,     # Mid-range from new bounds (0.01-0.2)
                "balance_weight": 0.8,            # High to avoid degenerate solutions
                "target_balance_ratio": 0.3,      # Mid-range from new bounds (0.20-0.40)
                "adaptive_scaling": True,
            }
        }
    ]
    
    for i, config in enumerate(test_configs, 1):
        method = config["method"]
        name = config["name"]
        params = config["params"]
        
        print(f"{i}. {name}:")
        print(f"   Key params: lookforward={params['lookforward_window']}")
        if method == "triple_barrier":
            print(f"   Barrier width: {params['barrier_width']:.8f} ({params['barrier_width']*100:.4f}%)")
        else:
            print(f"   Scaling factor: {params['scaling_factor']:.1f}")
            print(f"   Min threshold: {params['min_exceedance_threshold']:.2f}")
        
        try:
            # Generate labels
            generator = TargetGeneratorFactory.create(method, **params)
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Analyze labels
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            balance_score = min(percentages) / max(percentages) * 100 if len(percentages) > 1 else 100
            
            print(f"   Classes: {len(unique_labels)} ({set(unique_labels)})")
            print(f"   Distribution: {dict(zip(unique_labels, percentages.round(1)))}")
            print(f"   Balance score: {balance_score:.1f}%")
            
            # Check if degenerate (single class)
            if len(unique_labels) == 1:
                print(f"   ⚠️  DEGENERATE: Only one class generated!")
            else:
                print(f"   ✅ Multi-class solution generated!")
            
            # Calculate PnL using exact optimization logic
            try:
                fees_config = FeesConfig(
                    lp_transaction_fees=0.00007,
                    sp_transaction_fees=0.00007,
                )
                returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
                
                # Convert labels to TStrends format
                labels_int = labels.astype(int)
                unique_set = set(np.unique(labels_int))
                
                # Both methods should use {-1, 0, 1} format
                if unique_set.issubset({-1, 0, 1}):
                    labels_tstrends = labels_int
                else:
                    # Fallback if needed
                    labels_tstrends = labels_int
                
                total_pnl = returns_estimator.estimate_return(
                    prices.tolist(),
                    labels_tstrends.tolist()
                )
                
                # Count trades
                num_trades = sum(1 for j in range(1, len(labels_tstrends)) 
                               if labels_tstrends[j] != labels_tstrends[j-1])
                
                mean_return = total_pnl / num_trades if num_trades > 0 else 0
                print(f"   PnL: {total_pnl:.6f} ({total_pnl*100:.2f}%)")
                print(f"   Trades: {num_trades:,}, Mean: {mean_return:.8f}")
                
                # Check if meaningful improvement
                if total_pnl > 0.001:  # > 0.1%
                    print(f"   🎯 POSITIVE returns achieved!")
                elif num_trades > 10 and len(unique_labels) > 1:
                    print(f"   📈 Multi-class with trading activity")
                else:
                    print(f"   ⚠️  Still needs improvement")
                
            except Exception as e:
                print(f"   PnL calculation failed: {e}")
                
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")
        
        print()


def compare_old_vs_new():
    """Compare old degenerate parameters vs new bounds."""
    print("📊 Old vs New Bounds Comparison:")
    print("=" * 60)
    
    print("Triple Barrier:")
    print("  OLD: barrier_width=(0.0001, 0.005) → found 0.3035% → degenerate")  
    print("  NEW: barrier_width=(0.000005, 0.0001) → should find ~0.002%")
    print()
    
    print("Triple Exceedance:")
    print("  OLD: scaling_factor=(2.0, 20.0) → found 19.3x → degenerate")
    print("  NEW: scaling_factor=(1.1, 5.0) → should find ~2.5x")
    print()
    
    print("Expected improvements:")
    print("  ✅ Multi-class label generation (not 100% neutral)")
    print("  ✅ More trading activity (not 0 trades)")
    print("  ✅ Parameters proportional to micro-volatility")
    print("  ⚠️  May still be challenging due to transaction costs")


def main():
    """Run the updated bounds validation."""
    try:
        compare_old_vs_new()
        print()
        test_new_bounds()
        
        print("💡 VALIDATION SUMMARY:")
        print("=" * 60)
        print("The updated bounds should:")
        print("1. Generate multi-class solutions (not degenerate single-class)")
        print("2. Create meaningful trading activity") 
        print("3. Scale appropriately with micro-volatility characteristics")
        print("4. Avoid the 100% neutral label trap")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()