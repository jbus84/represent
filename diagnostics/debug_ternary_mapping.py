#!/usr/bin/env python3
"""
Debug Ternary CTL Label Mapping Issue

Compare optimization logic vs enhanced output logic for ternary labels.
The issue appears to be in how we handle degenerate ternary labels.
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


def test_ternary_label_mapping():
    """Test the exact difference in label mapping between optimization and enhanced output."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("🔬 Ternary CTL Label Mapping Debug")
    print("=" * 50)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(10000)
    prices = test_df["mid_price"].to_numpy()
    
    # Generate Ternary CTL labels with optimized parameters
    optimized_params = {
        "marginal_change_thres": 0.0446,
        "window_size": 501
    }
    
    generator = TargetGeneratorFactory.create("ternary_ctl", **optimized_params)
    targets_df = generator.generate_targets(test_df)
    target_info = generator.get_target_info()
    target_col = target_info['target_names'][0]
    our_labels = targets_df[target_col].to_numpy()
    
    print(f"Generated labels shape: {our_labels.shape}")
    unique_labels, counts = np.unique(our_labels, return_counts=True)
    percentages = counts / len(our_labels) * 100
    print(f"Label distribution: {dict(zip(unique_labels, percentages.round(1)))}")
    print()
    
    # Now test both mapping approaches
    
    # OPTIMIZATION LOGIC (from lines 900-903)
    labels_int = our_labels.astype(int)
    unique_labels_set = np.unique(labels_int[~np.isnan(labels_int)])
    
    print("🎯 OPTIMIZATION MAPPING LOGIC:")
    print(f"Unique labels detected: {set(unique_labels_set)}")
    
    if set(unique_labels_set).issubset({0, 1, 2}):
        # Ternary labels: {0, 1, 2} → {-1, 0, 1}
        labels_tstrends_opt = labels_int - 1
        print(f"Applied ternary mapping: {set(unique_labels_set)} → {set(labels_tstrends_opt)}")
    else:
        labels_tstrends_opt = labels_int
        print("No mapping applied")
    
    # ENHANCED OUTPUT LOGIC (current)
    print("\n🔧 ENHANCED OUTPUT MAPPING LOGIC:")
    if set(unique_labels_set).issubset({0, 1, 2}):
        labels_tstrends_enh = labels_int - 1
        print(f"Applied ternary mapping: {set(unique_labels_set)} → {set(labels_tstrends_enh)}")
    elif len(unique_labels_set) == 2 and set(unique_labels_set).issubset({0, 1}):
        labels_tstrends_enh = labels_int  # This is the problem!
        print(f"Applied binary mapping: {set(unique_labels_set)} → {set(labels_tstrends_enh)}")
    else:
        labels_tstrends_enh = labels_int
        print("No mapping applied")
    
    print()
    print("🔍 COMPARISON:")
    print(f"Optimization labels: {set(labels_tstrends_opt)} (unique: {len(set(labels_tstrends_opt))})")
    print(f"Enhanced output labels: {set(labels_tstrends_enh)} (unique: {len(set(labels_tstrends_enh))})")
    
    # Check if they're different
    if not np.array_equal(labels_tstrends_opt, labels_tstrends_enh):
        print("❌ MAPPINGS ARE DIFFERENT!")
        
        # Count differences
        diff_count = np.sum(labels_tstrends_opt != labels_tstrends_enh)
        print(f"   Different values: {diff_count:,} out of {len(labels_tstrends_opt):,}")
        
        # Show some examples
        diff_indices = np.where(labels_tstrends_opt != labels_tstrends_enh)[0][:5]
        for idx in diff_indices:
            print(f"   Index {idx}: opt={labels_tstrends_opt[idx]}, enh={labels_tstrends_enh[idx]} (orig={our_labels[idx]})")
    else:
        print("✅ Mappings are identical")
    
    # Calculate PnL with both mappings
    print("\n💰 PnL COMPARISON:")
    
    fees_config = FeesConfig(
        lp_transaction_fees=0.00007,
        sp_transaction_fees=0.00007,
    )
    returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
    
    try:
        # Optimization PnL
        pnl_opt = returns_estimator.estimate_return(
            prices.tolist(),
            labels_tstrends_opt.tolist()
        )
        
        # Enhanced output PnL  
        pnl_enh = returns_estimator.estimate_return(
            prices.tolist(),
            labels_tstrends_enh.tolist()
        )
        
        print(f"Optimization PnL: {pnl_opt:.8f}")
        print(f"Enhanced output PnL: {pnl_enh:.8f}")
        print(f"Difference: {pnl_opt - pnl_enh:.8f}")
        
        if abs(pnl_opt - pnl_enh) > 1e-10:
            print("❌ PnL calculations are different!")
        else:
            print("✅ PnL calculations match")
            
    except Exception as e:
        print(f"❌ PnL calculation failed: {e}")


def main():
    """Run the debug analysis."""
    try:
        test_ternary_label_mapping()
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()