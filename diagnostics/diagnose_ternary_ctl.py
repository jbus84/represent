#!/usr/bin/env python3
"""
Diagnose Ternary CTL Issues

Investigate why Ternary CTL shows:
1. Extreme class imbalance (1.2% balance score with 0.6% middle class)
2. Same negative mean return per trade as Binary CTL (-0.000028)

This suggests the optimized parameters may be creating degenerate labels.
"""

import numpy as np
import polars as pl
from pathlib import Path

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    from tstrends.trend_labelling import TernaryCTL
    from tstrends.returns_estimation import ReturnsEstimatorWithFees
    from tstrends.returns_estimation.fees_config import FeesConfig
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


def analyze_ternary_ctl_parameters():
    """Analyze the optimized Ternary CTL parameters and their effects."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("🔬 Ternary CTL Parameter Analysis")
    print("=" * 50)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(10000)
    prices = test_df["mid_price"].to_numpy()
    
    print(f"Test data: {len(test_df):,} samples")
    print(f"Price range: {prices.min():.6f} to {prices.max():.6f}")
    print(f"Price std: {prices.std():.8f}")
    print()
    
    # Test different parameter combinations
    print("📊 PARAMETER TESTING:")
    print("-" * 50)
    
    # Optimized parameters (from optimization result)
    optimized_params = {
        "marginal_change_thres": 0.0446,  # From optimization
        "window_size": 501                # From optimization
    }
    
    # Test with different thresholds to understand the issue
    test_params = [
        {"marginal_change_thres": 0.0446, "window_size": 501, "name": "OPTIMIZED"},
        {"marginal_change_thres": 0.001, "window_size": 10, "name": "LOW_THRESH"},
        {"marginal_change_thres": 0.01, "window_size": 50, "name": "MEDIUM_THRESH"},
        {"marginal_change_thres": 0.02, "window_size": 100, "name": "DEFAULT"},
    ]
    
    for i, params in enumerate(test_params, 1):
        name = params.pop("name")
        print(f"{i}. {name} (thres={params['marginal_change_thres']:.4f}, win={params['window_size']}):")
        
        try:
            # Generate labels using our wrapped generator
            generator = TargetGeneratorFactory.create("ternary_ctl", **params)
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            our_labels = targets_df[target_col].to_numpy()
            
            # Generate labels using raw TStrends for comparison
            price_list = [float(p) for p in prices.tolist()]
            raw_labeller = TernaryCTL(**params)
            raw_labels = np.array(raw_labeller.get_labels(price_list), dtype=np.int32)
            
            # Analyze distributions
            our_unique, our_counts = np.unique(our_labels, return_counts=True)
            raw_unique, raw_counts = np.unique(raw_labels, return_counts=True)
            
            our_pcts = our_counts / len(our_labels) * 100
            raw_pcts = raw_counts / len(raw_labels) * 100
            
            print(f"   Our labels {set(our_unique)}: {dict(zip(our_unique, our_pcts.round(1)))}")
            print(f"   Raw labels {set(raw_unique)}: {dict(zip(raw_unique, raw_pcts.round(1)))}")
            
            # Check if we have extreme imbalance
            min_pct = min(our_pcts)
            max_pct = max(our_pcts)
            balance_score = min_pct / max_pct * 100
            print(f"   Balance score: {balance_score:.1f}%")
            
            # Calculate PnL using exact optimization logic
            try:
                fees_config = FeesConfig(
                    lp_transaction_fees=0.00007,
                    sp_transaction_fees=0.00007,
                )
                returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
                
                # Convert our labels {0,1,2} to TStrends format {-1,0,1}
                labels_tstrends = our_labels.astype(int) - 1
                
                total_pnl = returns_estimator.estimate_return(
                    prices.tolist(),
                    labels_tstrends.tolist()
                )
                
                # Count trades
                num_trades = sum(1 for j in range(1, len(labels_tstrends)) 
                               if labels_tstrends[j] != labels_tstrends[j-1])
                
                mean_return = total_pnl / num_trades if num_trades > 0 else 0
                print(f"   PnL: {total_pnl:.6f}, Trades: {num_trades:,}, Mean: {mean_return:.8f}")
                
            except Exception as e:
                print(f"   PnL calculation failed: {e}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        print()
    
    print("💡 ANALYSIS:")
    print("If optimized parameters create extreme class imbalance,")
    print("it suggests the threshold is too high for this data's volatility.")
    print("The high window size (501) may also cause over-smoothing.")


def test_micro_volatility_impact():
    """Test how micro-volatility affects Ternary CTL performance."""
    if not LIBRARIES_AVAILABLE:
        return
    
    print("\n📈 MICRO-VOLATILITY ANALYSIS:")
    print("-" * 50)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(50000)  # Larger sample
    prices = test_df["mid_price"].to_numpy()
    
    # Calculate price change statistics
    price_changes = np.diff(prices)
    zero_changes = np.sum(price_changes == 0)
    non_zero_changes = np.sum(price_changes != 0)
    zero_pct = zero_changes / len(price_changes) * 100
    
    print(f"Price change analysis over {len(price_changes):,} periods:")
    print(f"  Zero changes: {zero_changes:,} ({zero_pct:.1f}%)")
    print(f"  Non-zero changes: {non_zero_changes:,} ({100-zero_pct:.1f}%)")
    print(f"  Std of non-zero changes: {price_changes[price_changes != 0].std():.8f}")
    print(f"  Mean absolute change: {np.abs(price_changes).mean():.8f}")
    print()
    
    # Test if threshold is too high for this volatility
    threshold = 0.0446  # Optimized threshold
    mean_abs_change = np.abs(price_changes).mean()
    
    print(f"Threshold vs Actual Volatility:")
    print(f"  Optimized threshold: {threshold:.4f}")
    print(f"  Mean absolute change: {mean_abs_change:.8f}")
    print(f"  Ratio (thresh/volatility): {threshold/mean_abs_change:.1f}x")
    print()
    
    if threshold > mean_abs_change * 10:
        print("🚨 ISSUE IDENTIFIED:")
        print("The optimized threshold is WAY too high for this data's volatility!")
        print("This explains the extreme class imbalance.")


def main():
    """Run comprehensive Ternary CTL diagnosis."""
    try:
        analyze_ternary_ctl_parameters()
        test_micro_volatility_impact()
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()