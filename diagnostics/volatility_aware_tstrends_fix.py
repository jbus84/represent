#!/usr/bin/env python3
"""
Volatility-Aware TStrends Parameter Scaling Fix

This script implements automatic parameter scaling for TStrends methods
based on the actual volatility characteristics of the input data.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Tuple

try:
    from tstrends.trend_labelling import BinaryCTL, TernaryCTL, OracleBinaryTrendLabeller, OracleTernaryTrendLabeller
    TSTRENDS_AVAILABLE = True
except ImportError:
    TSTRENDS_AVAILABLE = False


def analyze_data_volatility(df: pl.DataFrame) -> dict:
    """
    Analyze the volatility characteristics of price data.
    
    Returns key metrics needed for parameter scaling.
    """
    prices = df["mid_price"].to_numpy()
    
    # Price changes
    price_changes = np.diff(prices)
    abs_changes = np.abs(price_changes)
    
    # Percentage changes  
    pct_changes = price_changes[1:] / prices[:-2]  # Avoid division by first price
    abs_pct_changes = np.abs(pct_changes)
    
    # Filter out zero changes for meaningful statistics
    nonzero_changes = abs_changes[abs_changes > 0]
    nonzero_pct_changes = abs_pct_changes[abs_pct_changes > 0]
    
    volatility_metrics = {
        # Basic statistics
        'total_samples': len(prices),
        'zero_change_pct': np.sum(price_changes == 0) / len(price_changes) * 100,
        'mean_abs_change': np.mean(abs_changes),
        'std_abs_change': np.std(abs_changes),
        'median_abs_change': np.median(abs_changes),
        
        # Non-zero change statistics
        'mean_nonzero_change': np.mean(nonzero_changes) if len(nonzero_changes) > 0 else 0,
        'std_nonzero_change': np.std(nonzero_changes) if len(nonzero_changes) > 0 else 0,
        
        # Percentage change statistics
        'mean_abs_pct_change': np.mean(abs_pct_changes),
        'std_abs_pct_change': np.std(abs_pct_changes),
        'mean_nonzero_pct_change': np.mean(nonzero_pct_changes) if len(nonzero_pct_changes) > 0 else 0,
        
        # Volatility percentiles
        'p10_abs_change': np.percentile(abs_changes, 10),
        'p25_abs_change': np.percentile(abs_changes, 25),
        'p50_abs_change': np.percentile(abs_changes, 50),
        'p75_abs_change': np.percentile(abs_changes, 75),
        'p90_abs_change': np.percentile(abs_changes, 90),
        'p99_abs_change': np.percentile(abs_changes, 99),
        
        # Rolling volatility (100-tick window)
        'rolling_vol_mean': 0,
        'rolling_vol_std': 0,
    }
    
    # Calculate rolling volatility
    window = min(100, len(prices) // 10)  # Use 100 or 10% of data, whichever is smaller
    rolling_vols = []
    for i in range(window, len(prices)):
        window_changes = price_changes[i-window:i]
        vol = np.std(window_changes)
        rolling_vols.append(vol)
    
    if rolling_vols:
        volatility_metrics['rolling_vol_mean'] = np.mean(rolling_vols)
        volatility_metrics['rolling_vol_std'] = np.std(rolling_vols)
    
    return volatility_metrics


def calculate_volatility_scaled_parameters(volatility_metrics: dict) -> dict:
    """
    Calculate volatility-aware parameter recommendations for TStrends methods.
    
    Based on the actual data volatility, scale parameters to work with micro-movements.
    """
    # Base scaling factors
    mean_change = volatility_metrics['mean_nonzero_change']
    std_change = volatility_metrics['std_nonzero_change'] 
    p75_change = volatility_metrics['p75_abs_change']
    p90_change = volatility_metrics['p90_abs_change']
    
    # For Binary CTL omega scaling
    # Use a fraction of typical absolute changes to capture meaningful movements
    binary_ctl_omega_candidates = [
        0.0,  # Zero threshold - capture all movements
        mean_change * 0.1,  # 10% of mean change
        mean_change * 0.25, # 25% of mean change  
        mean_change * 0.5,  # 50% of mean change
        p75_change * 0.5,   # 50% of 75th percentile
        std_change * 0.5,   # 50% of standard deviation
    ]
    
    # For Ternary CTL marginal change threshold scaling
    # Use percentage-based thresholds scaled to actual volatility
    mean_pct_change = volatility_metrics['mean_nonzero_pct_change']
    ternary_ctl_thres_candidates = [
        mean_pct_change * 0.1,   # 10% of mean percentage change
        mean_pct_change * 0.25,  # 25% of mean percentage change
        mean_pct_change * 0.5,   # 50% of mean percentage change
        mean_pct_change * 1.0,   # 100% of mean percentage change
        mean_pct_change * 2.0,   # 200% of mean percentage change
    ]
    
    # For Oracle transaction costs
    # Scale transaction costs to be meaningful relative to typical movements
    oracle_tc_candidates = [
        mean_change * 0.01,  # 1% of mean change
        mean_change * 0.05,  # 5% of mean change
        mean_change * 0.1,   # 10% of mean change
        0.00007,             # Our standard 0.7 pip cost
        mean_change * 0.25,  # 25% of mean change
    ]
    
    return {
        'binary_ctl_omega': sorted([max(0, x) for x in binary_ctl_omega_candidates]),
        'ternary_ctl_marginal_change_thres': sorted([max(0, x) for x in ternary_ctl_thres_candidates]),
        'oracle_transaction_costs': sorted([max(0.000001, x) for x in oracle_tc_candidates]),  # Min 0.1 pip
        'ternary_ctl_window_sizes': [10, 50, 100, 250, 500, 1000],  # Scale window sizes appropriately
    }


def test_scaled_parameters():
    """Test the volatility-scaled parameters against our problematic data."""
    if not TSTRENDS_AVAILABLE:
        print("❌ TStrends library not available")
        return
    
    # Load data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    print("🔍 Analyzing data volatility...")
    volatility_metrics = analyze_data_volatility(df)
    
    print("\n📊 VOLATILITY ANALYSIS RESULTS:")
    print(f"Total samples: {volatility_metrics['total_samples']:,}")
    print(f"Zero changes: {volatility_metrics['zero_change_pct']:.2f}%")
    print(f"Mean absolute change: {volatility_metrics['mean_abs_change']:.8f}")
    print(f"Mean non-zero change: {volatility_metrics['mean_nonzero_change']:.8f}")
    print(f"P75 absolute change: {volatility_metrics['p75_abs_change']:.8f}")
    print(f"P90 absolute change: {volatility_metrics['p90_abs_change']:.8f}")
    print(f"Mean percentage change: {volatility_metrics['mean_nonzero_pct_change']:.6f}")
    
    # Calculate scaled parameters
    scaled_params = calculate_volatility_scaled_parameters(volatility_metrics)
    
    print("\n🎯 VOLATILITY-SCALED PARAMETERS:")
    print(f"Binary CTL omega candidates: {[f'{x:.8f}' for x in scaled_params['binary_ctl_omega']]}")
    print(f"Ternary CTL threshold candidates: {[f'{x:.6f}' for x in scaled_params['ternary_ctl_marginal_change_thres']]}")
    print(f"Oracle transaction cost candidates: {[f'{x:.6f}' for x in scaled_params['oracle_transaction_costs']]}")
    
    # Test with sample data
    print("\n🧪 TESTING SCALED PARAMETERS:")
    test_sample = df.head(10000)
    prices = test_sample["mid_price"].to_numpy()
    price_list = [float(p) for p in prices.tolist()]
    
    # Test Binary CTL with scaled omega values
    print("\n=== BINARY CTL WITH SCALED OMEGA ===")
    for omega in scaled_params['binary_ctl_omega'][:5]:  # Test first 5
        try:
            labeller = BinaryCTL(omega=omega)
            raw_labels = labeller.get_labels(price_list)
            labels = np.array(raw_labels)
            unique, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            balance_score = min(percentages) / max(percentages) * 100  # Balance metric
            print(f"Omega {omega:10.8f}: Labels {unique} = {percentages.round(1)}% (Balance: {balance_score:.1f}%)")
        except Exception as e:
            print(f"Omega {omega:10.8f}: ERROR - {e}")
    
    # Test Ternary CTL with scaled parameters  
    print("\n=== TERNARY CTL WITH SCALED PARAMETERS ===")
    for thres in scaled_params['ternary_ctl_marginal_change_thres'][:3]:  # Test first 3
        for window in [100, 500]:  # Test key window sizes
            try:
                labeller = TernaryCTL(marginal_change_thres=thres, window_size=window)
                raw_labels = labeller.get_labels(price_list)
                labels = np.array(raw_labels)
                unique, counts = np.unique(labels, return_counts=True)
                percentages = counts / len(labels) * 100
                balance_score = min(percentages) / max(percentages) * 100
                print(f"Thres {thres:.6f}, Win {window:3d}: Labels {unique} = {percentages.round(1)}% (Balance: {balance_score:.1f}%)")
            except Exception as e:
                print(f"Thres {thres:.6f}, Win {window:3d}: ERROR - {e}")
    
    # Test Oracle with scaled transaction costs
    print("\n=== ORACLE BINARY WITH SCALED TRANSACTION COSTS ===")
    for tc in scaled_params['oracle_transaction_costs'][:5]:  # Test first 5
        try:
            labeller = OracleBinaryTrendLabeller(transaction_cost=tc)
            raw_labels = labeller.get_labels(price_list)
            labels = np.array(raw_labels)
            unique, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            balance_score = min(percentages) / max(percentages) * 100
            print(f"TC {tc:8.6f}: Labels {unique} = {percentages.round(1)}% (Balance: {balance_score:.1f}%)")
        except Exception as e:
            print(f"TC {tc:8.6f}: ERROR - {e}")


def main():
    """Run volatility-aware parameter scaling diagnostic."""
    print("🔧 Volatility-Aware TStrends Parameter Scaling")
    print("=" * 60)
    
    try:
        test_scaled_parameters()
        
        print("\n✅ SOLUTION SUMMARY:")
        print("- Use omega ≈ 0 for Binary CTL to capture micro-movements")
        print("- Scale Ternary CTL thresholds to 10-50% of actual price volatility")
        print("- Reduce Oracle transaction costs to 1-10% of typical price changes")
        print("- Consider larger window sizes (500-1000) for Ternary CTL")
        print("\n🎯 Next: Implement these scaled parameters in optimization bounds")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()