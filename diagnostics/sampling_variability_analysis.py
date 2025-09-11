#!/usr/bin/env python3
"""
Sampling Variability Analysis and Fix

This script investigates and addresses the high sampling variability issues
discovered in our comprehensive diagnostic (75-518% performance differences 
between full dataset and windowed sampling).
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    REPRESENT_AVAILABLE = True
except ImportError:
    REPRESENT_AVAILABLE = False

try:
    from tstrends.returns_estimation import ReturnsEstimatorWithFees
    from tstrends.returns_estimation.fees_config import FeesConfig
    TSTRENDS_AVAILABLE = True
except ImportError:
    TSTRENDS_AVAILABLE = False


def calculate_pnl_with_fees(prices: np.ndarray, labels: np.ndarray, 
                           transaction_cost: float = 0.00007) -> float:
    """
    Calculate PnL with transaction costs using consistent methodology.
    
    Args:
        prices: Price array
        labels: Trading labels (-1, 0, 1 or remapped equivalents)
        transaction_cost: Transaction cost per trade
        
    Returns:
        Total PnL including transaction costs
    """
    if len(prices) <= 1 or len(labels) == 0:
        return 0.0
    
    # Ensure we have the same length for prices and labels
    min_len = min(len(prices), len(labels))
    prices = prices[:min_len]
    labels = labels[:min_len]
    
    # Convert labels to trading positions {-1, 0, 1}
    # Handle both original TStrends format and our remapped format
    positions = np.array(labels, dtype=np.float64)
    
    # If labels are in {0, 1} format (Binary), remap to {-1, 1}
    if set(np.unique(labels)) <= {0, 1}:
        positions = np.where(labels == 0, -1, 1)
    # If labels are in {0, 1, 2} format (Ternary), remap to {-1, 0, 1}  
    elif set(np.unique(labels)) <= {0, 1, 2}:
        positions = labels - 1
    
    total_pnl = 0.0
    current_position = 0
    
    for i in range(1, len(prices)):
        target_position = positions[i-1]  # Use previous label for current period
        
        # Calculate position change and associated costs
        if target_position != current_position:
            # Transaction cost for position change
            total_pnl -= transaction_cost
            current_position = target_position
        
        # Calculate return for current position
        if current_position != 0:
            price_return = (prices[i] - prices[i-1]) / prices[i-1]
            total_pnl += current_position * price_return
    
    return total_pnl


def analyze_sampling_variability(method_name: str, method_params: Dict[str, Any], 
                                test_df: pl.DataFrame, num_trials: int = 10) -> Dict[str, Any]:
    """
    Analyze sampling variability for a specific method across multiple sampling strategies.
    
    Args:
        method_name: Target generation method name
        method_params: Method parameters
        test_df: Test DataFrame
        num_trials: Number of sampling trials to run
        
    Returns:
        Comprehensive sampling variability analysis
    """
    results = {
        "method": method_name,
        "parameters": method_params,
        "trials": num_trials,
        "full_dataset_analysis": {},
        "window_sampling_analysis": {},
        "variability_metrics": {}
    }
    
    # Generate labels for full dataset
    try:
        generator = TargetGeneratorFactory.create(method_name, **method_params)
        targets_df = generator.generate_targets(test_df)
        target_info = generator.get_target_info()
        target_col = target_info['target_names'][0]
        labels = targets_df[target_col].to_numpy()
        prices = test_df["mid_price"].to_numpy()
        
        # Full dataset analysis
        full_pnl = calculate_pnl_with_fees(prices, labels)
        unique_labels, counts = np.unique(labels, return_counts=True)
        full_balance = min(counts) / max(counts) * 100 if len(counts) > 1 else 100
        
        results["full_dataset_analysis"] = {
            "samples": len(test_df),
            "pnl": full_pnl,
            "unique_labels": len(unique_labels),
            "balance_score": full_balance,
            "label_distribution": dict(zip(unique_labels.astype(str), counts))
        }
        
        # Window sampling analysis
        window_sizes = [5000, 10000, 25000, 50000]
        window_results = {}
        
        for window_size in window_sizes:
            if window_size >= len(test_df):
                continue
                
            trial_pnls = []
            trial_balances = []
            trial_label_counts = []
            
            # Run multiple trials for this window size
            for trial in range(num_trials):
                # Random sampling
                start_idx = np.random.randint(0, len(test_df) - window_size)
                window_df = test_df[start_idx:start_idx + window_size]
                
                # Generate labels for window
                window_targets_df = generator.generate_targets(window_df)
                window_labels = window_targets_df[target_col].to_numpy()
                window_prices = window_df["mid_price"].to_numpy()
                
                # Calculate metrics
                window_pnl = calculate_pnl_with_fees(window_prices, window_labels)
                unique_window_labels, window_counts = np.unique(window_labels, return_counts=True)
                window_balance = min(window_counts) / max(window_counts) * 100 if len(window_counts) > 1 else 100
                
                trial_pnls.append(window_pnl)
                trial_balances.append(window_balance)
                trial_label_counts.append(len(unique_window_labels))
            
            # Calculate variability statistics
            pnl_mean = np.mean(trial_pnls)
            pnl_std = np.std(trial_pnls)
            pnl_cv = (pnl_std / abs(pnl_mean)) * 100 if pnl_mean != 0 else float('inf')
            
            balance_mean = np.mean(trial_balances)
            balance_std = np.std(trial_balances)
            
            window_results[window_size] = {
                "pnls": trial_pnls,
                "pnl_mean": pnl_mean,
                "pnl_std": pnl_std,
                "pnl_cv": pnl_cv,
                "balance_mean": balance_mean,
                "balance_std": balance_std,
                "avg_label_count": np.mean(trial_label_counts)
            }
        
        results["window_sampling_analysis"] = window_results
        
        # Calculate overall variability metrics
        if window_results:
            # Find the most stable window size
            stable_window = min(window_results.keys(), 
                              key=lambda w: window_results[w]["pnl_cv"] if np.isfinite(window_results[w]["pnl_cv"]) else float('inf'))
            
            # Compare full dataset vs most stable window
            stable_pnl = window_results[stable_window]["pnl_mean"]
            variability_ratio = abs((full_pnl - stable_pnl) / full_pnl) * 100 if full_pnl != 0 else float('inf')
            
            results["variability_metrics"] = {
                "most_stable_window_size": stable_window,
                "full_vs_stable_variability": variability_ratio,
                "recommended_min_window": stable_window,
                "high_variability": variability_ratio > 50,  # Flag if >50% difference
            }
        
    except Exception as e:
        results["error"] = str(e)
    
    return results


def run_comprehensive_sampling_analysis():
    """Run comprehensive sampling variability analysis across all methods."""
    if not REPRESENT_AVAILABLE:
        print("❌ Represent library not available")
        return
    
    print("📊 Comprehensive Sampling Variability Analysis")
    print("=" * 60)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Use larger test set for variability analysis
    test_df = df.head(100000)  # 100K samples
    print(f"Testing with {len(test_df):,} samples")
    print()
    
    # Test configurations with optimized parameters
    test_methods = [
        ("binary_ctl", {"omega": 0.00001}),
        ("ternary_ctl", {"marginal_change_thres": 0.00002, "window_size": 500}),
        ("oracle_binary", {"transaction_cost": 0.0001}),
        ("oracle_ternary", {"transaction_cost": 0.0001, "neutral_reward_factor": 0.5}),
    ]
    
    all_results = {}
    
    for method_name, params in test_methods:
        print(f"🔍 Testing {method_name.upper()} sampling variability...")
        start_time = time.time()
        
        result = analyze_sampling_variability(method_name, params, test_df, num_trials=10)
        all_results[method_name] = result
        
        elapsed = time.time() - start_time
        print(f"   Completed in {elapsed:.1f}s")
        
        if "error" in result:
            print(f"   ❌ ERROR: {result['error']}")
            continue
        
        # Report key metrics
        full_pnl = result["full_dataset_analysis"]["pnl"]
        full_balance = result["full_dataset_analysis"]["balance_score"]
        
        print(f"   📈 Full dataset: PnL={full_pnl:.6f}, Balance={full_balance:.1f}%")
        
        # Report window variability
        if result["window_sampling_analysis"]:
            print(f"   📊 Window sampling variability:")
            for window_size, metrics in result["window_sampling_analysis"].items():
                pnl_mean = metrics["pnl_mean"]
                pnl_cv = metrics["pnl_cv"]
                balance_mean = metrics["balance_mean"]
                print(f"      {window_size:5d} samples: PnL={pnl_mean:.6f} (CV: {pnl_cv:.1f}%), Balance={balance_mean:.1f}%")
        
        # Report variability summary
        if result["variability_metrics"]:
            vm = result["variability_metrics"]
            print(f"   🎯 Most stable window: {vm['most_stable_window_size']} samples")
            print(f"   📈 Full vs stable variability: {vm['full_vs_stable_variability']:.1f}%")
            if vm["high_variability"]:
                print(f"   ⚠️  HIGH VARIABILITY DETECTED (>{vm['full_vs_stable_variability']:.0f}% difference)")
        
        print()
    
    # Summary of findings
    print("📋 SAMPLING VARIABILITY SUMMARY:")
    print("=" * 40)
    
    high_variability_methods = []
    stable_methods = []
    
    for method_name, result in all_results.items():
        if "error" in result:
            continue
            
        if result.get("variability_metrics", {}).get("high_variability", False):
            variability = result["variability_metrics"]["full_vs_stable_variability"]
            high_variability_methods.append((method_name, variability))
        else:
            stable_methods.append(method_name)
    
    print(f"✅ Stable methods ({len(stable_methods)}): {', '.join(stable_methods)}")
    if high_variability_methods:
        print(f"⚠️  High variability methods ({len(high_variability_methods)}):")
        for method, var in high_variability_methods:
            print(f"   {method}: {var:.1f}% variability")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"- Use minimum 50K sample windows for stable evaluation")
    print(f"- Run multiple trials (5-10) and average results")
    print(f"- Monitor coefficient of variation (CV) - aim for <30%")
    if high_variability_methods:
        print(f"- Review high-variability methods for parameter optimization stability")


def main():
    """Run the comprehensive sampling variability analysis."""
    try:
        run_comprehensive_sampling_analysis()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()