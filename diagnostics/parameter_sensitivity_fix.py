#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis and Fix

This script analyzes and fixes parameter sensitivity issues discovered in our
comprehensive diagnostic, focusing on proper parameter ranges for micro-volatility data.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Any

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    REPRESENT_AVAILABLE = True
except ImportError:
    REPRESENT_AVAILABLE = False


def analyze_parameter_sensitivity_detailed(method_name: str, param_configs: List[Dict[str, Any]], 
                                          test_df: pl.DataFrame) -> Dict[str, Any]:
    """
    Analyze parameter sensitivity with detailed ranges.
    
    Args:
        method_name: Name of the target generation method
        param_configs: List of parameter configurations to test
        test_df: Test DataFrame
        
    Returns:
        Detailed sensitivity analysis results
    """
    results = {}
    
    for i, config in enumerate(param_configs):
        try:
            generator = TargetGeneratorFactory.create(method_name, **config)
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            unique, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            balance_score = min(percentages) / max(percentages) * 100 if len(percentages) > 1 else 100
            
            # Calculate entropy for distribution uniformity
            probs = percentages / 100
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            
            results[f"config_{i}"] = {
                "parameters": config,
                "unique_labels": len(unique),
                "label_distribution": dict(zip(unique.astype(str), counts)),
                "balance_score": balance_score,
                "entropy": entropy,
                "percentages": dict(zip(unique.astype(str), percentages.round(2)))
            }
            
        except Exception as e:
            results[f"config_{i}"] = {
                "parameters": config,
                "error": str(e)
            }
    
    return results


def test_binary_ctl_sensitivity():
    """Test Binary CTL with micro-scale parameter sensitivity."""
    print("🔍 BINARY CTL PARAMETER SENSITIVITY (MICRO-SCALE)")
    print("-" * 60)
    
    # Test micro-scale omega values around the working range
    omega_configs = [
        {"omega": 0.0},           # Known good
        {"omega": 0.000001},      # Ultra-micro
        {"omega": 0.000005},      # Volatility-scaled
        {"omega": 0.00001},       # 2x volatility
        {"omega": 0.00002},       # 4x volatility  
        {"omega": 0.00005},       # 10x volatility
        {"omega": 0.0001},        # 20x volatility (should start failing)
        {"omega": 0.0005},        # 100x volatility (should fail)
        {"omega": 0.001},         # 200x volatility (should fail)
    ]
    
    return omega_configs


def test_ternary_ctl_sensitivity():
    """Test Ternary CTL with micro-scale parameter sensitivity."""
    print("🔍 TERNARY CTL PARAMETER SENSITIVITY (MICRO-SCALE)")
    print("-" * 60)
    
    # Test micro-scale threshold and window combinations
    ternary_configs = []
    
    # Micro-scale thresholds based on actual volatility (mean_pct_change ~0.000078)
    thresholds = [0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002]
    windows = [50, 100, 250, 500, 1000]
    
    # Test key combinations
    for thres in thresholds[:4]:  # Test first 4 thresholds
        for window in [100, 500]:  # Test key window sizes
            ternary_configs.append({
                "marginal_change_thres": thres,
                "window_size": window
            })
    
    return ternary_configs


def test_oracle_sensitivity():
    """Test Oracle methods with micro-scale transaction cost sensitivity."""
    print("🔍 ORACLE METHODS PARAMETER SENSITIVITY (MICRO-SCALE)")
    print("-" * 60)
    
    # Test micro-scale transaction costs
    oracle_binary_configs = []
    transaction_costs = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    
    for tc in transaction_costs:
        oracle_binary_configs.append({"transaction_cost": tc})
    
    # Test Oracle Ternary with transaction cost and neutral factor combinations
    oracle_ternary_configs = []
    for tc in [0.000005, 0.00001, 0.00007]:  # Key transaction costs
        for nrf in [0.1, 0.3, 0.5, 0.7, 0.9]:  # Neutral reward factors
            oracle_ternary_configs.append({
                "transaction_cost": tc,
                "neutral_reward_factor": nrf
            })
    
    return oracle_binary_configs, oracle_ternary_configs


def run_comprehensive_sensitivity_analysis():
    """Run comprehensive parameter sensitivity analysis with micro-scale ranges."""
    if not REPRESENT_AVAILABLE:
        print("❌ Represent library not available")
        return
    
    print("🔬 Comprehensive Parameter Sensitivity Analysis")
    print("=" * 70)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(10000)
    
    print(f"Testing with {len(test_df):,} samples")
    print()
    
    # Test Binary CTL sensitivity
    print("1️⃣ BINARY CTL SENSITIVITY ANALYSIS")
    omega_configs = test_binary_ctl_sensitivity()
    binary_results = analyze_parameter_sensitivity_detailed("binary_ctl", omega_configs, test_df)
    
    for config_name, result in binary_results.items():
        if "error" in result:
            print(f"   {result['parameters']}: ERROR - {result['error']}")
        else:
            params = result['parameters']
            balance = result['balance_score']
            labels = len(result['label_distribution'])
            percentages = result['percentages']
            print(f"   Omega {params['omega']:9.6f}: {labels} labels, Balance: {balance:5.1f}%, Dist: {percentages}")
    
    print()
    
    # Test Ternary CTL sensitivity
    print("2️⃣ TERNARY CTL SENSITIVITY ANALYSIS")
    ternary_configs = test_ternary_ctl_sensitivity()
    ternary_results = analyze_parameter_sensitivity_detailed("ternary_ctl", ternary_configs, test_df)
    
    for config_name, result in ternary_results.items():
        if "error" in result:
            print(f"   {result['parameters']}: ERROR - {result['error']}")
        else:
            params = result['parameters']
            balance = result['balance_score']
            labels = len(result['label_distribution'])
            percentages = result['percentages']
            thres = params['marginal_change_thres']
            window = params['window_size']
            print(f"   Thres {thres:.6f}, Win {window:4d}: {labels} labels, Balance: {balance:5.1f}%, Dist: {percentages}")
    
    print()
    
    # Test Oracle Binary sensitivity
    print("3️⃣ ORACLE BINARY SENSITIVITY ANALYSIS")
    oracle_binary_configs, oracle_ternary_configs = test_oracle_sensitivity()
    oracle_binary_results = analyze_parameter_sensitivity_detailed("oracle_binary", oracle_binary_configs, test_df)
    
    for config_name, result in oracle_binary_results.items():
        if "error" in result:
            print(f"   {result['parameters']}: ERROR - {result['error']}")
        else:
            params = result['parameters']
            balance = result['balance_score']
            labels = len(result['label_distribution'])
            percentages = result['percentages']
            tc = params['transaction_cost']
            print(f"   TC {tc:8.6f}: {labels} labels, Balance: {balance:5.1f}%, Dist: {percentages}")
    
    print()
    
    # Test Oracle Ternary sensitivity (first few configs)
    print("4️⃣ ORACLE TERNARY SENSITIVITY ANALYSIS (Sample)")
    oracle_ternary_sample = oracle_ternary_configs[:10]  # Test first 10 configs
    oracle_ternary_results = analyze_parameter_sensitivity_detailed("oracle_ternary", oracle_ternary_sample, test_df)
    
    for config_name, result in oracle_ternary_results.items():
        if "error" in result:
            print(f"   {result['parameters']}: ERROR - {result['error']}")
        else:
            params = result['parameters']
            balance = result['balance_score']
            labels = len(result['label_distribution'])
            percentages = result['percentages']
            tc = params['transaction_cost']
            nrf = params['neutral_reward_factor']
            print(f"   TC {tc:.6f}, NRF {nrf:.1f}: {labels} labels, Balance: {balance:5.1f}%, Dist: {percentages}")
    
    print()
    
    # Summarize sensitivity findings
    print("📊 SENSITIVITY ANALYSIS SUMMARY:")
    print("=" * 40)
    
    # Find optimal ranges
    binary_working = [r for r in binary_results.values() if "error" not in r and r['balance_score'] > 80]
    ternary_working = [r for r in ternary_results.values() if "error" not in r and r['balance_score'] > 50]
    oracle_binary_working = [r for r in oracle_binary_results.values() if "error" not in r and r['balance_score'] > 25]
    
    print(f"Binary CTL: {len(binary_working)}/{len(binary_results)} configs work (Balance > 80%)")
    if binary_working:
        omega_range = [r['parameters']['omega'] for r in binary_working]
        print(f"  Working omega range: {min(omega_range):.6f} to {max(omega_range):.6f}")
    
    print(f"Ternary CTL: {len(ternary_working)}/{len(ternary_results)} configs work (Balance > 50%)")
    if ternary_working:
        thres_range = [r['parameters']['marginal_change_thres'] for r in ternary_working]
        print(f"  Working threshold range: {min(thres_range):.6f} to {max(thres_range):.6f}")
    
    print(f"Oracle Binary: {len(oracle_binary_working)}/{len(oracle_binary_results)} configs work (Balance > 25%)")
    if oracle_binary_working:
        tc_range = [r['parameters']['transaction_cost'] for r in oracle_binary_working]
        print(f"  Working transaction cost range: {min(tc_range):.6f} to {max(tc_range):.6f}")
    
    print("\n✅ RECOMMENDED OPTIMIZED PARAMETER RANGES:")
    if binary_working:
        print(f"  Binary CTL omega: ({min(omega_range):.6f}, {max(omega_range):.6f})")
    if ternary_working:
        print(f"  Ternary CTL threshold: ({min(thres_range):.6f}, {max(thres_range):.6f})")
    if oracle_binary_working:
        print(f"  Oracle transaction cost: ({min(tc_range):.6f}, {max(tc_range):.6f})")


def main():
    """Run the comprehensive parameter sensitivity analysis."""
    try:
        run_comprehensive_sensitivity_analysis()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()