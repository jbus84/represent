#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Transaction Cost Analysis - Debug negative returns despite optimization
"""

import polars as pl
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any

from represent.target_generators.factory import TargetGeneratorFactory
from represent.modular_dataset_builder import ModularDatasetBuilder


def load_optimized_params() -> Dict[str, Dict[str, Any]]:
    """Load all optimized parameters"""
    params = {}
    
    # GA parameters
    try:
        with open("optimized_ga_params_corrected.json", "r") as f:
            params["ga_labeling"] = json.load(f)
    except FileNotFoundError:
        params["ga_labeling"] = {
            "population_size": 50, "max_generations": 75,
            "lookforward_window": 250, "transaction_cost": 0.00007
        }
    
    # CTL parameters
    try:
        with open("optimized_binary_ctl_params.json", "r") as f:
            params["binary_ctl"] = json.load(f)
    except FileNotFoundError:
        params["binary_ctl"] = {"omega": 0.0}
    
    try:
        with open("optimized_ternary_ctl_params.json", "r") as f:
            params["ternary_ctl"] = json.load(f)
    except FileNotFoundError:
        params["ternary_ctl"] = {"marginal_change_thres": 0.0446, "window_size": 501}
    
    # Oracle parameters
    try:
        with open("optimized_oracle_binary_params.json", "r") as f:
            params["oracle_binary"] = json.load(f)
    except FileNotFoundError:
        params["oracle_binary"] = {"transaction_cost": 9.3e-07}
    
    try:
        with open("optimized_oracle_ternary_params.json", "r") as f:
            params["oracle_ternary"] = json.load(f)
    except FileNotFoundError:
        params["oracle_ternary"] = {"transaction_cost": 0.008, "neutral_reward_factor": 0.18}
    
    return params


def analyze_transaction_costs():
    """Analyze the transaction cost settings"""
    params = load_optimized_params()
    
    print("=" * 60)
    print("TRANSACTION COST ANALYSIS")
    print("=" * 60)
    
    for method, method_params in params.items():
        tc = method_params.get("transaction_cost", "N/A")
        if tc != "N/A":
            pips = tc * 100000 if isinstance(tc, float) else "N/A"
            print(f"{method:20}: {tc:.2e} ({pips:.1f} pips)" if pips != "N/A" else f"{method:20}: {tc}")
        else:
            print(f"{method:20}: No transaction cost")
    
    print("\n" + "=" * 60)
    print("ISSUES IDENTIFIED:")
    print("=" * 60)
    
    # Check for problematic transaction costs
    ga_tc = params["ga_labeling"]["transaction_cost"]
    oracle_binary_tc = params["oracle_binary"]["transaction_cost"] 
    oracle_ternary_tc = params["oracle_ternary"]["transaction_cost"]
    
    print(f"1. GA Labeling: {ga_tc:.2e} = {ga_tc*100000:.1f} pips (reasonable)")
    print(f"2. Oracle Binary: {oracle_binary_tc:.2e} = {oracle_binary_tc*100000:.4f} pips (TOO LOW!)")
    print(f"3. Oracle Ternary: {oracle_ternary_tc:.2e} = {oracle_ternary_tc*100000:.1f} pips (TOO HIGH!)")
    
    return params


def test_with_zero_transaction_costs(data_file: str = None):
    """Test performance with zero transaction costs to isolate the issue"""
    
    # Use small sample if no file provided
    if data_file is None:
        print("\nGenerating synthetic test data...")
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic price movement
        price_changes = np.random.normal(0, 0.0001, n_samples)
        mid_prices = 1.1000 + np.cumsum(price_changes)
        timestamps = range(n_samples)
        
        test_data = pl.DataFrame({
            "ts_event": timestamps,
            "mid_price": mid_prices,
            "symbol": ["EURUSD"] * n_samples
        })
    else:
        # Use provided file (first 1000 rows for speed)
        test_data = pl.read_parquet(data_file).head(1000)
    
    print(f"\nTesting with {len(test_data)} samples...")
    print(f"Price range: {test_data['mid_price'].min():.6f} - {test_data['mid_price'].max():.6f}")
    
    # Test scenarios
    scenarios = [
        ("Zero TC", {"transaction_cost": 0.0}),
        ("Low TC", {"transaction_cost": 0.00001}),  # 0.1 pips
        ("Medium TC", {"transaction_cost": 0.00005}),  # 0.5 pips  
        ("Current TC", {"transaction_cost": 0.00007}),  # 0.7 pips
        ("High TC", {"transaction_cost": 0.0001}),  # 1.0 pips
    ]
    
    results = {}
    
    for scenario_name, tc_params in scenarios:
        print(f"\n--- Testing {scenario_name} ---")
        
        try:
            # Test GA Labeling
            ga_params = {
                "population_size": 20,  # Smaller for speed
                "max_generations": 10,   # Fewer generations for speed
                "lookforward_window": 100,  # Shorter window for speed
                "min_trades": 5,  # Lower threshold
                **tc_params
            }
            
            generator = TargetGeneratorFactory.create("ga_labeling", **ga_params)
            
            # Generate targets
            targets = generator.generate(test_data)
            
            # Calculate basic stats
            if "ga_signal" in targets.columns:
                signals = targets["ga_signal"].to_numpy()
                n_trades = np.sum(np.abs(np.diff(signals, prepend=0)) > 0)
                unique_signals = targets["ga_signal"].unique().sort()
                
                print(f"  Signals: {unique_signals.to_list()}")
                print(f"  Trades: {n_trades}")
                print(f"  Signal distribution: {targets['ga_signal'].value_counts().sort('ga_signal')}")
                
                # Simple return calculation
                if n_trades > 0:
                    price_changes = test_data["mid_price"].diff().fill_null(0).to_numpy()
                    position_returns = signals[1:] * price_changes[1:]  # Shift for forward-looking
                    gross_return = np.sum(position_returns)
                    transaction_costs = n_trades * tc_params["transaction_cost"]
                    net_return = gross_return - transaction_costs
                    
                    print(f"  Gross return: {gross_return:.6f}")
                    print(f"  Transaction costs: {transaction_costs:.6f}")
                    print(f"  Net return: {net_return:.6f}")
                    
                    results[scenario_name] = {
                        "gross_return": gross_return,
                        "transaction_costs": transaction_costs,
                        "net_return": net_return,
                        "n_trades": n_trades
                    }
                else:
                    print("  No trades generated!")
                    results[scenario_name] = {"error": "No trades"}
            else:
                print("  No ga_signal column found!")
                results[scenario_name] = {"error": "No signal column"}
                
        except Exception as e:
            print(f"  Error: {e}")
            results[scenario_name] = {"error": str(e)}
    
    return results, test_data


def main():
    """Main analysis function"""
    print("Starting transaction cost analysis...")
    
    # Step 1: Analyze current parameters
    params = analyze_transaction_costs()
    
    # Step 2: Test with different transaction costs
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT TRANSACTION COST SCENARIOS")
    print("=" * 60)
    
    results, test_data = test_with_zero_transaction_costs()
    
    # Step 3: Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for scenario, result in results.items():
        if "error" in result:
            print(f"{scenario:15}: ERROR - {result['error']}")
        else:
            print(f"{scenario:15}: Net={result['net_return']:8.6f}, "
                  f"Gross={result['gross_return']:8.6f}, "
                  f"TC={result['transaction_costs']:8.6f}, "
                  f"Trades={result['n_trades']:3d}")
    
    # Step 4: Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if results:
        zero_tc = results.get("Zero TC", {})
        current_tc = results.get("Current TC", {})
        
        if "net_return" in zero_tc and "net_return" in current_tc:
            tc_impact = zero_tc["net_return"] - current_tc["net_return"]
            print(f"1. Transaction cost impact: {tc_impact:.6f}")
            
            if zero_tc["net_return"] > 0:
                print("2. Strategy is profitable WITHOUT transaction costs")
                print("3. Need to optimize for fewer trades or higher alpha per trade")
            else:
                print("2. Strategy is UNPROFITABLE even without transaction costs")
                print("3. Core strategy needs improvement, not just transaction cost reduction")
        
        print("\n4. Suggested actions:")
        print("   - Use standardized 0.7 pips (0.00007) for all methods")
        print("   - Re-optimize with consistent transaction costs")
        print("   - Focus on trade frequency reduction")
        print("   - Consider larger lookforward windows for GA")


if __name__ == "__main__":
    main()