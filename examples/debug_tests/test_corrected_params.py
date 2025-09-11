#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Test GA Labeling with Corrected Parameters
"""

import polars as pl
import numpy as np
import json

from represent.target_generators.factory import TargetGeneratorFactory


def test_corrected_parameters():
    """Test GA with proper 1 pip transaction cost and 1000-5000 tick lookforward"""
    
    print("=" * 60)
    print("TESTING CORRECTED GA PARAMETERS")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    n_samples = 2000  # Larger dataset for longer lookforward windows
    
    # More realistic FX price simulation
    base_price = 1.1000
    returns = np.random.normal(0, 0.0001, n_samples)  # Realistic FX volatility
    prices = base_price + np.cumsum(returns)
    
    test_data = pl.DataFrame({
        "ts_event": range(n_samples),
        "mid_price": prices,
        "symbol": ["EURUSD"] * n_samples
    })
    
    print(f"Test data: {len(test_data)} samples")
    print(f"Price range: {test_data['mid_price'].min():.6f} - {test_data['mid_price'].max():.6f}")
    print(f"Price volatility: {test_data['mid_price'].std():.6f}")
    
    # Test scenarios
    scenarios = [
        ("Original (WRONG)", {
            "population_size": 50, "max_generations": 75,
            "lookforward_window": 250, "transaction_cost": 0.00007
        }),
        ("Corrected Params", {
            "population_size": 50, "max_generations": 30,  # Reduced for speed
            "lookforward_window": 2500, "transaction_cost": 0.0001, 
            "min_trades": 10  # Lower for testing
        }),
        ("Conservative Test", {
            "population_size": 20, "max_generations": 10,  # Very fast
            "lookforward_window": 1500, "transaction_cost": 0.0001,
            "min_trades": 5
        })
    ]
    
    print("\nTesting different parameter sets:")
    print("-" * 80)
    print(f"{'Scenario':>20} {'Window':>8} {'TC(pips)':>8} {'Trades':>8} {'Gross':>10} {'Net':>10} {'Status':>12}")
    print("-" * 80)
    
    for scenario_name, params in scenarios:
        try:
            generator = TargetGeneratorFactory.create("ga_labeling", **params)
            targets = generator.generate_targets(test_data)
            
            if "ga_long_labels" in targets.columns and "ga_short_labels" in targets.columns:
                long_signals = targets["ga_long_labels"].to_numpy()
                short_signals = targets["ga_short_labels"].to_numpy()
                
                # Calculate position changes (trades)
                long_trades = np.sum(np.abs(np.diff(long_signals, prepend=0)) > 0)
                short_trades = np.sum(np.abs(np.diff(short_signals, prepend=0)) > 0)
                total_trades = long_trades + short_trades
                
                if total_trades > 0:
                    # Calculate returns
                    returns = test_data["mid_price"].pct_change().fill_null(0).to_numpy()
                    positions = long_signals.astype(float) - short_signals.astype(float)
                    position_returns = positions[1:] * returns[1:]
                    
                    gross_return = np.sum(position_returns)
                    transaction_costs = total_trades * params["transaction_cost"]
                    net_return = gross_return - transaction_costs
                    
                    tc_pips = params["transaction_cost"] * 100000
                    
                    # Determine status
                    if net_return > 0:
                        status = "PROFITABLE"
                    elif gross_return > 0:
                        status = "TC_KILLED"
                    else:
                        status = "UNPROFITABLE"
                    
                    print(f"{scenario_name:>20} {params['lookforward_window']:>8d} {tc_pips:>8.1f} {total_trades:>8d} {gross_return:>10.6f} {net_return:>10.6f} {status:>12}")
                    
                    # Additional analysis for corrected params
                    if "Corrected" in scenario_name and net_return > 0:
                        print(f"  >>> SUCCESS! Trade frequency: {total_trades/len(test_data)*100:.1f}%")
                        print(f"  >>> Cost ratio: {transaction_costs/abs(gross_return)*100:.1f}% of gross returns")
                        
                else:
                    print(f"{scenario_name:>20} {params['lookforward_window']:>8d} {params['transaction_cost']*100000:>8.1f} {'0':>8} {'0.000000':>10} {'0.000000':>10} {'NO_TRADES':>12}")
                    
            else:
                print(f"{scenario_name:>20} {'N/A':>8} {'N/A':>8} {'ERROR':>8} {'NO_COLUMNS':>10} {'NO_COLUMNS':>10} {'ERROR':>12}")
                
        except Exception as e:
            print(f"{scenario_name:>20} {'ERROR':>8} {'ERROR':>8} {'ERROR':>8} {str(e)[:10]:>10} {str(e)[:10]:>10} {'ERROR':>12}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("Expected behavior with corrected parameters:")
    print("1. Lookforward window 2500 ticks should reduce trade frequency")
    print("2. 1 pip transaction cost (vs 0.7 pip) should be more realistic")  
    print("3. Strategy should have fewer, higher-quality trades")
    print("4. Net returns should be positive if strategy has alpha")
    
    # Test with the saved corrected parameters
    print("\n" + "=" * 60)
    print("TESTING WITH SAVED CORRECTED PARAMETERS")
    print("=" * 60)
    
    try:
        with open("corrected_ga_params.json", "r") as f:
            corrected_params = json.load(f)
            
        print(f"Loaded parameters: {corrected_params}")
        
        # Quick test with reduced generations for speed
        test_params = corrected_params.copy()
        test_params["max_generations"] = 20
        test_params["min_trades"] = 5
        
        generator = TargetGeneratorFactory.create("ga_labeling", **test_params)
        targets = generator.generate_targets(test_data)
        
        if "ga_long_labels" in targets.columns:
            long_signals = targets["ga_long_labels"].to_numpy()
            short_signals = targets["ga_short_labels"].to_numpy()
            
            long_trades = np.sum(np.abs(np.diff(long_signals, prepend=0)) > 0)
            short_trades = np.sum(np.abs(np.diff(short_signals, prepend=0)) > 0)
            total_trades = long_trades + short_trades
            
            print(f"Total trades: {total_trades} (L:{long_trades}, S:{short_trades})")
            print(f"Trade frequency: {total_trades/len(test_data)*100:.1f}%")
            
            if total_trades > 0:
                returns = test_data["mid_price"].pct_change().fill_null(0).to_numpy()
                positions = long_signals.astype(float) - short_signals.astype(float)
                position_returns = positions[1:] * returns[1:]
                
                gross_return = np.sum(position_returns)
                transaction_costs = total_trades * corrected_params["transaction_cost"]
                net_return = gross_return - transaction_costs
                
                print(f"Gross return: {gross_return:.6f}")
                print(f"Transaction costs: {transaction_costs:.6f}")
                print(f"Net return: {net_return:.6f}")
                print(f"Return per trade: {net_return/total_trades:.6f}")
                
                if net_return > 0:
                    print(">>> CORRECTED PARAMETERS ARE WORKING! <<<")
                else:
                    print(">>> Still negative, but trade frequency should be much lower <<<")
            else:
                print(">>> No trades - parameters may be too conservative <<<")
        
    except Exception as e:
        print(f"Error testing corrected parameters: {e}")


if __name__ == "__main__":
    test_corrected_parameters()