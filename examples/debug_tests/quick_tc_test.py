#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Quick Transaction Cost Test - Simple analysis with small dataset
"""

import polars as pl
import numpy as np
from pathlib import Path

from represent.target_generators.factory import TargetGeneratorFactory


def quick_test():
    """Quick test with minimal data to understand the issue"""
    
    print("=" * 50)
    print("QUICK TRANSACTION COST ANALYSIS")
    print("=" * 50)
    
    # Create minimal test data
    np.random.seed(42)
    n_samples = 500
    
    # More realistic price simulation
    base_price = 1.1000
    returns = np.random.normal(0, 0.0001, n_samples)
    prices = base_price + np.cumsum(returns)
    
    test_data = pl.DataFrame({
        "ts_event": range(n_samples),
        "mid_price": prices,
        "symbol": ["EURUSD"] * n_samples
    })
    
    print(f"Test data: {len(test_data)} samples")
    print(f"Price range: {test_data['mid_price'].min():.6f} - {test_data['mid_price'].max():.6f}")
    print(f"Price volatility: {test_data['mid_price'].std():.6f}")
    
    # Test different transaction costs with GA labeling
    transaction_costs = [0.0, 0.00001, 0.00005, 0.00007, 0.0001]
    
    print("\nTesting GA Labeling with different transaction costs:")
    print("-" * 70)
    print(f"{'TC (pips)':>10} {'Trades':>8} {'Signals':>15} {'Comments':>25}")
    print("-" * 70)
    
    for tc in transaction_costs:
        tc_pips = tc * 100000
        
        try:
            # Simple GA parameters for speed
            generator = TargetGeneratorFactory.create(
                "ga_labeling",
                population_size=10,
                max_generations=5, 
                lookforward_window=50,
                transaction_cost=tc,
                min_trades=3
            )
            
            targets = generator.generate_targets(test_data)
            
            if "ga_long_labels" in targets.columns:
                long_signals = targets["ga_long_labels"].to_numpy()
                short_signals = targets["ga_short_labels"].to_numpy() if "ga_short_labels" in targets.columns else np.zeros_like(long_signals)
                
                # Calculate trades for both long and short
                long_trades = np.sum(np.abs(np.diff(long_signals, prepend=0)) > 0)
                short_trades = np.sum(np.abs(np.diff(short_signals, prepend=0)) > 0)
                total_trades = long_trades + short_trades
                
                # Format signal distribution
                long_dist = targets["ga_long_labels"].value_counts().sort("ga_long_labels")
                long_str = ", ".join([f"L{row['ga_long_labels']}:{row['count']}" for row in long_dist.to_dicts()])
                
                if "ga_short_labels" in targets.columns:
                    short_dist = targets["ga_short_labels"].value_counts().sort("ga_short_labels")
                    short_str = ", ".join([f"S{row['ga_short_labels']}:{row['count']}" for row in short_dist.to_dicts()])
                    signal_dist = f"{long_str}|{short_str}"
                else:
                    signal_dist = long_str
                
                print(f"{tc_pips:8.1f} {total_trades:8d} {signal_dist[:25]:>25} {'OK' if total_trades > 0 else 'No trades':>25}")
                
            else:
                print(f"{tc_pips:8.1f} {'ERROR':>8} {'No GA labels':>25} {'Missing columns':>25}")
                
        except Exception as e:
            print(f"{tc_pips:8.1f} {'ERROR':>8} {str(e)[:15]:>15} {str(e)[-25:]:>25}")
    
    print("\n" + "=" * 50)
    print("ANALYSIS OF OPTIMIZED PARAMETERS")
    print("=" * 50)
    
    # Load and analyze optimized parameters
    import json
    
    try:
        with open("optimized_ga_params_corrected.json", "r") as f:
            ga_params = json.load(f)
            tc = ga_params["transaction_cost"]
            print(f"GA Optimized TC: {tc:.2e} = {tc*100000:.1f} pips")
            
            # Test with optimized parameters
            print(f"\nTesting with OPTIMIZED parameters:")
            generator = TargetGeneratorFactory.create("ga_labeling", **ga_params)
            targets = generator.generate_targets(test_data)
            
            if "ga_long_labels" in targets.columns:
                long_signals = targets["ga_long_labels"].to_numpy()
                short_signals = targets["ga_short_labels"].to_numpy() if "ga_short_labels" in targets.columns else np.zeros_like(long_signals)
                
                long_trades = np.sum(np.abs(np.diff(long_signals, prepend=0)) > 0)
                short_trades = np.sum(np.abs(np.diff(short_signals, prepend=0)) > 0)
                total_trades = long_trades + short_trades
                
                print(f"Optimized result: {total_trades} trades (long: {long_trades}, short: {short_trades})")
                print(f"Long signals: {targets['ga_long_labels'].value_counts().sort('ga_long_labels')}")
                if "ga_short_labels" in targets.columns:
                    print(f"Short signals: {targets['ga_short_labels'].value_counts().sort('ga_short_labels')}")
                
                if total_trades > 0:
                    # Calculate simple return using combined long/short strategy
                    returns = test_data["mid_price"].pct_change().fill_null(0).to_numpy()
                    
                    # Long positions: +1 when long_signal=1, 0 otherwise
                    # Short positions: -1 when short_signal=1, 0 otherwise
                    positions = long_signals.astype(float) - short_signals.astype(float)
                    position_returns = positions[1:] * returns[1:]  # Forward-looking
                    
                    gross_return = np.sum(position_returns)
                    costs = total_trades * tc
                    net_return = gross_return - costs
                    
                    print(f"Gross return: {gross_return:.6f}")
                    print(f"Transaction costs: {costs:.6f} (trades: {total_trades}, tc: {tc:.6f})")
                    print(f"Net return: {net_return:.6f}")
                    
                    if net_return < 0:
                        if gross_return > 0:
                            print(">>> ISSUE: Profitable strategy killed by transaction costs!")
                            print(f">>> Cost impact: {costs/abs(gross_return)*100:.1f}% of gross returns")
                        else:
                            print(">>> ISSUE: Strategy is fundamentally unprofitable!")
                    else:
                        print(">>> Strategy is profitable after costs")
                else:
                    print(">>> No trades generated - optimization too conservative!")
                        
            else:
                print("No GA label columns found in targets!")
                        
    except FileNotFoundError:
        print("No optimized parameters file found")
    except Exception as e:
        print(f"Error testing optimized parameters: {e}")
    
    print("\n" + "=" * 50)
    print("CONCLUSIONS")
    print("=" * 50)
    print("1. Check if optimized parameters are generating any trades")
    print("2. If no trades: optimization may have over-constrained the strategy")
    print("3. If negative returns: need to balance trade frequency vs alpha")
    print("4. Transaction cost of 7 pips might be too high for the lookforward window")


if __name__ == "__main__":
    quick_test()