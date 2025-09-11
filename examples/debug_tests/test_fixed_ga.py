#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Test Fixed GA Labeling - Verify trade frequency penalties work
"""

import polars as pl
import numpy as np
import json

from represent.target_generators.factory import TargetGeneratorFactory


def test_fixed_ga():
    """Test the fixed GA with proper trade frequency penalties"""
    
    print("=" * 60)
    print("TESTING FIXED GA LABELING")
    print("=" * 60)
    
    # Create larger test dataset for proper testing
    np.random.seed(42)
    n_samples = 3000  # Larger for realistic lookforward windows
    
    # Generate realistic FX price data with volatility clustering
    base_price = 1.1000
    volatility = 0.0001
    returns = np.random.normal(0, volatility, n_samples)
    
    # Add some trend and mean reversion for more realistic patterns
    trend = np.sin(np.arange(n_samples) * 0.002) * 0.00005  # Gentle trending
    returns += trend
    
    prices = base_price + np.cumsum(returns)
    
    test_data = pl.DataFrame({
        "ts_event": range(n_samples),
        "mid_price": prices,
        "symbol": ["EURUSD"] * n_samples
    })
    
    print(f"Test data: {len(test_data)} samples")
    print(f"Price range: {test_data['mid_price'].min():.6f} - {test_data['mid_price'].max():.6f}")
    print(f"Price volatility: {test_data['mid_price'].std():.6f}")
    
    # Test scenarios comparing old vs new fitness function
    scenarios = [
        ("OLD (Broken)", {
            "population_size": 20, "max_generations": 10,  # Fast for comparison
            "lookforward_window": 250, "transaction_cost": 0.00007,
            "max_trade_frequency": 0.95,  # Almost no penalty (old behavior)
            "min_trades": 5
        }),
        ("FIXED (Conservative)", {
            "population_size": 20, "max_generations": 15,
            "lookforward_window": 2500, "transaction_cost": 0.0001,
            "max_trade_frequency": 0.05,  # 5% max trade frequency
            "min_trades": 10
        }),
        ("FIXED (Moderate)", {
            "population_size": 20, "max_generations": 15,
            "lookforward_window": 1500, "transaction_cost": 0.0001,
            "max_trade_frequency": 0.08,  # 8% max trade frequency
            "min_trades": 15
        }),
    ]
    
    print("\nComparing OLD vs FIXED GA approaches:")
    print("-" * 90)
    print(f"{'Scenario':>20} {'Window':>7} {'MaxFreq':>8} {'Trades':>7} {'TradeFreq':>9} {'Gross':>10} {'Net':>10} {'Status':>10}")
    print("-" * 90)
    
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
                
                trade_frequency = total_trades / len(test_data)
                
                if total_trades > 0:
                    # Calculate returns
                    returns = test_data["mid_price"].pct_change().fill_null(0).to_numpy()
                    positions = long_signals.astype(float) - short_signals.astype(float)
                    position_returns = positions[1:] * returns[1:]
                    
                    gross_return = np.sum(position_returns)
                    transaction_costs = total_trades * params["transaction_cost"]
                    net_return = gross_return - transaction_costs
                    
                    # Determine status
                    if net_return > 0:
                        status = "PROFIT"
                    elif gross_return > 0 and net_return < 0:
                        status = "TC_KILLED"
                    else:
                        status = "UNPROFITABLE"
                    
                    print(f"{scenario_name:>20} {params['lookforward_window']:>7d} {params['max_trade_frequency']:>8.1%} {total_trades:>7d} {trade_frequency:>9.1%} {gross_return:>10.6f} {net_return:>10.6f} {status:>10}")
                    
                    if "FIXED" in scenario_name:
                        if trade_frequency <= params['max_trade_frequency'] * 1.2:  # Allow 20% tolerance
                            print(f"  ✓ Trade frequency constraint respected!")
                        else:
                            print(f"  ⚠ Trade frequency too high: {trade_frequency:.1%} > {params['max_trade_frequency']:.1%}")
                        
                        if net_return > 0:
                            print(f"  🎉 FIXED GA IS PROFITABLE! Net return: {net_return:.6f}")
                            print(f"     Return per trade: {net_return/total_trades:.6f}")
                        elif gross_return > 0:
                            print(f"  📈 Strategy has alpha but needs optimization")
                        
                else:
                    print(f"{scenario_name:>20} {params['lookforward_window']:>7d} {params['max_trade_frequency']:>8.1%} {'0':>7} {'0.0%':>9} {'0.000000':>10} {'0.000000':>10} {'NO_TRADES':>10}")
                    
            else:
                print(f"{scenario_name:>20} {'ERROR':>7} {'ERROR':>8} {'ERROR':>7} {'ERROR':>9} {'ERROR':>10} {'ERROR':>10} {'ERROR':>10}")
                
        except Exception as e:
            print(f"{scenario_name:>20} {'ERROR':>7} {'ERROR':>8} {'ERROR':>7} {'ERROR':>9} {str(e)[:10]:>10} {'ERROR':>10} {'ERROR':>10}")
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("TESTING WITH SAVED FIXED PARAMETERS")
    print("=" * 60)
    
    try:
        with open("fixed_ga_params.json", "r") as f:
            fixed_params = json.load(f)
            
        print(f"Fixed parameters: {fixed_params}")
        
        # Test with saved parameters (reduced generations for speed)
        test_params = fixed_params.copy()
        test_params["max_generations"] = 20
        test_params["min_trades"] = 10
        
        print(f"\nTesting with reduced generations for speed...")
        generator = TargetGeneratorFactory.create("ga_labeling", **test_params)
        targets = generator.generate_targets(test_data)
        
        if "ga_long_labels" in targets.columns:
            long_signals = targets["ga_long_labels"].to_numpy()
            short_signals = targets["ga_short_labels"].to_numpy()
            
            long_trades = np.sum(np.abs(np.diff(long_signals, prepend=0)) > 0)
            short_trades = np.sum(np.abs(np.diff(short_signals, prepend=0)) > 0)
            total_trades = long_trades + short_trades
            trade_frequency = total_trades / len(test_data)
            
            print(f"Results:")
            print(f"  Total trades: {total_trades} (L:{long_trades}, S:{short_trades})")
            print(f"  Trade frequency: {trade_frequency:.1%}")
            print(f"  Max allowed: {fixed_params['max_trade_frequency']:.1%}")
            
            if total_trades > 0:
                returns = test_data["mid_price"].pct_change().fill_null(0).to_numpy()
                positions = long_signals.astype(float) - short_signals.astype(float)
                position_returns = positions[1:] * returns[1:]
                
                gross_return = np.sum(position_returns)
                transaction_costs = total_trades * fixed_params["transaction_cost"]
                net_return = gross_return - transaction_costs
                
                print(f"  Gross return: {gross_return:.6f}")
                print(f"  Transaction costs: {transaction_costs:.6f}")
                print(f"  Net return: {net_return:.6f}")
                print(f"  Return per trade: {net_return/total_trades:.6f}")
                
                if trade_frequency <= fixed_params['max_trade_frequency'] * 1.5:
                    print(f"  ✅ SUCCESS! Trade frequency is under control")
                else:
                    print(f"  ❌ Trade frequency still too high")
                    
                if net_return > 0:
                    print(f"  🎉 PROFITABLE after transaction costs!")
                elif gross_return > 0:
                    print(f"  📊 Has alpha but needs more optimization")
                else:
                    print(f"  📉 No alpha detected")
            else:
                print(f"  ⚠️ No trades generated - may be too conservative")
        
    except Exception as e:
        print(f"Error testing fixed parameters: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Expected improvements:")
    print("1. ✓ Much lower trade frequency (< 10% vs >90%)")
    print("2. ✓ Proper transaction cost (1 pip = 0.0001)")
    print("3. ✓ Longer lookforward window (1500-2500 vs 250)")
    print("4. ✓ Strong penalties for overtrading in fitness function")
    print("5. → Should result in fewer, higher-quality trades")
    print("6. → Net returns should be positive if strategy has alpha")


if __name__ == "__main__":
    test_fixed_ga()