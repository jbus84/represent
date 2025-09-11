#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Test Oracle and CTL Methods - Investigate why they're not profitable
"""

import polars as pl
import numpy as np
import json
from pathlib import Path

from represent.target_generators.factory import TargetGeneratorFactory


def test_oracle_and_ctl_methods():
    """Test Oracle and CTL methods with proper transaction costs"""
    
    print("=" * 70)
    print("INVESTIGATING ORACLE AND CTL METHOD ISSUES")
    print("=" * 70)
    
    # Create test data with clear trend patterns
    np.random.seed(42)
    n_samples = 2000
    
    # Create data with clear profitable patterns
    base_price = 1.1000
    
    # Add strong trending component
    trend_component = np.linspace(0, 0.01, n_samples)  # 100 pip uptrend
    noise = np.random.normal(0, 0.0001, n_samples)     # Small noise
    
    prices = base_price + trend_component + noise
    
    test_data = pl.DataFrame({
        "ts_event": range(n_samples),
        "mid_price": prices,
        "symbol": ["EURUSD"] * n_samples
    })
    
    print(f"Test data: {n_samples} samples")
    print(f"Price range: {test_data['mid_price'].min():.6f} - {test_data['mid_price'].max():.6f}")
    print(f"Total price move: {(test_data['mid_price'].max() - test_data['mid_price'].min()) * 100000:.1f} pips")
    print(f"This should be VERY profitable for Oracle methods!")
    
    # Test different methods with corrected parameters
    methods_to_test = [
        # Oracle methods with corrected transaction costs
        ("Oracle Binary (FIXED)", "oracle_binary", {"transaction_cost": 0.0001}),  # 1 pip
        ("Oracle Ternary (FIXED)", "oracle_ternary", {"transaction_cost": 0.0001, "neutral_reward_factor": 0.5}),  # 1 pip
        
        # Oracle methods with saved (broken) parameters
        ("Oracle Binary (SAVED)", "oracle_binary", {"transaction_cost": 9.3e-07}),  # ~0 pips
        ("Oracle Ternary (SAVED)", "oracle_ternary", {"transaction_cost": 0.008, "neutral_reward_factor": 0.18}),  # 800 pips!
        
        # CTL methods with current parameters
        ("Binary CTL (SAVED)", "binary_ctl", {"omega": 0.0}),
        ("Ternary CTL (SAVED)", "ternary_ctl", {"marginal_change_thres": 0.0446, "window_size": 501}),
        
        # CTL methods with corrected parameters  
        ("Binary CTL (CORRECTED)", "binary_ctl", {"omega": 0.01}),  # Small threshold
        ("Ternary CTL (CORRECTED)", "ternary_ctl", {"marginal_change_thres": 0.001, "window_size": 100}),  # Sensitive
    ]
    
    print("\nTesting different methods:")
    print("-" * 90)
    print(f"{'Method':>25} {'Trades':>7} {'TradeFreq':>9} {'Unique':>8} {'GrossRet':>10} {'NetRet':>10} {'Status':>12}")
    print("-" * 90)
    
    for method_name, method_type, params in methods_to_test:
        try:
            generator = TargetGeneratorFactory.create(method_type, **params)
            targets = generator.generate_targets(test_data)
            
            # Find the target column
            target_cols = [col for col in targets.columns if col not in ["row_idx", "symbol", "timestamp"]]
            if not target_cols:
                print(f"{method_name:>25} {'ERROR':>7} {'ERROR':>9} {'ERROR':>8} {'NO_TARGETS':>10} {'ERROR':>10} {'ERROR':>12}")
                continue
                
            target_col = target_cols[0]  # Use first target column
            signals = targets[target_col].to_numpy()
            
            # Calculate basic stats
            unique_vals = sorted(targets[target_col].unique().to_list())
            unique_str = ",".join(map(str, unique_vals))
            
            # Calculate position changes (trades)
            position_changes = np.sum(np.abs(np.diff(signals, prepend=signals[0])) > 0)
            trade_frequency = position_changes / len(test_data)
            
            # Calculate returns (simple approach)
            returns = test_data["mid_price"].pct_change().fill_null(0).to_numpy()
            
            # For classification: convert to position (assume binary: 0->-1, 1->+1 or similar)
            if method_type in ["oracle_binary", "binary_ctl"]:
                # Binary: 0 = short/sell, 1 = long/buy
                positions = (signals * 2) - 1  # Convert 0,1 -> -1,+1
            elif method_type in ["oracle_ternary", "ternary_ctl"]:
                # Ternary: typically -1, 0, 1 or 0, 1, 2
                if min(unique_vals) == 0:
                    positions = signals - 1  # Convert 0,1,2 -> -1,0,1
                else:
                    positions = signals  # Already -1,0,1
            else:
                positions = signals  # Default
            
            # Calculate gross returns
            position_returns = positions[1:] * returns[1:]  # Forward-looking
            gross_return = np.sum(position_returns)
            
            # Estimate transaction costs
            if "transaction_cost" in params:
                tc = params["transaction_cost"]
                net_return = gross_return - (position_changes * tc)
            else:
                tc = 0.0001  # Default 1 pip for CTL methods
                net_return = gross_return - (position_changes * tc)
            
            # Status
            if net_return > 0:
                status = "PROFITABLE"
            elif gross_return > 0:
                status = "TC_KILLED"
            else:
                status = "UNPROFITABLE"
            
            print(f"{method_name:>25} {position_changes:>7d} {trade_frequency:>9.1%} {unique_str:>8} {gross_return:>10.6f} {net_return:>10.6f} {status:>12}")
            
            # Special analysis for Oracle methods
            if "Oracle" in method_name and "FIXED" in method_name:
                if net_return <= 0:
                    print(f"  ⚠️  ORACLE METHOD NOT PROFITABLE - This indicates a serious issue!")
                    if gross_return > 0:
                        print(f"      Transaction cost impact: {params['transaction_cost'] * position_changes:.6f}")
                        print(f"      Cost ratio: {(params['transaction_cost'] * position_changes) / gross_return * 100:.1f}% of gross")
                    else:
                        print(f"      Gross return is negative - Oracle implementation may be broken!")
                else:
                    print(f"  ✅ Oracle method working correctly")
            
            # Analysis for saved parameters
            if "SAVED" in method_name:
                if "Oracle Binary" in method_name:
                    print(f"      TC = {params['transaction_cost']:.2e} = {params['transaction_cost']*100000:.4f} pips (TOO LOW)")
                elif "Oracle Ternary" in method_name:
                    print(f"      TC = {params['transaction_cost']:.2e} = {params['transaction_cost']*100000:.1f} pips (TOO HIGH)")
                
        except Exception as e:
            print(f"{method_name:>25} {'ERROR':>7} {'ERROR':>9} {'ERROR':>8} {str(e)[:10]:>10} {'ERROR':>10} {'ERROR':>12}")
            print(f"  Error: {e}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    print("Expected behavior:")
    print("1. Oracle methods with 1 pip TC should be HIGHLY profitable on trending data")
    print("2. Oracle Binary (saved) with ~0 TC should be profitable (no costs)")
    print("3. Oracle Ternary (saved) with 800 pips TC should be unprofitable (extreme costs)")
    print("4. CTL methods should show reasonable performance with proper parameters")
    print("")
    print("If Oracle methods aren't profitable:")
    print("A) Implementation bug in Oracle labeling logic")
    print("B) Incorrect position mapping (0,1 -> -1,+1)")
    print("C) Transaction cost calculation error")
    print("D) Lookforward window issues")
    
    print("\n" + "=" * 70)
    print("DETAILED ORACLE INVESTIGATION")
    print("=" * 70)
    
    # Deep dive into Oracle Binary with minimal transaction costs
    try:
        print("Testing Oracle Binary with near-zero transaction costs...")
        oracle_params = {"transaction_cost": 1e-8}  # Essentially zero
        generator = TargetGeneratorFactory.create("oracle_binary", **oracle_params)
        targets = generator.generate_targets(test_data.head(500))  # Smaller dataset for analysis
        
        target_col = [col for col in targets.columns if col not in ["row_idx", "symbol", "timestamp"]][0]
        signals = targets[target_col].to_numpy()
        
        print(f"Oracle signals: {np.unique(signals, return_counts=True)}")
        print(f"Signal distribution: {dict(zip(*np.unique(signals, return_counts=True)))}")
        
        # Check if Oracle is actually using future information correctly
        prices_subset = test_data.head(500)["mid_price"].to_numpy()
        returns_subset = np.diff(prices_subset) / prices_subset[:-1]
        
        # Oracle should buy before up moves, sell before down moves
        signal_vs_future_return = []
        for i in range(len(signals)-1):
            if i < len(returns_subset):
                signal_vs_future_return.append((signals[i], returns_subset[i]))
        
        if signal_vs_future_return:
            signal_0_returns = [ret for sig, ret in signal_vs_future_return if sig == 0]
            signal_1_returns = [ret for sig, ret in signal_vs_future_return if sig == 1]
            
            if signal_0_returns and signal_1_returns:
                print(f"Average return when signal=0: {np.mean(signal_0_returns):.6f}")
                print(f"Average return when signal=1: {np.mean(signal_1_returns):.6f}")
                
                if np.mean(signal_1_returns) > np.mean(signal_0_returns):
                    print("✅ Oracle correctly predicts: signal=1 leads to higher returns")
                else:
                    print("❌ Oracle prediction WRONG: signal=1 should lead to higher returns")
        
    except Exception as e:
        print(f"Oracle investigation failed: {e}")


if __name__ == "__main__":
    test_oracle_and_ctl_methods()