#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Diagnose GA Trading Frequency Issue
"""

import polars as pl
import numpy as np
from represent.target_generators.ga_labeling import GALabelingGenerator


def diagnose_ga_issue():
    """Diagnose why GA is still overtrading despite penalties"""
    
    print("=" * 60)
    print("DIAGNOSING GA OVERTRADING ISSUE") 
    print("=" * 60)
    
    # Small test case
    np.random.seed(42)
    n_samples = 500
    
    prices = 1.1 + np.cumsum(np.random.normal(0, 0.0001, n_samples))
    test_data = pl.DataFrame({
        "ts_event": range(n_samples),
        "mid_price": prices,
        "symbol": ["EURUSD"] * n_samples
    })
    
    # Test with very strict parameters
    strict_params = {
        "population_size": 10,
        "max_generations": 5, 
        "lookforward_window": 100,  # Short for testing
        "transaction_cost": 0.0001,
        "max_trade_frequency": 0.05,  # Only 5% trades allowed
        "min_trades": 5,
        "min_win_rate": 0.1,  # Very relaxed
        "max_win_rate": 0.9,  # Very relaxed
        "min_profit_factor": 0.5,  # Very relaxed
        "verbose": True
    }
    
    print(f"Test data: {n_samples} samples")
    print(f"Max allowed trades: {n_samples * 0.05:.0f} (5% of {n_samples})")
    print(f"Expected behavior: GA should find strategies with ≤25 trades")
    
    generator = GALabelingGenerator(**strict_params)
    
    # Let's manually test the fitness function
    print("\n" + "=" * 40)
    print("TESTING FITNESS FUNCTION DIRECTLY")
    print("=" * 40)
    
    price_array = test_data["mid_price"].to_numpy()
    
    # Test different trading patterns
    test_patterns = [
        ("Low frequency (10 trades)", np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])),
        ("Medium frequency (50 trades)", np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])),
        ("High frequency (250 trades)", np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])),
        ("Extreme frequency (400+ trades)", np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]))
    ]
    
    for pattern_name, chromosome in test_patterns:
        try:
            fitness = generator._evaluate_fitness(chromosome, price_array, "long")
            
            # Calculate actual trades
            trades = generator._simulate_specialized_trading(chromosome, price_array, "long") 
            n_trades = len(trades)
            
            print(f"{pattern_name:30}: {n_trades:3d} trades, fitness = {fitness:8.2f}")
            
            if n_trades > n_samples * 0.2:  # More than 20%
                print(f"  → Should get -10000 penalty: {fitness == -10000.0}")
            elif n_trades > n_samples * 0.05:  # More than 5%
                print(f"  → Should get heavy penalty")
            else:
                print(f"  → Should be acceptable")
                
        except Exception as e:
            print(f"{pattern_name:30}: ERROR - {e}")
    
    print("\n" + "=" * 40)
    print("RUNNING ACTUAL GA OPTIMIZATION")  
    print("=" * 40)
    
    try:
        targets = generator.generate_targets(test_data)
        
        if "ga_long_labels" in targets.columns:
            long_signals = targets["ga_long_labels"].to_numpy()
            short_signals = targets["ga_short_labels"].to_numpy()
            
            long_trades = np.sum(np.abs(np.diff(long_signals, prepend=0)) > 0)
            short_trades = np.sum(np.abs(np.diff(short_signals, prepend=0)) > 0)
            total_trades = long_trades + short_trades
            
            print(f"GA Result:")
            print(f"  Long trades: {long_trades}")
            print(f"  Short trades: {short_trades}")
            print(f"  Total trades: {total_trades}")
            print(f"  Trade frequency: {total_trades/n_samples:.1%}")
            print(f"  Expected max: {n_samples * 0.05:.0f} trades (5%)")
            
            if total_trades > n_samples * 0.2:
                print(f"  ❌ MAJOR ISSUE: GA chose extreme overtrading despite -10000 fitness!")
            elif total_trades > n_samples * 0.05:
                print(f"  ⚠️ GA exceeded target frequency")
            else:
                print(f"  ✅ GA respected trade frequency constraint!")
                
        else:
            print("No GA labels found!")
            
    except Exception as e:
        print(f"GA generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print("If GA still overtrades despite -10000 fitness penalties:")
    print("1. ALL chromosomes might be getting penalties → no good options")  
    print("2. Dual model approach might double the trading frequency")
    print("3. Trading simulation might have bugs")
    print("4. Chromosome → trading signal conversion might be wrong")
    print()
    print("Next steps:")
    print("- Check if ANY chromosomes avoid the penalty")
    print("- Test single model (not dual) approach")
    print("- Verify trading simulation logic")


if __name__ == "__main__":
    diagnose_ga_issue()