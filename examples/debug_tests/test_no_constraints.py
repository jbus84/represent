#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Test GA with minimal constraints to isolate trade frequency penalty
"""

import polars as pl
import numpy as np
from represent.target_generators.ga_labeling import GALabelingGenerator


def test_no_constraints():
    """Test GA with only trade frequency constraints, no win rate/profit constraints"""
    
    print("=" * 60)
    print("TESTING GA WITH ONLY TRADE FREQUENCY CONSTRAINTS")
    print("=" * 60)
    
    # Small test case
    np.random.seed(42)
    n_samples = 200
    
    prices = 1.1 + np.cumsum(np.random.normal(0, 0.0001, n_samples))
    test_data = pl.DataFrame({
        "ts_event": range(n_samples),
        "mid_price": prices,
        "symbol": ["EURUSD"] * n_samples
    })
    
    # Minimal constraints - only trade frequency matters
    minimal_params = {
        "population_size": 8,
        "max_generations": 3, 
        "lookforward_window": 50,  # Short for testing
        "transaction_cost": 0.0001,
        "max_trade_frequency": 0.1,  # 10% max
        "min_trades": 1,  # Very low
        "min_win_rate": 0.0,   # No constraint
        "max_win_rate": 1.0,   # No constraint  
        "min_profit_factor": 0.0,  # No constraint
        "verbose": True
    }
    
    print(f"Test data: {n_samples} samples")
    print(f"Max allowed trades: {n_samples * 0.1:.0f} (10% of {n_samples})")
    print(f"No win rate or profit factor constraints")
    
    generator = GALabelingGenerator(**minimal_params)
    
    # Test fitness function directly with no constraints
    print("\n" + "=" * 40)
    print("TESTING FITNESS WITH MINIMAL CONSTRAINTS")
    print("=" * 40)
    
    price_array = test_data["mid_price"].to_numpy()
    
    # Test patterns to find where -10000 penalty kicks in
    test_patterns = [
        ("All hold (0 trades)", np.zeros(n_samples, dtype=int)),
        ("Low frequency (~10 trades)", np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])),
        ("Target limit (20 trades)", np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])),
        ("Above limit (30 trades)", np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])),
        ("2x limit (40+ trades)", np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])),
        ("High frequency (100 trades)", np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])),
        ("Extreme (160+ trades)", np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]))
    ]
    
    for pattern_name, chromosome in test_patterns:
        try:
            fitness = generator._evaluate_fitness(chromosome, price_array, "long")
            
            # Calculate actual trades
            trades = generator._simulate_specialized_trading(chromosome, price_array, "long") 
            n_trades = len(trades)
            
            print(f"{pattern_name:30}: {n_trades:3d} trades, fitness = {fitness:8.2f}")
            
            max_allowed = n_samples * 0.1  # 10%
            if n_trades == 0:
                print(f"  → No trades penalty (-1000)")
            elif n_trades > max_allowed:
                print(f"  → SHOULD get overtrading penalty (>{max_allowed:.0f} trades)")
            else:
                print(f"  → Should be acceptable (<={max_allowed:.0f} trades)")
                
        except Exception as e:
            print(f"{pattern_name:30}: ERROR - {e}")
    
    print("\n" + "=" * 40)
    print("RUNNING GA WITH MINIMAL CONSTRAINTS")
    print("=" * 40)
    
    try:
        targets = generator.generate_targets(test_data)
        
        if "ga_long_labels" in targets.columns:
            long_signals = targets["ga_long_labels"].to_numpy()
            short_signals = targets["ga_short_labels"].to_numpy()
            
            long_trades = np.sum(np.abs(np.diff(long_signals, prepend=0)) > 0)
            short_trades = np.sum(np.abs(np.diff(short_signals, prepend=0)) > 0)
            total_trades = long_trades + short_trades
            
            max_expected = n_samples * 0.1  # 10%
            
            print(f"Results:")
            print(f"  Long trades: {long_trades} (max {max_expected:.0f})")
            print(f"  Short trades: {short_trades} (max {max_expected:.0f})")
            print(f"  Total: {total_trades} (max {max_expected*2:.0f} combined)")
            print(f"  Trade frequency: {total_trades/n_samples:.1%}")
            
            if long_trades <= max_expected and short_trades <= max_expected:
                print(f"  ✅ SUCCESS! Trade frequency constraints respected!")
            else:
                print(f"  ❌ STILL FAILED: Overtrading despite penalties")
                print(f"  This suggests the fitness function penalties aren't working")
                
        else:
            print("No GA labels found!")
            
    except Exception as e:
        print(f"GA generation failed: {e}")


if __name__ == "__main__":
    test_no_constraints()