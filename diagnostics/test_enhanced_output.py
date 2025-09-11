#!/usr/bin/env python3
"""
Test Enhanced Output for Symbol Optimization Runner

Quick test to verify the new class balance, number of trades, and mean return per trade outputs.
"""

import sys
from pathlib import Path
import numpy as np

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the function we want to test
from examples.symbol_optimization_runner import calculate_additional_metrics

def test_additional_metrics():
    """Test the calculate_additional_metrics function."""
    print("🧪 Testing Enhanced Output Metrics")
    print("=" * 50)
    
    # Create sample price data
    np.random.seed(42)
    base_price = 0.65
    n_samples = 5000
    price_changes = np.random.normal(0, 0.00001, n_samples)  # Micro-volatility like our data
    prices = np.cumsum(price_changes) + base_price
    
    print(f"Generated {n_samples:,} price samples")
    print(f"Price range: {prices.min():.6f} to {prices.max():.6f}")
    print()
    
    # Test different methods with optimized parameters
    test_methods = [
        ("binary_ctl", {"omega": 0.00001}),
        ("ternary_ctl", {"marginal_change_thres": 0.00002, "window_size": 500}),
        ("oracle_binary", {"transaction_cost": 0.0001}),
        ("oracle_ternary", {"transaction_cost": 0.0001, "neutral_reward_factor": 0.5}),
    ]
    
    for method_name, params in test_methods:
        print(f"🎯 Testing {method_name.upper()}")
        print("-" * 40)
        
        metrics = calculate_additional_metrics(prices, method_name, params)
        
        if "error" in metrics:
            print(f"   ❌ Error: {metrics['error']}")
        else:
            print(f"   ⚖️  Class balance: {metrics['class_balance_score']:.1f}% ({metrics['num_classes']} classes)")
            print(f"   📊 Label distribution: {metrics['label_distribution']}")
            print(f"   🔄 Trades: {metrics['num_trades']:,} ({metrics['trading_frequency']:.2f}% frequency)")
            print(f"   💰 Mean return/trade: {metrics['mean_return_per_trade']:.6f}")
            print(f"   📈 Total PnL: {metrics['total_pnl']:.6f}")
        
        print()
    
    print("✅ Enhanced output metrics test complete!")


if __name__ == "__main__":
    test_additional_metrics()