#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Test Triple Barrier Method Implementation
"""

import polars as pl
import numpy as np
from represent.target_generators.factory import TargetGeneratorFactory
from represent.parameter_optimization import ParameterOptimizer


def test_triple_barrier_implementation():
    """Comprehensive test of Triple Barrier Method"""
    
    print("=" * 70)
    print("TESTING TRIPLE BARRIER METHOD IMPLEMENTATION")
    print("=" * 70)
    
    # Create test data with various market patterns
    np.random.seed(42)
    n_samples = 1500
    
    # Create realistic FX price data with trend and volatility clustering
    base_price = 1.1000
    
    # Add trending component
    trend = np.linspace(0, 0.005, n_samples)  # 50 pip uptrend
    
    # Add volatility clustering (GARCH-like)
    returns = []
    volatility = 0.0001  # Base volatility
    for i in range(n_samples):
        # Volatility clustering
        volatility = 0.7 * volatility + 0.3 * 0.0001 + 0.1 * (returns[-1]**2 if returns else 0)
        ret = np.random.normal(0, volatility)
        returns.append(ret)
    
    prices = base_price + trend + np.cumsum(returns)
    
    test_data = pl.DataFrame({
        "ts_event": range(n_samples),
        "mid_price": prices,
        "symbol": ["EURUSD"] * n_samples
    })
    
    print(f"Test data: {n_samples} samples")
    print(f"Price range: {test_data['mid_price'].min():.6f} - {test_data['mid_price'].max():.6f}")
    print(f"Total trend: {(test_data['mid_price'].max() - test_data['mid_price'].min()) * 100000:.1f} pips")
    
    # Test different Triple Barrier configurations
    test_configs = [
        ("Conservative (20 pip)", {
            "lookforward_window": 500, 
            "barrier_width": 0.0002,  # 2 pips
            "transaction_cost": 0.0001
        }),
        ("Moderate (50 pip)", {
            "lookforward_window": 300,
            "barrier_width": 0.0005,  # 5 pips
            "transaction_cost": 0.0001
        }),
        ("Aggressive (100 pip)", {
            "lookforward_window": 200,
            "barrier_width": 0.001,   # 10 pips
            "transaction_cost": 0.0001
        }),
        ("Volatility Normalized", {
            "lookforward_window": 400,
            "barrier_width": 0.0003,  # 3 pips base
            "normalize_by_volatility": True,
            "volatility_window": 50,
            "transaction_cost": 0.0001
        }),
        ("Asymmetric Barriers", {
            "lookforward_window": 350,
            "upper_barrier": 0.0008,   # 8 pips profit target
            "lower_barrier": 0.0004,   # 4 pips stop loss
            "transaction_cost": 0.0001
        })
    ]
    
    print("\nTesting different Triple Barrier configurations:")
    print("-" * 100)
    print(f"{'Config':>20} {'Window':>7} {'Barriers':>12} {'Labels':>15} {'TradeFreq':>9} {'ProfitRate':>10} {'Status':>12}")
    print("-" * 100)
    
    for config_name, params in test_configs:
        try:
            generator = TargetGeneratorFactory.create("triple_barrier", **params)
            targets = generator.generate_targets(test_data)
            
            # Get the target column
            target_col = [col for col in targets.columns 
                         if col not in ["row_idx", "symbol", "timestamp"] and not col.endswith("_return") and not col.endswith("_barrier_width")][0]
            
            labels = targets[target_col].to_numpy()
            returns_col = f"{target_col}_return"
            
            # Calculate statistics
            label_counts = dict(zip(*np.unique(labels, return_counts=True)))
            total_labels = len(labels)
            
            # Format label distribution
            label_str = ", ".join([f"{label}:{count}" for label, count in sorted(label_counts.items())])
            
            # Calculate trade frequency (non-zero labels)
            trading_labels = np.sum(labels != 0)
            trade_frequency = trading_labels / total_labels
            
            # Calculate profit rate (positive returns)
            if returns_col in targets.columns:
                returns = targets[returns_col].to_numpy()
                profitable_trades = np.sum(returns > 0)
                profit_rate = profitable_trades / total_labels if total_labels > 0 else 0
                
                # Calculate expected return
                expected_return = np.mean(returns)
                status = "PROFITABLE" if expected_return > 0 else "UNPROFITABLE"
            else:
                profit_rate = 0
                status = "NO_RETURNS"
            
            # Format barrier description
            if "upper_barrier" in params and "lower_barrier" in params:
                barrier_desc = f"+{params['upper_barrier']*100000:.0f}/-{params['lower_barrier']*100000:.0f}"
            else:
                barrier_width = params.get("barrier_width", 0.0005)
                barrier_desc = f"±{barrier_width*100000:.0f}pip"
            
            print(f"{config_name:>20} {params['lookforward_window']:>7d} {barrier_desc:>12} {label_str:>15} {trade_frequency:>9.1%} {profit_rate:>10.1%} {status:>12}")
            
            # Additional analysis for interesting cases
            if trade_frequency > 0.5:
                print(f"    ⚠️ High trading frequency - consider larger barriers or longer windows")
            if profit_rate > 0.6:
                print(f"    ✅ High profit rate - good barrier calibration for trending market")
            if "Volatility" in config_name and params.get("normalize_by_volatility"):
                print(f"    📊 Using volatility normalization with {params['volatility_window']} tick window")
                
        except Exception as e:
            print(f"{config_name:>20} {'ERROR':>7} {'ERROR':>12} {'ERROR':>15} {'ERROR':>9} {'ERROR':>10} {'ERROR':>12}")
            print(f"    Error: {e}")
    
    print("\n" + "=" * 70)
    print("TESTING TRIPLE BARRIER OPTIMIZATION")
    print("=" * 70)
    
    # Test the optimization process
    print("Running Bayesian optimization for Triple Barrier parameters...")
    print("(Using small dataset and few iterations for speed)")
    
    try:
        # Use subset of data for faster optimization
        test_prices = test_data.head(800)["mid_price"].to_numpy()
        
        optimizer = ParameterOptimizer(n_calls=10, verbose=True)  # Reduced iterations for testing
        
        # Custom bounds for testing
        custom_bounds = {
            'lookforward_window': (100, 500),   # Smaller range for testing
            'barrier_width': (0.0002, 0.001),   # 2-10 pips
            'min_return_threshold': (0.00001, 0.00005),
            'volatility_window': (20, 50),
            'normalize_by_volatility': (0, 1),
        }
        
        result = optimizer.optimize_triple_barrier(test_prices, custom_bounds)
        
        print(f"\n🎯 Optimization Results:")
        print(f"   Method: {result['method']}")
        print(f"   Maximum Returns: {result['maximum_returns']:.4f}")
        print(f"   Optimal Parameters:")
        for param, value in result['optimal_params'].items():
            if param == 'barrier_width':
                print(f"     {param}: {value:.6f} ({value*100000:.1f} pips)")
            elif param == 'transaction_cost':
                print(f"     {param}: {value:.6f} ({value*100000:.1f} pips)")
            elif param == 'normalize_by_volatility':
                print(f"     {param}: {value} ({'Enabled' if value else 'Disabled'})")
            else:
                print(f"     {param}: {value}")
        
        # Test optimized parameters
        print(f"\n🧪 Testing optimized parameters...")
        opt_generator = TargetGeneratorFactory.create("triple_barrier", **result['optimal_params'])
        opt_targets = opt_generator.generate_targets(test_data.head(800))
        
        opt_target_col = [col for col in opt_targets.columns 
                         if col not in ["row_idx", "symbol", "timestamp"] and not col.endswith("_return") and not col.endswith("_barrier_width")][0]
        opt_labels = opt_targets[opt_target_col].to_numpy()
        opt_returns = opt_targets[f"{opt_target_col}_return"].to_numpy()
        
        opt_trade_freq = np.sum(opt_labels != 0) / len(opt_labels)
        opt_profit_rate = np.sum(opt_returns > 0) / len(opt_returns)
        opt_expected_return = np.mean(opt_returns)
        
        print(f"   Optimized Performance:")
        print(f"     Trade Frequency: {opt_trade_freq:.1%}")
        print(f"     Profit Rate: {opt_profit_rate:.1%}")
        print(f"     Expected Return: {opt_expected_return:.6f}")
        print(f"     Status: {'✅ PROFITABLE' if opt_expected_return > 0 else '❌ UNPROFITABLE'}")
        
        # Save optimized parameters
        import json
        with open("optimized_triple_barrier_params.json", "w") as f:
            json.dump(result['optimal_params'], f, indent=2)
        print(f"   💾 Saved optimized parameters to optimized_triple_barrier_params.json")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TRIPLE BARRIER ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("Key findings:")
    print("1. Triple Barrier method provides structured risk/reward labeling")
    print("2. Barrier width controls trade frequency vs signal strength tradeoff")
    print("3. Lookforward window affects time barrier effectiveness")
    print("4. Volatility normalization adapts barriers to market conditions")
    print("5. Asymmetric barriers allow different profit target vs stop loss")
    print("")
    print("Optimization benefits:")
    print("- Automatically finds optimal barrier/window combinations")
    print("- Balances trade frequency with profitability")
    print("- Accounts for transaction costs in parameter selection")
    print("- Adapts to specific market characteristics")


if __name__ == "__main__":
    test_triple_barrier_implementation()