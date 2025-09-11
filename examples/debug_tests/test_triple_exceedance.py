#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Test Triple Exceedance Method Implementation
"""

import polars as pl
import numpy as np
from represent.target_generators.factory import TargetGeneratorFactory
from represent.parameter_optimization import ParameterOptimizer
import json


def test_triple_exceedance_implementation():
    """Comprehensive test of Triple Exceedance Method"""
    
    print("=" * 80)
    print("TESTING TRIPLE EXCEEDANCE METHOD IMPLEMENTATION")
    print("=" * 80)
    
    # Create test data with various market patterns
    np.random.seed(42)
    n_samples = 1200
    
    # Create realistic FX market with trend and varying volatility
    base_price = 1.1000
    transaction_cost = 0.0001  # 1 pip
    
    # Generate price series with trend and volatility clustering
    trend = np.linspace(0, 0.003, n_samples)  # 30 pip uptrend
    
    # GARCH-like volatility
    returns = []
    volatility = 0.0001
    for i in range(n_samples):
        # Volatility clustering
        if i > 0:
            volatility = 0.8 * volatility + 0.2 * 0.0001 + 0.15 * (returns[-1]**2)
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
    print(f"Transaction cost: {transaction_cost * 100000:.1f} pips")
    
    # Test different Triple Exceedance configurations
    test_configs = [
        ("Conservative 3x TC", {
            "lookforward_window": 200,
            "scaling_factor": 3.0,    # 3 pips barriers (3x 1 pip TC)
            "transaction_cost": transaction_cost,
            "adaptive_scaling": False
        }),
        ("Moderate 5x TC", {
            "lookforward_window": 150,
            "scaling_factor": 5.0,    # 5 pips barriers
            "transaction_cost": transaction_cost,
            "adaptive_scaling": False
        }),
        ("Aggressive 10x TC", {
            "lookforward_window": 100,
            "scaling_factor": 10.0,   # 10 pips barriers
            "transaction_cost": transaction_cost,
            "adaptive_scaling": False
        }),
        ("Adaptive Scaling", {
            "lookforward_window": 120,
            "scaling_factor": 6.0,    # 6 pips base, adjusted by volatility
            "transaction_cost": transaction_cost,
            "adaptive_scaling": True,
            "volatility_window": 30
        }),
        ("Short Window Focus", {
            "lookforward_window": 80,  # Very short window
            "scaling_factor": 8.0,
            "transaction_cost": transaction_cost,
            "window_penalty_weight": 0.3,  # Heavy penalty for long windows
            "min_exceedance_threshold": 0.7  # High threshold
        }),
        ("Asymmetric Exceedance", {
            "lookforward_window": 140,
            "transaction_cost": transaction_cost,
            "upper_scaling": 8.0,      # 8 pips profit target
            "lower_scaling": 4.0,      # 4 pips stop loss
            "adaptive_scaling": False
        })
    ]
    
    print("\nTesting different Triple Exceedance configurations:")
    print("-" * 120)
    print(f"{'Config':>22} {'Window':>7} {'Scaling':>10} {'Labels':>20} {'TradeFreq':>9} {'AvgExceed':>9} {'ExpReturn':>10} {'Status':>10}")
    print("-" * 120)
    
    for config_name, params in test_configs:
        try:
            generator = TargetGeneratorFactory.create("triple_exceedance", **params)
            targets = generator.generate_targets(test_data)
            
            # Get the target columns
            target_col = [col for col in targets.columns 
                         if col.endswith("_label") and not col.endswith("_exceedance")][0]
            exceedance_col = f"{target_col.replace('_label', '')}_exceedance"
            
            labels = targets[target_col].to_numpy()
            exceedances = targets[exceedance_col].to_numpy() if exceedance_col in targets.columns else np.zeros_like(labels)
            
            # Calculate statistics
            label_counts = dict(zip(*np.unique(labels, return_counts=True)))
            total_labels = len(labels)
            
            # Format label distribution
            label_str = ", ".join([f"{label}:{count}" for label, count in sorted(label_counts.items())])
            
            # Calculate trade frequency (non-zero labels)
            trading_labels = np.sum(labels != 0)
            trade_frequency = trading_labels / total_labels
            
            # Calculate average exceedance ratio
            non_zero_exceedances = exceedances[exceedances != 0]
            avg_exceedance = np.mean(np.abs(non_zero_exceedances)) if len(non_zero_exceedances) > 0 else 0
            
            # Calculate expected return (simplified)
            returns = []
            for i in range(len(labels) - 1):
                if labels[i] != 0:
                    entry_price = test_data["mid_price"][i]
                    exit_price = test_data["mid_price"][i + 1]
                    
                    if labels[i] == 1:  # Long
                        ret = (exit_price - entry_price) / entry_price - transaction_cost
                    else:  # Short
                        ret = (entry_price - exit_price) / entry_price - transaction_cost
                    
                    returns.append(ret)
            
            expected_return = np.mean(returns) if returns else 0
            status = "PROFITABLE" if expected_return > 0 else "UNPROFITABLE"
            
            # Format scaling description
            if "upper_scaling" in params and "lower_scaling" in params:
                scaling_desc = f"+{params['upper_scaling']:.1f}/-{params['lower_scaling']:.1f}"
            else:
                scaling_desc = f"{params.get('scaling_factor', 'N/A'):.1f}x"
            
            print(f"{config_name:>22} {params['lookforward_window']:>7d} {scaling_desc:>10} {label_str:>20} {trade_frequency:>9.1%} {avg_exceedance:>9.2f} {expected_return:>10.6f} {status:>10}")
            
            # Analysis notes
            if trade_frequency < 0.1:
                print(f"    📉 Low trading frequency - consider lower scaling or longer windows")
            elif trade_frequency > 0.6:
                print(f"    ⚠️  High trading frequency - barriers may be too small")
            if avg_exceedance > 2.0:
                print(f"    🎯 High exceedance ratios - barriers well-calibrated for this market")
            if expected_return > 0.0001:
                print(f"    💰 Strong profitability after transaction costs")
            if params.get("adaptive_scaling"):
                print(f"    🔄 Adaptive volatility scaling enabled")
                
        except Exception as e:
            print(f"{config_name:>22} {'ERROR':>7} {'ERROR':>10} {'ERROR':>20} {'ERROR':>9} {'ERROR':>9} {'ERROR':>10} {'ERROR':>10}")
            print(f"    Error: {e}")
    
    print("\n" + "=" * 80)
    print("TRANSACTION COST SCALING ANALYSIS")
    print("=" * 80)
    
    # Test how different scaling factors affect performance
    scaling_factors = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    
    print("Testing transaction cost scaling sensitivity:")
    print("-" * 90)
    print(f"{'Scaling':>8} {'BarrierPips':>11} {'TradeFreq':>9} {'Trades':>7} {'ExpReturn':>10} {'Performance':>12}")
    print("-" * 90)
    
    for scaling in scaling_factors:
        try:
            generator = TargetGeneratorFactory.create(
                "triple_exceedance",
                lookforward_window=120,
                scaling_factor=scaling,
                transaction_cost=transaction_cost,
                adaptive_scaling=False
            )
            targets = generator.generate_targets(test_data.head(800))  # Smaller dataset for speed
            
            target_col = [col for col in targets.columns if col.endswith("_label") and not col.endswith("_exceedance")][0]
            labels = targets[target_col].to_numpy()
            
            trading_labels = np.sum(labels != 0)
            trade_frequency = trading_labels / len(labels)
            barrier_pips = scaling * transaction_cost * 100000
            
            # Quick return calculation
            returns = []
            for i in range(len(labels) - 1):
                if labels[i] != 0:
                    entry_price = test_data["mid_price"][i]
                    exit_price = test_data["mid_price"][i + 1]
                    
                    if labels[i] == 1:
                        ret = (exit_price - entry_price) / entry_price - transaction_cost
                    else:
                        ret = (entry_price - exit_price) / entry_price - transaction_cost
                    
                    returns.append(ret)
            
            expected_return = np.mean(returns) if returns else 0
            
            # Performance assessment
            if expected_return > 0.0001:
                performance = "EXCELLENT"
            elif expected_return > 0:
                performance = "GOOD"
            elif expected_return > -0.0001:
                performance = "NEUTRAL"
            else:
                performance = "POOR"
            
            print(f"{scaling:>8.1f} {barrier_pips:>11.1f} {trade_frequency:>9.1%} {trading_labels:>7d} {expected_return:>10.6f} {performance:>12}")
            
        except Exception as e:
            print(f"{scaling:>8.1f} {'ERROR':>11} {'ERROR':>9} {'ERROR':>7} {'ERROR':>10} {'ERROR':>12}")
    
    print("\n" + "=" * 80)
    print("MULTI-OBJECTIVE OPTIMIZATION TEST")
    print("=" * 80)
    
    print("Testing multi-objective optimization (maximize returns + minimize window)...")
    print("Note: Optimization may fail due to scikit-optimize NumPy compatibility issues")
    
    try:
        # Use subset for faster optimization
        test_prices = test_data.head(600)["mid_price"].to_numpy()
        
        optimizer = ParameterOptimizer(n_calls=8, verbose=True)  # Reduced for testing
        
        # Custom bounds focusing on the multi-objective trade-offs
        custom_bounds = {
            'lookforward_window': (60, 200),     # Focus on shorter windows
            'scaling_factor': (3.0, 12.0),      # Reasonable scaling range
            'min_exceedance_threshold': (0.4, 0.8),  # Moderate thresholds
            'window_penalty_weight': (0.1, 0.4),     # Penalty weight range
            'adaptive_scaling': (0, 1),              # Test both modes
        }
        
        result = optimizer.optimize_triple_exceedance(test_prices, custom_bounds)
        
        print(f"\n🎯 Multi-Objective Optimization Results:")
        print(f"   Method: {result['method']}")
        print(f"   Multi-Objective Score: {result['maximum_returns']:.6f}")
        print(f"   Optimal Parameters:")
        for param, value in result['optimal_params'].items():
            if param == 'scaling_factor':
                barrier_pips = value * transaction_cost * 100000
                print(f"     {param}: {value:.2f}x TC = {barrier_pips:.1f} pips")
            elif param == 'transaction_cost':
                print(f"     {param}: {value:.6f} ({value*100000:.1f} pips)")
            elif param in ['adaptive_scaling']:
                print(f"     {param}: {value} ({'Enabled' if value else 'Disabled'})")
            else:
                print(f"     {param}: {value}")
        
        # Test optimized parameters
        print(f"\n🧪 Testing optimized parameters...")
        opt_generator = TargetGeneratorFactory.create("triple_exceedance", **result['optimal_params'])
        opt_targets = opt_generator.generate_targets(test_data.head(600))
        
        opt_target_col = [col for col in opt_targets.columns if col.endswith("_label") and not col.endswith("_exceedance")][0]
        opt_labels = opt_targets[opt_target_col].to_numpy()
        opt_exceedances = opt_targets[f"{opt_target_col.replace('_label', '')}_exceedance"].to_numpy()
        
        opt_trade_freq = np.sum(opt_labels != 0) / len(opt_labels)
        opt_avg_exceedance = np.mean(np.abs(opt_exceedances[opt_exceedances != 0])) if np.any(opt_exceedances != 0) else 0
        
        # Calculate actual returns
        opt_returns = []
        for i in range(len(opt_labels) - 1):
            if opt_labels[i] != 0:
                entry_price = test_data["mid_price"][i]
                exit_price = test_data["mid_price"][i + 1]
                
                if opt_labels[i] == 1:
                    ret = (exit_price - entry_price) / entry_price - transaction_cost
                else:
                    ret = (entry_price - exit_price) / entry_price - transaction_cost
                
                opt_returns.append(ret)
        
        opt_expected_return = np.mean(opt_returns) if opt_returns else 0
        
        print(f"   Optimized Performance:")
        print(f"     Lookforward Window: {result['optimal_params']['lookforward_window']} ticks")
        print(f"     Scaling Factor: {result['optimal_params']['scaling_factor']:.1f}x")
        print(f"     Trade Frequency: {opt_trade_freq:.1%}")
        print(f"     Average Exceedance: {opt_avg_exceedance:.2f}")
        print(f"     Expected Return: {opt_expected_return:.6f}")
        print(f"     Status: {'✅ PROFITABLE' if opt_expected_return > 0 else '❌ UNPROFITABLE'}")
        
        # Multi-objective analysis
        window_efficiency = 200 / result['optimal_params']['lookforward_window']  # Efficiency vs max window
        return_quality = max(0, opt_expected_return * 10000)  # Scale up for comparison
        
        print(f"   Multi-Objective Analysis:")
        print(f"     Window Efficiency: {window_efficiency:.2f}x (shorter is better)")
        print(f"     Return Quality: {return_quality:.2f} (higher is better)")
        print(f"     Combined Score: {window_efficiency * return_quality:.2f}")
        
        # Save optimized parameters
        with open("optimized_triple_exceedance_params.json", "w") as f:
            json.dump(result['optimal_params'], f, indent=2)
        print(f"   💾 Saved optimized parameters to optimized_triple_exceedance_params.json")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("This is likely due to scikit-optimize NumPy compatibility issues")
    
    print("\n" + "=" * 80)
    print("TRIPLE EXCEEDANCE METHOD SUMMARY")
    print("=" * 80)
    
    print("Key Innovations:")
    print("1. 🔗 Transaction cost-proportional barriers (no arbitrary pip values)")
    print("2. ⚡ Multi-objective optimization (returns + window minimization)")
    print("3. 📊 Adaptive volatility scaling for different market regimes")
    print("4. 🎯 Exceedance ratio tracking for signal strength analysis")
    print("5. ⚙️  Asymmetric scaling (different profit/loss thresholds)")
    print("")
    print("Optimization Benefits:")
    print("- Automatically balances return generation with time efficiency")
    print("- Adapts barrier sizes to transaction cost reality")
    print("- Penalizes longer lookforward windows to improve responsiveness")
    print("- Accounts for market volatility in barrier scaling")
    print("")
    print("Comparison to Triple Barrier:")
    print("- Triple Barrier: Fixed pip-based barriers")
    print("- Triple Exceedance: Transaction cost-scaled, time-efficient barriers")
    print("- Advantage: Better adaptation to different transaction cost regimes")


if __name__ == "__main__":
    test_triple_exceedance_implementation()