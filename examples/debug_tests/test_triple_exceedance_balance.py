#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Test Triple Exceedance Method with Multi-Objective Optimization
Focus: Returns + Window Length + Class Balance
"""

import polars as pl
import numpy as np
from represent.target_generators.factory import TargetGeneratorFactory
from represent.parameter_optimization import ParameterOptimizer
import json


def test_multi_objective_exceedance():
    """Test Triple Exceedance with three objectives: returns, window, balance"""
    
    print("=" * 90)
    print("TESTING TRIPLE EXCEEDANCE - THREE OBJECTIVE OPTIMIZATION")
    print("Objectives: 1) Maximize Returns  2) Minimize Window  3) Maximize Class Balance")
    print("=" * 90)
    
    # Create test data with mixed market conditions
    np.random.seed(42)
    n_samples = 1000
    
    # Create complex price pattern: trend + mean reversion + volatility clustering
    base_price = 1.1000
    transaction_cost = 0.0001
    
    # Multi-regime market
    t = np.linspace(0, 4*np.pi, n_samples)
    trend = 0.002 * np.sin(t * 0.3)  # Slow trending
    mean_reversion = 0.0005 * np.sin(t * 2)  # Faster oscillation
    
    # Volatility clustering
    returns = []
    volatility = 0.0001
    for i in range(n_samples):
        if i > 0:
            volatility = 0.9 * volatility + 0.1 * 0.0001 + 0.2 * (returns[-1]**2)
        ret = np.random.normal(0, volatility)
        returns.append(ret)
    
    prices = base_price + np.cumsum(trend + mean_reversion + returns)
    
    test_data = pl.DataFrame({
        "ts_event": range(n_samples),
        "mid_price": prices,
        "symbol": ["EURUSD"] * n_samples
    })
    
    print(f"Test data: {n_samples} samples")
    print(f"Price range: {test_data['mid_price'].min():.6f} - {test_data['mid_price'].max():.6f}")
    print(f"Transaction cost: {transaction_cost * 100000:.1f} pips")
    
    # Test different balance weight configurations
    balance_configs = [
        ("No Balance Focus", {"balance_weight": 0.0, "target_balance_ratio": 0.33}),
        ("Light Balance", {"balance_weight": 0.2, "target_balance_ratio": 0.33}),
        ("Moderate Balance", {"balance_weight": 0.5, "target_balance_ratio": 0.33}),
        ("Heavy Balance", {"balance_weight": 1.0, "target_balance_ratio": 0.33}),
        ("Strict Balance", {"balance_weight": 2.0, "target_balance_ratio": 0.33}),
    ]
    
    print("\nTesting different class balance objectives:")
    print("-" * 130)
    print(f"{'Config':>15} {'Window':>7} {'Scaling':>8} {'BalWt':>6} {'Class Distribution':>20} {'Balance':>8} {'Returns':>9} {'Score':>8}")
    print("-" * 130)
    
    results = []
    
    for config_name, balance_params in balance_configs:
        try:
            # Base parameters with varying balance focus
            params = {
                "lookforward_window": 150,
                "scaling_factor": 6.0,
                "transaction_cost": transaction_cost,
                "window_penalty_weight": 0.2,
                "adaptive_scaling": False,
                **balance_params
            }
            
            generator = TargetGeneratorFactory.create("triple_exceedance", **params)
            targets = generator.generate_targets(test_data)
            
            # Get labels
            target_col = [col for col in targets.columns if col.endswith("_label")][0]
            labels = targets[target_col].to_numpy()
            
            # Calculate class balance metrics
            balance_metrics = generator.calculate_class_balance_metrics(labels)
            
            # Calculate returns
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
            
            # Multi-objective fitness score
            fitness_score = generator.calculate_fitness_score(test_data["mid_price"].to_numpy())
            
            # Format class distribution
            props = balance_metrics["proportions"]
            class_dist = f"{props[-1]:.2f}|{props[0]:.2f}|{props[1]:.2f}"
            
            print(f"{config_name:>15} {params['lookforward_window']:>7d} {params['scaling_factor']:>8.1f} {params['balance_weight']:>6.1f} {class_dist:>20} {balance_metrics['balance_score']:>8.3f} {expected_return:>9.6f} {fitness_score:>8.2f}")
            
            results.append({
                "config": config_name,
                "params": params,
                "balance_metrics": balance_metrics,
                "expected_return": expected_return,
                "fitness_score": fitness_score,
                "labels": labels
            })
            
            # Analysis notes
            if balance_metrics["balance_score"] > 0.8:
                print(f"    ✅ Excellent class balance (score: {balance_metrics['balance_score']:.3f})")
            if balance_metrics["normalized_entropy"] > 0.9:
                print(f"    📊 High entropy = {balance_metrics['normalized_entropy']:.3f} (well-distributed)")
            if expected_return > 0:
                print(f"    💰 Profitable after transaction costs")
                
        except Exception as e:
            print(f"{config_name:>15} {'ERROR':>7} {'ERROR':>8} {'ERROR':>6} {'ERROR':>20} {'ERROR':>8} {'ERROR':>9} {'ERROR':>8}")
            print(f"    Error: {e}")
    
    print("\n" + "=" * 90)
    print("CLASS BALANCE ANALYSIS")
    print("=" * 90)
    
    if results:
        best_balance = max(results, key=lambda x: x["balance_metrics"]["balance_score"])
        best_returns = max(results, key=lambda x: x["expected_return"])
        best_fitness = max(results, key=lambda x: x["fitness_score"])
        
        print(f"🏆 Best Class Balance: {best_balance['config']}")
        print(f"   Balance Score: {best_balance['balance_metrics']['balance_score']:.3f}")
        print(f"   Entropy: {best_balance['balance_metrics']['normalized_entropy']:.3f}")
        print(f"   Distribution: {best_balance['balance_metrics']['proportions']}")
        
        print(f"\n💰 Best Returns: {best_returns['config']}")
        print(f"   Expected Return: {best_returns['expected_return']:.6f}")
        print(f"   Balance Score: {best_returns['balance_metrics']['balance_score']:.3f}")
        
        print(f"\n🎯 Best Multi-Objective Score: {best_fitness['config']}")
        print(f"   Fitness Score: {best_fitness['fitness_score']:.2f}")
        print(f"   Expected Return: {best_fitness['expected_return']:.6f}")
        print(f"   Balance Score: {best_fitness['balance_metrics']['balance_score']:.3f}")
        print(f"   Window Length: {best_fitness['params']['lookforward_window']} ticks")
    
    print("\n" + "=" * 90)
    print("WINDOW LENGTH vs CLASS BALANCE TRADE-OFF")
    print("=" * 90)
    
    # Test how window length affects class balance
    window_lengths = [80, 120, 180, 250, 350]
    
    print("Testing window length impact on class balance:")
    print("-" * 100)
    print(f"{'Window':>7} {'Class Distribution':>20} {'Balance':>8} {'Entropy':>8} {'Returns':>9} {'TradeFreq':>9}")
    print("-" * 100)
    
    for window in window_lengths:
        try:
            generator = TargetGeneratorFactory.create(
                "triple_exceedance",
                lookforward_window=window,
                scaling_factor=7.0,
                transaction_cost=transaction_cost,
                balance_weight=0.5,
                target_balance_ratio=0.33,
                window_penalty_weight=0.3,
                adaptive_scaling=False
            )
            
            targets = generator.generate_targets(test_data)
            target_col = [col for col in targets.columns if col.endswith("_label")][0]
            labels = targets[target_col].to_numpy()
            
            balance_metrics = generator.calculate_class_balance_metrics(labels)
            
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
            trade_freq = np.sum(labels != 0) / len(labels)
            
            props = balance_metrics["proportions"]
            class_dist = f"{props[-1]:.2f}|{props[0]:.2f}|{props[1]:.2f}"
            
            print(f"{window:>7d} {class_dist:>20} {balance_metrics['balance_score']:>8.3f} {balance_metrics['normalized_entropy']:>8.3f} {expected_return:>9.6f} {trade_freq:>9.1%}")
            
        except Exception as e:
            print(f"{window:>7d} {'ERROR':>20} {'ERROR':>8} {'ERROR':>8} {'ERROR':>9} {'ERROR':>9}")
    
    print("\n" + "=" * 90)
    print("COMPLETE MULTI-OBJECTIVE OPTIMIZATION TEST")
    print("=" * 90)
    
    print("Running full three-objective Bayesian optimization...")
    print("Objectives: 1) Maximize Returns  2) Minimize Window  3) Maximize Class Balance")
    
    try:
        test_prices = test_data.head(700)["mid_price"].to_numpy()
        
        optimizer = ParameterOptimizer(n_calls=12, verbose=True, random_state=42)
        
        # Focused bounds for three-objective optimization
        custom_bounds = {
            'lookforward_window': (80, 300),      # Balance efficiency vs effectiveness
            'scaling_factor': (4.0, 15.0),       # Reasonable scaling range
            'min_exceedance_threshold': (0.5, 0.8),  # Moderate selectivity
            'balance_weight': (0.2, 1.5),        # Significant balance focus
            'target_balance_ratio': (0.30, 0.35), # Near-perfect balance
            'window_penalty_weight': (0.1, 0.4),  # Window efficiency focus
            'adaptive_scaling': (0, 1),           # Test both modes
        }
        
        result = optimizer.optimize_triple_exceedance(test_prices, custom_bounds)
        
        print(f"\n🎯 THREE-OBJECTIVE OPTIMIZATION RESULTS:")
        print(f"   Method: {result['method']}")
        print(f"   Multi-Objective Score: {result['maximum_returns']:.4f}")
        print(f"   Optimal Parameters:")
        
        for param, value in result['optimal_params'].items():
            if param == 'scaling_factor':
                barrier_pips = value * transaction_cost * 100000
                print(f"     {param}: {value:.2f}x TC = {barrier_pips:.1f} pips")
            elif param == 'target_balance_ratio':
                perfect_balance = 1.0/3
                deviation = abs(value - perfect_balance) * 100
                print(f"     {param}: {value:.3f} (±{deviation:.1f}% from perfect balance)")
            elif param in ['adaptive_scaling']:
                print(f"     {param}: {value} ({'Enabled' if value else 'Disabled'})")
            else:
                print(f"     {param}: {value:.3f}")
        
        # Test optimized parameters
        print(f"\n🧪 Testing three-objective optimized parameters...")
        opt_generator = TargetGeneratorFactory.create("triple_exceedance", **result['optimal_params'])
        opt_targets = opt_generator.generate_targets(test_data.head(700))
        
        opt_target_col = [col for col in opt_targets.columns if col.endswith("_label")][0]
        opt_labels = opt_targets[opt_target_col].to_numpy()
        
        # Comprehensive analysis
        opt_balance_metrics = opt_generator.calculate_class_balance_metrics(opt_labels)
        
        # Calculate returns
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
        opt_trade_freq = np.sum(opt_labels != 0) / len(opt_labels)
        
        print(f"   Three-Objective Performance:")
        print(f"     📈 Returns Objective:")
        print(f"       Expected Return: {opt_expected_return:.6f}")
        print(f"       Total Trades: {len(opt_returns)}")
        print(f"       Trade Frequency: {opt_trade_freq:.1%}")
        
        print(f"     ⚡ Window Objective:")
        print(f"       Lookforward Window: {result['optimal_params']['lookforward_window']:.0f} ticks")
        print(f"       Window Penalty Weight: {result['optimal_params']['window_penalty_weight']:.3f}")
        print(f"       Time Efficiency Score: {300/result['optimal_params']['lookforward_window']:.2f}x")
        
        print(f"     ⚖️  Balance Objective:")
        print(f"       Class Distribution: {opt_balance_metrics['proportions']}")
        print(f"       Balance Score: {opt_balance_metrics['balance_score']:.3f}")
        print(f"       Normalized Entropy: {opt_balance_metrics['normalized_entropy']:.3f}")
        print(f"       Max Deviation: {opt_balance_metrics['max_deviation']:.3f}")
        print(f"       Balance Weight: {result['optimal_params']['balance_weight']:.3f}")
        
        # Overall assessment
        objectives_met = 0
        if opt_expected_return > 0:
            objectives_met += 1
            print(f"     ✅ Returns: POSITIVE ({opt_expected_return:.6f})")
        else:
            print(f"     ❌ Returns: NEGATIVE ({opt_expected_return:.6f})")
            
        if result['optimal_params']['lookforward_window'] < 200:
            objectives_met += 1
            print(f"     ✅ Window: EFFICIENT ({result['optimal_params']['lookforward_window']:.0f} < 200 ticks)")
        else:
            print(f"     ❌ Window: LONG ({result['optimal_params']['lookforward_window']:.0f} ≥ 200 ticks)")
            
        if opt_balance_metrics['balance_score'] > 0.7:
            objectives_met += 1
            print(f"     ✅ Balance: GOOD ({opt_balance_metrics['balance_score']:.3f} > 0.7)")
        else:
            print(f"     ❌ Balance: POOR ({opt_balance_metrics['balance_score']:.3f} ≤ 0.7)")
        
        print(f"\n   🏆 OBJECTIVES MET: {objectives_met}/3")
        if objectives_met == 3:
            print(f"   🎉 ALL THREE OBJECTIVES ACHIEVED!")
        elif objectives_met >= 2:
            print(f"   👍 MAJORITY OBJECTIVES ACHIEVED")
        else:
            print(f"   ⚠️  OPTIMIZATION NEEDS IMPROVEMENT")
        
        # Save results
        with open("optimized_triple_exceedance_multi_objective_params.json", "w") as f:
            json.dump(result['optimal_params'], f, indent=2)
        print(f"   💾 Saved parameters to optimized_triple_exceedance_multi_objective_params.json")
        
    except Exception as e:
        print(f"❌ Three-objective optimization failed: {e}")
        print("This may be due to scikit-optimize compatibility issues or complex objective interactions")
    
    print("\n" + "=" * 90)
    print("MULTI-OBJECTIVE OPTIMIZATION SUMMARY")
    print("=" * 90)
    
    print("🎯 Three-Objective Innovation:")
    print("1. 📈 Returns Maximization: Optimize for profitable trading signals")
    print("2. ⚡ Window Minimization: Reduce lookforward time for faster signals")  
    print("3. ⚖️  Balance Maximization: Ensure even class distribution for ML training")
    print()
    print("🔧 Technical Implementation:")
    print("- Weighted multi-objective fitness function")
    print("- Class balance entropy and deviation metrics")
    print("- Transaction cost-scaled barriers")
    print("- Adaptive volatility scaling")
    print()
    print("💡 Key Insights:")
    print("- Balance weight controls trade-off between profit and class distribution")
    print("- Shorter windows may improve balance through reduced time bias")
    print("- Transaction cost scaling ensures realistic profitability constraints")
    print("- Multi-objective optimization finds Pareto-optimal solutions")
    print()
    print("🚀 Applications:")
    print("- Balanced datasets for ML model training")
    print("- Time-efficient trading signal generation") 
    print("- Transaction cost-aware strategy development")
    print("- Multi-regime market adaptation")


if __name__ == "__main__":
    test_multi_objective_exceedance()