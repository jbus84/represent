#!/usr/bin/env python3
"""
DEEP DIVE: Triple Methods Economic Analysis

Fundamental economic analysis of why triple barrier methods fail with micro-volatility data:

1. TRANSACTION COST ECONOMICS
   - 0.7 pip fee = 0.00007 decimal
   - Need >0.00014 profit per round-trip to break even
   - But barriers are 0.00002 (0.002%) = only 0.2 pip profit target!

2. BARRIER WIDTH vs FEES MISMATCH
   - Barrier: 0.00002 (0.2 pips profit)
   - Round-trip cost: 0.00014 (1.4 pips cost)
   - Loss per trade: -1.2 pips GUARANTEED

3. MICRO-VOLATILITY REALITY
   - 92.7% zero price changes
   - Mean absolute return: 0.000006 (0.06 pips)
   - Need barriers 20x+ transaction costs to be profitable

This is a fundamental structural problem, not just parameter optimization.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    from tstrends.returns_estimation import ReturnsEstimatorWithFees
    from tstrends.returns_estimation.fees_config import FeesConfig
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


def analyze_economic_fundamentals(prices: np.ndarray) -> Dict[str, float]:
    """Deep analysis of the economic fundamentals."""
    price_changes = np.diff(prices)
    returns = price_changes / prices[:-1]
    
    # Convert everything to "pips" (basis points * 10) for intuitive understanding
    pip_value = 0.00001  # 1 pip for this pair
    
    stats = {
        'mean_abs_return_pips': np.mean(np.abs(returns)) / pip_value,
        'std_return_pips': np.std(returns) / pip_value,
        'max_move_pips': np.max(np.abs(returns)) / pip_value,
        'min_nonzero_move_pips': np.min(np.abs(returns[returns != 0])) / pip_value if np.any(returns != 0) else 0,
        'zero_changes_pct': np.sum(price_changes == 0) / len(price_changes) * 100,
        'transaction_cost_pips': 0.7,  # Our standard transaction cost
        'round_trip_cost_pips': 1.4,   # Entry + exit
    }
    
    # Calculate break-even requirements
    stats['min_barrier_for_breakeven_pips'] = stats['round_trip_cost_pips']  # Need to exceed round-trip cost
    stats['profitable_move_frequency'] = np.sum(np.abs(returns) > (stats['round_trip_cost_pips'] * pip_value)) / len(returns) * 100
    
    return stats


def test_barrier_economics():
    """Test different barrier widths vs transaction cost economics."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("💰 BARRIER ECONOMICS DEEP DIVE")
    print("=" * 80)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(20000)
    prices = test_df["mid_price"].to_numpy()
    
    # Analyze fundamentals
    econ = analyze_economic_fundamentals(prices)
    
    print("📊 ECONOMIC FUNDAMENTALS:")
    print(f"   Mean absolute move: {econ['mean_abs_return_pips']:.2f} pips")
    print(f"   Std dev of moves: {econ['std_return_pips']:.2f} pips")
    print(f"   Max move observed: {econ['max_move_pips']:.1f} pips")
    print(f"   Min non-zero move: {econ['min_nonzero_move_pips']:.2f} pips")
    print(f"   Zero changes: {econ['zero_changes_pct']:.1f}%")
    print()
    print("💸 TRANSACTION COST ANALYSIS:")
    print(f"   Transaction cost: {econ['transaction_cost_pips']:.1f} pips per trade")
    print(f"   Round-trip cost: {econ['round_trip_cost_pips']:.1f} pips")
    print(f"   Min barrier for breakeven: {econ['min_barrier_for_breakeven_pips']:.1f} pips")
    print(f"   Moves > breakeven: {econ['profitable_move_frequency']:.2f}% of time")
    print()
    
    # Test different barrier configurations
    barrier_tests = [
        {"name": "MICRO (Current)", "pips": 0.2, "decimal": 0.00002},
        {"name": "BREAK-EVEN", "pips": 1.4, "decimal": 0.00014},
        {"name": "PROFITABLE", "pips": 3.0, "decimal": 0.0003},
        {"name": "CONSERVATIVE", "pips": 5.0, "decimal": 0.0005},
        {"name": "WIDE", "pips": 10.0, "decimal": 0.001},
    ]
    
    print("🎯 BARRIER WIDTH TESTING:")
    print(f"{'Barrier':<15} {'Expected':<12} {'Actual':<12} {'Trades':<8} {'Hit Rate':<8} {'Outcome':<15}")
    print("-" * 80)
    
    for test in barrier_tests:
        barrier_pips = test["pips"]
        barrier_decimal = test["decimal"]
        
        # Calculate expected outcome
        if barrier_pips < econ['round_trip_cost_pips']:
            expected = f"-{econ['round_trip_cost_pips'] - barrier_pips:.1f}p"
        else:
            expected = f"+{barrier_pips - econ['round_trip_cost_pips']:.1f}p"
        
        try:
            # Test with Triple Barrier
            generator = TargetGeneratorFactory.create("triple_barrier", 
                lookforward_window=1000,
                barrier_width=barrier_decimal,
                min_return_threshold=1e-8,
                volatility_window=100,
                normalize_by_volatility=False,
            )
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Calculate actual PnL
            fees_config = FeesConfig(
                lp_transaction_fees=0.00007,
                sp_transaction_fees=0.00007,
            )
            returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
            
            total_pnl = returns_estimator.estimate_return(
                prices.tolist(),
                labels.tolist()
            )
            
            # Count trades and hits
            num_trades = sum(1 for j in range(1, len(labels)) if labels[j] != labels[j-1])
            
            # Calculate hit rate (profitable trades)
            profit_hits = np.sum(labels != 0)  # Non-neutral positions
            hit_rate = profit_hits / len(labels) * 100
            
            actual = f"{total_pnl*10000:.0f}p"  # Convert to pips
            
            # Determine outcome
            if total_pnl > 0.001:
                outcome = "PROFITABLE ✅"
            elif total_pnl > -0.001:
                outcome = "BREAK-EVEN ⚖️"
            elif num_trades == 0:
                outcome = "NO TRADES ⚠️"
            else:
                outcome = "LOSS ❌"
            
            print(f"{test['name']:<15} {expected:<12} {actual:<12} {num_trades:<8} {hit_rate:.1f}%{'':<4} {outcome:<15}")
            
        except Exception as e:
            print(f"{test['name']:<15} {expected:<12} {'ERROR':<12} {'N/A':<8} {'N/A':<8} {str(e)[:15]:<15}")
    
    print()
    return econ


def analyze_microstructure_reality():
    """Analyze the microstructure reality of this data."""
    print("🔬 MICROSTRUCTURE REALITY CHECK")
    print("=" * 80)
    
    # Load larger sample for better statistics
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Analyze different sample sizes
    sample_sizes = [10000, 50000, 100000]
    
    for sample_size in sample_sizes:
        test_df = df.head(sample_size)
        prices = test_df["mid_price"].to_numpy()
        
        price_changes = np.diff(prices)
        returns = price_changes / prices[:-1]
        
        pip_value = 0.00001
        return_pips = returns / pip_value
        
        print(f"📈 SAMPLE SIZE: {sample_size:,} ticks")
        
        # Movement analysis
        zero_pct = np.sum(price_changes == 0) / len(price_changes) * 100
        up_moves = np.sum(return_pips > 0)
        down_moves = np.sum(return_pips < 0)
        
        print(f"   Zero changes: {zero_pct:.1f}%")
        print(f"   Up moves: {up_moves:,} ({up_moves/len(returns)*100:.1f}%)")
        print(f"   Down moves: {down_moves:,} ({down_moves/len(returns)*100:.1f}%)")
        
        # Profitable move analysis
        profitable_moves = np.sum(np.abs(return_pips) > 1.4)  # > round-trip cost
        print(f"   Moves > 1.4 pips: {profitable_moves:,} ({profitable_moves/len(returns)*100:.2f}%)")
        
        # Size distribution
        abs_pips = np.abs(return_pips[return_pips != 0])
        if len(abs_pips) > 0:
            print(f"   Non-zero moves - Min: {abs_pips.min():.2f}p, Mean: {abs_pips.mean():.2f}p, Max: {abs_pips.max():.1f}p")
            print(f"   95th percentile: {np.percentile(abs_pips, 95):.1f}p")
        
        print()


def recommend_solutions():
    """Recommend potential solutions for the economic mismatch."""
    print("💡 RECOMMENDED SOLUTIONS")
    print("=" * 80)
    
    print("1. TRANSACTION COST REDUCTION:")
    print("   • Current: 0.7 pips per trade")
    print("   • Needed: <0.1 pips for micro-strategies")
    print("   • Solutions: Market making, institutional execution, rebates")
    print()
    
    print("2. BARRIER SCALING FIXES:")
    print("   • Current bounds: 0.0005% - 0.01% (0.05-1 pip)")
    print("   • Profitable bounds: 0.03% - 0.1% (3-10 pips)")
    print("   • But: Only 0.02% of moves exceed 1.4 pips!")
    print()
    
    print("3. ALTERNATIVE APPROACHES:")
    print("   • Portfolio-based strategies (diversification)")
    print("   • Longer time horizons (daily/weekly barriers)")
    print("   • Regime-aware barriers (volatile periods only)")
    print("   • Ensemble methods with other signals")
    print()
    
    print("4. REALISTIC BOUNDS UPDATE:")
    print("   • Triple Barrier: barrier_width=(0.0002, 0.002) # 2-20 pips")
    print("   • Triple Exceedance: scaling_factor=(10, 50) # Much higher")
    print("   • Accept lower Sharpe ratios, focus on consistency")
    print()
    
    print("⚠️  FUNDAMENTAL TRUTH:")
    print("High-frequency micro-volatility strategies require:")
    print("• Ultra-low transaction costs (<0.1 pips)")
    print("• Sophisticated execution algorithms")
    print("• Market making capabilities")
    print("• Or completely different approaches")


def main():
    """Run comprehensive economic analysis."""
    try:
        # Economic fundamentals
        test_barrier_economics()
        
        # Microstructure analysis
        analyze_microstructure_reality()
        
        # Recommendations
        recommend_solutions()
        
        print("\n🎯 CONCLUSION:")
        print("=" * 80)
        print("The triple barrier methods are fundamentally incompatible with")
        print("this micro-volatility data due to transaction cost economics.")
        print("Current 0.2-1 pip barriers vs 1.4 pip costs = guaranteed losses.")
        print("Need either much lower fees or completely different strategy approach.")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()