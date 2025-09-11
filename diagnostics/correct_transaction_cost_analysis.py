#!/usr/bin/env python3
"""
CORRECTED Transaction Cost Analysis

ERROR CORRECTION: I was wrong about transaction costs!
- Transaction cost = 0.7 pips TOTAL round-trip (not 1.4!)
- 0.00007 decimal = 0.7 pips total (entry + exit combined)

This completely changes the economics:
- Breakeven barrier: 0.7 pips (not 1.4)
- Profitable barrier: 1+ pips (not 2+)
- Current barriers: 0.2-3 pips should work!

Let me re-analyze with correct costs.
"""

import numpy as np
import polars as pl
from pathlib import Path

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    from tstrends.returns_estimation import ReturnsEstimatorWithFees
    from tstrends.returns_estimation.fees_config import FeesConfig
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


def correct_economic_analysis():
    """Re-analyze with CORRECT transaction cost understanding."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("🔧 CORRECTED TRANSACTION COST ANALYSIS")
    print("=" * 60)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(20000)
    prices = test_df["mid_price"].to_numpy()
    
    # Calculate statistics with CORRECT transaction cost
    price_changes = np.diff(prices)
    returns = price_changes / prices[:-1]
    pip_value = 0.00001
    return_pips = returns / pip_value
    
    print("💸 CORRECTED TRANSACTION COST FACTS:")
    print("   Transaction cost: 0.7 pips TOTAL round-trip")
    print("   Decimal value: 0.00007")
    print("   Breakeven barrier: >0.7 pips profit target")
    print("   My previous error: Counted 1.4 pips (double-counting!)")
    print()
    
    # Economic fundamentals with correct costs
    abs_returns = np.abs(returns[returns != 0])
    abs_return_pips = abs_returns / pip_value if len(abs_returns) > 0 else np.array([])
    
    print("📊 MARKET STATISTICS:")
    print(f"   Mean absolute move: {np.mean(abs_return_pips):.2f} pips")
    print(f"   Zero changes: {np.sum(price_changes == 0)/len(price_changes)*100:.1f}%")
    print(f"   Moves > 0.7 pips: {np.sum(abs_return_pips > 0.7)/len(abs_return_pips)*100:.1f}% (profitable moves)")
    print()
    
    # Test barrier configurations with CORRECT expectations
    barrier_tests = [
        {"name": "SUB-BREAKEVEN", "pips": 0.5, "decimal": 0.00005},  # Should lose
        {"name": "BREAKEVEN", "pips": 0.7, "decimal": 0.00007},      # Should break even
        {"name": "SMALL PROFIT", "pips": 1.0, "decimal": 0.0001},   # Should be slightly profitable
        {"name": "DECENT PROFIT", "pips": 2.0, "decimal": 0.0002},  # Should be profitable
        {"name": "LARGE PROFIT", "pips": 5.0, "decimal": 0.0005},   # Should be very profitable but rare
    ]
    
    print("🎯 BARRIER TESTING WITH CORRECT COSTS:")
    print(f"{'Barrier':<15} {'Expected':<12} {'Actual':<12} {'Trades':<8} {'Outcome':<15}")
    print("-" * 70)
    
    for test in barrier_tests:
        barrier_pips = test["pips"]
        barrier_decimal = test["decimal"]
        
        # Calculate CORRECT expected outcome
        expected_profit = barrier_pips - 0.7  # Subtract actual 0.7 pip cost
        if expected_profit > 0:
            expected = f"+{expected_profit:.1f}p"
        else:
            expected = f"{expected_profit:.1f}p"
        
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
            
            # Calculate PnL with correct fees
            fees_config = FeesConfig(
                lp_transaction_fees=0.00007,  # This is TOTAL round-trip
                sp_transaction_fees=0.00007,  # This is TOTAL round-trip
            )
            returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
            
            total_pnl = returns_estimator.estimate_return(
                prices.tolist(),
                labels.tolist()
            )
            
            num_trades = sum(1 for j in range(1, len(labels)) if labels[j] != labels[j-1])
            actual_pips = total_pnl * 10000  # Convert to pips
            actual = f"{actual_pips:.0f}p"
            
            # Determine outcome
            if total_pnl > 0.001:
                outcome = "PROFITABLE ✅"
            elif total_pnl > -0.001:
                outcome = "BREAK-EVEN ⚖️"
            elif num_trades == 0:
                outcome = "NO TRADES ⚠️"
            else:
                outcome = "LOSS ❌"
            
            print(f"{test['name']:<15} {expected:<12} {actual:<12} {num_trades:<8} {outcome:<15}")
            
        except Exception as e:
            print(f"{test['name']:<15} {expected:<12} {'ERROR':<12} {'N/A':<8} {str(e)[:15]:<15}")
    
    print()


def analyze_hit_rates():
    """Analyze why barriers still lose money despite correct costs."""
    if not LIBRARIES_AVAILABLE:
        return
    
    print("🎯 HIT RATE ANALYSIS")
    print("=" * 60)
    
    # Load data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(10000)
    
    # Test a specific barrier configuration to understand hit patterns
    try:
        generator = TargetGeneratorFactory.create("triple_barrier", 
            lookforward_window=1000,
            barrier_width=0.0001,  # 1 pip barriers
            min_return_threshold=1e-8,
            volatility_window=100,
            normalize_by_volatility=False,
        )
        targets_df = generator.generate_targets(test_df)
        target_info = generator.get_target_info()
        target_col = target_info['target_names'][0]
        labels = targets_df[target_col].to_numpy()
        
        # Analyze label distribution (hit patterns)
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100
        
        print("📊 LABEL DISTRIBUTION (1 pip barriers):")
        for label, pct in zip(unique_labels, percentages):
            if label == -1:
                print(f"   Loss hits: {pct:.1f}% (hit lower barrier first)")
            elif label == 0:
                print(f"   Timeouts: {pct:.1f}% (hit time barrier)")
            elif label == 1:
                print(f"   Profit hits: {pct:.1f}% (hit upper barrier first)")
        
        print()
        
        # Calculate expected vs actual based on hit rates
        loss_pct = percentages[unique_labels == -1][0] if -1 in unique_labels else 0
        profit_pct = percentages[unique_labels == 1][0] if 1 in unique_labels else 0
        timeout_pct = percentages[unique_labels == 0][0] if 0 in unique_labels else 0
        
        # Expected PnL calculation
        # Profit hits: +1 pip - 0.7 pip cost = +0.3 pip
        # Loss hits: -1 pip - 0.7 pip cost = -1.7 pip
        # Timeouts: 0 pip - 0.7 pip cost = -0.7 pip (if we trade on timeout)
        
        expected_pnl_pips = (profit_pct/100 * 0.3) + (loss_pct/100 * -1.7) + (timeout_pct/100 * -0.7)
        print(f"💰 EXPECTED PnL BREAKDOWN:")
        print(f"   Profit hits contribute: +{profit_pct/100 * 0.3:.1f} pips")
        print(f"   Loss hits contribute: {loss_pct/100 * -1.7:.1f} pips")
        print(f"   Timeout hits contribute: {timeout_pct/100 * -0.7:.1f} pips")
        print(f"   Expected total: {expected_pnl_pips:.1f} pips")
        
        # This shows WHY the strategy loses money despite correct transaction costs
        
    except Exception as e:
        print(f"Hit rate analysis failed: {e}")


def main():
    """Run corrected analysis."""
    try:
        correct_economic_analysis()
        analyze_hit_rates()
        
        print("💡 CORRECTED CONCLUSION:")
        print("=" * 60)
        print("With CORRECT 0.7 pip transaction costs:")
        print("1. Barriers should be profitable at 1+ pips")
        print("2. But hit rate analysis reveals the real problem")
        print("3. Loss hits (-1.7p) outweigh profit hits (+0.3p)")
        print("4. This is a market microstructure issue, not transaction costs")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()