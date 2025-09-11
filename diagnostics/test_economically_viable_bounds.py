#!/usr/bin/env python3
"""
Test Economically Viable Bounds

Test the updated bounds that are designed to overcome the transaction cost hurdle:
- Triple Barrier: 3-20 pip barriers (vs 0.2 pip before)
- Triple Exceedance: 4x-25x scaling (3-18 pip barriers)

These should finally be profitable (or at least break-even).
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


def test_economically_viable_bounds():
    """Test the new economically viable bounds."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("💰 TESTING ECONOMICALLY VIABLE BOUNDS")
    print("=" * 60)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(20000)
    prices = test_df["mid_price"].to_numpy()
    
    print(f"Test data: {len(test_df):,} samples")
    print()
    
    # Test configurations with economically viable bounds
    test_configs = [
        {
            "method": "triple_barrier",
            "name": "VIABLE Triple Barrier (3 pips)",
            "params": {
                "lookforward_window": 2000,
                "barrier_width": 0.0003,        # 3 pips - above breakeven
                "min_return_threshold": 1e-5,
                "volatility_window": 200,
                "normalize_by_volatility": False,
            },
            "expected_pips": 3.0 - 1.4  # 3 pip target - 1.4 pip cost
        },
        {
            "method": "triple_barrier",
            "name": "CONSERVATIVE Triple Barrier (10 pips)",
            "params": {
                "lookforward_window": 3000,
                "barrier_width": 0.001,         # 10 pips - well above breakeven
                "min_return_threshold": 1e-5,
                "volatility_window": 300,
                "normalize_by_volatility": False,
            },
            "expected_pips": 10.0 - 1.4  # 10 pip target - 1.4 pip cost
        },
        {
            "method": "triple_exceedance",
            "name": "VIABLE Triple Exceedance (5x scaling)",
            "params": {
                "lookforward_window": 2000,
                "scaling_factor": 5.0,          # 5x × 0.7 = 3.5 pip barriers
                "min_exceedance_threshold": 0.5,
                "volatility_window": 200,
                "window_penalty_weight": 0.1,
                "balance_weight": 0.8,
                "target_balance_ratio": 0.33,
                "adaptive_scaling": True,
            },
            "expected_pips": 3.5 - 1.4  # 3.5 pip barriers - 1.4 pip cost
        },
        {
            "method": "triple_exceedance",
            "name": "CONSERVATIVE Triple Exceedance (15x scaling)",
            "params": {
                "lookforward_window": 3000,
                "scaling_factor": 15.0,         # 15x × 0.7 = 10.5 pip barriers
                "min_exceedance_threshold": 0.6,
                "volatility_window": 300,
                "window_penalty_weight": 0.1,
                "balance_weight": 0.8,
                "target_balance_ratio": 0.33,
                "adaptive_scaling": True,
            },
            "expected_pips": 10.5 - 1.4  # 10.5 pip barriers - 1.4 pip cost
        }
    ]
    
    print(f"{'Configuration':<35} {'Expected':<12} {'Actual':<12} {'Trades':<8} {'Classes':<8} {'Outcome':<15}")
    print("-" * 100)
    
    for config in test_configs:
        method = config["method"]
        name = config["name"]
        params = config["params"]
        expected = f"+{config['expected_pips']:.1f}p"
        
        try:
            # Generate labels
            generator = TargetGeneratorFactory.create(method, **params)
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Analyze classes
            unique_labels, counts = np.unique(labels, return_counts=True)
            num_classes = len(unique_labels)
            
            # Calculate PnL
            fees_config = FeesConfig(
                lp_transaction_fees=0.00007,
                sp_transaction_fees=0.00007,
            )
            returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
            
            total_pnl = returns_estimator.estimate_return(
                prices.tolist(),
                labels.tolist()
            )
            
            # Count trades
            num_trades = sum(1 for j in range(1, len(labels)) if labels[j] != labels[j-1])
            
            actual = f"{total_pnl*10000:.0f}p"  # Convert to pips
            
            # Determine outcome
            if total_pnl > 0.005:  # > 0.5%
                outcome = "PROFITABLE ✅"
            elif total_pnl > -0.001:  # > -0.1%
                outcome = "BREAK-EVEN ⚖️"
            elif num_trades == 0:
                outcome = "NO TRADES ⚠️"
            else:
                outcome = "LOSS ❌"
            
            print(f"{name:<35} {expected:<12} {actual:<12} {num_trades:<8} {num_classes:<8} {outcome:<15}")
            
            # Show detailed analysis for profitable ones
            if total_pnl > 0:
                percentages = counts / len(labels) * 100
                balance_score = min(percentages) / max(percentages) * 100 if len(percentages) > 1 else 100
                print(f"    📊 Class balance: {balance_score:.1f}%, Distribution: {dict(zip(unique_labels, percentages.round(1)))}")
                mean_return = total_pnl / num_trades if num_trades > 0 else 0
                print(f"    💰 Mean return/trade: {mean_return:.6f} ({mean_return*10000:.1f}p)")
                print()
            
        except Exception as e:
            print(f"{name:<35} {expected:<12} {'ERROR':<12} {'N/A':<8} {'N/A':<8} {str(e)[:15]:<15}")
    
    print()
    print("💡 ANALYSIS:")
    print("These bounds should finally overcome the 1.4 pip transaction cost hurdle")
    print("by targeting 3+ pip profit per trade, making the strategies economically viable.")


def compare_bounds_evolution():
    """Show the evolution of bounds understanding."""
    print("📈 BOUNDS EVOLUTION")
    print("=" * 60)
    
    print("TRIPLE BARRIER barrier_width:")
    print("  Original: (0.0001, 0.005) → 1-50 pip barriers")
    print("  Micro-optimized: (0.000005, 0.0001) → 0.05-1 pip barriers")
    print("  ECONOMICALLY VIABLE: (0.0003, 0.002) → 3-20 pip barriers")
    print()
    
    print("TRIPLE EXCEEDANCE scaling_factor:")  
    print("  Original: (2.0, 20.0) → 1.4-14 pip barriers")
    print("  Micro-optimized: (1.1, 5.0) → 0.8-3.5 pip barriers")
    print("  ECONOMICALLY VIABLE: (4.0, 25.0) → 3-18 pip barriers")
    print()
    
    print("KEY INSIGHT:")
    print("The 'micro-optimized' bounds were actually TOO SMALL!")
    print("They created barriers smaller than transaction costs = guaranteed losses.")
    print("Economic viability requires barriers >2x transaction costs.")


def main():
    """Test economically viable bounds."""
    try:
        compare_bounds_evolution()
        print()
        test_economically_viable_bounds()
        
        print("🎯 CONCLUSION:")
        print("=" * 60)
        print("These economically viable bounds should finally enable")
        print("the triple barrier methods to find profitable parameters")
        print("by ensuring barriers exceed transaction cost hurdles.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()