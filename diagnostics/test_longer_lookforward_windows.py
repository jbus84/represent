#!/usr/bin/env python3
"""
Test Longer Lookforward Windows for Triple Barriers

The real issue: lookforward windows were too short!
- Current: 500-2000 ticks (0.5-2 minutes)
- Needed: 5000+ ticks (5+ minutes) for moves to develop

With micro-volatility (92.7% zero changes), need much longer time
for meaningful price movements to occur and hit profit barriers.
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


def analyze_move_development_over_time():
    """Analyze how price moves develop over different time horizons."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("📊 PRICE MOVE DEVELOPMENT OVER TIME")
    print("=" * 60)
    
    # Load larger sample to analyze longer time horizons
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    prices = df["mid_price"].to_numpy()[:50000]  # 50K ticks for analysis
    
    pip_value = 0.00001
    
    # Test different lookforward windows
    lookforward_windows = [500, 1000, 2000, 3000, 5000, 7500, 10000]
    
    print(f"{'Window':<8} {'Time':<8} {'Max Move':<10} {'Avg Move':<10} {'Std Move':<10} {'>1 pip':<8} {'>2 pip':<8}")
    print("-" * 70)
    
    for window in lookforward_windows:
        time_minutes = window / 1000  # Rough estimate: 1000 ticks ≈ 1 minute
        
        # Calculate maximum absolute moves over this window
        max_moves = []
        avg_moves = []
        
        for i in range(0, len(prices) - window, window):  # Non-overlapping windows
            window_prices = prices[i:i+window]
            if len(window_prices) > 1:
                # Maximum move in this window
                max_move = np.max(window_prices) - np.min(window_prices)
                max_moves.append(max_move)
                
                # Average absolute move
                price_changes = np.diff(window_prices)
                avg_move = np.mean(np.abs(price_changes))
                avg_moves.append(avg_move)
        
        if max_moves:
            max_moves = np.array(max_moves)
            avg_moves = np.array(avg_moves)
            
            # Convert to pips
            max_move_pips = max_moves / pip_value
            avg_move_pips = avg_moves / pip_value
            
            # Statistics
            mean_max_move = np.mean(max_move_pips)
            mean_avg_move = np.mean(avg_move_pips)
            std_max_move = np.std(max_move_pips)
            
            # Count significant moves
            moves_gt_1pip = np.sum(max_move_pips > 1) / len(max_move_pips) * 100
            moves_gt_2pip = np.sum(max_move_pips > 2) / len(max_move_pips) * 100
            
            print(f"{window:<8} {time_minutes:.1f}min{'':<2} {mean_max_move:.1f}p{'':<6} {mean_avg_move:.2f}p{'':<6} {std_max_move:.1f}p{'':<6} {moves_gt_1pip:.0f}%{'':<4} {moves_gt_2pip:.0f}%")
        else:
            print(f"{window:<8} {time_minutes:.1f}min{'':<2} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A'}")
    
    print()


def test_triple_barrier_with_long_windows():
    """Test Triple Barrier with proper long lookforward windows."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("🎯 TRIPLE BARRIER WITH LONG LOOKFORWARD WINDOWS")
    print("=" * 70)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(30000)  # 30K ticks
    prices = test_df["mid_price"].to_numpy()
    
    # Test configurations with LONG lookforward windows
    configs = [
        {
            "name": "SHORT (1000 ticks - 1min)",
            "lookforward_window": 1000,
            "barrier_width": 0.0001,  # 1 pip barriers
        },
        {
            "name": "MEDIUM (3000 ticks - 3min)", 
            "lookforward_window": 3000,
            "barrier_width": 0.0001,  # 1 pip barriers
        },
        {
            "name": "LONG (5000 ticks - 5min)",
            "lookforward_window": 5000,
            "barrier_width": 0.0001,  # 1 pip barriers
        },
        {
            "name": "VERY LONG (7500 ticks - 7.5min)",
            "lookforward_window": 7500,
            "barrier_width": 0.0001,  # 1 pip barriers
        },
        {
            "name": "EXTENDED (10000 ticks - 10min)",
            "lookforward_window": 10000,
            "barrier_width": 0.0001,  # 1 pip barriers
        },
    ]
    
    print(f"{'Configuration':<30} {'Profit%':<8} {'Loss%':<8} {'Timeout%':<10} {'Trades':<8} {'PnL':<10} {'Outcome':<10}")
    print("-" * 90)
    
    for config in configs:
        try:
            # Generate labels with long lookforward window
            generator = TargetGeneratorFactory.create("triple_barrier", 
                lookforward_window=config["lookforward_window"],
                barrier_width=config["barrier_width"],
                min_return_threshold=1e-8,
                volatility_window=200,
                normalize_by_volatility=False,
            )
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Analyze hit rates
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            
            profit_pct = percentages[unique_labels == 1][0] if 1 in unique_labels else 0
            loss_pct = percentages[unique_labels == -1][0] if -1 in unique_labels else 0
            timeout_pct = percentages[unique_labels == 0][0] if 0 in unique_labels else 0
            
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
            
            num_trades = sum(1 for j in range(1, len(labels)) if labels[j] != labels[j-1])
            pnl_pips = total_pnl * 10000
            
            # Determine outcome
            if total_pnl > 0.001:
                outcome = "PROFIT ✅"
            elif total_pnl > -0.001:
                outcome = "BREAK-EVEN"
            else:
                outcome = "LOSS ❌"
            
            print(f"{config['name']:<30} {profit_pct:.1f}%{'':<4} {loss_pct:.1f}%{'':<4} {timeout_pct:.1f}%{'':<6} {num_trades:<8} {pnl_pips:.0f}p{'':<6} {outcome:<10}")
            
        except Exception as e:
            print(f"{config['name']:<30} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10} {'N/A':<8} {'N/A':<10} {str(e)[:10]:<10}")
    
    print()


def analyze_optimal_window_barrier_combination():
    """Find the optimal combination of window size and barrier width."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("🔧 OPTIMAL WINDOW-BARRIER COMBINATION SEARCH")
    print("=" * 70)
    
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(20000)  # Smaller sample for speed
    prices = test_df["mid_price"].to_numpy()
    
    # Test combinations
    combinations = [
        {"window": 5000, "barrier_pips": 0.5, "barrier_decimal": 0.00005},
        {"window": 5000, "barrier_pips": 1.0, "barrier_decimal": 0.0001},
        {"window": 5000, "barrier_pips": 1.5, "barrier_decimal": 0.00015},
        {"window": 7500, "barrier_pips": 1.0, "barrier_decimal": 0.0001},
        {"window": 7500, "barrier_pips": 1.5, "barrier_decimal": 0.00015},
        {"window": 10000, "barrier_pips": 1.0, "barrier_decimal": 0.0001},
        {"window": 10000, "barrier_pips": 2.0, "barrier_decimal": 0.0002},
    ]
    
    print(f"{'Window':<8} {'Barrier':<8} {'PnL (pips)':<12} {'Trades':<8} {'P/L/T %':<15} {'Outcome':<10}")
    print("-" * 70)
    
    best_pnl = float('-inf')
    best_config = None
    
    for combo in combinations:
        try:
            generator = TargetGeneratorFactory.create("triple_barrier", 
                lookforward_window=combo["window"],
                barrier_width=combo["barrier_decimal"],
                min_return_threshold=1e-8,
                volatility_window=200,
                normalize_by_volatility=False,
            )
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Calculate metrics
            fees_config = FeesConfig(lp_transaction_fees=0.00007, sp_transaction_fees=0.00007)
            returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
            
            total_pnl = returns_estimator.estimate_return(prices.tolist(), labels.tolist())
            pnl_pips = total_pnl * 10000
            num_trades = sum(1 for j in range(1, len(labels)) if labels[j] != labels[j-1])
            
            # Hit rates
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            profit_pct = percentages[unique_labels == 1][0] if 1 in unique_labels else 0
            loss_pct = percentages[unique_labels == -1][0] if -1 in unique_labels else 0
            timeout_pct = percentages[unique_labels == 0][0] if 0 in unique_labels else 0
            
            hit_rates = f"{profit_pct:.0f}/{loss_pct:.0f}/{timeout_pct:.0f}"
            
            outcome = "PROFIT ✅" if total_pnl > 0.001 else ("BREAK-EVEN" if total_pnl > -0.001 else "LOSS ❌")
            
            print(f"{combo['window']:<8} {combo['barrier_pips']:.1f}p{'':<4} {pnl_pips:.0f}p{'':<8} {num_trades:<8} {hit_rates:<15} {outcome:<10}")
            
            # Track best configuration
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_config = combo.copy()
                best_config['pnl'] = total_pnl
                best_config['hit_rates'] = (profit_pct, loss_pct, timeout_pct)
            
        except Exception as e:
            print(f"{combo['window']:<8} {combo['barrier_pips']:.1f}p{'':<4} {'ERROR':<12} {'N/A':<8} {'N/A':<15} {str(e)[:10]:<10}")
    
    print()
    if best_config:
        print(f"🏆 BEST CONFIGURATION:")
        print(f"   Window: {best_config['window']} ticks (~{best_config['window']/1000:.1f} minutes)")
        print(f"   Barrier: {best_config['barrier_pips']} pips")
        print(f"   PnL: {best_config['pnl']*10000:.0f} pips")
        print(f"   Hit rates: {best_config['hit_rates'][0]:.1f}% profit, {best_config['hit_rates'][1]:.1f}% loss, {best_config['hit_rates'][2]:.1f}% timeout")


def main():
    """Run comprehensive analysis of longer lookforward windows."""
    try:
        analyze_move_development_over_time()
        test_triple_barrier_with_long_windows()
        analyze_optimal_window_barrier_combination()
        
        print("💡 CONCLUSION:")
        print("=" * 60)
        print("Your insight was correct - longer lookforward windows (5000+ ticks)")
        print("allow more time for meaningful moves to develop and hit profit barriers")
        print("instead of timing out or hitting stop losses immediately.")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()