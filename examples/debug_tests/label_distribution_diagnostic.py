#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Label Distribution Diagnostic

Analyze label distributions and PnL calculations to understand why returns are low.
Check if buy/sell signals are canceling each other out.
"""

import polars as pl
import numpy as np
from represent.target_generators.factory import TargetGeneratorFactory


def create_trending_data(n_samples=5000, trend_strength=0.00005):
    """Create trending price data to test directional strategies."""
    np.random.seed(42)
    base_price = 1.1000
    
    # Create upward trend with noise
    trend = np.cumsum(np.random.normal(trend_strength, 0.0001, n_samples))
    prices = base_price + trend
    
    return pl.DataFrame({
        "ts_event": range(n_samples),
        "mid_price": prices,
        "symbol": ["EURUSD"] * n_samples
    })


def analyze_labels_and_pnl(method_name, params, test_data, description=""):
    """Analyze label distribution and calculate PnL step by step."""
    print(f"\n{'='*60}")
    print(f"📊 ANALYZING: {method_name.upper()} {description}")
    print(f"{'='*60}")
    
    try:
        generator = TargetGeneratorFactory.create(method_name, **params)
        targets = generator.generate_targets(test_data)
        
        if targets is None or len(targets) == 0:
            print("❌ No targets generated")
            return None
            
        # Get first label column
        label_columns = [col for col in targets.columns if "label" in col.lower()]
        if not label_columns:
            print("❌ No label columns found")
            return None
            
        labels = targets[label_columns[0]].to_numpy()
        prices = test_data["mid_price"].to_numpy()
        
        print(f"📈 Price range: {prices.min():.6f} to {prices.max():.6f}")
        print(f"📈 Price trend: {prices[-1]/prices[0]-1:.4%} total return")
        
        # Analyze label distribution
        unique_labels = np.unique(labels)
        print(f"🎯 Label values: {unique_labels}")
        
        label_dist = {}
        for label in unique_labels:
            count = np.sum(labels == label)
            pct = count / len(labels) * 100
            label_dist[label] = (count, pct)
            print(f"   Label {label}: {count:,} samples ({pct:.1f}%)")
        
        # Calculate PnL step by step with detailed analysis
        pnl = 0.0
        position = 0  # -1 short, 0 flat, 1 long
        position_changes = 0
        total_fees = 0.0
        fee = 0.00007 / 2.0  # 0.7 pips total round-trip, divided by 2 for entry/exit
        
        long_returns = []
        short_returns = []
        position_history = []
        
        print(f"\n📊 PnL CALCULATION (fee={fee:.5f} per trade):")
        
        for t in range(1, len(prices)):
            ret = (prices[t] - prices[t-1]) / prices[t-1]
            old_position = position
            
            # Position change logic
            if labels[t] != position:
                # Exit cost
                if position != 0:
                    pnl -= fee
                    total_fees += fee
                # Entry cost
                if labels[t] != 0:
                    pnl -= fee
                    total_fees += fee
                
                position_changes += 1
                position = labels[t]
            
            # Accrue returns
            position_pnl = ret * position
            pnl += position_pnl
            
            # Track returns by position type
            if position == 1:  # Long
                long_returns.append(ret)
            elif position == -1:  # Short
                short_returns.append(-ret)  # Short profits from negative price moves
            
            position_history.append(position)
        
        # Calculate statistics
        gross_return = pnl + total_fees  # Returns before fees
        net_return = pnl  # Returns after fees
        
        print(f"   Position changes: {position_changes:,}")
        print(f"   Total fees paid: {total_fees:.6f} ({total_fees*10000:.1f} pips)")
        print(f"   Gross return: {gross_return:.6f} ({gross_return*10000:.1f} pips)")
        print(f"   Net return: {net_return:.6f} ({net_return*10000:.1f} pips)")
        print(f"   Return/trade: {net_return/max(position_changes,1):.6f}")
        
        # Analyze by position type
        if long_returns:
            long_avg = np.mean(long_returns)
            long_total = np.sum(long_returns)
            print(f"   Long positions: {len(long_returns)} periods, avg return: {long_avg:.6f}, total: {long_total:.6f}")
        
        if short_returns:
            short_avg = np.mean(short_returns)
            short_total = np.sum(short_returns)
            print(f"   Short positions: {len(short_returns)} periods, avg return: {short_avg:.6f}, total: {short_total:.6f}")
        
        # Position analysis
        position_dist = {}
        for pos in [-1, 0, 1]:
            count = np.sum(np.array(position_history) == pos)
            pct = count / len(position_history) * 100 if position_history else 0
            position_dist[pos] = (count, pct)
            pos_name = {-1: "Short", 0: "Flat", 1: "Long"}[pos]
            print(f"   {pos_name} periods: {count:,} ({pct:.1f}%)")
        
        # Analysis summary
        print(f"\n🔍 ANALYSIS:")
        if position_changes == 0:
            print("   ⚠️  No position changes - static strategy")
        elif total_fees > abs(gross_return):
            print("   💸 OVER-TRADING: Fees exceed gross returns")
        elif abs(net_return) < total_fees * 0.1:
            print("   ⚖️  BALANCED TRADING: Long/short returns likely canceling out")
        elif net_return > 0:
            print("   📈 PROFITABLE STRATEGY")
        else:
            print("   📉 UNPROFITABLE STRATEGY")
        
        return {
            'labels': labels,
            'label_dist': label_dist,
            'position_changes': position_changes,
            'total_fees': total_fees,
            'gross_return': gross_return,
            'net_return': net_return,
            'long_periods': len(long_returns),
            'short_periods': len(short_returns)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Run comprehensive label distribution analysis."""
    print("🔍 LABEL DISTRIBUTION & PnL DIAGNOSTIC")
    print("="*80)
    print("Analyzing if buy/sell signals are canceling each other out")
    
    # Create trending test data (should favor long positions)
    test_data = create_trending_data(3000, trend_strength=0.00005)
    
    print(f"\n📊 Test Data: {len(test_data)} samples")
    prices = test_data["mid_price"].to_numpy()
    total_return = prices[-1]/prices[0] - 1
    print(f"📈 Price movement: {prices[0]:.6f} → {prices[-1]:.6f} ({total_return:.4%})")
    
    # Test methods with optimal parameters
    methods_to_test = [
        ('binary_ctl', {'omega': 0.0}, "Academic binary trend"),
        ('binary_ctl', {'omega': 0.05}, "Academic binary (high omega)"),
        ('ternary_ctl', {'marginal_change_thres': 0.0446, 'window_size': 501}, "Academic ternary"),
        ('oracle_binary', {'transaction_cost': 0.0001}, "Oracle binary (perfect foresight)"),
        ('oracle_ternary', {'transaction_cost': 0.0001, 'neutral_reward_factor': 0.2}, "Oracle ternary"),
        ('triple_barrier', {'barrier_width': 0.0005, 'transaction_cost': 0.0001, 'lookforward_window': 200}, "Barrier method"),
    ]
    
    results = {}
    
    for method, params, desc in methods_to_test:
        result = analyze_labels_and_pnl(method, params, test_data, desc)
        if result:
            results[method] = result
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("📊 SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Method':<15} {'PosChg':<8} {'Fees':<10} {'GrossRet':<10} {'NetRet':<10} {'Diagnosis':<20}")
    print("-"*80)
    
    for method, result in results.items():
        pos_chg = result['position_changes']
        fees = result['total_fees'] * 10000  # in pips
        gross_ret = result['gross_return'] * 10000  # in pips
        net_ret = result['net_return'] * 10000  # in pips
        
        if pos_chg == 0:
            diagnosis = "No trades"
        elif fees > abs(gross_ret):
            diagnosis = "Over-trading"
        elif abs(net_ret) < fees * 0.1:
            diagnosis = "Balanced cancel"
        elif net_ret > 0:
            diagnosis = "Profitable"
        else:
            diagnosis = "Unprofitable"
        
        print(f"{method:<15} {pos_chg:<8} {fees:<10.1f} {gross_ret:<10.1f} {net_ret:<10.1f} {diagnosis:<20}")
    
    print(f"\n{'='*80}")
    print("🎯 DIAGNOSTIC CONCLUSIONS")
    print("="*80)
    
    balanced_methods = [m for m, r in results.items() if abs(r['net_return']) < r['total_fees'] * 0.1 and r['position_changes'] > 0]
    overtrading_methods = [m for m, r in results.items() if r['total_fees'] > abs(r['gross_return'])]
    
    if balanced_methods:
        print(f"⚖️  BALANCED TRADING (returns canceling): {balanced_methods}")
        print("   → These methods generate balanced long/short signals")
        print("   → Consider directional bias or market regime filtering")
    
    if overtrading_methods:
        print(f"💸 OVER-TRADING (fees > gross returns): {overtrading_methods}")
        print("   → These methods trade too frequently")
        print("   → Need higher signal quality thresholds")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("1. Test with longer lookforward windows for less frequent trading")
    print("2. Add directional bias based on market trend")
    print("3. Use regime-aware parameters (trending vs mean-reverting)")
    print("4. Consider ensemble methods combining multiple signals")


if __name__ == "__main__":
    main()