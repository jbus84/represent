#!/usr/bin/env python3
"""
Simple Label Time Series Plots

Create time series plots showing how optimization methods assign labels.
No GUI display to avoid timeout issues.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.patches as mpatches
from pathlib import Path

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


def create_binary_ctl_plot():
    """Create Binary CTL time series plot."""
    if not LIBRARIES_AVAILABLE:
        return
    
    print("📊 Creating Binary CTL visualization...")
    
    # Load data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Small sample for speed
    sample_size = 1000
    test_df = df.slice(15000, sample_size)
    prices = test_df["mid_price"].to_numpy()
    
    # Generate Binary CTL labels with optimized omega=0.0
    generator = TargetGeneratorFactory.create("binary_ctl", omega=0.0)
    targets_df = generator.generate_targets(test_df)
    target_info = generator.get_target_info()
    target_col = target_info['target_names'][0]
    labels = targets_df[target_col].to_numpy()
    
    # Convert {0,1} to {-1,1} for display
    labels_display = np.where(labels == 0, -1, 1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    time_axis = np.arange(len(prices))
    
    # Plot price
    ax.plot(time_axis, prices, 'k-', linewidth=1.5, alpha=0.8, label='Price', zorder=2)
    
    # Color background by label
    current_label = labels_display[0]
    start_idx = 0
    
    for j in range(1, len(labels_display) + 1):
        if j == len(labels_display) or labels_display[j] != current_label:
            # End of current region
            end_idx = j - 1 if j < len(labels_display) else j - 1
            
            if current_label == 1:
                ax.axvspan(start_idx, end_idx, alpha=0.3, color='lightgreen', zorder=1)
            else:  # -1
                ax.axvspan(start_idx, end_idx, alpha=0.3, color='lightcoral', zorder=1)
            
            if j < len(labels_display):
                start_idx = j
                current_label = labels_display[j]
    
    # Calculate statistics
    unique_labels, counts = np.unique(labels_display, return_counts=True)
    percentages = counts / len(labels_display) * 100
    num_changes = np.sum(labels_display[1:] != labels_display[:-1])
    
    # Formatting
    ax.set_title('Binary CTL Labels (ω=0.0) - Optimized Parameter', fontweight='bold', fontsize=14)
    ax.set_xlabel('Time (ticks)', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=1.5, label='Price'),
        mpatches.Patch(color='lightgreen', alpha=0.3, label=f'Long ({percentages[unique_labels==1][0]:.1f}%)'),
        mpatches.Patch(color='lightcoral', alpha=0.3, label=f'Short ({percentages[unique_labels==-1][0]:.1f}%)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Statistics
    stats = f"Label Changes: {num_changes} | Avg Hold Period: {len(labels_display)//max(num_changes,1)} ticks"
    ax.text(0.98, 0.02, stats, transform=ax.transAxes, ha='right', va='bottom',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9), fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = "examples/binary_ctl_labels_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Binary CTL plot: {output_path}")
    plt.close()
    
    return {
        "num_changes": num_changes,
        "label_distribution": dict(zip(unique_labels, percentages)),
        "sample_size": len(labels_display)
    }


def create_returns_analysis():
    """Create returns analysis plot."""
    if not LIBRARIES_AVAILABLE:
        return
    
    print("📈 Creating returns analysis...")
    
    # Load data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Sample data
    sample_size = 1000
    test_df = df.slice(25000, sample_size)
    prices = test_df["mid_price"].to_numpy()
    
    # Generate labels
    generator = TargetGeneratorFactory.create("binary_ctl", omega=0.0)
    targets_df = generator.generate_targets(test_df)
    target_info = generator.get_target_info()
    target_col = target_info['target_names'][0]
    labels = targets_df[target_col].to_numpy()
    
    # Convert for returns calculation: {0,1} → {-1,1}
    labels_returns = np.where(labels == 0, -1, 1)
    
    # Calculate tick-by-tick returns
    price_changes = np.diff(prices)
    tick_returns = price_changes * labels_returns[1:]  # Skip first label
    cumulative_returns = np.cumsum(tick_returns)
    
    # Create dual plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    time_axis = np.arange(len(prices))
    
    # Top: Price with label regions
    ax1.plot(time_axis, prices, 'k-', linewidth=1.5, label='Price')
    
    # Color regions
    for j in range(len(labels)):
        color = 'lightgreen' if labels[j] == 1 else 'lightcoral'
        if j < len(labels) - 1:
            ax1.axvspan(j, j+1, alpha=0.3, color=color)
    
    ax1.set_title('Binary CTL: Price with Position Labels', fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bottom: Cumulative returns
    ax2.plot(time_axis[1:], cumulative_returns, 'b-', linewidth=2, label='Cumulative Returns')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Break-even')
    ax2.set_title('Strategy Cumulative Returns (Before Fees)')
    ax2.set_xlabel('Time (ticks)')
    ax2.set_ylabel('Cumulative Return')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Statistics
    final_return = cumulative_returns[-1]
    num_trades = np.sum(labels_returns[1:] != labels_returns[:-1])
    
    stats_text = f'Final Return: {final_return:.6f}\nTrades: {num_trades}\nReturn/Trade: {final_return/max(num_trades,1):.6f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    
    # Save
    output_path = "examples/binary_ctl_returns_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved returns analysis: {output_path}")
    plt.close()
    
    return {
        "final_return": final_return,
        "num_trades": num_trades,
        "return_per_trade": final_return / max(num_trades, 1)
    }


def main():
    """Create label visualization plots."""
    try:
        print("🎯 CREATING LABEL TIME SERIES VISUALIZATIONS")
        print("=" * 60)
        
        # Create plots
        binary_stats = create_binary_ctl_plot()
        returns_stats = create_returns_analysis()
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print("=" * 40)
        if binary_stats:
            print(f"Binary CTL Statistics:")
            print(f"  Sample size: {binary_stats['sample_size']} ticks")
            print(f"  Label changes: {binary_stats['num_changes']}")
            print(f"  Label distribution: {binary_stats['label_distribution']}")
        
        if returns_stats:
            print(f"\nReturns Analysis:")
            print(f"  Final return: {returns_stats['final_return']:.6f}")
            print(f"  Number of trades: {returns_stats['num_trades']}")
            print(f"  Return per trade: {returns_stats['return_per_trade']:.6f}")
        
        print(f"\n✅ VISUALIZATION COMPLETE")
        print("Check examples/ folder for:")
        print("  - binary_ctl_labels_timeseries.png")  
        print("  - binary_ctl_returns_analysis.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()