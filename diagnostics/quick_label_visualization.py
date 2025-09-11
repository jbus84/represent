#!/usr/bin/env python3
"""
Quick Label Time Series Visualization

Fast visualization of how optimized labeling methods assign labels.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


def create_quick_plots():
    """Create quick time series plots for key methods."""
    if not LIBRARIES_AVAILABLE:
        return
    
    print("📊 CREATING QUICK LABEL TIME SERIES")
    print("=" * 50)
    
    # Load data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Smaller sample for speed
    sample_size = 2000
    test_df = df.slice(10000, sample_size)
    prices = test_df["mid_price"].to_numpy()
    
    # Focus on the most important methods
    methods = [
        {
            "name": "Binary CTL (ω=0.0)", 
            "method": "binary_ctl",
            "params": {"omega": 0.0}
        },
        {
            "name": "Ternary CTL (Optimized)",
            "method": "ternary_ctl", 
            "params": {"marginal_change_thres": 0.0446, "window_size": 501}
        }
    ]
    
    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 6 * len(methods)))
    if len(methods) == 1:
        axes = [axes]
    
    time_axis = np.arange(len(prices))
    
    for i, method_config in enumerate(methods):
        ax = axes[i]
        
        try:
            print(f"Processing {method_config['name']}...")
            
            # Generate labels
            generator = TargetGeneratorFactory.create(
                method_config["method"],
                **method_config["params"]
            )
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Plot price
            ax.plot(time_axis, prices, 'k-', linewidth=1.5, alpha=0.8, label='Price')
            
            # Color regions by label
            if method_config["method"] == "binary_ctl":
                # Convert {0,1} to {-1,1} for display
                labels_display = np.where(labels == 0, -1, 1)
                colors = {-1: 'lightcoral', 1: 'lightgreen'}
                names = {-1: 'Short', 1: 'Long'}
            else:  # ternary_ctl
                # Keep {0,1,2} format
                labels_display = labels
                colors = {0: 'lightcoral', 1: 'lightgray', 2: 'lightgreen'}
                names = {0: 'Down', 1: 'Neutral', 2: 'Up'}
            
            # Create colored background
            current_label = labels_display[0]
            start_idx = 0
            
            for j in range(1, len(labels_display)):
                if labels_display[j] != current_label:
                    # Color previous region
                    if current_label in colors:
                        ax.axvspan(start_idx, j-1, alpha=0.3, color=colors[current_label])
                    start_idx = j
                    current_label = labels_display[j]
            
            # Color final region
            if current_label in colors:
                ax.axvspan(start_idx, len(labels_display)-1, alpha=0.3, color=colors[current_label])
            
            # Statistics
            unique_labels, counts = np.unique(labels_display, return_counts=True)
            percentages = counts / len(labels_display) * 100
            num_changes = np.sum(labels_display[1:] != labels_display[:-1])
            
            # Formatting
            ax.set_title(f"{method_config['name']}", fontweight='bold', fontsize=14)
            ax.set_ylabel('Price', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Legend
            legend_elements = [plt.Line2D([0], [0], color='black', linewidth=1.5, label='Price')]
            for label_val, pct in zip(unique_labels, percentages):
                if label_val in colors:
                    name = names.get(label_val, str(label_val))
                    legend_elements.append(
                        mpatches.Patch(color=colors[label_val], alpha=0.3, label=f'{name} ({pct:.1f}%)')
                    )
            
            ax.legend(handles=legend_elements, loc='upper left')
            
            # Stats box
            stats = f"Changes: {num_changes} | Avg Hold: {len(labels_display)//max(num_changes,1)} ticks"
            ax.text(0.98, 0.02, stats, transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            
            print(f"✅ {method_config['name']}: {dict(zip(unique_labels, percentages))}")
            
        except Exception as e:
            print(f"❌ Error with {method_config['name']}: {e}")
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
    
    axes[-1].set_xlabel('Time (ticks)', fontsize=12)
    plt.tight_layout()
    
    # Save
    output_path = "examples/quick_label_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.show()


def create_returns_overlay():
    """Create a plot showing cumulative returns overlaid on labels."""
    if not LIBRARIES_AVAILABLE:
        return
        
    print(f"\n📈 BINARY CTL RETURNS VISUALIZATION")
    print("=" * 50)
    
    # Data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    test_df = df.slice(20000, 1500)  # Small sample
    prices = test_df["mid_price"].to_numpy()
    
    try:
        # Generate Binary CTL labels
        generator = TargetGeneratorFactory.create("binary_ctl", omega=0.0)
        targets_df = generator.generate_targets(test_df)
        target_info = generator.get_target_info()
        target_col = target_info['target_names'][0]
        labels = targets_df[target_col].to_numpy()
        
        # Convert for returns: {0,1} → {-1,1}
        labels_returns = np.where(labels == 0, -1, 1)
        
        # Calculate returns
        price_changes = np.diff(prices)
        tick_returns = price_changes * labels_returns[1:]
        cumulative_returns = np.cumsum(tick_returns)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        time_axis = np.arange(len(prices))
        
        # Top: Price + Labels
        ax1.plot(time_axis, prices, 'k-', linewidth=1.5, label='Price')
        
        # Color background
        for j in range(len(labels)):
            color = 'lightgreen' if labels[j] == 1 else 'lightcoral'
            if j < len(labels) - 1:
                ax1.axvspan(j, j+1, alpha=0.3, color=color)
        
        ax1.set_title('Binary CTL Labels (ω=0.0)', fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Bottom: Cumulative Returns
        ax2.plot(time_axis[1:], cumulative_returns, 'b-', linewidth=2, label='Cumulative Returns')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Strategy Cumulative Returns')
        ax2.set_xlabel('Time (ticks)')
        ax2.set_ylabel('Cumulative Return')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Final return
        final_return = cumulative_returns[-1]
        ax2.text(0.02, 0.98, f'Final: {final_return:.6f}', transform=ax2.transAxes, 
                va='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        output_path = "examples/binary_ctl_returns.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.show()
        
    except Exception as e:
        print(f"❌ Returns visualization error: {e}")


if __name__ == "__main__":
    try:
        create_quick_plots()
        create_returns_overlay()
        print(f"\n✅ VISUALIZATION COMPLETE")
        print("Check examples/ folder for generated plots")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()