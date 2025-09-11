#!/usr/bin/env python3
"""
Full Time Series Debugging Plots

Create comprehensive time series plots showing exactly how labels are assigned
over the full 100K optimization windows. This reveals the true temporal patterns
that statistics alone can't capture.
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


def simulate_optimization_sampling(df: pl.DataFrame, window_size: int = 100000, n_windows: int = 5):
    """Simulate the exact sampling strategy used in optimization."""
    total_samples = len(df)
    
    if total_samples < window_size:
        return [df]
    
    # Create multiple windows as done in optimization
    windows = []
    step_size = max(1, (total_samples - window_size) // (n_windows - 1))
    
    for i in range(n_windows):
        start_idx = min(i * step_size, total_samples - window_size)
        end_idx = start_idx + window_size
        windows.append(df.slice(start_idx, window_size))
        
        print(f"Window {i+1}: samples {start_idx:,} to {end_idx-1:,}")
    
    return windows


def create_full_timeseries_comparison():
    """Create full time series plots comparing all methods across optimization windows."""
    if not LIBRARIES_AVAILABLE:
        return
    
    print("📊 CREATING FULL TIME SERIES DEBUGGING PLOTS")
    print("=" * 70)
    
    # Load data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Use ACTUAL optimization sampling
    viz_window_size = 100000  # Full optimization window size
    windows = simulate_optimization_sampling(df, viz_window_size, n_windows=3)  # Use 3 windows for clarity
    
    # Methods to analyze
    methods = [
        {
            "name": "Binary CTL (ω=0.0)",
            "method": "binary_ctl", 
            "params": {"omega": 0.0},
            "color_map": {-1: "red", 1: "green"},
            "label_names": {-1: "Short", 1: "Long"}
        },
        {
            "name": "Triple Barrier (1K ticks)",
            "method": "triple_barrier",
            "params": {"lookforward_window": 1000, "barrier_width": 0.0001, "normalize_by_volatility": False},
            "color_map": {-1: "red", 0: "gray", 1: "green"},
            "label_names": {-1: "Loss", 0: "Timeout", 1: "Profit"}
        },
        {
            "name": "Triple Barrier (5K ticks)", 
            "method": "triple_barrier",
            "params": {"lookforward_window": 5000, "barrier_width": 0.0001, "normalize_by_volatility": False},
            "color_map": {-1: "red", 0: "gray", 1: "green"},
            "label_names": {-1: "Loss", 0: "Timeout", 1: "Profit"}
        }
    ]
    
    # Create plots for each window
    for window_idx, window_df in enumerate(windows):
        create_window_timeseries_plot(window_df, methods, window_idx + 1)


def create_window_timeseries_plot(window_df: pl.DataFrame, methods: list, window_num: int):
    """Create time series plot for a specific window showing all methods."""
    print(f"\n🔍 Creating time series for Window {window_num}...")
    
    # Get prices
    prices = window_df["mid_price"].to_numpy()
    
    # Subsample for plotting (target ~2000 points for clarity)
    step = max(1, len(prices) // 2000)
    prices_plot = prices[::step]
    time_axis = np.arange(0, len(prices), step)[:len(prices_plot)]
    
    # Create subplots for each method
    fig, axes = plt.subplots(len(methods), 1, figsize=(20, 5 * len(methods)))
    if len(methods) == 1:
        axes = [axes]
    
    fig.suptitle(f"Full Time Series Analysis - Window {window_num} (100K samples)", fontsize=16, fontweight='bold')
    
    for i, method_config in enumerate(methods):
        ax = axes[i]
        
        try:
            print(f"  Processing {method_config['name']}...")
            
            # Generate labels
            generator = TargetGeneratorFactory.create(
                method_config["method"],
                **method_config["params"]
            )
            targets_df = generator.generate_targets(window_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Convert labels for display
            if method_config["method"] == "binary_ctl":
                # Convert {0,1} to {-1,1} for display
                labels_display = np.where(labels == 0, -1, 1)
            else:  # triple_barrier
                labels_display = labels
            
            # Subsample labels for plotting
            labels_plot = labels_display[::step] if len(labels_display) >= len(prices) else labels_display
            labels_plot = labels_plot[:len(prices_plot)]  # Ensure same length
            
            # Plot price line
            ax.plot(time_axis, prices_plot, 'black', linewidth=0.8, alpha=0.7, label='Price', zorder=3)
            
            # Add barrier bounds if applicable
            if method_config.get("method") == "triple_barrier":
                barrier_width = method_config["params"]["barrier_width"]
                
                # Calculate barrier bounds
                upper_barrier = prices_plot + barrier_width
                lower_barrier = prices_plot - barrier_width
                
                # Plot barrier zones
                ax.fill_between(time_axis, lower_barrier, upper_barrier, 
                               alpha=0.1, color='orange', label=f'±{barrier_width*10000:.0f} pip barriers', zorder=1)
            
            # Color background regions by label - FULL TEMPORAL ANALYSIS
            current_label = labels_plot[0] if len(labels_plot) > 0 else 0
            start_idx = 0
            
            for j in range(1, len(labels_plot) + 1):
                if j == len(labels_plot) or (j < len(labels_plot) and labels_plot[j] != current_label):
                    end_idx = j - 1
                    
                    # Color this region
                    if current_label in method_config["color_map"]:
                        color = method_config["color_map"][current_label]
                        ax.axvspan(time_axis[start_idx], time_axis[end_idx] if end_idx < len(time_axis) else time_axis[-1], 
                                  alpha=0.3, color=color, zorder=0)
                    
                    if j < len(labels_plot):
                        start_idx = j
                        current_label = labels_plot[j]
            
            # Calculate statistics
            unique_labels, counts = np.unique(labels_display, return_counts=True)
            percentages = counts / len(labels_display) * 100
            num_changes = np.sum(labels_display[1:] != labels_display[:-1])
            
            # Title and formatting
            ax.set_title(f"{method_config['name']} - Full 100K Time Series", fontweight='bold', fontsize=14)
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            
            # Legend
            legend_elements = [plt.Line2D([0], [0], color='black', linewidth=0.8, label='Price')]
            
            if method_config.get("method") == "triple_barrier":
                barrier_width = method_config["params"]["barrier_width"]
                legend_elements.append(mpatches.Patch(color='orange', alpha=0.1, 
                                                    label=f'±{barrier_width*10000:.0f} pip barriers'))
            
            for label_val, pct in zip(unique_labels, percentages):
                if label_val in method_config["color_map"]:
                    name = method_config["label_names"].get(label_val, str(label_val))
                    legend_elements.append(
                        mpatches.Patch(color=method_config["color_map"][label_val], alpha=0.3, 
                                     label=f'{name} ({pct:.1f}%)')
                    )
            
            ax.legend(handles=legend_elements, loc='upper left')
            
            # Statistics box
            avg_hold = len(labels_display) // max(num_changes, 1)
            stats_text = f'Label Changes: {num_changes:,}\nAvg Hold: {avg_hold:,} ticks\nSample: {len(labels_display):,} total'
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9), fontsize=10)
            
            # Print detailed analysis
            print(f"    {method_config['name']}:")
            for label_val, pct in zip(unique_labels, percentages):
                name = method_config["label_names"].get(label_val, str(label_val))
                print(f"      {name}: {pct:.1f}%")
            print(f"      Changes: {num_changes:,}, Avg Hold: {avg_hold:,} ticks")
            
        except Exception as e:
            print(f"❌ Error with {method_config['name']}: {e}")
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes, color='red')
            ax.set_title(f"{method_config['name']}: ERROR", color='red')
    
    axes[-1].set_xlabel('Time (ticks)', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    output_path = f"examples/full_timeseries_window_{window_num}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_comparative_analysis():
    """Create side-by-side comparison showing the key differences."""
    if not LIBRARIES_AVAILABLE:
        return
        
    print(f"\n📈 CREATING COMPARATIVE ANALYSIS")
    print("=" * 50)
    
    # Load data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Take one representative window
    test_df = df.slice(50000, 100000)  # Middle section
    prices = test_df["mid_price"].to_numpy()
    
    # Focus window for detailed analysis (5K samples)
    focus_start = 25000
    focus_end = 30000  
    focus_df = test_df.slice(focus_start, 5000)
    focus_prices = focus_df["mid_price"].to_numpy()
    focus_time = np.arange(focus_start, focus_end)
    
    # Generate labels for comparison
    methods_to_compare = [
        {
            "name": "Binary CTL",
            "method": "binary_ctl",
            "params": {"omega": 0.0}
        },
        {
            "name": "Triple Barrier (1K)",
            "method": "triple_barrier", 
            "params": {"lookforward_window": 1000, "barrier_width": 0.0001}
        }
    ]
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    for i, method_config in enumerate(methods_to_compare):
        ax = axes[i]
        
        try:
            # Generate labels
            generator = TargetGeneratorFactory.create(
                method_config["method"],
                **method_config["params"]
            )
            targets_df = generator.generate_targets(focus_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Plot price
            ax.plot(focus_time, focus_prices, 'k-', linewidth=1.5, label='Price', zorder=3)
            
            # Color by labels
            if method_config["method"] == "binary_ctl":
                labels_display = np.where(labels == 0, -1, 1)
                colors = {-1: 'lightcoral', 1: 'lightgreen'}
                names = {-1: 'Short', 1: 'Long'}
            else:
                labels_display = labels  
                colors = {-1: 'lightcoral', 0: 'lightgray', 1: 'lightgreen'}
                names = {-1: 'Loss', 0: 'Timeout', 1: 'Profit'}
            
            # Add colored background
            for j in range(len(labels_display)):
                if j < len(labels_display) - 1:
                    color = colors.get(labels_display[j], 'white')
                    ax.axvspan(focus_time[j], focus_time[j+1], alpha=0.3, color=color, zorder=1)
            
            # Add barriers for triple barrier
            if method_config["method"] == "triple_barrier":
                barrier_width = 0.0001
                upper_barrier = focus_prices + barrier_width
                lower_barrier = focus_prices - barrier_width
                ax.plot(focus_time, upper_barrier, '--', color='orange', alpha=0.8, linewidth=1)
                ax.plot(focus_time, lower_barrier, '--', color='orange', alpha=0.8, linewidth=1)
            
            ax.set_title(f"{method_config['name']} - Detailed 5K Sample Analysis", fontweight='bold')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            
            # Statistics
            unique, counts = np.unique(labels_display, return_counts=True)
            percentages = counts / len(labels_display) * 100
            changes = np.sum(labels_display[1:] != labels_display[:-1])
            
            stats = []
            for label_val, pct in zip(unique, percentages):
                name = names.get(label_val, str(label_val))
                stats.append(f"{name}: {pct:.1f}%")
            stats.append(f"Changes: {changes}")
            
            ax.text(0.02, 0.98, " | ".join(stats), transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
    
    axes[-1].set_xlabel('Time (ticks)')
    plt.tight_layout()
    
    output_path = "examples/comparative_timeseries_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparative analysis: {output_path}")
    plt.close()


def main():
    """Create all full time series debugging plots."""
    try:
        create_full_timeseries_comparison()
        create_comparative_analysis()
        
        print(f"\n🎯 FULL TIME SERIES DEBUGGING COMPLETE")
        print("=" * 70)
        print("Created comprehensive time series plots showing:")
        print("✅ Full 100K sample optimization windows")
        print("✅ Temporal label assignment patterns")
        print("✅ Barrier interaction visualization")
        print("✅ Method comparison analysis")
        print("\nUse these plots to understand:")
        print("• How labels change over time")
        print("• Whether methods respond to price movements")
        print("• Temporal clustering of label types")
        print("• True optimization behavior patterns")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()