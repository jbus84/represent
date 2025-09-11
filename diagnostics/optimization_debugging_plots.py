#!/usr/bin/env python3
"""
Optimization Debugging Visualization

Create plots showing exactly what the optimization sees during evaluation:
- Full sample windows used in optimization (100k samples × 5 windows)
- Barrier bounds for triple barrier methods
- Label distributions across different windows
- This is for debugging optimization behavior
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
        # Single window if not enough data
        return [df]
    
    # Create multiple windows as done in optimization
    windows = []
    step_size = max(1, (total_samples - window_size) // (n_windows - 1))
    
    for i in range(n_windows):
        start_idx = min(i * step_size, total_samples - window_size)
        end_idx = start_idx + window_size
        windows.append(df.slice(start_idx, window_size))
        
        print(f"Window {i+1}: samples {start_idx} to {end_idx-1} ({window_size:,} samples)")
    
    return windows


def create_optimization_window_plots():
    """Create plots showing optimization sampling windows with labels."""
    if not LIBRARIES_AVAILABLE:
        return
    
    print("🔍 CREATING OPTIMIZATION DEBUGGING PLOTS")
    print("=" * 60)
    
    # Load data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Use ACTUAL optimization sampling size for debugging  
    viz_window_size = 100000  # Full optimization window size
    windows = simulate_optimization_sampling(df, viz_window_size, n_windows=5)  # Full 5 windows
    
    # Methods to debug
    methods = [
        {
            "name": "Binary CTL",
            "method": "binary_ctl", 
            "params": {"omega": 0.0},
            "has_barriers": False
        },
        {
            "name": "Triple Barrier (Short)",
            "method": "triple_barrier",
            "params": {"lookforward_window": 1000, "barrier_width": 0.0001, "normalize_by_volatility": False},
            "has_barriers": True,
            "barrier_width": 0.0001
        },
        {
            "name": "Triple Barrier (Long)",
            "method": "triple_barrier", 
            "params": {"lookforward_window": 5000, "barrier_width": 0.0001, "normalize_by_volatility": False},
            "has_barriers": True,
            "barrier_width": 0.0001
        }
    ]
    
    for method_config in methods:
        create_method_debugging_plot(method_config, windows)


def create_method_debugging_plot(method_config: dict, windows: list):
    """Create debugging plot for a specific method across windows."""
    print(f"\n📊 Creating plot for {method_config['name']}...")
    
    fig, axes = plt.subplots(len(windows), 1, figsize=(15, 4 * len(windows)))
    if len(windows) == 1:
        axes = [axes]
    
    fig.suptitle(f"Optimization Debugging: {method_config['name']}", fontsize=16, fontweight='bold')
    
    for i, window_df in enumerate(windows):
        ax = axes[i]
        
        try:
            # Generate labels for this window
            generator = TargetGeneratorFactory.create(
                method_config["method"],
                **method_config["params"]
            )
            targets_df = generator.generate_targets(window_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Convert labels for display first
            if method_config["method"] == "binary_ctl":
                # Convert {0,1} to {-1,1} for display
                labels_display = np.where(labels == 0, -1, 1)
                colors = {-1: 'lightcoral', 1: 'lightgreen'}
                names = {-1: 'Short', 1: 'Long'}
            else:  # triple_barrier
                labels_display = labels
                colors = {-1: 'lightcoral', 0: 'lightgray', 1: 'lightgreen'}  
                names = {-1: 'Loss', 0: 'Timeout', 1: 'Profit'}
            
            # Get prices (full dataset for statistics, subsampled for plotting)
            prices_full = window_df["mid_price"].to_numpy()
            labels_full = labels.copy()
            
            # Subsample for plotting (every 100th point for 100K samples = 1K plot points)
            step = max(1, len(prices_full) // 1000)  # Target ~1000 plot points
            prices = prices_full[::step]
            labels_display_plot = labels_display[::step] if len(labels_display) >= len(prices_full) else labels_display
            time_axis = np.arange(0, len(prices_full), step)[:len(prices)]
            
            # Plot price line
            ax.plot(time_axis, prices, 'k-', linewidth=1, alpha=0.8, label='Price', zorder=3)
            
            # Add barrier bounds if applicable
            if method_config.get("has_barriers", False):
                barrier_width = method_config["barrier_width"]
                
                # Calculate barrier bounds for each point
                upper_barrier = prices + barrier_width
                lower_barrier = prices - barrier_width
                
                # Plot barrier zones as shaded areas
                ax.fill_between(time_axis, lower_barrier, upper_barrier, 
                               alpha=0.15, color='orange', label=f'Barrier Zone (±{barrier_width*10000:.0f} pips)', zorder=1)
                
                # Plot barrier lines
                ax.plot(time_axis, upper_barrier, '--', color='orange', alpha=0.7, linewidth=0.8, zorder=2)
                ax.plot(time_axis, lower_barrier, '--', color='orange', alpha=0.7, linewidth=0.8, zorder=2)
            
            # Color regions by label (colors and names already defined above)
            
            # Create colored background regions
            if len(np.unique(labels_display)) > 1:  # Only if we have multiple labels
                current_label = labels_display[0]
                start_idx = 0
                
                for j in range(1, len(labels_display) + 1):
                    if j == len(labels_display) or (j < len(labels_display) and labels_display[j] != current_label):
                        end_idx = j - 1
                        
                        if current_label in colors:
                            ax.axvspan(start_idx, end_idx, alpha=0.25, color=colors[current_label], zorder=0)
                        
                        if j < len(labels_display):
                            start_idx = j
                            current_label = labels_display[j]
            
            # Statistics (from FULL dataset, not subsampled)
            unique_labels, counts = np.unique(labels_display, return_counts=True)
            percentages = counts / len(labels_display) * 100
            num_changes = np.sum(labels_display[1:] != labels_display[:-1])
            
            # For plotting, use subsampled labels
            labels_display_plot = labels_display[::step] if len(labels_display) >= len(prices_full) else labels_display
            
            # Title and labels
            window_start = i * (len(prices_full) // len(windows)) if len(windows) > 1 else 0
            ax.set_title(f"Window {i+1} (samples {window_start:,} to {window_start + len(prices_full):,}) - Full 100K Dataset", fontweight='bold')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            
            # Legend
            legend_elements = [plt.Line2D([0], [0], color='black', linewidth=1, label='Price')]
            
            if method_config.get("has_barriers", False):
                legend_elements.append(mpatches.Patch(color='orange', alpha=0.15, 
                                                    label=f'Barrier Zone (±{barrier_width*10000:.0f} pips)'))
            
            for label_val, pct in zip(unique_labels, percentages):
                if label_val in colors:
                    name = names.get(label_val, str(label_val))
                    legend_elements.append(
                        mpatches.Patch(color=colors[label_val], alpha=0.25, label=f'{name} ({pct:.1f}%)')
                    )
            
            ax.legend(handles=legend_elements, loc='upper left')
            
            # Statistics box
            stats = f"Changes: {num_changes} | Labels: {len(unique_labels)} classes"
            if method_config.get("has_barriers", False):
                profit_pct = percentages[unique_labels == 1][0] if 1 in unique_labels else 0
                loss_pct = percentages[unique_labels == -1][0] if -1 in unique_labels else 0
                timeout_pct = percentages[unique_labels == 0][0] if 0 in unique_labels else 0
                stats += f"\nP/L/T: {profit_pct:.1f}%/{loss_pct:.1f}%/{timeout_pct:.1f}%"
            
            ax.text(0.98, 0.02, stats, transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9), fontsize=9)
            
            print(f"  Window {i+1}: {dict(zip(unique_labels, percentages))}")
            
        except Exception as e:
            print(f"❌ Error with window {i+1}: {e}")
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes, color='red')
            ax.set_title(f"Window {i+1}: ERROR", color='red')
    
    axes[-1].set_xlabel('Time (ticks)')
    plt.tight_layout()
    
    # Save plot
    method_name_clean = method_config['name'].replace(' ', '_').replace('(', '').replace(')', '').lower()
    output_path = f"examples/debug_{method_name_clean}_optimization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_barrier_analysis_plot():
    """Create detailed barrier analysis for triple barrier methods."""
    if not LIBRARIES_AVAILABLE:
        return
    
    print(f"\n🎯 CREATING BARRIER ANALYSIS PLOT")
    print("=" * 50)
    
    # Load data  
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Take focused sample for barrier analysis
    sample_size = 2000
    test_df = df.slice(30000, sample_size)
    prices = test_df["mid_price"].to_numpy()
    
    # Compare different barrier widths
    barrier_configs = [
        {"width": 0.00005, "name": "0.5 pips", "color": "lightblue"},
        {"width": 0.0001, "name": "1.0 pips", "color": "lightgreen"},
        {"width": 0.0002, "name": "2.0 pips", "color": "lightyellow"}
    ]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    time_axis = np.arange(len(prices))
    
    # Plot price
    ax.plot(time_axis, prices, 'k-', linewidth=2, label='Price', zorder=5)
    
    # Plot multiple barrier zones
    for config in barrier_configs:
        barrier_width = config["width"]
        upper_barrier = prices + barrier_width
        lower_barrier = prices - barrier_width
        
        ax.fill_between(time_axis, lower_barrier, upper_barrier,
                       alpha=0.15, color=config["color"], 
                       label=f'±{config["name"]} Barrier', zorder=1)
    
    # Add price statistics
    price_volatility = np.std(np.diff(prices))
    mean_move = np.mean(np.abs(np.diff(prices)))
    
    ax.set_title('Triple Barrier Analysis: Barrier Widths vs Price Movement', fontweight='bold', fontsize=14)
    ax.set_xlabel('Time (ticks)')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Statistics
    stats = f'Price Volatility: {price_volatility:.6f}\nMean Move: {mean_move:.6f}\nTransaction Cost: 0.00007 (0.7 pips)'
    ax.text(0.98, 0.98, stats, transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    
    output_path = "examples/debug_barrier_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved barrier analysis: {output_path}")
    plt.close()


def main():
    """Create all debugging visualization plots."""
    try:
        create_optimization_window_plots()
        create_barrier_analysis_plot()
        
        print(f"\n🎯 DEBUGGING PLOTS COMPLETE")
        print("=" * 60)
        print("Created plots showing:")
        print("✅ Optimization sampling windows with labels")
        print("✅ Barrier zones for triple barrier methods") 
        print("✅ Label distributions across different windows")
        print("✅ Statistical analysis for debugging")
        print("\nUse these plots to:")
        print("• Debug unexpected optimization results")
        print("• Verify barrier sizes vs price movements")
        print("• Check label consistency across windows")
        print("• Understand parameter sensitivity")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()