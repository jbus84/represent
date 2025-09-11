#!/usr/bin/env python3
"""
Optimization Label Time Series Visualization

Create time series plots showing how different labeling methods 
assign labels over time during optimization, overlaid on price data.
This helps visualize what the optimization is actually learning.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


def create_label_timeseries_plots():
    """Create time series plots showing labels overlaid on price data."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("📊 CREATING LABEL TIME SERIES VISUALIZATIONS")
    print("=" * 60)
    
    # Load sample data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Take a representative window for visualization
    sample_size = 5000  # 5K ticks for clear visualization
    start_idx = 10000   # Skip initial data
    test_df = df.slice(start_idx, sample_size)
    prices = test_df["mid_price"].to_numpy()
    
    # Methods to visualize with their optimized parameters
    methods = [
        {
            "name": "Binary CTL",
            "method": "binary_ctl",
            "params": {"omega": 0.0},  # Optimized: omega=0.0
            "color_map": {-1: "red", 1: "green"},
            "label_names": {-1: "Short", 1: "Long"}
        },
        {
            "name": "Ternary CTL", 
            "method": "ternary_ctl",
            "params": {"marginal_change_thres": 0.0446, "window_size": 501},  # Optimized
            "color_map": {0: "red", 1: "gray", 2: "green"},
            "label_names": {0: "Down", 1: "Neutral", 2: "Up"}
        },
        {
            "name": "Triple Barrier (Short Window)",
            "method": "triple_barrier", 
            "params": {"lookforward_window": 1000, "barrier_width": 0.0001, "normalize_by_volatility": False},
            "color_map": {-1: "red", 0: "gray", 1: "green"},
            "label_names": {-1: "Loss", 0: "Timeout", 1: "Profit"}
        },
        {
            "name": "Triple Barrier (Long Window)", 
            "method": "triple_barrier",
            "params": {"lookforward_window": 5000, "barrier_width": 0.0001, "normalize_by_volatility": False},
            "color_map": {-1: "red", 0: "gray", 1: "green"}, 
            "label_names": {-1: "Loss", 0: "Timeout", 1: "Profit"}
        }
    ]
    
    # Create subplots
    fig, axes = plt.subplots(len(methods), 1, figsize=(15, 4 * len(methods)))
    if len(methods) == 1:
        axes = [axes]
    
    fig.suptitle("Label Time Series: Optimized Parameters on Sample Data", fontsize=16, fontweight='bold')
    
    # Time axis (simplified)
    time_axis = np.arange(len(prices))
    
    for i, method_config in enumerate(methods):
        ax = axes[i]
        
        try:
            print(f"Generating labels for {method_config['name']}...")
            
            # Generate labels
            generator = TargetGeneratorFactory.create(
                method_config["method"],
                **method_config["params"]
            )
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Convert labels if needed (for TStrends compatibility)
            if method_config["method"] in ["ternary_ctl"] and set(np.unique(labels)).issubset({0, 1, 2}):
                # Keep as {0, 1, 2} for visualization
                pass
            elif method_config["method"] in ["binary_ctl"] and set(np.unique(labels)).issubset({0, 1}):
                # Convert {0, 1} to {-1, 1} for visualization
                labels = np.where(labels == 0, -1, 1)
            
            # Plot price line
            ax.plot(time_axis, prices, color='black', linewidth=0.8, alpha=0.7, label='Price')
            
            # Create colored background regions based on labels
            unique_labels = np.unique(labels)
            current_label = labels[0]
            start_idx = 0
            
            for j in range(1, len(labels)):
                if labels[j] != current_label or j == len(labels) - 1:
                    # End of current label region
                    end_idx = j if j < len(labels) - 1 else len(labels) - 1
                    
                    # Color background based on label
                    if current_label in method_config["color_map"]:
                        color = method_config["color_map"][current_label]
                        ax.axvspan(start_idx, end_idx, alpha=0.2, color=color)
                    
                    # Update for next region
                    start_idx = j
                    current_label = labels[j] if j < len(labels) else current_label
            
            # Formatting
            ax.set_title(f"{method_config['name']}: {method_config['params']}", fontweight='bold')
            ax.set_xlabel("Time (ticks)")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)
            
            # Create legend for labels
            legend_elements = [mpatches.Patch(color='black', alpha=0.7, label='Price')]
            for label_val, color in method_config["color_map"].items():
                if label_val in unique_labels:
                    label_name = method_config["label_names"].get(label_val, str(label_val))
                    count = np.sum(labels == label_val)
                    pct = count / len(labels) * 100
                    legend_elements.append(
                        mpatches.Patch(color=color, alpha=0.2, 
                                     label=f'{label_name} ({pct:.1f}%)')
                    )
            
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Add statistics text
            num_changes = np.sum(labels[1:] != labels[:-1])
            stats_text = f"Label changes: {num_changes} | Avg hold: {len(labels)//max(num_changes,1)} ticks"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            print(f"✅ {method_config['name']}: {len(unique_labels)} unique labels, {num_changes} changes")
            
        except Exception as e:
            print(f"❌ Error with {method_config['name']}: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f"{method_config['name']}: ERROR", fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save plot
    output_path = "examples/optimization_label_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: {output_path}")
    plt.show()


def create_comparison_with_returns():
    """Create a comparison plot showing labels with cumulative returns."""
    if not LIBRARIES_AVAILABLE:
        return
    
    print(f"\n📈 CREATING LABEL VS RETURNS COMPARISON")
    print("=" * 60)
    
    # Load data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    sample_size = 3000
    test_df = df.slice(15000, sample_size)  # Different window
    prices = test_df["mid_price"].to_numpy()
    
    # Focus on Binary CTL with optimized parameters
    method_config = {
        "name": "Binary CTL (Optimized)",
        "method": "binary_ctl", 
        "params": {"omega": 0.0},
    }
    
    try:
        # Generate labels
        generator = TargetGeneratorFactory.create(
            method_config["method"],
            **method_config["params"]
        )
        targets_df = generator.generate_targets(test_df)
        target_info = generator.get_target_info()
        target_col = target_info['target_names'][0]
        labels = targets_df[target_col].to_numpy()
        
        # Convert {0,1} to {-1,1} for returns calculation
        labels_returns = np.where(labels == 0, -1, 1)
        
        # Calculate cumulative returns
        price_changes = np.diff(prices)
        tick_returns = price_changes * labels_returns[1:]  # Skip first label
        cumulative_returns = np.cumsum(tick_returns)
        
        # Create dual-axis plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle("Binary CTL: Labels vs Cumulative Returns", fontsize=16, fontweight='bold')
        
        time_axis = np.arange(len(prices))
        
        # Top plot: Price with labels
        ax1.plot(time_axis, prices, color='black', linewidth=1, label='Price')
        
        # Color background based on labels
        current_label = labels[0]
        start_idx = 0
        
        for j in range(1, len(labels)):
            if labels[j] != current_label or j == len(labels) - 1:
                end_idx = j if j < len(labels) - 1 else len(labels) - 1
                
                color = 'lightgreen' if current_label == 1 else 'lightcoral'
                ax1.axvspan(start_idx, end_idx, alpha=0.3, color=color)
                
                start_idx = j
                current_label = labels[j] if j < len(labels) else current_label
        
        ax1.set_title("Price with Binary CTL Labels")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Bottom plot: Cumulative returns
        ax2.plot(time_axis[1:], cumulative_returns, color='blue', linewidth=2, label='Cumulative Returns')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title("Cumulative Strategy Returns")
        ax2.set_xlabel("Time (ticks)")
        ax2.set_ylabel("Cumulative Return")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add final return stat
        final_return = cumulative_returns[-1]
        final_return_pct = final_return / prices[0] * 100
        ax2.text(0.98, 0.95, f'Final Return: {final_return:.6f}\n({final_return_pct:.3f}%)', 
                transform=ax2.transAxes, ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        output_path = "examples/binary_ctl_returns_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {output_path}")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating returns comparison: {e}")


def main():
    """Create all label visualization plots."""
    try:
        create_label_timeseries_plots()
        create_comparison_with_returns()
        
        print(f"\n💡 VISUALIZATION SUMMARY")
        print("=" * 60)
        print("✅ Time series plots show how labels are assigned over time")
        print("✅ Background colors indicate label regions")
        print("✅ Statistics show label distribution and holding periods")
        print("✅ Returns analysis shows actual profitability mechanics")
        print()
        print("🎯 USE THESE PLOTS TO:")
        print("   - Verify labels make sense with price movements")
        print("   - Understand why certain parameters are optimal")
        print("   - Debug any unexpected optimization results")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()