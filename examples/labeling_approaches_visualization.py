#!/usr/bin/env python3
"""
Labeling Approaches Visualization

This script loads real market data from DBN files and visualizes all the different
labeling approaches available in the represent package, including both traditional
and academic tstrends-based methods.
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False

from represent import (
    ModularDatasetBuilder,
    QuantileClassificationGenerator,
    DirectionalMFEGenerator,
    PriceMovementGenerator,
    VolatilityGenerator,
)

# Try to import tstrends generators
try:
    from represent.target_generators.tstrends_labeling import (
        BinaryCTLGenerator,
        TernaryCTLGenerator,
        OracleBinaryTrendGenerator,
        OracleTernaryTrendGenerator,
        TSTRENDS_AVAILABLE
    )
except ImportError:
    TSTRENDS_AVAILABLE = False


def load_market_data_from_dbn(data_dir: str = "data", max_samples: int = 5000) -> pl.DataFrame:
    """Load market data from DBN files."""
    data_path = Path(data_dir)
    
    # Find DBN files
    dbn_files = list(data_path.glob("*.dbn.zst"))
    
    if not dbn_files:
        print(f"⚠️  No DBN files found in {data_path}")
        return create_synthetic_market_data(max_samples)
    
    print(f"📂 Found {len(dbn_files)} DBN files in {data_path}")
    
    if not DATABENTO_AVAILABLE:
        print("⚠️  databento not available, using synthetic data")
        return create_synthetic_market_data(max_samples)
    
    # Load data from first DBN file
    dbn_file = dbn_files[0]
    print(f"📊 Loading data from: {dbn_file.name}")
    
    try:
        # Read DBN file
        client = db.Historical()
        data = client.read(dbn_file)
        
        # Convert to DataFrame
        df = data.to_df()
        
        if len(df) == 0:
            print("⚠️  Empty DBN file, using synthetic data")
            return create_synthetic_market_data(max_samples)
        
        # Calculate mid prices
        df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2
        df['volume'] = df.get('size', np.random.exponential(1000, len(df)))
        
        # Filter valid prices and limit samples
        df = df[df['mid_price'] > 0].head(max_samples)
        
        # Create polars DataFrame
        market_data = pl.DataFrame({
            'timestamp': df['ts_event'].values,
            'mid_price': df['mid_price'].values,
            'volume': df['volume'].values,
        })
        
        print(f"✅ Loaded {len(market_data)} samples from {dbn_file.name}")
        return market_data
        
    except Exception as e:
        print(f"❌ Error loading DBN file: {e}")
        print("🔄 Using synthetic data instead")
        return create_synthetic_market_data(max_samples)


def create_synthetic_market_data(n_samples: int = 5000) -> pl.DataFrame:
    """Create realistic synthetic market data for demonstration."""
    print(f"🎲 Creating synthetic market data with {n_samples} samples")
    
    np.random.seed(42)  # For reproducible results
    
    # Create realistic price series with trends and noise
    base_price = 0.6500  # AUDUSD-like price
    
    # Generate trending components
    trend_periods = [500, 800, 1200]  # Different trend periods
    trends = np.zeros(n_samples)
    
    for i, period in enumerate(trend_periods):
        start_idx = i * (n_samples // len(trend_periods))
        end_idx = min(start_idx + period, n_samples)
        
        # Alternate between up and down trends
        trend_direction = 1 if i % 2 == 0 else -1
        trend_strength = 0.0005 * trend_direction
        
        trends[start_idx:end_idx] = trend_strength
    
    # Add noise and random walk
    noise = np.random.normal(0, 0.0001, n_samples)
    random_walk = np.random.normal(0, 0.0002, n_samples)
    
    # Combine components
    price_changes = trends + noise + random_walk
    prices = base_price + np.cumsum(price_changes)
    
    # Ensure prices stay positive
    prices = np.maximum(prices, 0.5000)
    
    # Create timestamps
    timestamps = np.arange(n_samples) * 1000  # Millisecond timestamps
    
    # Create volume data
    volume = np.random.exponential(1000, n_samples)
    
    return pl.DataFrame({
        'timestamp': timestamps,
        'mid_price': prices,
        'volume': volume,
    })


def apply_all_labeling_approaches(market_data: pl.DataFrame) -> Dict[str, np.ndarray]:
    """Apply all available labeling approaches to the market data."""
    print("\n🎯 Applying all labeling approaches...")
    
    generators = []
    
    # Traditional represent generators
    generators.extend([
        QuantileClassificationGenerator(nbins=13, target_name="quantile_13class"),
        QuantileClassificationGenerator(nbins=5, target_name="quantile_5class"),
        DirectionalMFEGenerator(
            lookforward_horizon=200, 
            target_names=("mfe_buy", "mfe_sell")
        ),
        PriceMovementGenerator(
            lookforward_window=100,
            target_name="price_movement"
        ),
        VolatilityGenerator(
            window_size=50,
            target_name="volatility"
        ),
    ])
    
    # TStrends generators (if available) - simplified for demo
    if TSTRENDS_AVAILABLE:
        print("   📚 Including TStrends academic approaches")
        try:
            generators.extend([
                BinaryCTLGenerator(omega=0.02, target_name="binary_ctl"),
                TernaryCTLGenerator(
                    marginal_change_thres=0.02,
                    window_size=10,
                    target_name="ternary_ctl"
                ),
            ])
            print("   ✅ Added Binary and Ternary CTL generators")
        except Exception as e:
            print(f"   ⚠️  TStrends generators failed: {e}")
    else:
        print("   ⚠️  TStrends not available - install with: uv add git+https://github.com/agpenas/tstrends.git")
    
    # Build dataset with all generators
    builder = ModularDatasetBuilder(generators, verbose=False)
    
    try:
        dataset = builder.build_dataset(market_data)
        print(f"   ✅ Generated {len(dataset.columns) - 3} target columns")
        
        # Extract all target arrays
        targets = {}
        for col in dataset.columns:
            if col not in ['timestamp', 'mid_price', 'volume']:
                targets[col] = dataset[col].to_numpy()
        
        return targets
        
    except Exception as e:
        print(f"   ❌ Error applying labeling approaches: {e}")
        return {}


def create_comprehensive_visualization(
    market_data: pl.DataFrame, 
    targets: Dict[str, np.ndarray],
    output_dir: str = "examples"
) -> List[str]:
    """Create comprehensive visualization of all labeling approaches."""
    print("\n📊 Creating comprehensive visualization...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    prices = market_data["mid_price"].to_numpy()
    timestamps = np.arange(len(prices))
    
    saved_files = []
    
    # 1. Classification Approaches Comparison
    classification_targets = {
        name: labels for name, labels in targets.items() 
        if any(keyword in name.lower() for keyword in ['class', 'ctl', 'oracle'])
    }
    
    if classification_targets:
        fig_path = create_classification_comparison_plot(
            prices, timestamps, classification_targets, output_path
        )
        saved_files.append(fig_path)
    
    # 2. Regression Approaches Comparison  
    regression_targets = {
        name: values for name, values in targets.items()
        if any(keyword in name.lower() for keyword in ['mfe', 'movement', 'volatility'])
    }
    
    if regression_targets:
        fig_path = create_regression_comparison_plot(
            prices, timestamps, regression_targets, output_path
        )
        saved_files.append(fig_path)
    
    # 3. Academic vs Traditional Comparison
    if TSTRENDS_AVAILABLE:
        fig_path = create_academic_vs_traditional_plot(
            prices, timestamps, targets, output_path
        )
        saved_files.append(fig_path)
    
    # 4. Complete Overview
    fig_path = create_complete_overview_plot(
        prices, timestamps, targets, output_path
    )
    saved_files.append(fig_path)
    
    return saved_files


def create_classification_comparison_plot(
    prices: np.ndarray,
    timestamps: np.ndarray, 
    classification_targets: Dict[str, np.ndarray],
    output_path: Path
) -> str:
    """Create comparison plot for classification approaches."""
    
    n_approaches = len(classification_targets)
    fig, axes = plt.subplots(n_approaches + 1, 1, figsize=(15, 3 * (n_approaches + 1)))
    
    if n_approaches == 0:
        return ""
    
    # Plot price series at top
    axes[0].plot(timestamps, prices, 'k-', linewidth=1, alpha=0.8)
    axes[0].set_title('Market Price Series', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Color maps for different approaches
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    # Plot each classification approach
    for i, (name, labels) in enumerate(classification_targets.items()):
        ax = axes[i + 1]
        
        # Handle NaN values
        valid_mask = ~np.isnan(labels)
        valid_timestamps = timestamps[valid_mask]
        valid_labels = labels[valid_mask]
        
        if len(valid_labels) == 0:
            ax.text(0.5, 0.5, 'No valid labels', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name.replace("_", " ").title()} - No Data')
            continue
        
        # Create scatter plot with colors for each class
        unique_labels = np.unique(valid_labels)
        for j, label in enumerate(unique_labels):
            mask = valid_labels == label
            if np.any(mask):
                color_idx = int(label) % len(colors)
                ax.scatter(
                    valid_timestamps[mask], 
                    valid_labels[mask],
                    c=[colors[color_idx]], 
                    s=1, 
                    alpha=0.6,
                    label=f'Class {int(label)}'
                )
        
        ax.set_title(f'{name.replace("_", " ").title()}', fontsize=12)
        ax.set_ylabel('Label')
        ax.grid(True, alpha=0.3)
        
        # Add legend if not too many classes
        if len(unique_labels) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    axes[-1].set_xlabel('Time Steps')
    plt.tight_layout()
    
    # Save plot
    fig_path = output_path / "classification_approaches_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Saved classification comparison: {fig_path.name}")
    return str(fig_path)


def create_regression_comparison_plot(
    prices: np.ndarray,
    timestamps: np.ndarray,
    regression_targets: Dict[str, np.ndarray], 
    output_path: Path
) -> str:
    """Create comparison plot for regression approaches."""
    
    n_approaches = len(regression_targets)
    fig, axes = plt.subplots(n_approaches + 1, 1, figsize=(15, 3 * (n_approaches + 1)))
    
    if n_approaches == 0:
        return ""
    
    # Ensure axes is always a list
    if n_approaches == 0:
        axes = [axes]
    
    # Plot price series at top
    axes[0].plot(timestamps, prices, 'k-', linewidth=1, alpha=0.8)
    axes[0].set_title('Market Price Series', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Plot each regression approach
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (name, values) in enumerate(regression_targets.items()):
        ax = axes[i + 1]
        
        # Handle NaN values
        valid_mask = ~np.isnan(values)
        valid_timestamps = timestamps[valid_mask]
        valid_values = values[valid_mask]
        
        if len(valid_values) == 0:
            ax.text(0.5, 0.5, 'No valid values', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name.replace("_", " ").title()} - No Data')
            continue
        
        color = colors[i % len(colors)]
        ax.plot(valid_timestamps, valid_values, color=color, linewidth=1, alpha=0.7)
        
        # Add zero line for reference
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{name.replace("_", " ").title()}', fontsize=12)
        ax.set_ylabel('Value (BPS)' if 'mfe' in name.lower() or 'movement' in name.lower() else 'Value')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        ax.text(0.02, 0.98, f'μ={mean_val:.2f}, σ={std_val:.2f}', 
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Time Steps')
    plt.tight_layout()
    
    # Save plot
    fig_path = output_path / "regression_approaches_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Saved regression comparison: {fig_path.name}")
    return str(fig_path)


def create_academic_vs_traditional_plot(
    prices: np.ndarray,
    timestamps: np.ndarray,
    targets: Dict[str, np.ndarray],
    output_path: Path
) -> str:
    """Create comparison between academic and traditional approaches."""
    
    # Separate academic (tstrends) vs traditional approaches
    academic_targets = {
        name: labels for name, labels in targets.items()
        if any(keyword in name.lower() for keyword in ['ctl', 'oracle'])
    }
    
    traditional_targets = {
        name: labels for name, labels in targets.items()
        if any(keyword in name.lower() for keyword in ['quantile', 'mfe'])
    }
    
    if not academic_targets or not traditional_targets:
        return ""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Price series (top left)
    axes[0, 0].plot(timestamps, prices, 'k-', linewidth=1, alpha=0.8)
    axes[0, 0].set_title('Market Price Series', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Traditional approaches (top right)
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(traditional_targets)))
    
    for i, (name, values) in enumerate(traditional_targets.items()):
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            if 'quantile' in name.lower():
                # Classification - use scatter
                ax.scatter(timestamps[valid_mask], values[valid_mask], 
                          c=[colors[i]], s=0.5, alpha=0.6, label=name.replace('_', ' '))
            else:
                # Regression - use line
                ax.plot(timestamps[valid_mask], values[valid_mask], 
                       color=colors[i], linewidth=1, alpha=0.7, label=name.replace('_', ' '))
    
    ax.set_title('Traditional Represent Approaches', fontsize=14, fontweight='bold')
    ax.set_ylabel('Values')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Academic approaches (bottom left)
    ax = axes[1, 0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(academic_targets)))
    
    for i, (name, values) in enumerate(academic_targets.items()):
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax.scatter(timestamps[valid_mask], values[valid_mask],
                      c=[colors[i]], s=0.5, alpha=0.6, label=name.replace('_', ' '))
    
    ax.set_title('Academic TStrends Approaches', fontsize=14, fontweight='bold')
    ax.set_ylabel('Label Classes')
    ax.set_xlabel('Time Steps')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Comparison statistics (bottom right)
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create comparison text
    comparison_text = "📊 Approach Comparison\n\n"
    comparison_text += "Traditional Represent:\n"
    for name in traditional_targets.keys():
        comparison_text += f"  • {name.replace('_', ' ').title()}\n"
    
    comparison_text += "\nAcademic TStrends:\n"
    for name in academic_targets.keys():
        comparison_text += f"  • {name.replace('_', ' ').title()}\n"
    
    comparison_text += "\n🎯 Benefits:\n"
    comparison_text += "• Traditional: High performance, practical\n"
    comparison_text += "• Academic: Research-backed, optimal benchmarks\n"
    comparison_text += "• Combined: Best of both worlds!"
    
    ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes, 
            fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    fig_path = output_path / "academic_vs_traditional_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Saved academic vs traditional: {fig_path.name}")
    return str(fig_path)


def create_complete_overview_plot(
    prices: np.ndarray,
    timestamps: np.ndarray,
    targets: Dict[str, np.ndarray],
    output_path: Path
) -> str:
    """Create complete overview plot of all approaches."""
    
    # Create a comprehensive overview
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Main price plot (top row, spans all columns)
    ax_price = fig.add_subplot(gs[0, :])
    ax_price.plot(timestamps, prices, 'k-', linewidth=2, alpha=0.8)
    ax_price.set_title('Market Price Series with All Labeling Approaches', 
                      fontsize=16, fontweight='bold')
    ax_price.set_ylabel('Price', fontsize=12)
    ax_price.grid(True, alpha=0.3)
    
    # Separate targets by type
    classification_targets = {}
    regression_targets = {}
    
    for name, values in targets.items():
        if any(keyword in name.lower() for keyword in ['class', 'ctl', 'oracle']) and 'mfe' not in name.lower():
            classification_targets[name] = values
        else:
            regression_targets[name] = values
    
    # Classification approaches (second row)
    if classification_targets:
        for i, (name, labels) in enumerate(list(classification_targets.items())[:3]):
            ax = fig.add_subplot(gs[1, i])
            
            valid_mask = ~np.isnan(labels)
            if np.any(valid_mask):
                unique_labels = np.unique(labels[valid_mask])
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                
                for j, label in enumerate(unique_labels):
                    mask = (labels == label) & valid_mask
                    if np.any(mask):
                        ax.scatter(timestamps[mask], labels[mask], 
                                 c=[colors[j]], s=1, alpha=0.6, label=f'Class {int(label)}')
            
            ax.set_title(name.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Labels')
            ax.grid(True, alpha=0.3)
            if len(unique_labels) <= 5:
                ax.legend(fontsize=8)
    
    # Regression approaches (third row)
    if regression_targets:
        colors = ['blue', 'red', 'green']
        for i, (name, values) in enumerate(list(regression_targets.items())[:3]):
            ax = fig.add_subplot(gs[2, i])
            
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                ax.plot(timestamps[valid_mask], values[valid_mask], 
                       color=colors[i % len(colors)], linewidth=1, alpha=0.7)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_title(name.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Values')
            ax.grid(True, alpha=0.3)
    
    # Summary statistics (bottom row)
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    # Create summary table
    summary_text = "📊 LABELING APPROACHES SUMMARY\n\n"
    
    summary_text += f"🎯 Total Approaches Applied: {len(targets)}\n"
    summary_text += f"📈 Classification Methods: {len(classification_targets)}\n"
    summary_text += f"📊 Regression Methods: {len(regression_targets)}\n"
    summary_text += f"📋 Data Points: {len(prices):,}\n\n"
    
    summary_text += "🔧 Available Approaches:\n"
    for name in targets.keys():
        approach_type = "Classification" if name in classification_targets else "Regression"
        library = "TStrends" if any(kw in name.lower() for kw in ['ctl', 'oracle']) else "Represent"
        summary_text += f"  • {name.replace('_', ' ').title()} ({approach_type}, {library})\n"
    
    if TSTRENDS_AVAILABLE:
        summary_text += "\n✅ TStrends academic approaches available"
    else:
        summary_text += "\n⚠️  TStrends not installed - install with: uv add git+https://github.com/agpenas/tstrends.git"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, va='top', ha='left', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Save plot
    fig_path = output_path / "complete_labeling_overview.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Saved complete overview: {fig_path.name}")
    return str(fig_path)


def main():
    """Main execution function."""
    print("🚀 LABELING APPROACHES VISUALIZATION")
    print("=" * 60)
    print("This script demonstrates all available labeling approaches")
    print("in the represent package using real market data.")
    print()
    
    try:
        # Load market data
        market_data = load_market_data_from_dbn()
        
        if len(market_data) < 100:
            print("❌ Insufficient data for meaningful visualization")
            return
        
        # Apply all labeling approaches
        targets = apply_all_labeling_approaches(market_data)
        
        if not targets:
            print("❌ No targets generated")
            return
        
        # Create visualizations
        saved_files = create_comprehensive_visualization(market_data, targets)
        
        print(f"\n🎉 VISUALIZATION COMPLETE!")
        print(f"📊 Generated {len(targets)} different target types")
        print(f"📈 Created {len(saved_files)} visualization plots")
        print("\n📁 Saved files:")
        for file_path in saved_files:
            print(f"   • {Path(file_path).name}")
        
        print(f"\n💡 All plots saved to: examples/")
        print("🔗 These plots will be added to the README for documentation")
        
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()