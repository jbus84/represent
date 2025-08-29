#!/usr/bin/env python3
"""
Simple Labeling Approaches Demo

A simplified version that focuses on the core represent generators
and creates clean visualizations for the README.
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent import (
    ModularDatasetBuilder,
    QuantileClassificationGenerator,
    DirectionalMFEGenerator,
    PriceMovementGenerator,
    VolatilityGenerator,
)


def create_realistic_market_data(n_samples: int = 2000) -> pl.DataFrame:
    """Create realistic market data with clear trends."""
    np.random.seed(42)
    
    # Create AUDUSD-like price series with clear patterns
    base_price = 0.6500
    
    # Create different market regimes
    regime_length = n_samples // 4
    
    # Regime 1: Uptrend
    uptrend = np.linspace(0, 0.02, regime_length)
    uptrend += np.random.normal(0, 0.0005, regime_length)
    
    # Regime 2: Sideways with volatility
    sideways = np.random.normal(0, 0.001, regime_length)
    
    # Regime 3: Downtrend
    downtrend = np.linspace(0, -0.015, regime_length)
    downtrend += np.random.normal(0, 0.0005, regime_length)
    
    # Regime 4: Recovery
    recovery = np.linspace(0, 0.01, n_samples - 3 * regime_length)
    recovery += np.random.normal(0, 0.0003, len(recovery))
    
    # Combine regimes
    price_changes = np.concatenate([uptrend, sideways, downtrend, recovery])
    prices = base_price + np.cumsum(price_changes)
    
    # Ensure positive prices
    prices = np.maximum(prices, 0.5000)
    
    return pl.DataFrame({
        'timestamp': np.arange(n_samples) * 1000,
        'mid_price': prices,
        'volume': np.random.exponential(1000, n_samples),
    })


def create_labeling_showcase():
    """Create a comprehensive showcase of labeling approaches."""
    print("🚀 Creating Labeling Approaches Showcase")
    print("=" * 50)
    
    # Create market data
    market_data = create_realistic_market_data(2000)
    prices = market_data["mid_price"].to_numpy()
    timestamps = np.arange(len(prices))
    
    print(f"📊 Generated {len(prices)} price points")
    
    # Create generators
    generators = [
        QuantileClassificationGenerator(nbins=5, target_name="quantile_5class"),
        QuantileClassificationGenerator(nbins=13, target_name="quantile_13class"),
        DirectionalMFEGenerator(
            lookforward_horizon=100,
            target_names=("mfe_buy", "mfe_sell")
        ),
        PriceMovementGenerator(
            lookforward_window=50,
            target_name="price_movement"
        ),
        VolatilityGenerator(
            window_size=30,
            target_name="volatility"
        ),
    ]
    
    # Build dataset
    builder = ModularDatasetBuilder(generators, verbose=False)
    dataset = builder.build_dataset(market_data)
    
    print(f"✅ Generated {len(dataset.columns) - 3} target types")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    
    # 1. Price series (top, full width)
    ax_price = fig.add_subplot(gs[0, :])
    ax_price.plot(timestamps, prices, 'k-', linewidth=1.5, alpha=0.8)
    ax_price.set_title('Market Price Series (AUDUSD-like)', fontsize=14, fontweight='bold')
    ax_price.set_ylabel('Price')
    ax_price.grid(True, alpha=0.3)
    
    # Add regime annotations
    regime_length = len(prices) // 4
    regimes = ['Uptrend', 'Sideways', 'Downtrend', 'Recovery']
    colors = ['green', 'orange', 'red', 'blue']
    
    for i, (regime, color) in enumerate(zip(regimes, colors)):
        start = i * regime_length
        end = start + regime_length if i < 3 else len(prices)
        ax_price.axvspan(start, end, alpha=0.1, color=color, label=regime)
    
    ax_price.legend(loc='upper left', fontsize=10)
    
    # 2. Classification approaches
    # 5-class quantile
    ax = fig.add_subplot(gs[1, 0])
    labels_5 = dataset["quantile_5class"].to_numpy()
    colors_5 = plt.cm.Set1(np.linspace(0, 1, 5))
    
    for i in range(5):
        mask = labels_5 == i
        if np.any(mask):
            ax.scatter(timestamps[mask], labels_5[mask], c=[colors_5[i]], 
                      s=2, alpha=0.7, label=f'Class {i}')
    
    ax.set_title('5-Class Quantile Classification', fontsize=12, fontweight='bold')
    ax.set_ylabel('Class Label')
    ax.legend(fontsize=8, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.05))
    ax.grid(True, alpha=0.3)
    
    # 13-class quantile
    ax = fig.add_subplot(gs[1, 1])
    labels_13 = dataset["quantile_13class"].to_numpy()
    colors_13 = plt.cm.tab20(np.linspace(0, 1, 13))
    
    for i in range(13):
        mask = labels_13 == i
        if np.any(mask):
            ax.scatter(timestamps[mask], labels_13[mask], c=[colors_13[i]], 
                      s=1, alpha=0.6)
    
    ax.set_title('13-Class Quantile Classification', fontsize=12, fontweight='bold')
    ax.set_ylabel('Class Label')
    ax.grid(True, alpha=0.3)
    
    # 3. MFE regression targets
    ax = fig.add_subplot(gs[2, 0])
    mfe_buy = dataset["mfe_buy"].to_numpy()
    mfe_sell = dataset["mfe_sell"].to_numpy()
    
    valid_buy = ~np.isnan(mfe_buy)
    valid_sell = ~np.isnan(mfe_sell)
    
    ax.plot(timestamps[valid_buy], mfe_buy[valid_buy], 'g-', linewidth=1, alpha=0.7, label='MFE Buy')
    ax.plot(timestamps[valid_sell], mfe_sell[valid_sell], 'r-', linewidth=1, alpha=0.7, label='MFE Sell')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title('Directional MFE Targets', fontsize=12, fontweight='bold')
    ax.set_ylabel('MFE (Basis Points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Price movement
    ax = fig.add_subplot(gs[2, 1])
    price_movement = dataset["price_movement"].to_numpy()
    valid_movement = ~np.isnan(price_movement)
    
    ax.plot(timestamps[valid_movement], price_movement[valid_movement], 'b-', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title('Price Movement Targets', fontsize=12, fontweight='bold')
    ax.set_ylabel('Movement (Basis Points)')
    ax.grid(True, alpha=0.3)
    
    # 5. Volatility
    ax = fig.add_subplot(gs[3, 0])
    volatility = dataset["volatility"].to_numpy()
    valid_vol = ~np.isnan(volatility)
    
    ax.plot(timestamps[valid_vol], volatility[valid_vol], 'purple', linewidth=1, alpha=0.7)
    
    ax.set_title('Rolling Volatility Targets', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volatility (Basis Points)')
    ax.set_xlabel('Time Steps')
    ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax = fig.add_subplot(gs[3, 1])
    ax.axis('off')
    
    # Calculate statistics
    stats_text = "📊 TARGET STATISTICS\n\n"
    stats_text += f"5-Class Distribution:\n"
    for i in range(5):
        count = np.sum(labels_5 == i)
        pct = count / len(labels_5) * 100
        stats_text += f"  Class {i}: {count:,} ({pct:.1f}%)\n"
    
    stats_text += f"\nMFE Statistics:\n"
    stats_text += f"  Buy Mean: {np.nanmean(mfe_buy):.1f} BPS\n"
    stats_text += f"  Sell Mean: {np.nanmean(mfe_sell):.1f} BPS\n"
    
    stats_text += f"\nVolatility:\n"
    stats_text += f"  Mean: {np.nanmean(volatility):.1f} BPS\n"
    stats_text += f"  Max: {np.nanmax(volatility):.1f} BPS\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=9, va='top', ha='left', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Represent Package: Modular Target Generation Showcase', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    output_path = Path("labeling_approaches_showcase.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Saved showcase plot: {output_path}")
    
    # Create a simpler comparison plot
    create_simple_comparison_plot(market_data, dataset)
    
    return str(output_path)


def create_simple_comparison_plot(market_data, dataset):
    """Create a simple comparison plot for README."""
    
    prices = market_data["mid_price"].to_numpy()
    timestamps = np.arange(len(prices))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Price with classification overlay
    ax = axes[0]
    ax.plot(timestamps, prices, 'k-', linewidth=1, alpha=0.8, label='Price')
    
    # Overlay 5-class classification as background colors
    labels_5 = dataset["quantile_5class"].to_numpy()
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    
    for i in range(5):
        mask = labels_5 == i
        if np.any(mask):
            ax.scatter(timestamps[mask], prices[mask], c=colors[i], 
                      s=3, alpha=0.6, label=f'Class {i}')
    
    ax.set_title('Price Series with 5-Class Quantile Classification', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. MFE targets
    ax = axes[1]
    mfe_buy = dataset["mfe_buy"].to_numpy()
    mfe_sell = dataset["mfe_sell"].to_numpy()
    
    valid_buy = ~np.isnan(mfe_buy)
    valid_sell = ~np.isnan(mfe_sell)
    
    ax.plot(timestamps[valid_buy], mfe_buy[valid_buy], 'g-', linewidth=1.5, alpha=0.8, label='MFE Buy (Long)')
    ax.plot(timestamps[valid_sell], mfe_sell[valid_sell], 'r-', linewidth=1.5, alpha=0.8, label='MFE Sell (Short)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(timestamps[valid_buy], 0, mfe_buy[valid_buy], alpha=0.2, color='green')
    ax.fill_between(timestamps[valid_sell], 0, mfe_sell[valid_sell], alpha=0.2, color='red')
    
    ax.set_title('Directional MFE (Maximum Favorable Excursion) Targets', fontsize=14, fontweight='bold')
    ax.set_ylabel('MFE (Basis Points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Volatility
    ax = axes[2]
    volatility = dataset["volatility"].to_numpy()
    valid_vol = ~np.isnan(volatility)
    
    ax.plot(timestamps[valid_vol], volatility[valid_vol], 'purple', linewidth=1.5, alpha=0.8)
    ax.fill_between(timestamps[valid_vol], 0, volatility[valid_vol], alpha=0.3, color='purple')
    
    ax.set_title('Rolling Volatility Target', fontsize=14, fontweight='bold')
    ax.set_ylabel('Volatility (Basis Points)')
    ax.set_xlabel('Time Steps')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("target_generation_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Saved comparison plot: {output_path}")
    return str(output_path)


def main():
    """Main execution function."""
    print("🎯 SIMPLE LABELING DEMO")
    print("=" * 40)
    
    try:
        showcase_path = create_labeling_showcase()
        
        print(f"\n🎉 DEMO COMPLETE!")
        print(f"📊 Created visualization plots for README")
        print(f"📁 Files saved in examples/")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()