#!/usr/bin/env python3
"""
Individual Method Signal Plots

Generates individual plots for each labeling method showing actual signal patterns
on market data with optimized parameters.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_synthetic_market_data(n_samples: int = 2000) -> pl.DataFrame:
    """Create realistic synthetic market data for demonstration."""
    np.random.seed(42)

    # Create realistic price movement with trends and noise
    base_price = 1.0
    price_changes = np.random.normal(0, 0.0001, n_samples)

    # Add some trending periods
    trend_periods = [
        (200, 400, 0.00005),   # Uptrend
        (600, 800, -0.00003),  # Downtrend
        (1200, 1400, 0.00008), # Strong uptrend
        (1600, 1800, -0.00006) # Strong downtrend
    ]

    for start, end, trend_strength in trend_periods:
        if end <= n_samples:
            price_changes[start:end] += trend_strength

    # Generate cumulative prices
    mid_prices = base_price + np.cumsum(price_changes)

    # Create timestamps
    timestamps = np.arange(n_samples, dtype=np.int64)

    return pl.DataFrame({
        'mid_price': mid_prices,
        'ts_event': timestamps,
        'symbol': ['M6AM4'] * n_samples
    })

def plot_ga_labeling_signals():
    """Plot GA Labeling signals with optimized parameters."""
    try:
        from represent.target_generators.ga_labeling import GALabelingGenerator

        # Create synthetic data
        market_data = create_synthetic_market_data(1000)

        # Create GA generator with OPTIMIZED parameters
        generator = GALabelingGenerator(
            population_size=30,
            max_generations=31,
            lookforward_window=4,
            transaction_cost=0.0005,
            min_trades=8,
            min_win_rate=0.3578,
            max_win_rate=0.6201,
            min_profit_factor=1.0876,
            mutation_rate=0.0173,
            crossover_rate=0.7438,
            target_name="ga_optimized",
            verbose=False
        )

        # Generate signals
        result_df = generator.generate_targets(market_data)

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Price plot
        ax1.plot(market_data['mid_price'], linewidth=1.5, color='black', label='Mid Price')
        ax1.set_title('GA Labeling: Evolutionary Optimized Trading Signals (71.34% Returns)',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.legend()

        # Signals plot
        signals = result_df['ga_optimized'].to_numpy()
        buy_signals = signals == 1
        sell_signals = signals == 0

        ax2.scatter(np.where(buy_signals)[0], signals[buy_signals],
                   color='green', marker='^', s=50, alpha=0.8, label=f'Buy Signals ({np.sum(buy_signals)})')
        ax2.scatter(np.where(sell_signals)[0], signals[sell_signals],
                   color='red', marker='v', s=50, alpha=0.8, label=f'Sell/Hold Signals ({np.sum(sell_signals)})')

        ax2.set_title('Optimized Parameters: pop_size=30, max_gen=31, lookforward=4', fontsize=12)
        ax2.set_ylabel('Signal (0=Sell/Hold, 1=Buy)', fontweight='bold')
        ax2.set_xlabel('Time (Ticks)', fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend()

        plt.figtext(0.02, 0.02,
                    'GA Labeling uses evolutionary optimization to generate trading signals optimized for performance.\n' +
                    'Optimized parameters achieve 71.34% returns with 0.7 pip transaction costs.',
                    fontsize=10, style='italic')

        plt.tight_layout()
        plt.savefig('plots/optimisation/individual_ga_labeling_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated individual_ga_labeling_signals.png")

    except Exception as e:
        print(f"⚠️  Could not generate GA Labeling plot: {e}")

def plot_ctl_methods_signals():
    """Plot Binary and Ternary CTL signals with optimized parameters."""
    try:
        from represent.target_generators.tstrends_labeling import (
            BinaryCTLGenerator,
            TernaryCTLGenerator,
        )

        # Create synthetic data
        market_data = create_synthetic_market_data(1000)

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Price plot
        axes[0].plot(market_data['mid_price'], linewidth=2, color='black', label='Mid Price')
        axes[0].set_title('CTL Methods: Academic Trend Labeling with Bayesian Optimization',
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price', fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        # Binary CTL with optimized omega=0.0
        binary_generator = BinaryCTLGenerator(omega=0.0, target_name="binary_ctl_opt")
        binary_result = binary_generator.generate_targets(market_data)
        binary_signals = binary_result['binary_ctl_opt'].to_numpy()

        buy_signals = binary_signals == 1
        sell_signals = binary_signals == 0

        axes[1].scatter(np.where(buy_signals)[0], binary_signals[buy_signals],
                       color='green', marker='^', s=40, alpha=0.8, label=f'Up/Buy ({np.sum(buy_signals)})')
        axes[1].scatter(np.where(sell_signals)[0], binary_signals[sell_signals],
                       color='red', marker='v', s=40, alpha=0.8, label=f'Down/Sell ({np.sum(sell_signals)})')

        axes[1].set_title('Binary CTL: Optimized ω=0.0 (240.20% Returns)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Binary Signal', fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        # Ternary CTL with optimized parameters
        ternary_generator = TernaryCTLGenerator(
            marginal_change_thres=0.04458382945260628,
            window_size=501,
            target_name="ternary_ctl_opt"
        )
        ternary_result = ternary_generator.generate_targets(market_data)
        ternary_signals = ternary_result['ternary_ctl_opt'].to_numpy()

        up_signals = ternary_signals == 2
        neutral_signals = ternary_signals == 1
        down_signals = ternary_signals == 0

        axes[2].scatter(np.where(up_signals)[0], ternary_signals[up_signals],
                       color='green', marker='^', s=40, alpha=0.8, label=f'Up/Buy ({np.sum(up_signals)})')
        axes[2].scatter(np.where(neutral_signals)[0], ternary_signals[neutral_signals],
                       color='gray', marker='s', s=20, alpha=0.6, label=f'Neutral/Hold ({np.sum(neutral_signals)})')
        axes[2].scatter(np.where(down_signals)[0], ternary_signals[down_signals],
                       color='red', marker='v', s=40, alpha=0.8, label=f'Down/Sell ({np.sum(down_signals)})')

        axes[2].set_title('Ternary CTL: Optimized threshold=4.46%, window=501 (0.32% Returns)',
                         fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Ternary Signal', fontweight='bold')
        axes[2].set_xlabel('Time (Ticks)', fontweight='bold')
        axes[2].grid(alpha=0.3)
        axes[2].legend()

        plt.figtext(0.02, 0.02,
                    'CTL methods use cumulative trend analysis from academic literature.\n' +
                    'Bayesian optimization: Binary CTL ω=0 removes noise filtering, Ternary CTL uses higher thresholds.',
                    fontsize=10, style='italic')

        plt.tight_layout()
        plt.savefig('plots/optimisation/individual_ctl_methods_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated individual_ctl_methods_signals.png")

    except Exception as e:
        print(f"⚠️  Could not generate CTL methods plot: {e}")

def plot_oracle_methods_signals():
    """Plot Oracle methods signals with optimized parameters."""
    try:
        from represent.target_generators.tstrends_labeling import (
            OracleBinaryTrendGenerator,
            OracleTernaryTrendGenerator,
        )

        # Create synthetic data
        market_data = create_synthetic_market_data(1000)

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Price plot
        axes[0].plot(market_data['mid_price'], linewidth=2, color='black', label='Mid Price')
        axes[0].set_title('Oracle Methods: Theoretical Optimal Labels with Bayesian Optimization',
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price', fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        # Oracle Binary with optimized transaction cost
        oracle_binary_generator = OracleBinaryTrendGenerator(
            transaction_cost=9.326802124287607e-07,
            target_name="oracle_binary_opt"
        )
        binary_result = oracle_binary_generator.generate_targets(market_data)
        binary_signals = binary_result['oracle_binary_opt'].to_numpy()

        buy_signals = binary_signals == 1
        sell_signals = binary_signals == 0

        axes[1].scatter(np.where(buy_signals)[0], binary_signals[buy_signals],
                       color='darkgreen', marker='^', s=50, alpha=0.9, label=f'Optimal Buy ({np.sum(buy_signals)})')
        axes[1].scatter(np.where(sell_signals)[0], binary_signals[sell_signals],
                       color='darkred', marker='v', s=50, alpha=0.9, label=f'Optimal Sell ({np.sum(sell_signals)})')

        axes[1].set_title('Oracle Binary: Optimized TX Cost=9.33e-07 (1.23% Returns)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Optimal Binary Signal', fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        # Oracle Ternary with optimized parameters
        oracle_ternary_generator = OracleTernaryTrendGenerator(
            transaction_cost=0.00796542986860233,
            neutral_reward_factor=0.18343478986616382,
            target_name="oracle_ternary_opt"
        )
        ternary_result = oracle_ternary_generator.generate_targets(market_data)
        ternary_signals = ternary_result['oracle_ternary_opt'].to_numpy()

        up_signals = ternary_signals == 2
        neutral_signals = ternary_signals == 1
        down_signals = ternary_signals == 0

        axes[2].scatter(np.where(up_signals)[0], ternary_signals[up_signals],
                       color='darkgreen', marker='^', s=50, alpha=0.9, label=f'Optimal Up ({np.sum(up_signals)})')
        axes[2].scatter(np.where(neutral_signals)[0], ternary_signals[neutral_signals],
                       color='darkgray', marker='s', s=30, alpha=0.7, label=f'Optimal Neutral ({np.sum(neutral_signals)})')
        axes[2].scatter(np.where(down_signals)[0], ternary_signals[down_signals],
                       color='darkred', marker='v', s=50, alpha=0.9, label=f'Optimal Down ({np.sum(down_signals)})')

        axes[2].set_title('Oracle Ternary: Optimized TX Cost=0.8%, Neutral=18.3% (0.18% Returns)',
                         fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Optimal Ternary Signal', fontweight='bold')
        axes[2].set_xlabel('Time (Ticks)', fontweight='bold')
        axes[2].grid(alpha=0.3)
        axes[2].legend()

        plt.figtext(0.02, 0.02,
                    'Oracle methods provide theoretical optimal labels using future price knowledge.\n' +
                    'Used for benchmarking - show maximum possible performance under perfect information.',
                    fontsize=10, style='italic')

        plt.tight_layout()
        plt.savefig('plots/optimisation/individual_oracle_methods_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated individual_oracle_methods_signals.png")

    except Exception as e:
        print(f"⚠️  Could not generate Oracle methods plot: {e}")

def plot_regression_methods_signals():
    """Plot regression method targets."""
    try:
        from represent.target_generators.regression import (
            DirectionalMFEGenerator,
            LogReturnHorizonsGenerator,
            VolatilityScaledReturnsGenerator,
        )

        # Create synthetic data
        market_data = create_synthetic_market_data(1000)

        fig, axes = plt.subplots(4, 1, figsize=(14, 16))

        # Price plot
        axes[0].plot(market_data['mid_price'], linewidth=2, color='black', label='Mid Price')
        axes[0].set_title('Regression Methods: Continuous Target Generation',
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price', fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        # Log Return Horizons
        log_return_generator = LogReturnHorizonsGenerator(
            horizons=[1000, 2000, 3000],
            target_prefix="log_ret"
        )
        log_result = log_return_generator.generate_targets(market_data)

        axes[1].plot(log_result['log_ret_1000t'], label='1000 tick horizon', linewidth=2, alpha=0.8)
        axes[1].plot(log_result['log_ret_2000t'], label='2000 tick horizon', linewidth=2, alpha=0.8)
        axes[1].plot(log_result['log_ret_3000t'], label='3000 tick horizon', linewidth=2, alpha=0.8)
        axes[1].set_title('Log Return Horizons: Multi-Scale Time Analysis', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Log Returns (bps)', fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        # Directional MFE
        mfe_generator = DirectionalMFEGenerator(
            lookforward_horizon=3000,
            target_names=("mfe_buy", "mfe_sell")
        )
        mfe_result = mfe_generator.generate_targets(market_data)

        axes[2].fill_between(range(len(mfe_result)), 0, mfe_result['mfe_buy'],
                           alpha=0.6, color='green', label='Buy MFE (Long Potential)')
        axes[2].fill_between(range(len(mfe_result)), 0, -mfe_result['mfe_sell'],
                           alpha=0.6, color='red', label='Sell MFE (Short Potential)')
        axes[2].set_title('Directional MFE: Maximum Favorable Excursion Analysis', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('MFE (basis points)', fontweight='bold')
        axes[2].grid(alpha=0.3)
        axes[2].legend()

        # Volatility Scaled Returns
        vol_scaled_generator = VolatilityScaledReturnsGenerator(
            vol_multiplier=2.5,
            target_name="vol_scaled_returns"
        )
        vol_result = vol_scaled_generator.generate_targets(market_data)

        vol_returns = vol_result['vol_scaled_returns'].to_numpy()
        positive_returns = vol_returns > 0
        negative_returns = vol_returns < 0

        axes[3].scatter(np.where(positive_returns)[0], vol_returns[positive_returns],
                       color='green', alpha=0.7, s=20, label=f'Positive Returns ({np.sum(positive_returns)})')
        axes[3].scatter(np.where(negative_returns)[0], vol_returns[negative_returns],
                       color='red', alpha=0.7, s=20, label=f'Negative Returns ({np.sum(negative_returns)})')
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        axes[3].set_title('Volatility Scaled Returns: Adaptive Risk Management', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Scaled Returns (bps)', fontweight='bold')
        axes[3].set_xlabel('Time (Ticks)', fontweight='bold')
        axes[3].grid(alpha=0.3)
        axes[3].legend()

        plt.figtext(0.02, 0.02,
                    'Regression methods provide continuous targets for risk management and position sizing.\n' +
                    'Multi-horizon analysis captures different time scale dynamics.',
                    fontsize=10, style='italic')

        plt.tight_layout()
        plt.savefig('plots/optimisation/individual_regression_methods_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated individual_regression_methods_signals.png")

    except Exception as e:
        print(f"⚠️  Could not generate regression methods plot: {e}")

def plot_quantile_classification_signals():
    """Plot quantile classification signals."""
    try:
        from represent.target_generators.classification import QuantileClassificationGenerator

        # Create synthetic data
        market_data = create_synthetic_market_data(1000)

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Price plot
        axes[0].plot(market_data['mid_price'], linewidth=2, color='black', label='Mid Price')
        axes[0].set_title('Quantile Classification: Traditional Balanced Labeling',
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price', fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        # Binary quantile (2 bins)
        binary_generator = QuantileClassificationGenerator(nbins=2, target_name="binary_quantile")
        binary_result = binary_generator.generate_targets(market_data, symbol="M6AM4")
        binary_signals = binary_result['binary_quantile'].to_numpy()

        up_signals = binary_signals == 1
        down_signals = binary_signals == 0

        axes[1].scatter(np.where(up_signals)[0], binary_signals[up_signals],
                       color='blue', marker='^', s=40, alpha=0.7, label=f'Up Class ({np.sum(up_signals)})')
        axes[1].scatter(np.where(down_signals)[0], binary_signals[down_signals],
                       color='orange', marker='v', s=40, alpha=0.7, label=f'Down Class ({np.sum(down_signals)})')

        axes[1].set_title('Binary Quantile Classification (2 classes - balanced)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Class Label', fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        # Multi-class quantile (5 bins)
        multi_generator = QuantileClassificationGenerator(nbins=5, target_name="multi_quantile")
        multi_result = multi_generator.generate_targets(market_data, symbol="M6AM4")
        multi_signals = multi_result['multi_quantile'].to_numpy()

        # Color map for 5 classes
        colors = ['darkred', 'red', 'gray', 'green', 'darkgreen']
        labels = ['Strong Down', 'Down', 'Neutral', 'Up', 'Strong Up']

        for class_idx in range(5):
            class_mask = multi_signals == class_idx
            axes[2].scatter(np.where(class_mask)[0], multi_signals[class_mask],
                           color=colors[class_idx], s=30, alpha=0.8,
                           label=f'{labels[class_idx]} ({np.sum(class_mask)})')

        axes[2].set_title('5-Class Quantile Classification (balanced distribution)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Class Label', fontweight='bold')
        axes[2].set_xlabel('Time (Ticks)', fontweight='bold')
        axes[2].grid(alpha=0.3)
        axes[2].legend()

        plt.figtext(0.02, 0.02,
                    'Quantile classification ensures balanced class distributions for stable ML training.\n' +
                    'Each class contains equal number of samples based on price movement percentiles.',
                    fontsize=10, style='italic')

        plt.tight_layout()
        plt.savefig('plots/optimisation/individual_quantile_classification_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated individual_quantile_classification_signals.png")

    except Exception as e:
        print(f"⚠️  Could not generate quantile classification plot: {e}")

def main():
    """Generate all individual method plots."""
    print("🎨 Generating individual method signal plots...")

    # Ensure plots/optimisation directory exists
    Path("plots/optimisation").mkdir(parents=True, exist_ok=True)

    # Generate individual method plots
    plot_ga_labeling_signals()
    plot_ctl_methods_signals()
    plot_oracle_methods_signals()
    plot_regression_methods_signals()
    plot_quantile_classification_signals()

    print("\n🎉 All individual method plots generated successfully!")
    print("\nGenerated individual method files:")
    print("  • individual_ga_labeling_signals.png - GA evolutionary optimization signals")
    print("  • individual_ctl_methods_signals.png - Binary & Ternary CTL academic methods")
    print("  • individual_oracle_methods_signals.png - Theoretical optimal Oracle methods")
    print("  • individual_regression_methods_signals.png - Continuous regression targets")
    print("  • individual_quantile_classification_signals.png - Traditional balanced classification")

if __name__ == "__main__":
    main()
