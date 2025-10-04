"""
Analysis script for LookbackLookforwardReturnsGenerator.

This script demonstrates the new lookback/lookforward window regression features:
1. Loads real market data from processed parquet files
2. Generates targets for multiple window lengths
3. Visualizes distributions of all output metrics
4. Shows relationships between window means and current price
5. Compares log returns across different window lengths
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from represent.target_generators.regression import LookbackLookforwardReturnsGenerator


def load_real_data(
    data_dir: str = "/Users/danielfisher/data/databento/symbol_datasets/inputs",
    max_rows: int = 100000,
) -> pl.DataFrame:
    """
    Load real market data from processed parquet files.

    Args:
        data_dir: Directory containing processed parquet files
        max_rows: Maximum number of rows to load (for performance)

    Returns:
        DataFrame with mid_price and ts_event columns
    """
    # Find all parquet files
    parquet_files = glob.glob(f"{data_dir}/*.parquet")

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    # Load first file (or combine multiple if needed)
    print(f"   Found {len(parquet_files)} parquet files")
    print(f"   Loading: {parquet_files[0].split('/')[-1]}")

    df = pl.read_parquet(parquet_files[0])

    # Filter to valid mid_price rows
    df = df.filter(pl.col("mid_price").is_not_null())

    print(f"   Total valid rows: {len(df):,}")

    # Sample for performance
    if len(df) > max_rows:
        # Take contiguous sample from middle of dataset
        start_idx = (len(df) - max_rows) // 2
        df = df.slice(start_idx, max_rows)
        print(f"   Sampled {max_rows:,} contiguous rows for analysis")
    else:
        print(f"   Using all {len(df):,} rows")

    # Ensure required columns exist
    if "mid_price" not in df.columns:
        raise ValueError("mid_price column not found in data")

    if "ts_event" not in df.columns:
        # Create sequential index if ts_event missing
        df = df.with_columns(pl.lit(pl.int_range(len(df))).alias("ts_event"))

    return df.select(["mid_price", "ts_event"])


def create_sample_data(n_samples: int = 30000, use_realistic_walk: bool = True) -> pl.DataFrame:
    """
    Create sample market data with various price patterns (for testing only).

    Args:
        n_samples: Number of samples to generate
        use_realistic_walk: If True, use geometric Brownian motion (GBM) for realistic FX simulation
    """
    np.random.seed(42)

    if use_realistic_walk:
        # Geometric Brownian Motion - realistic FX price simulation
        # dS = μ*S*dt + σ*S*dW
        prices = []
        current_price = 1.2345

        # Realistic FX parameters
        mu = 0.0  # Drift (zero for mean-reverting FX)
        sigma = 0.0001  # Volatility (10 bps per tick)
        dt = 1.0  # Time step

        for _ in range(n_samples):
            # Geometric Brownian motion step
            dW = np.random.normal(0, np.sqrt(dt))
            drift = mu * current_price * dt
            diffusion = sigma * current_price * dW

            current_price += drift + diffusion
            prices.append(current_price)
    else:
        # Simple random walk with trend changes (original)
        prices = []
        current_price = 1.2345
        trend = 0.0

        for i in range(n_samples):
            # Occasional trend changes every 5000 ticks
            if i % 5000 == 0:
                trend = np.random.normal(0, 0.00001)

            # Price evolution with trend and noise
            current_price += trend + np.random.normal(0, 0.00005)
            prices.append(current_price)

    return pl.DataFrame({
        "mid_price": prices,
        "ts_event": range(n_samples),
    })


def analyze_lookback_lookforward_returns(use_real_data: bool = True):
    """
    Analyze lookback/lookforward returns generator outputs.

    Args:
        use_real_data: If True, load real market data; if False, use simulated data
    """
    print("=" * 80)
    print("Lookback/Lookforward Returns Generator Analysis")
    print("=" * 80)

    # Load data
    if use_real_data:
        print("\n1. Loading real market data...")
        df = load_real_data()
    else:
        print("\n1. Creating simulated data...")
        df = create_sample_data(n_samples=30000)
        print(f"   Generated {len(df)} samples")

    # Create generator with all default window lengths
    print("\n2. Generating targets...")
    generator = LookbackLookforwardReturnsGenerator(
        window_lengths=[1000, 2500, 5000, 10000],
        target_prefix="lf"
    )

    targets = generator.generate_targets(df)
    print(f"   Generated {len(targets.columns)} columns")

    # Print target info
    info = generator.get_target_info()
    print("\n3. Generator Info:")
    print(f"   Type: {info['target_type']}")
    print(f"   Description: {info['description']}")
    print(f"   Window lengths: {info['parameters']['window_lengths']}")
    print(f"   Target columns: {len(info['target_names'])}")

    # Analyze each window length
    window_lengths = [1000, 2500, 5000, 10000]

    print("\n4. Statistical Summary by Window Length:")
    print("-" * 80)

    for window_len in window_lengths:
        print(f"\n   Window Length: {window_len} ticks")

        # Extract columns for this window
        log_return = targets[f"lf_log_return_{window_len}t"]
        current_vs_lookback = targets[f"lf_current_vs_lookback_mean_{window_len}t"]
        current_vs_lookforward = targets[f"lf_current_vs_lookforward_mean_{window_len}t"]

        # Get valid data
        valid_mask = ~log_return.is_nan()
        valid_count = valid_mask.sum()
        valid_pct = (valid_count / len(df)) * 100

        print(f"   Valid samples: {valid_count:,} ({valid_pct:.1f}%)")

        if valid_count > 0:
            # Log return statistics
            valid_log_returns = log_return.filter(valid_mask)
            print("   Log Return (bps):")
            print(f"     Mean: {valid_log_returns.mean():.2f}")
            print(f"     Std:  {valid_log_returns.std():.2f}")
            print(f"     Min:  {valid_log_returns.min():.2f}")
            print(f"     Max:  {valid_log_returns.max():.2f}")

            # Current vs lookback statistics
            valid_cvl = current_vs_lookback.filter(valid_mask)
            print("   Current vs Lookback Mean (bps):")
            print(f"     Mean: {valid_cvl.mean():.2f}")
            print(f"     Std:  {valid_cvl.std():.2f}")

            # Current vs lookforward statistics
            valid_cvf = current_vs_lookforward.filter(valid_mask)
            print("   Current vs Lookforward Mean (bps):")
            print(f"     Mean: {valid_cvf.mean():.2f}")
            print(f"     Std:  {valid_cvf.std():.2f}")

    # Create visualization
    print("\n5. Creating visualizations...")
    create_analysis_plots(targets, window_lengths)

    print("\n" + "=" * 80)
    print("Analysis complete! Plots saved to 'outputs/lookback_lookforward_analysis.png'")
    print("=" * 80)


def create_analysis_plots(targets: pl.DataFrame, window_lengths: list[int]):
    """Create comprehensive analysis plots."""
    plt.figure(figsize=(20, 12))

    # Row 1: Log Return Distributions with Normal Fit
    for idx, window_len in enumerate(window_lengths):
        ax = plt.subplot(4, 4, idx + 1)

        log_return = targets[f"lf_log_return_{window_len}t"]
        valid_data = log_return.filter(~log_return.is_nan()).to_numpy()

        if len(valid_data) > 0:
            # Plot histogram
            n, bins, patches = ax.hist(valid_data, bins=50, alpha=0.7, color='steelblue',
                                       edgecolor='black', density=True, label='Data')

            # Fit normal distribution
            mu, sigma = stats.norm.fit(valid_data)

            # Plot fitted normal distribution
            x = np.linspace(valid_data.min(), valid_data.max(), 100)
            fitted_normal = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, fitted_normal, 'r-', linewidth=2, label='Normal Fit')

            # Add vertical line at zero
            ax.axvline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

            # Add text box with parameters
            textstr = f'μ = {mu:.2f}\nσ = {sigma:.2f}'
            props = {"boxstyle": 'round', "facecolor": 'wheat', "alpha": 0.8}
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right', bbox=props)

            ax.set_xlabel('Log Return (bps)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Log Return Distribution\nWindow: {window_len}t', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)

    # Row 2: Current vs Lookback Mean Distributions with Normal Fit
    for idx, window_len in enumerate(window_lengths):
        ax = plt.subplot(4, 4, idx + 5)

        current_vs_lookback = targets[f"lf_current_vs_lookback_mean_{window_len}t"]
        valid_data = current_vs_lookback.filter(~current_vs_lookback.is_nan()).to_numpy()

        if len(valid_data) > 0:
            # Plot histogram
            n, bins, patches = ax.hist(valid_data, bins=50, alpha=0.7, color='forestgreen',
                                       edgecolor='black', density=True, label='Data')

            # Fit normal distribution
            mu, sigma = stats.norm.fit(valid_data)

            # Plot fitted normal distribution
            x = np.linspace(valid_data.min(), valid_data.max(), 100)
            fitted_normal = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, fitted_normal, 'r-', linewidth=2, label='Normal Fit')

            # Add vertical line at zero
            ax.axvline(0, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)

            # Add text box with parameters
            textstr = f'μ = {mu:.2f}\nσ = {sigma:.2f}'
            props = {"boxstyle": 'round', "facecolor": 'wheat', "alpha": 0.8}
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right', bbox=props)

            ax.set_xlabel('Current vs Lookback Mean (bps)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Current vs Lookback Mean\nWindow: {window_len}t', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)

    # Row 3: Current vs Lookforward Mean Distributions with Normal Fit
    for idx, window_len in enumerate(window_lengths):
        ax = plt.subplot(4, 4, idx + 9)

        current_vs_lookforward = targets[f"lf_current_vs_lookforward_mean_{window_len}t"]
        valid_data = current_vs_lookforward.filter(~current_vs_lookforward.is_nan()).to_numpy()

        if len(valid_data) > 0:
            # Plot histogram
            n, bins, patches = ax.hist(valid_data, bins=50, alpha=0.7, color='coral',
                                       edgecolor='black', density=True, label='Data')

            # Fit normal distribution
            mu, sigma = stats.norm.fit(valid_data)

            # Plot fitted normal distribution
            x = np.linspace(valid_data.min(), valid_data.max(), 100)
            fitted_normal = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, fitted_normal, 'r-', linewidth=2, label='Normal Fit')

            # Add vertical line at zero
            ax.axvline(0, color='purple', linestyle='--', linewidth=1.5, alpha=0.7)

            # Add text box with parameters
            textstr = f'μ = {mu:.2f}\nσ = {sigma:.2f}'
            props = {"boxstyle": 'round', "facecolor": 'wheat', "alpha": 0.8}
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right', bbox=props)

            ax.set_xlabel('Current vs Lookforward Mean (bps)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Current vs Lookforward Mean\nWindow: {window_len}t', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)

    # Row 4: Comparative Analysis

    # Plot 4.1: Log Returns across all window lengths (box plot)
    ax = plt.subplot(4, 4, 13)
    log_returns_data = []
    labels = []
    for window_len in window_lengths:
        log_return = targets[f"lf_log_return_{window_len}t"]
        valid_data = log_return.filter(~log_return.is_nan()).to_numpy()
        if len(valid_data) > 0:
            log_returns_data.append(valid_data)
            labels.append(f'{window_len}t')

    if log_returns_data:
        ax.boxplot(log_returns_data, tick_labels=labels)
        ax.set_xlabel('Window Length', fontsize=10)
        ax.set_ylabel('Log Return (bps)', fontsize=10)
        ax.set_title('Log Returns Comparison', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

    # Plot 4.2: Standard deviation vs window length
    ax = plt.subplot(4, 4, 14)
    stds = []
    window_labels = []
    for window_len in window_lengths:
        log_return = targets[f"lf_log_return_{window_len}t"]
        valid_data = log_return.filter(~log_return.is_nan()).to_numpy()
        if len(valid_data) > 0:
            stds.append(np.std(valid_data))
            window_labels.append(window_len)

    if stds:
        ax.plot(window_labels, stds, marker='o', linewidth=2, markersize=8, color='darkblue')
        ax.set_xlabel('Window Length (ticks)', fontsize=10)
        ax.set_ylabel('Std Dev (bps)', fontsize=10)
        ax.set_title('Volatility vs Window Length', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Plot 4.3: Scatter plot - 1000t vs 10000t log returns
    ax = plt.subplot(4, 4, 15)
    log_1000 = targets["lf_log_return_1000t"]
    log_10000 = targets["lf_log_return_10000t"]

    valid_mask = ~log_1000.is_nan() & ~log_10000.is_nan()
    if valid_mask.any():
        valid_1000 = log_1000.filter(valid_mask).to_numpy()
        valid_10000 = log_10000.filter(valid_mask).to_numpy()

        # Sample for visualization (max 5000 points)
        if len(valid_1000) > 5000:
            sample_indices = np.random.choice(len(valid_1000), 5000, replace=False)
            valid_1000 = valid_1000[sample_indices]
            valid_10000 = valid_10000[sample_indices]

        ax.scatter(valid_1000, valid_10000, alpha=0.3, s=10, color='purple')
        ax.set_xlabel('1000t Log Return (bps)', fontsize=10)
        ax.set_ylabel('10000t Log Return (bps)', fontsize=10)
        ax.set_title('Short vs Long Window Returns', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add diagonal line
        min_val = min(valid_1000.min(), valid_10000.min())
        max_val = max(valid_1000.max(), valid_10000.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, alpha=0.5)

    # Plot 4.4: Time series sample (last 2000 points)
    ax = plt.subplot(4, 4, 16)
    sample_len = 2000
    log_2500 = targets["lf_log_return_2500t"]

    # Get last N valid points
    valid_indices = np.where(~log_2500.is_nan().to_numpy())[0]
    if len(valid_indices) >= sample_len:
        sample_indices = valid_indices[-sample_len:]
        sample_data = log_2500.to_numpy()[sample_indices]

        ax.plot(range(len(sample_data)), sample_data, linewidth=1, color='teal', alpha=0.7)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Time (ticks)', fontsize=10)
        ax.set_ylabel('Log Return (bps)', fontsize=10)
        ax.set_title('2500t Log Return Time Series\n(Last 2000 samples)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    import os
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/lookback_lookforward_analysis.png", dpi=150, bbox_inches='tight')
    print("   Saved plot to outputs/lookback_lookforward_analysis.png")


if __name__ == "__main__":
    analyze_lookback_lookforward_returns()
