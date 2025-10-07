"""
Analysis script for LookbackLookforwardReturnsGenerator.

This script demonstrates the lookback/lookforward window return features:
1. Loads real market data from processed parquet files
2. Generates targets for multiple window lengths
3. Visualizes distributions of percentage return metrics
4. Summarizes cross-window statistics and correlations
5. Highlights behaviour across scaled and current percentage returns
"""

import glob
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from represent.target_generators.regression import LookbackLookforwardReturnsGenerator

AUTO_SAMPLE_Z = 1.96  # 95% confidence level


def _determine_sample_size(
    total_rows: int,
    mean_price: float,
    std_price: float,
    max_window: int,
    target_rel_error: float,
    min_sample: int,
    max_sample: int | None,
) -> int:
    """Heuristically determine rows needed for representative sampling."""
    if total_rows <= min_sample:
        return total_rows

    if mean_price <= 0 or std_price <= 0:
        base = max(min_sample, max_window * 10)
        if max_sample is not None:
            base = min(base, max_sample)
        return min(total_rows, base)

    coeff_var = std_price / mean_price
    required = int((AUTO_SAMPLE_Z * coeff_var / target_rel_error) ** 2)
    required = max(required, max_window * 10, min_sample)
    if max_sample is not None:
        required = min(required, max_sample)
    return min(total_rows, required)


def _row_select(df: pl.DataFrame, indices: np.ndarray) -> pl.DataFrame:
    """Select rows by integer indices with compatibility across Polars versions."""
    df_any = cast(Any, df)

    if hasattr(df_any, "take"):
        return cast(pl.DataFrame, df_any.take(indices))

    if hasattr(df_any, "gather"):
        return cast(pl.DataFrame, df_any.gather(indices.tolist()))

    # Fallback: join on row counts
    selector = pl.Series("__row_idx", indices)
    return (
        df.with_row_count("__row_idx")
        .join(pl.DataFrame({"__row_idx": selector}), on="__row_idx", how="inner")
        .sort("__row_idx")
        .drop("__row_idx")
    )


def load_real_data(
    data_dir: str = "/Users/danielfisher/data/databento/symbol_datasets/inputs",
    max_rows: int | None = None,
    symbol_file: str | None = None,
    sample_method: str = "strided",
    random_seed: int | None = 42,
    max_window: int = 1000,
    target_rel_error: float = 0.005,
    min_sample: int = 150_000,
    max_sample: int | None = 1_500_000,
) -> tuple[pl.DataFrame, str]:
    """
    Load real market data from processed parquet files.

    Args:
        data_dir: Directory containing processed parquet files
        max_rows: Explicit cap on rows to load. If None, an automatic target is computed.
        symbol_file: Specific parquet file to load (if None, loads first file)
        sample_method: Sampling approach when truncating the dataset. Options:
            "strided" (default) picks evenly spaced rows across the file,
            "random" draws a random subset,
            "contiguous" takes a single middle slice.
        random_seed: Optional seed for the random sampler
        max_window: Longest lookback window used by downstream analysis
        target_rel_error: Desired relative error for mean estimates (e.g., 0.005 → 0.5%)
        min_sample: Minimum sample size when subsampling
        max_sample: Optional hard ceiling for sample size

    Returns:
        Tuple of (DataFrame with mid_price and ts_event columns, symbol name)
    """
    # Find all parquet files
    parquet_files = sorted(glob.glob(f"{data_dir}/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    # Select file to load
    if symbol_file:
        file_path = symbol_file
    else:
        file_path = parquet_files[0]

    # Extract symbol name from filename
    filename = file_path.split('/')[-1]
    symbol = filename.split('_')[0]  # Extract symbol (e.g., M6AZ4 from M6AZ4_inputs_only...)

    print(f"   Loading: {filename}")

    df = pl.read_parquet(file_path)

    # Filter to valid mid_price rows
    df = df.filter(pl.col("mid_price").is_not_null())

    print(f"   Total valid rows: {len(df):,}")

    total_rows = len(df)

    stats_df = df.select(
        [
            pl.col("mid_price").mean().alias("mean"),
            pl.col("mid_price").std().alias("std"),
        ]
    )
    mean_price, std_price = stats_df.row(0)

    target_rows = max_rows
    if target_rows is None:
        target_rows = _determine_sample_size(
            total_rows,
            mean_price,
            std_price,
            max_window=max_window,
            target_rel_error=target_rel_error,
            min_sample=min_sample,
            max_sample=max_sample,
        )

    # Sample for performance
    if target_rows is not None and total_rows > target_rows:
        if sample_method == "strided":
            indices = np.linspace(0, total_rows - 1, num=target_rows, dtype=int)
            df = _row_select(df, indices)
            stride = max(int(total_rows / target_rows), 1)
            print(
                f"   Sampled {target_rows:,} evenly spaced rows (approx. stride {stride:,}) for analysis"
            )
        elif sample_method == "random":
            rng = np.random.default_rng(random_seed)
            indices = np.sort(rng.choice(total_rows, size=target_rows, replace=False))
            df = _row_select(df, indices)
            print(f"   Sampled {target_rows:,} random rows for analysis")
        else:  # contiguous fallback
            start_idx = (total_rows - target_rows) // 2
            df = df.slice(start_idx, target_rows)
            print(f"   Sampled {target_rows:,} contiguous rows from dataset centre")
    else:
        print(f"   Using all {total_rows:,} rows")

    # Ensure required columns exist
    if "mid_price" not in df.columns:
        raise ValueError("mid_price column not found in data")

    if "ts_event" not in df.columns:
        # Create sequential index if ts_event missing
        df = df.with_columns(pl.lit(pl.int_range(len(df))).alias("ts_event"))

    return df.select(["mid_price", "ts_event"]), symbol


def get_all_symbol_files(data_dir: str = "/Users/danielfisher/data/databento/symbol_datasets/inputs") -> list[str]:
    """Get all parquet files for processing."""
    return sorted(glob.glob(f"{data_dir}/*.parquet"))


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


def analyze_lookback_lookforward_returns(
    use_real_data: bool = True,
    process_all_symbols: bool = True,
    max_rows: int | None = None,
    sample_method: str = "strided",
    random_seed: int | None = 42,
    max_window: int = 1000,
    target_rel_error: float = 0.005,
    min_sample: int = 150_000,
    max_sample: int | None = 1_500_000,
):
    """
    Analyze lookback/lookforward returns generator outputs.

    Args:
        use_real_data: If True, load real market data; if False, use simulated data
        process_all_symbols: If True, process all symbol files; if False, process only first
    """
    print("=" * 80)
    print("Lookback/Lookforward Returns Generator Analysis")
    print("=" * 80)

    if use_real_data and process_all_symbols:
        # Process all symbols
        symbol_files = get_all_symbol_files()
        print(f"\n1. Found {len(symbol_files)} symbol files to process")

        for idx, symbol_file in enumerate(symbol_files):
            print(f"\n{'=' * 80}")
            print(f"Processing symbol {idx + 1}/{len(symbol_files)}")
            print(f"{'=' * 80}")
            analyze_single_symbol(
                symbol_file,
                use_real_data=True,
                max_rows=max_rows,
                sample_method=sample_method,
                random_seed=random_seed,
                max_window=max_window,
                target_rel_error=target_rel_error,
                min_sample=min_sample,
                max_sample=max_sample,
            )

    else:
        # Process single symbol or simulated data
        analyze_single_symbol(
            symbol_file=None,
            use_real_data=use_real_data,
            max_rows=max_rows,
            sample_method=sample_method,
            random_seed=random_seed,
            max_window=max_window,
            target_rel_error=target_rel_error,
            min_sample=min_sample,
            max_sample=max_sample,
        )


def analyze_single_symbol(
    symbol_file: str | None = None,
    use_real_data: bool = True,
    max_rows: int | None = None,
    sample_method: str = "strided",
    random_seed: int | None = 42,
    max_window: int = 1000,
    target_rel_error: float = 0.005,
    min_sample: int = 150_000,
    max_sample: int | None = 1_500_000,
):
    """Analyze a single symbol."""
    # Load data
    if use_real_data:
        print("\n1. Loading real market data...")
        df, symbol = load_real_data(
            symbol_file=symbol_file,
            max_rows=max_rows,
            sample_method=sample_method,
            random_seed=random_seed,
            max_window=max_window,
            target_rel_error=target_rel_error,
            min_sample=min_sample,
            max_sample=max_sample,
        )
    else:
        print("\n1. Creating simulated data...")
        df = create_sample_data(n_samples=30000)
        symbol = "SIMULATED"
        print(f"   Generated {len(df)} samples")

    # Create generator with all default window lengths
    print("\n2. Generating targets...")
    generator = LookbackLookforwardReturnsGenerator(
        window_lengths=[100, 250, 500, 1000],
        target_prefix="lf",
        scale_factor=0.5,
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
    window_lengths = generator.window_lengths

    print("\n4. Statistical Summary by Window Length:")
    print("-" * 80)

    for window_len in window_lengths:
        print(f"\n   Window Length: {window_len} ticks")

        # Extract columns for this window
        base_return = targets[f"lf_return_{window_len}t"]
        scaled_return = targets[f"lf_scaled_return_{window_len}t"]
        current_return = targets[f"lf_current_return_{window_len}t"]

        # Get valid data for the base return
        valid_mask = ~base_return.is_nan()
        valid_count = valid_mask.sum()
        valid_pct = (valid_count / len(df)) * 100

        print(f"   Valid samples: {valid_count:,} ({valid_pct:.1f}%)")

        if valid_count > 0:
            base_pct = base_return.filter(valid_mask).to_numpy() * 100
            print("   Return (%):")
            print(f"     Mean: {base_pct.mean():.3f}")
            print(f"     Std:  {base_pct.std():.3f}")
            print(f"     Min:  {base_pct.min():.3f}")
            print(f"     Max:  {base_pct.max():.3f}")

            scaled_valid = scaled_return.filter(~scaled_return.is_nan()).to_numpy() * 100
            if scaled_valid.size:
                print("   Scaled Return (%):")
                print(f"     Mean: {scaled_valid.mean():.3f}")
                print(f"     Std:  {scaled_valid.std():.3f}")

            current_valid = current_return.filter(~current_return.is_nan()).to_numpy() * 100
            if current_valid.size:
                print("   Current Return (%):")
                print(f"     Mean: {current_valid.mean():.3f}")
                print(f"     Std:  {current_valid.std():.3f}")

    # Create visualization
    print("\n5. Creating visualizations...")
    create_analysis_plots(targets, window_lengths, symbol, df)

    print("\n" + "=" * 80)
    print(f"Analysis complete! Plots saved to 'outputs/lookback_lookforward_analysis_{symbol}.png'")
    print("=" * 80)


def create_analysis_plots(targets: pl.DataFrame, window_lengths: list[int], symbol: str, df: pl.DataFrame):
    """Create analysis plots highlighting percentage return distributions."""
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Lookback/Lookforward Percentage Returns: {symbol}', fontsize=16, fontweight='bold', y=0.97)

    # Helper for formatting mean ± std strings
    def _format_stats(data: np.ndarray) -> str:
        if data.size == 0:
            return "–"
        return f"{data.mean():.3f} ± {data.std():.3f}"

    # Precompute percentage return arrays per window
    base_returns: dict[int, np.ndarray] = {}
    scaled_returns: dict[int, np.ndarray] = {}
    current_returns: dict[int, np.ndarray] = {}
    summary_rows: list[list[str]] = []

    for window_len in window_lengths:
        base_series = targets[f"lf_return_{window_len}t"]
        scaled_series = targets[f"lf_scaled_return_{window_len}t"]
        current_series = targets[f"lf_current_return_{window_len}t"]

        base_data = base_series.filter(~base_series.is_nan()).to_numpy() * 100
        scaled_data = scaled_series.filter(~scaled_series.is_nan()).to_numpy() * 100
        current_data = current_series.filter(~current_series.is_nan()).to_numpy() * 100

        base_returns[window_len] = base_data
        scaled_returns[window_len] = scaled_data
        current_returns[window_len] = current_data

        summary_rows.append([
            f"{window_len}t",
            f"{base_data.size:,}",
            _format_stats(base_data),
            _format_stats(scaled_data),
            _format_stats(current_data),
        ])

    # Summary table spanning the first row
    table_ax = plt.subplot2grid((5, 4), (0, 0), colspan=4, fig=fig)
    table_ax.axis('off')
    table = table_ax.table(
        cellText=summary_rows,
        colLabels=['Window', 'Valid Samples', 'Return μ±σ (%)', 'Scaled μ±σ (%)', 'Current μ±σ (%)'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor('#2E7D32')
            cell.set_text_props(weight='bold', color='white')
        elif row_idx % 2 == 0:
            cell.set_facecolor('#f4f6f6')

    # Histogram rows
    for idx, window_len in enumerate(window_lengths):
        ax = plt.subplot2grid((5, 4), (1, idx), fig=fig)
        data = base_returns[window_len]
        if data.size:
            ax.hist(data, bins=50, alpha=0.75, color='steelblue', edgecolor='black', density=True)
            mu, sigma = stats.norm.fit(data)
            x = np.linspace(data.min(), data.max(), 200)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_title(f'Mean Return\nWindow {window_len}t', fontsize=11, fontweight='bold')
            ax.set_xlabel('Return (%)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.text(0.95, 0.9, f'μ={mu:.3f}\nσ={sigma:.3f}', transform=ax.transAxes,
                    fontsize=9, bbox={"boxstyle": 'round', "facecolor": 'white', "alpha": 0.8},
                    ha='right', va='top')

    for idx, window_len in enumerate(window_lengths):
        ax = plt.subplot2grid((5, 4), (2, idx), fig=fig)
        data = scaled_returns[window_len]
        if data.size:
            ax.hist(data, bins=50, alpha=0.75, color='mediumseagreen', edgecolor='black', density=True)
            mu, sigma = stats.norm.fit(data)
            x = np.linspace(data.min(), data.max(), 200)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_title(f'Scaled Mean Return\nWindow {window_len}t', fontsize=11, fontweight='bold')
            ax.set_xlabel('Return (%)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.text(0.95, 0.9, f'μ={mu:.3f}\nσ={sigma:.3f}', transform=ax.transAxes,
                    fontsize=9, bbox={"boxstyle": 'round', "facecolor": 'white', "alpha": 0.8},
                    ha='right', va='top')

    for idx, window_len in enumerate(window_lengths):
        ax = plt.subplot2grid((5, 4), (3, idx), fig=fig)
        data = current_returns[window_len]
        if data.size:
            ax.hist(data, bins=50, alpha=0.75, color='darkorange', edgecolor='black', density=True)
            mu, sigma = stats.norm.fit(data)
            x = np.linspace(data.min(), data.max(), 200)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_title(f'Price vs Forward Return\nWindow {window_len}t', fontsize=11, fontweight='bold')
            ax.set_xlabel('Return (%)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.text(0.95, 0.9, f'μ={mu:.3f}\nσ={sigma:.3f}', transform=ax.transAxes,
                    fontsize=9, bbox={"boxstyle": 'round', "facecolor": 'white', "alpha": 0.8},
                    ha='right', va='top')

    # Bottom row comparisons
    ax_box_base = plt.subplot2grid((5, 4), (4, 0), fig=fig)
    base_data_for_box = [base_returns[w] for w in window_lengths if base_returns[w].size]
    base_labels = [f'{w}t' for w in window_lengths if base_returns[w].size]
    if base_data_for_box:
        ax_box_base.boxplot(base_data_for_box, tick_labels=base_labels)
        ax_box_base.set_title('Mean Return Distribution', fontsize=11, fontweight='bold')
        ax_box_base.set_ylabel('Return (%)', fontsize=10)
        ax_box_base.grid(True, alpha=0.3)
        ax_box_base.axhline(0, color='red', linestyle='--', linewidth=1)
    else:
        ax_box_base.text(0.5, 0.5, 'No data available', transform=ax_box_base.transAxes,
                         ha='center', va='center')
        ax_box_base.axis('off')

    ax_box_scaled = plt.subplot2grid((5, 4), (4, 1), fig=fig)
    scaled_data_for_box = [scaled_returns[w] for w in window_lengths if scaled_returns[w].size]
    scaled_labels = [f'{w}t' for w in window_lengths if scaled_returns[w].size]
    if scaled_data_for_box:
        ax_box_scaled.boxplot(scaled_data_for_box, tick_labels=scaled_labels)
        ax_box_scaled.set_title('Scaled Return Distribution', fontsize=11, fontweight='bold')
        ax_box_scaled.set_ylabel('Return (%)', fontsize=10)
        ax_box_scaled.grid(True, alpha=0.3)
        ax_box_scaled.axhline(0, color='red', linestyle='--', linewidth=1)
    else:
        ax_box_scaled.text(0.5, 0.5, 'No data available', transform=ax_box_scaled.transAxes,
                           ha='center', va='center')
        ax_box_scaled.axis('off')

    ax_scatter = plt.subplot2grid((5, 4), (4, 2), fig=fig)
    if len(window_lengths) >= 2:
        first_window = window_lengths[0]
        last_window = window_lengths[-1]
        first_series = (targets[f"lf_return_{first_window}t"] * 100)
        last_series = (targets[f"lf_return_{last_window}t"] * 100)
        valid_mask = ~first_series.is_nan() & ~last_series.is_nan()
        if valid_mask.any():
            first_values = first_series.filter(valid_mask).to_numpy()
            last_values = last_series.filter(valid_mask).to_numpy()
            if len(first_values) > 5000:
                sample_idx = np.random.choice(len(first_values), 5000, replace=False)
                first_values = first_values[sample_idx]
                last_values = last_values[sample_idx]
            ax_scatter.scatter(first_values, last_values, alpha=0.3, s=8, color='purple')
            ax_scatter.set_title(f'{first_window}t vs {last_window}t Returns', fontsize=11, fontweight='bold')
            ax_scatter.set_xlabel(f'{first_window}t Return (%)', fontsize=10)
            ax_scatter.set_ylabel(f'{last_window}t Return (%)', fontsize=10)
            ax_scatter.grid(True, alpha=0.3)
            min_val = min(first_values.min(), last_values.min())
            max_val = max(first_values.max(), last_values.max())
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        else:
            ax_scatter.text(0.5, 0.5, 'Insufficient overlap', transform=ax_scatter.transAxes,
                            ha='center', va='center')
            ax_scatter.axis('off')
    else:
        ax_scatter.axis('off')

    ax_ts = plt.subplot2grid((5, 4), (4, 3), fig=fig)
    mid_idx = min(len(window_lengths) // 2, len(window_lengths) - 1)
    ts_window = window_lengths[mid_idx]
    ts_series = (targets[f"lf_return_{ts_window}t"] * 100)
    valid_indices = np.where(~ts_series.is_nan().to_numpy())[0]
    sample_len = 2000
    if valid_indices.size:
        selected_indices = valid_indices[-sample_len:] if valid_indices.size > sample_len else valid_indices
        series_values = ts_series.to_numpy()[selected_indices]
        ax_ts.plot(range(len(series_values)), series_values, linewidth=1, color='teal', alpha=0.7)
        ax_ts.axhline(0, color='red', linestyle='--', linewidth=1)
        ax_ts.set_title(f'{ts_window}t Return Time Series', fontsize=11, fontweight='bold')
        ax_ts.set_xlabel('Tick', fontsize=10)
        ax_ts.set_ylabel('Return (%)', fontsize=10)
        ax_ts.grid(True, alpha=0.3)
    else:
        ax_ts.text(0.5, 0.5, 'No valid samples', transform=ax_ts.transAxes,
                   ha='center', va='center')
        ax_ts.axis('off')

    plt.tight_layout(rect=(0, 0, 1, 0.94))

    import os
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/lookback_lookforward_analysis_{symbol}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    analyze_lookback_lookforward_returns()
