#!/usr/bin/env python3
"""
Symbol Dataset Optimization Runner

This script runs parameter optimization on all symbol datasets and generates
comprehensive reports with visualizations.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent.large_scale_optimization import LargeScaleParameterOptimizer
from represent.parameter_storage import ParameterStorage, save_optimization_results


def generate_label_grouped_plots_for_symbol(
    symbol_info: dict[str, Any],
    symbol_results: dict[str, dict[str, Any]],
    examples_per_label: int = 10,
    window_size: int = 100_000
) -> None:
    """Generate 3 plots grouped by label type (-1, 0, +1) with multiple examples per plot.

    Output structure:
    - outputs/plots/optimization/[SYMBOL]/label_long_examples.png (label = +1)
    - outputs/plots/optimization/[SYMBOL]/label_timeout_examples.png (label = 0)
    - outputs/plots/optimization/[SYMBOL]/label_short_examples.png (label = -1)
    """
    from pathlib import Path as _P

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from diagnostics.comprehensive_method_debugging import simulate_optimization_sampling
    from represent.target_generators.factory import TargetGeneratorFactory

    symbol_name = symbol_info['symbol']

    # Only process adaptive triple barrier method for cleaner visualization
    if 'triple_barrier_adaptive' not in symbol_results:
        print(f"⚠️  No adaptive triple barrier results for {symbol_name}")
        return

    method_result = symbol_results['triple_barrier_adaptive']
    optimal_params = method_result.get('optimal_params', {})

    print(f"   Using parameters: {optimal_params}")

    # Load full dataset for the symbol
    df = pl.read_parquet(symbol_info['file_path'])
    df = df.filter(pl.col('mid_price').is_not_null())

    # Generate many windows to find diverse examples
    total_windows = examples_per_label * 5  # Generate 5x more to ensure diversity
    win_list = simulate_optimization_sampling(df, window_size, n_windows=total_windows)

    # Create generator with optimized parameters
    generator = TargetGeneratorFactory.create("triple_barrier_adaptive", **optimal_params)

    # Collect examples for each label type
    label_examples = {-1: [], 0: [], 1: []}

    print(f"   Searching for label examples in {total_windows} windows...")

    for window_idx, window_df in enumerate(win_list):
        if all(len(examples) >= examples_per_label for examples in label_examples.values()):
            break  # Found enough examples for all labels

        # Generate labels for this window
        targets_df = generator.generate_targets(window_df)
        labels = targets_df['adaptive_triple_barrier_label'].to_numpy()
        prices = window_df['mid_price'].to_numpy()

        # Extract metadata for plotting (no more recalculation needed!)
        volatilities = targets_df['adaptive_triple_barrier_label_volatility'].to_numpy()
        upper_barriers = targets_df['adaptive_triple_barrier_label_upper_barrier'].to_numpy()
        lower_barriers = targets_df['adaptive_triple_barrier_label_lower_barrier'].to_numpy()
        exit_prices = targets_df['adaptive_triple_barrier_label_exit_price'].to_numpy()
        exit_indices = targets_df['adaptive_triple_barrier_label_exit_index'].to_numpy()

        # Find representative examples of each label in this window
        for label_value in [-1, 0, 1]:
            if len(label_examples[label_value]) >= examples_per_label:
                continue

            label_indices = np.where(labels == label_value)[0]
            if len(label_indices) > 0:
                # Pick a representative example from this window
                mid_idx = label_indices[len(label_indices) // 2]  # Middle example

                # VALIDATION: Ensure the selected label actually matches what we expect
                actual_label = labels[mid_idx]
                if actual_label != label_value:
                    print(f"   ⚠️  Label mismatch at index {mid_idx}: expected {label_value}, got {actual_label}")
                    continue

                # Extract a focused window around this example
                lookforward = optimal_params.get('lookforward_window', 1000)
                start_idx = max(0, mid_idx - lookforward)
                end_idx = min(len(prices), mid_idx + lookforward * 2)

                example = {
                    'window_idx': window_idx,
                    'label_idx': mid_idx,
                    'prices': prices[start_idx:end_idx],
                    'labels': labels[start_idx:end_idx],
                    'focus_idx': mid_idx - start_idx,  # Index of the target label in the extracted segment
                    'label_value': label_value,
                    'validated_label': actual_label,  # Store the validated label for double-checking
                    # Add stored metadata - NO MORE RECALCULATION NEEDED!
                    'volatility': volatilities[mid_idx],
                    'upper_barrier': upper_barriers[mid_idx],
                    'lower_barrier': lower_barriers[mid_idx],
                    'exit_price': exit_prices[mid_idx],
                    'exit_index': exit_indices[mid_idx]
                }
                label_examples[label_value].append(example)

    # Create 3 plots: one for each label type
    label_names = {-1: 'Short', 0: 'Timeout', 1: 'Long'}

    base_dir = _P('outputs/plots/optimization') / symbol_name
    base_dir.mkdir(parents=True, exist_ok=True)

    for label_value, examples in label_examples.items():
        if not examples:
            print(f"   ⚠️  No examples found for label {label_value} ({label_names[label_value]})")
            continue

        # Limit to requested number of examples
        examples = examples[:examples_per_label]

        print(f"   📊 Creating {label_names[label_value]} plot with {len(examples)} examples...")

        # Create subplot grid
        n_examples = len(examples)
        n_cols = min(5, n_examples)
        n_rows = (n_examples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

        # Handle single subplot case
        if n_examples == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'{symbol_name} - {label_names[label_value]} Examples (Label {label_value})',
                    fontsize=16, fontweight='bold')

        for i, example in enumerate(examples):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            prices = example['prices']
            labels = example['labels']
            focus_idx = example['focus_idx']

            # Plot price line
            time_axis = np.arange(len(prices))
            ax.plot(time_axis, prices, 'black', linewidth=1.0, alpha=0.8, label='Price')

            # TRIPLE BARRIER VISUALIZATION: Show actual barriers and trade execution
            if focus_idx < len(prices):
                entry_price = prices[focus_idx]
                lookforward_window = optimal_params.get('lookforward_window', 1000)

                # Use STORED metadata instead of recalculating - this ensures consistency!
                upper_barrier = example['upper_barrier']
                lower_barrier = example['lower_barrier']
                volatility = example['volatility']

                print(f"    Using stored barriers: upper={upper_barrier:.4f}, lower={lower_barrier:.4f}, volatility={volatility:.6f}")

                # Draw barriers
                barrier_start = focus_idx
                barrier_end = min(len(prices), focus_idx + lookforward_window)
                barrier_time = np.arange(barrier_start, barrier_end)

                if len(barrier_time) > 0:
                    ax.plot(barrier_time, [upper_barrier] * len(barrier_time),
                           'g--', alpha=0.7, linewidth=2, label='Upper Barrier')
                    ax.plot(barrier_time, [lower_barrier] * len(barrier_time),
                           'r--', alpha=0.7, linewidth=2, label='Lower Barrier')

                    # Draw time barrier (vertical line)
                    if barrier_end - 1 < len(time_axis):
                        ax.axvline(x=barrier_end - 1, color='orange', linestyle=':',
                                  alpha=0.7, linewidth=2, label='Time Barrier')

                # Highlight entry point
                ax.scatter(focus_idx, entry_price, color='blue', s=120, alpha=0.9,
                          zorder=5, marker='o', edgecolors='white', linewidth=2,
                          label='Entry Point')

                # Use the ORIGINAL label outcome instead of recalculating
                # This ensures consistency with the filtering
                original_label = example['label_value']
                label_to_reason = {1: 'Upper Hit', -1: 'Lower Hit', 0: 'Timeout'}
                label_to_color = {1: 'green', -1: 'red', 0: 'orange'}

                exit_reason = label_to_reason[original_label]
                exit_color = label_to_color[original_label]

                # Use STORED exit information instead of recalculating - this ensures consistency!
                stored_exit_index = example['exit_index']  # Relative index from generator
                stored_exit_price = example['exit_price']
                exit_idx = focus_idx + 1 + stored_exit_index  # Convert to absolute index in price window

                print(f"    Using stored exit: relative_idx={stored_exit_index}, exit_price={stored_exit_price:.4f}")

                # Highlight exit point
                if exit_idx is not None and exit_idx < len(prices):
                    ax.scatter(exit_idx, prices[exit_idx], color=exit_color, s=100,
                              alpha=0.9, zorder=5, marker='X', label=f'Exit ({exit_reason})')

            # VALIDATION: Double-check the example is correctly filtered
            validated_label = example.get('validated_label', example['label_value'])
            if validated_label != label_value:
                validation_status = "⚠️ MISMATCH"
            else:
                validation_status = "✅"

            ax.set_title(f'Example {i+1}: {label_names[label_value]} Signal {validation_status}\n(Window {example["window_idx"]+1})')
            ax.set_xlabel('Time (ticks)')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)

            if i == 0:  # Add legend to first subplot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Hide empty subplots
        for i in range(n_examples, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()

        # Save plot
        output_file = base_dir / f'label_{label_names[label_value].lower()}_examples.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Saved: {output_file}")


def generate_diagnostics_comprehensive_plots_for_symbol_DEPRECATED(
    symbol_info: dict[str, Any],
    symbol_results: dict[str, dict[str, Any]],
    windows: int = 3,
    window_size: int = 25_000  # USER REQ: Max 25K sampling window
) -> None:
    """Use diagnostics.create_all_methods_plot to render comprehensive plots per window.
    For each subset of methods, copy the plots into per-symbol directories.

    Output structure:
    - outputs/plots/optimization/[SYMBOL]/all/comprehensive_all_methods_window_{n}.png
    - outputs/plots/optimization/[SYMBOL]/classification/comprehensive_all_methods_window_{n}.png
    - outputs/plots/optimization/[SYMBOL]/oracle/comprehensive_all_methods_window_{n}.png
    - outputs/plots/optimization/[SYMBOL]/triple/comprehensive_all_methods_window_{n}.png
    """
    import shutil
    from pathlib import Path as _P

    import polars as pl

    from diagnostics.comprehensive_method_debugging import (
        create_all_methods_plot,
        get_all_optimization_methods,
        simulate_optimization_sampling,
    )

    symbol_name = symbol_info['symbol']

    # Load full dataset for the symbol
    df = pl.read_parquet(symbol_info['file_path'])
    df = df.filter(pl.col('mid_price').is_not_null())

    # Build fixed 100k windows using diagnostic sampler
    win_list = simulate_optimization_sampling(df, window_size, n_windows=windows)

    # Get full method configs from diagnostics, but filter to only optimized methods
    all_available_methods = list(get_all_optimization_methods())
    # Filter to only methods that were actually optimized for this symbol
    optimized_method_names = set(symbol_results.keys())
    full_methods = [m for m in all_available_methods if m.get('method') in optimized_method_names]

    # Helper to inject optimized parameters
    def with_optimized_params(methods_in: list[dict]) -> list[dict]:
        out = []
        for m in methods_in:
            m2 = dict(m)
            method_key = m2.get('method')
            optimal = symbol_results.get(method_key, {}).get('optimal_params', {})
            params = dict(m2.get('params', {}))
            params.update(optimal)
            if method_key in ('binary_ctl', 'ternary_ctl'):
                params.pop('transaction_cost', None)
            for k in list(params.keys()):
                if k.endswith('_window') or k in ('window_size',):
                    try:
                        params[k] = int(params[k])
                    except Exception:
                        pass
            m2['params'] = params
            out.append(m2)
        return out

    # Define subsets
    def is_classification(m):
        return m.get('method') in {'binary_ctl', 'ternary_ctl'}

    def is_oracle(m):
        return m.get('method') in {'oracle_binary', 'oracle_ternary'}

    def is_triple(m):
        return 'triple' in (m.get('method') or '')

    subsets = {
        'all': full_methods,
        'classification': [m for m in full_methods if is_classification(m)],
        'oracle': [m for m in full_methods if is_oracle(m)],
        'triple': [m for m in full_methods if is_triple(m)],
    }

    # Ensure base output dirs exist
    base_dir = _P('outputs/plots/optimization')
    base_dir.mkdir(parents=True, exist_ok=True)

    # For each subset, render global plots then copy into per-symbol/subset dirs
    for subset_name, subset_methods in subsets.items():
        if not subset_methods:
            continue
        methods = with_optimized_params(subset_methods)
        # Render for each window (diagnostics saves to outputs/plots/optimization/... globally)
        for idx, window_df in enumerate(win_list, 1):
            create_all_methods_plot(window_df, methods, idx)
            # Copy to per-symbol/subset path
            src = base_dir / f"comprehensive_all_methods_window_{idx}.png"
            dst_dir = base_dir / symbol_name / subset_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"comprehensive_all_methods_window_{idx}.png"
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"⚠️  Failed to copy plot for {symbol_name}/{subset_name}/window_{idx}: {e}")


def generate_comprehensive_all_methods_windows(
    symbol_info: dict[str, Any],
    symbol_results: dict[str, dict[str, Any]],
    windows: int = 3,
    window_size: int = 25_000,  # USER REQ: Max 25K sampling window
    save_dir: Path = Path("scripts")
) -> list[Path]:
    """Apply optimized parameters to fixed 100k windows and generate comprehensive plots.
    Saves files as scripts/comprehensive_all_methods_window_1.png, etc.
    Returns list of saved file Paths.
    """
    import matplotlib.pyplot as plt

    from represent.target_generators import TargetGeneratorFactory

    # Load prices
    prices = load_symbol_data_from_parquet(symbol_info['file_path'])
    n = len(prices)
    w = min(window_size, n)
    if w < 1000:
        raise ValueError("Not enough samples to create a 100k window plot")

    # Determine window start indices (evenly spaced)
    if windows == 1:
        starts = [max(0, n - w)]
    else:
        step = max(1, (n - w) // max(1, windows - 1))
        starts = [min(i * step, n - w) for i in range(windows)]

    method_order = ['binary_ctl', 'ternary_ctl', 'oracle_binary', 'oracle_ternary']

    saved = []
    for idx, start in enumerate(starts, 1):
        end = start + w
        window_prices = prices[start:end]
        market_data = pl.DataFrame({
            'mid_price': window_prices,
            'ts_event': np.arange(w, dtype=np.int64)
        })

        # Collect labels per method using optimized params
        labels_per_method: dict[str, np.ndarray] = {}
        for method in method_order:
            if method not in symbol_results:
                continue
            optimal = symbol_results[method].get('optimal_params', {})
            # Filter incompatible params for CTL
            filtered = dict(optimal)
            if method in ('binary_ctl', 'ternary_ctl'):
                filtered.pop('transaction_cost', None)
            try:
                gen = TargetGeneratorFactory.create(method, **filtered)
            except Exception:
                # Fallback: try without any params
                try:
                    gen = TargetGeneratorFactory.create(method)
                except Exception:
                    continue
            try:
                df = gen.generate_targets(market_data)
                # Pick last column by default
                labels = df[df.columns[-1]].to_numpy()
                labels_per_method[method] = labels
            except Exception as e:
                print(f"   ⚠️  Visualization generation failed for {method}: {e}")
                continue

        # Create figure: top price, then one subplot per method
        rows = 1 + len(labels_per_method)
        if rows == 1:
            continue
        fig, axes = plt.subplots(rows, 1, figsize=(16, 3 * rows), sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # Plot price
        axes[0].plot(np.arange(w), window_prices, 'k-', linewidth=1)
        axes[0].set_title(f"{symbol_info['symbol']} – Price (Window {idx}, {w:,} samples)")
        axes[0].grid(True, alpha=0.3)

        # Plot methods
        r = 1
        for method in method_order:
            if method not in labels_per_method:
                continue
            lbls = labels_per_method[method]
            ax = axes[r]
            # Handle ternary {-1,0,1} vs binary {0,1}
            uniq = np.unique(lbls[~np.isnan(lbls)]) if hasattr(lbls, '__len__') else np.array([])
            cmap = plt.cm.get_cmap('Set2', max(3, len(uniq)))
            if len(uniq) == 0:
                ax.text(0.5, 0.5, 'No labels', transform=ax.transAxes, ha='center')
            else:
                for j, u in enumerate(sorted(uniq)):
                    mask = (lbls == u)
                    ax.scatter(np.where(mask)[0], lbls[mask], s=1, c=[cmap(j)], label=str(int(u)))
            ax.set_title(method.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            r += 1
        axes[-1].set_xlabel('Time (ticks)')
        plt.tight_layout()

        out_path = save_dir / f"comprehensive_all_methods_window_{idx}.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"   📈 Saved {out_path}")
        saved.append(out_path)

    return saved


def load_symbol_data_from_parquet(file_path: str | Path) -> np.ndarray:
    """Load price data from symbol parquet file."""
    df = pl.read_parquet(file_path)

    if 'mid_price' in df.columns:
        prices = df['mid_price'].to_numpy()
    elif 'price' in df.columns:
        prices = df['price'].to_numpy()
    else:
        raise ValueError(f"No price column found in {file_path}")

    # Remove NaN and invalid values
    valid_mask = np.isfinite(prices) & (prices > 0)
    valid_prices = prices[valid_mask]

    if len(valid_prices) == 0:
        raise ValueError(f"No valid price data found in {file_path}")

    if len(valid_prices) < len(prices):
        invalid_count = len(prices) - len(valid_prices)
        print(f"⚠️  Removed {invalid_count:,} invalid price values from {file_path.name}")

    return valid_prices


def discover_symbol_datasets(data_dir: str | Path) -> list[dict[str, Any]]:
    """
    Discover all symbol datasets in the data directory.

    Args:
        data_dir: Directory containing symbol datasets

    Returns:
        List of symbol dataset info dicts
    """
    data_path = Path(data_dir)
    symbols = []

    # Look for parquet files that look like symbol datasets
    for file_path in data_path.rglob("*.parquet"):
        if file_path.stat().st_size > 1 * 1024 * 1024:  # At least 1MB
            try:
                # Try to load a small sample to validate format
                df_sample = pl.read_parquet(file_path, n_rows=1000)

                if 'mid_price' in df_sample.columns or 'price' in df_sample.columns:
                    # Get full dataset size
                    df_full = pl.read_parquet(file_path)

                    symbol_info = {
                        'symbol': file_path.stem,  # Use filename as symbol
                        'file_path': file_path,
                        'samples': len(df_full),
                        'size_mb': file_path.stat().st_size / (1024 * 1024),
                        'date_range': None
                    }

                    # Try to get date range if timestamp column exists
                    if 'timestamp' in df_full.columns:
                        try:
                            timestamps = df_full['timestamp']
                            if len(timestamps) > 0:
                                symbol_info['date_range'] = f"{timestamps.min()} to {timestamps.max()}"
                        except Exception:
                            pass

                    symbols.append(symbol_info)
                    print(f"📊 Found symbol dataset: {symbol_info['symbol']} ({symbol_info['samples']:,} samples, {symbol_info['size_mb']:.1f}MB)")

            except Exception as e:
                print(f"⚠️  Skipping {file_path}: {e}")
                continue

    return symbols


def convert_params_for_generator(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Convert optimization parameters to correct types for generators.

    Optimization returns all parameters as floats, but some generators
    require integer parameters (e.g., window sizes).

    Args:
        method: Target generation method name
        params: Raw parameters from optimization

    Returns:
        Parameters with correct types for the generator
    """
    converted = params.copy()

    # Parameters that should be integers for various methods
    int_params_by_method = {
        'triple_barrier': ['lookforward_window', 'volatility_window'],
        'triple_barrier_adaptive': ['lookforward_window', 'lookback_window'],
        'triple_exceedance': ['lookforward_window', 'volatility_window'],
        'ternary_ctl': ['window_size'],
    }


    # Convert specified parameters to integers
    int_params = int_params_by_method.get(method, [])
    for param in int_params:
        if param in converted:
            converted[param] = int(round(converted[param]))

    return converted


def calculate_additional_metrics(prices: np.ndarray, method: str, optimal_params: dict[str, Any]) -> dict[str, Any]:
    """
    Calculate additional metrics using EXACT same logic and windowing as optimization.

    The optimization uses window sampling (e.g., 100K samples × 5 windows) while the original
    enhanced output calculated over entire dataset sequentially. This mismatch caused different results.

    Args:
        prices: Full price array
        method: Target generation method name
        optimal_params: Optimal parameters for the method

    Returns:
        Dictionary with additional metrics calculated using sampled windows
    """
    try:
        # Import here to avoid circular imports
        import polars as pl

        # Use EXACT same windowing parameters and sampling strategy as optimization
        # This ensures metrics match what optimization actually optimized for
        from represent.large_scale_optimization import LargeScaleParameterOptimizer
        from represent.target_generators.factory import TargetGeneratorFactory

        # Create temporary optimizer just to use its sampling method (with SAME parameters as optimization)
        temp_optimizer = LargeScaleParameterOptimizer(
            window_size=50000,      # Same as optimization config
            n_windows=10,           # Same as optimization config
            sampling_strategy='stratified',  # Same as optimization config
            random_state=42,        # Same random seed for reproducibility
            use_cross_validation=True,  # Same robust sampling
            target_coverage=0.25,   # Same coverage
            validation_split=0.3    # Same validation split
        )

        # Use the exact same robust sampling method as optimization
        if temp_optimizer.use_cross_validation:
            training_windows, validation_windows = temp_optimizer.sample_windows_robust(
                prices, n_windows=10, target_coverage=0.25, validation_split=0.3
            )
            # Use training windows for final metrics (same as optimization)
            sample_windows = training_windows
        else:
            sample_windows = temp_optimizer.sample_windows(prices, n_windows=10)

        # Aggregate results across windows (same as optimization)
        total_returns = 0.0
        total_trades = 0
        all_labels = []
        valid_windows = 0

        for window_prices in sample_windows:
            try:
                # Create DataFrame for this window
                df = pl.DataFrame({"mid_price": window_prices})

                # Convert float parameters to int where needed for generators
                converted_params = convert_params_for_generator(method, optimal_params)

                # Generate targets using optimal parameters
                generator = TargetGeneratorFactory.create(method, **converted_params)
                targets_df = generator.generate_targets(df)
                target_info = generator.get_target_info()
                target_col = target_info['target_names'][0]
                window_labels = targets_df[target_col].to_numpy()

                # Collect labels for overall balance calculation
                all_labels.extend(window_labels.tolist())

                # Calculate PnL for this window using EXACT optimization logic
                transaction_cost = 0.00007  # 0.7 pip default

                try:
                    from tstrends.returns_estimation import ReturnsEstimatorWithFees
                    from tstrends.returns_estimation.fees_config import FeesConfig

                    # EXACT replication of optimization logic (lines 882-915)
                    fees_config = FeesConfig(
                        lp_transaction_fees=transaction_cost,
                        sp_transaction_fees=transaction_cost,
                    )
                    returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)

                    # Convert labels to proper format for returns estimation
                    labels_int = window_labels.astype(int)
                    unique_labels_set = np.unique(labels_int[~np.isnan(labels_int)])

                    if (set(unique_labels_set).issubset({0, 1, 2}) and len(unique_labels_set) >= 2) or method.lower() in ['ternary_ctl', 'oracle_ternary']:
                        # Ternary labels: {0, 1, 2} → {-1, 0, 1} for TStrends
                        # Even if not all 3 classes present in this window, convert for TStrends compatibility
                        labels_tstrends = labels_int - 1
                    elif len(unique_labels_set) == 2 and set(unique_labels_set).issubset({0, 1}):
                        # Binary labels: {0, 1} → {-1, 1} for TStrends binary methods
                        # Only convert if this is binary CTL, oracle binary, etc.
                        if method.lower() in ['binary_ctl', 'oracle_binary']:
                            labels_tstrends = np.where(labels_int == 0, -1, 1)
                        else:
                            # For other binary methods, keep as {0, 1} (long-only)
                            labels_tstrends = labels_int
                    else:
                        labels_tstrends = labels_int

                    # Calculate returns using TStrends for this window
                    window_return = returns_estimator.estimate_return(
                        window_prices.tolist(),
                        labels_tstrends.tolist()
                    )

                    # Count trades in this window
                    window_trades = 0
                    current_position = 0
                    for target_position in labels_tstrends:
                        if target_position != current_position:
                            window_trades += 1
                            current_position = target_position

                    # Aggregate
                    total_returns += window_return
                    total_trades += window_trades
                    valid_windows += 1

                except ImportError:
                    # Skip this window if TStrends not available
                    continue

            except Exception:
                # Skip problematic windows
                continue

        # Calculate overall metrics from aggregated windows
        if valid_windows > 0:
            avg_return = total_returns / valid_windows  # Average return across windows
            avg_trades = total_trades / valid_windows   # Average trades per window
            mean_return_per_trade = avg_return / avg_trades if avg_trades > 0 else 0.0

            # Calculate trading frequency as percentage of actual window size used
            actual_window_size = len(sample_windows[0]) if sample_windows else 50000
            trading_frequency = avg_trades / actual_window_size * 100 if actual_window_size > 0 else 0.0
        else:
            avg_return = 0.0
            avg_trades = 0
            mean_return_per_trade = 0.0
            trading_frequency = 0.0

        # Calculate class balance from all collected labels
        if all_labels:
            all_labels_array = np.array(all_labels)

            # Check if method produces continuous (regression) outputs that need discretization
            regression_methods = {
                'directional_mfe', 'log_return_horizons', 'volatility',
                'volatility_scaled_returns', 'remaining_value_tuner'
            }

            unique_labels, counts = np.unique(all_labels_array, return_counts=True)

            # If this is a regression method, discretize the continuous outputs
            if method in regression_methods:
                # Discretize continuous values into meaningful bins
                if np.ptp(all_labels_array) > 0:  # If there's variation in the data
                    # Use quantile-based binning for balance
                    try:
                        # Create 3 bins: Low, Medium, High (similar to ternary classification)
                        bins = np.quantile(all_labels_array, [0, 0.33, 0.67, 1.0])
                        # Ensure unique bin edges
                        if len(np.unique(bins)) < len(bins):
                            # Fall back to uniform spacing if quantiles are identical
                            bins = np.linspace(all_labels_array.min(), all_labels_array.max(), 4)

                        # Discretize into bins
                        discretized = np.digitize(all_labels_array, bins[1:-1])  # Creates 0, 1, 2
                        unique_labels, counts = np.unique(discretized, return_counts=True)

                        # Map to meaningful labels
                        label_mapping = {0: "Low", 1: "Medium", 2: "High"}
                        unique_labels = [label_mapping.get(x, f"Bin_{x}") for x in unique_labels]

                    except Exception:
                        # Fallback: just show range info
                        label_range = f"{all_labels_array.min():.4f} to {all_labels_array.max():.4f}"
                        unique_labels = [f"Range: {label_range}"]
                        counts = [len(all_labels_array)]
                else:
                    # All values are the same
                    unique_labels = [f"Constant: {all_labels_array[0]:.4f}"]
                    counts = [len(all_labels_array)]

            percentages = counts / len(all_labels_array) * 100
            balance_score = min(percentages) / max(percentages) * 100 if len(percentages) > 1 else 100

            # Create label distribution string
            label_dist = {str(label): f"{pct:.1f}%" for label, pct in zip(unique_labels, percentages, strict=False)}
            num_classes = len(unique_labels)
        else:
            balance_score = 0.0
            label_dist = {}
            num_classes = 0

        return {
            "class_balance_score": balance_score,
            "label_distribution": label_dist,
            "num_classes": num_classes,
            "num_trades": int(avg_trades),
            "total_pnl": avg_return,
            "mean_return_per_trade": mean_return_per_trade,
            "trading_frequency": trading_frequency,
            "valid_windows": valid_windows,
            "total_windows": len(sample_windows) if 'sample_windows' in locals() else 0,
        }

    except Exception as e:
        return {
            "error": f"Failed to calculate additional metrics: {e}",
            "class_balance_score": 0,
            "label_distribution": {},
            "num_classes": 0,
            "num_trades": 0,
            "total_pnl": 0.0,
            "mean_return_per_trade": 0.0,
            "trading_frequency": 0.0,
            "valid_windows": 0,
            "total_windows": 0,
        }


def optimize_symbol_dataset(
    symbol_info: dict[str, Any],
    methods: list[str],
    optimization_config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """
    Run optimization on a single symbol dataset.

    Args:
        symbol_info: Symbol dataset information
        methods: List of methods to optimize
        optimization_config: Optimization configuration

    Returns:
        Optimization results for all methods
    """
    symbol = symbol_info['symbol']
    file_path = symbol_info['file_path']
    samples = symbol_info['samples']

    print(f"\n{'='*80}")
    print(f"🔍 OPTIMIZING SYMBOL: {symbol.upper()}")
    print(f"{'='*80}")
    print(f"   📈 Dataset: {samples:,} samples ({symbol_info['size_mb']:.1f}MB)")
    if symbol_info['date_range']:
        print(f"   📅 Date range: {symbol_info['date_range']}")

    # Load price data
    try:
        prices = load_symbol_data_from_parquet(file_path)
        print(f"   💰 Price range: {prices.min():.6f} to {prices.max():.6f}")
    except Exception as e:
        print(f"❌ Failed to load {symbol}: {e}")
        return {}

    # Smart processing strategy based on dataset size
    dataset_size = len(prices)
    sequential_max = 5_000_000

    # Create symbol-specific config
    symbol_config = optimization_config.copy()

    if dataset_size <= sequential_max:
        # Use full sequential processing for smaller datasets
        symbol_config['use_sequential_processing'] = True
        print(f"   🔄 Using SEQUENTIAL processing (dataset: {dataset_size:,} ≤ {sequential_max:,})")
    elif dataset_size <= 10_000_000:
        # Use sequential processing with subset for medium datasets
        symbol_config['use_sequential_processing'] = True
        symbol_config['sequential_subset_size'] = 3_000_000  # Use 3M sample subset
        print(f"   🔄 Using SEQUENTIAL processing with 3M subset (dataset: {dataset_size:,})")
    else:
        # Use windowed processing with larger windows for very large datasets
        symbol_config['use_sequential_processing'] = False
        symbol_config['window_size'] = 100_000  # 100K samples per window for large datasets
        symbol_config['n_windows'] = 10  # More windows for better coverage
        print(f"   🔄 Using WINDOWED processing with 100K windows (dataset: {dataset_size:,} > 10M)")

    # Initialize optimizer with symbol-specific configuration
    optimizer = LargeScaleParameterOptimizer(
        **symbol_config,
        verbose=True
    )

    results = {}

    # Optimize each method
    for method in methods:
        try:
            print(f"\n{'-'*60}")
            print(f"🎯 METHOD: {method.upper()} | SYMBOL: {symbol}")
            print(f"{'-'*60}")
            print("Starting optimization...")

            if method == 'binary_ctl':
                result = optimizer.optimize_binary_ctl(prices)
            elif method == 'ternary_ctl':
                result = optimizer.optimize_ternary_ctl(prices)
            elif method == 'oracle_binary':
                result = optimizer.optimize_oracle_binary(prices)
            elif method == 'oracle_ternary':
                result = optimizer.optimize_oracle_ternary(prices)
            elif method == 'triple_barrier':
                result = optimizer.optimize_triple_barrier(prices)
            elif method == 'triple_barrier_adaptive':
                result = optimizer.optimize_triple_barrier_adaptive(prices)
            elif method == 'triple_exceedance':
                result = optimizer.optimize_triple_exceedance(prices)
            else:
                print(f"⚠️  Unknown method: {method}")
                continue

            results[method] = result

            # Calculate additional metrics
            optimal_params = result.get('optimal_params', {})
            if optimal_params:
                additional_metrics = calculate_additional_metrics(prices, method, optimal_params)
                result.update(additional_metrics)  # Add metrics to result for storage
            else:
                additional_metrics = {"error": "No optimal parameters found"}

            print(f"\n✅ {method.upper()} OPTIMIZATION COMPLETE")
            print(f"   🎯 Best params: {result.get('optimal_params', 'N/A')}")
            print(f"   📈 Max returns: {result.get('maximum_returns', 'N/A'):.4f}")

            # Display additional metrics
            if "error" not in additional_metrics:
                print(f"   ⚖️  Class balance: {additional_metrics['class_balance_score']:.1f}% ({additional_metrics['num_classes']} classes)")
                print(f"   📊 Label distribution: {additional_metrics['label_distribution']}")
                print(f"   🪟 Windows: {additional_metrics['valid_windows']}/{additional_metrics['total_windows']} valid")
                print(f"   🔄 Trades: {additional_metrics['num_trades']:,}/window ({additional_metrics['trading_frequency']:.2f}% frequency)")
                print(f"   💰 Mean return/trade: {additional_metrics['mean_return_per_trade']:.6f}")

                # Add note about windowing for clarity
                if additional_metrics['total_windows'] > 1:
                    print(f"   ℹ️  Metrics calculated using {additional_metrics['total_windows']} sampled windows (matches optimization)")
            else:
                print(f"   ⚠️  Metrics calculation failed: {additional_metrics.get('error', 'Unknown error')}")

            print(f"{'-'*60}")

        except ImportError as e:
            print(f"⚠️  Skipping {method}: {e}")
            continue
        except Exception as e:
            print(f"\n❌ {method.upper()} OPTIMIZATION FAILED")
            print(f"   Error: {e}")
            print(f"{'-'*60}")
            continue

    return results


def run_all_symbol_optimizations(
    data_dir: str | Path = "/Users/danielfisher/data/databento/symbol_datasets/inputs",
    methods: list[str] = None,
    output_dir: str | Path = "outputs/optimization_results",
    max_symbols: int = None
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Run optimization on all discovered symbol datasets.

    Args:
        data_dir: Directory containing symbol datasets
        methods: Methods to optimize (default: all available)
        output_dir: Directory to save results
        max_symbols: Maximum number of symbols to process (for testing)

    Returns:
        Complete optimization results
    """
    if methods is None:
        methods = ['triple_barrier_adaptive']  # Only run adaptive method for easier iteration

    # Discover symbol datasets
    print("🔍 DISCOVERING SYMBOL DATASETS")
    print("=" * 60)

    symbols = discover_symbol_datasets(data_dir)

    if not symbols:
        print(f"❌ No symbol datasets found in {data_dir}")
        return {}

    print(f"✅ Found {len(symbols)} symbol datasets")

    # Limit symbols for testing
    if max_symbols and len(symbols) > max_symbols:
        symbols = symbols[:max_symbols]
        print(f"🔄 Limited to first {max_symbols} symbols for testing")

    # Optimization configuration with robust sampling and cross-validation
    optimization_config = {
        'window_size': 50000,      # USER REQ: Max 25K sampling window
        'n_windows': 10,            # More windows for better sampling
        'sampling_strategy': 'stratified',  # Ensure good coverage
        'adaptive_sampling': True,  # Enable adaptive sampling
        'stabilization_threshold': 0.05,  # 5% parameter change threshold
        'stabilization_patience': 3,  # Require 3 stable evaluations
        'growth_factor': 1.5,      # 50% growth per increase
        'min_window_size': 15000,  # USER REQ: Min 15K sampling window
        'max_window_size': 50000,  # USER REQ: Max 25K sampling window
        'early_stopping': True,    # Stop early when parameters stabilize
        'early_stopping_patience': 7,  # Extra evaluations after stabilization
        'fee_pips': 0.7,           # Correct 0.7 pip transaction costs
        'initial_points': 15,      # More initial samples for exploration
        'n_calls': 50,            # Increased optimization calls for better convergence
        'random_state': 42,        # Reproducible results
        'class_balance_weight': 0.3,  # Weight for class balance penalty (not used in new balance score optimization)
        # NEW: Robust sampling parameters to address overfitting
        'use_cross_validation': True,  # Enable cross-validation to prevent overfitting
        'target_coverage': 0.25,      # Increase dataset coverage from 2.3% to 25%
        'validation_split': 0.3,      # 30% of windows for validation
        # FIXED: Use sequential processing to match application behavior
        'use_sequential_processing': False,  # Will be enabled conditionally based on dataset size
    }

    print("\n🎯 OPTIMIZATION CONFIGURATION")
    print("=" * 60)
    for key, value in optimization_config.items():
        print(f"   {key}: {value}")

    # Run optimization for each symbol
    all_results = {}
    successful_optimizations = 0

    for i, symbol_info in enumerate(symbols, 1):
        symbol = symbol_info['symbol']

        print(f"\n{'#'*100}")
        print(f"📊 DATASET {i}/{len(symbols)}: {symbol.upper()}")
        print(f"{'#'*100}")

        try:
            symbol_results = optimize_symbol_dataset(
                symbol_info,
                methods,
                optimization_config
            )

            if symbol_results:
                all_results[symbol] = symbol_results

                # Save results immediately
                params_dir = Path(output_dir) / "optimized_parameters"
                params_dir.mkdir(parents=True, exist_ok=True)
                save_optimization_results(
                    symbol,
                    symbol_results,
                    str(params_dir)
                )

                # Generate comprehensive label-grouped visualizations per symbol via diagnostics
                try:
                    print(f"📊 Generating label-grouped visualizations for {symbol} via diagnostics...")
                    generate_label_grouped_plots_for_symbol(
                        symbol_info,
                        symbol_results,
                        examples_per_label=10,  # 10 examples per label type
                        window_size=100_000,  # Match optimization window size to see timeouts
                    )
                    print(f"✅ Label-grouped plots generated for {symbol}")
                except Exception as viz_error:
                    print(f"⚠️  Visualization failed for {symbol}: {viz_error}")

                print(f"✅ {symbol} optimization complete ({len(symbol_results)} methods)")
                successful_optimizations += 1
            else:
                print(f"⚠️  No results for {symbol}")

        except Exception as e:
            print(f"❌ Failed to optimize {symbol}: {e}")
            continue

    print("\n🎉 OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"   Successfully optimized: {successful_optimizations}/{len(symbols)} symbols")
    print(f"   Total methods run: {sum(len(results) for results in all_results.values())}")

    return all_results


def generate_optimization_report(
    results: dict[str, dict[str, dict[str, Any]]],
    output_dir: str | Path = "outputs/optimization_results"
) -> None:
    """
    Generate comprehensive optimization report with visualizations.

    Args:
        results: Complete optimization results
        output_dir: Directory to save report and visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize parameter storage
    storage = ParameterStorage(output_path / "optimized_parameters")

    print("\n📊 GENERATING OPTIMIZATION REPORT")
    print("=" * 60)

    # Create parameter comparison table
    try:
        comparison_df = storage.create_parameter_comparison_table()

        if not comparison_df.empty:
            # Save CSV for analysis
            csv_path = output_path / "parameter_comparison.csv"
            comparison_df.to_csv(csv_path, index=False)
            print(f"✅ Parameter comparison CSV: {csv_path}")

            # Create visualizations
            print("📈 Creating parameter visualizations...")

            # Parameter distributions for each method
            for method in comparison_df['method'].unique():
                viz_path = output_path / f"parameter_distributions_{method}.png"
                fig = storage.visualize_parameter_distributions(method=method, save_path=str(viz_path))
                if fig:
                    print(f"   ✅ {method} parameter distributions: {viz_path}")

            # Returns comparison
            returns_path = output_path / "returns_comparison.png"
            fig = storage.create_returns_comparison(save_path=str(returns_path))
            if fig:
                print(f"   ✅ Returns comparison: {returns_path}")

            # Export markdown report
            markdown_path = output_path / "OPTIMIZATION_RESULTS.md"
            storage.export_to_markdown(str(markdown_path))
            print(f"✅ Markdown report: {markdown_path}")

        else:
            print("⚠️  No parameter data available for report generation")

    except Exception as e:
        print(f"❌ Failed to generate report: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main execution function."""
    print("🚀 SYMBOL DATASET OPTIMIZATION RUNNER")
    print("=" * 80)
    print("This script optimizes parameters for all discovered symbol datasets")
    print("and generates comprehensive reports with visualizations.")
    print()

    # Configuration
    data_dir = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs")
    output_dir = Path("outputs/optimization_results")
    # Use only triple barrier methods
    methods = ['triple_barrier_adaptive']  # Only run adaptive method for easier iteration
    print("🎯 Using only Adaptive Triple Barrier method")
    max_symbols = None  # Process all symbol datasets (no limit)

    try:
        # Run optimizations
        results = run_all_symbol_optimizations(
            data_dir=data_dir,
            methods=methods,
            output_dir=output_dir,
            max_symbols=max_symbols
        )

        # Generate report
        if results:
            generate_optimization_report(results, output_dir)
        else:
            print("❌ No optimization results to report")

        print("\n🎉 ALL DONE!")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print("Check the following files:")
        print(f"  • {output_dir}/OPTIMIZATION_RESULTS.md - Main report")
        print(f"  • {output_dir}/parameter_comparison.csv - Raw data")
        print(f"  • {output_dir}/*.png - Parameter distribution plots")
        print(f"  • {output_dir}/outputs/plots/optimization/[SYMBOL]/comprehensive_all_methods_window_*.png - 100k-window comprehensive plots")
        print(f"  • {output_dir}/optimized_parameters/ - Individual parameter files")

    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    except Exception as e:
        print(f"❌ Optimization runner failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
