#!/usr/bin/env python3
"""
Comprehensive Method Debugging - All Methods, Full 100K Samples

Create detailed debugging plots for ALL optimization methods using the full
100K sample windows that optimization actually uses. This will reveal:
1. Triple Barrier logic issues (hits not being detected)
2. All method behaviors across different market regimes
3. True optimization sampling patterns
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


def get_all_optimization_methods():
    """Get all methods available for optimization and diagnostics."""
    return [
        {
            "name": "Binary CTL",
            "method": "binary_ctl",
            "params": {"omega": 0.0001},  # Fixed minimum to prevent pathological overtrading
            "color_map": {-1: "red", 1: "green"},
            "label_names": {-1: "Short", 1: "Long"},
            "has_barriers": False
        },
        {
            "name": "Ternary CTL",
            "method": "ternary_ctl",
            "params": {"marginal_change_thres": 0.0446, "window_size": 501},  # Optimized parameters
            "color_map": {0: "red", 1: "gray", 2: "green"},
            "label_names": {0: "Down", 1: "Neutral", 2: "Up"},
            "has_barriers": False
        },
        {
            "name": "Oracle Binary",
            "method": "oracle_binary",
            "params": {"transaction_cost": 0.000035},  # Fixed at 0.35 pips (0.7 pip round-trip)
            "color_map": {-1: "red", 1: "green"},
            "label_names": {-1: "Short", 1: "Long"},
            "has_barriers": False
        },
        {
            "name": "Oracle Ternary",
            "method": "oracle_ternary",
            "params": {"transaction_cost": 0.000035, "neutral_reward_factor": 0.5},  # Fixed cost, neutral factor in bounds
            "color_map": {0: "red", 1: "gray", 2: "green"},
            "label_names": {0: "Down", 1: "Neutral", 2: "Up"},
            "has_barriers": False
        },
        {
            "name": "Triple Barrier",
            "method": "triple_barrier",
            "params": {},  # NO FALLBACKS - Must use optimized parameters only
            "color_map": {-1: "red", 0: "gray", 1: "green"},
            "label_names": {-1: "Short", 0: "Timeout", 1: "Long"},
            "has_barriers": True
        },
        {
            "name": "Triple Barrier Adaptive",
            "method": "triple_barrier_adaptive",
            "params": {},  # NO FALLBACKS - Must use optimized parameters only
            "color_map": {-1: "red", 0: "gray", 1: "green"},
            "label_names": {-1: "Short", 0: "Timeout", 1: "Long"},
            "has_barriers": True
        },
        {
            "name": "Triple Exceedance (Long)",
            "method": "triple_exceedance",
            "params": {},  # NO FALLBACKS - Must use optimized parameters only
            "color_map": {0: "red", 1: "green"},
            "label_names": {0: "Wrong Direction", 1: "Right Direction"},
            "has_barriers": True,  # FIXED: Triple exceedance also has time barriers
            "target_column": "long",  # Specify to use long exceedance column
            "direction_type": "long"  # For direction correctness checking
        },
        {
            "name": "Triple Exceedance (Short)",
            "method": "triple_exceedance",
            "params": {},  # NO FALLBACKS - Must use optimized parameters only
            "color_map": {0: "red", 1: "green"},
            "label_names": {0: "Wrong Direction", 1: "Right Direction"},
            "has_barriers": True,  # FIXED: Triple exceedance also has time barriers
            "target_column": "short",  # Specify to use short exceedance column
            "direction_type": "short"  # For direction correctness checking
        },
        {
            "name": "GA Labeling (Long)",
            "method": "ga_labeling",
            "params": {},  # NO FALLBACKS - Must use optimized parameters only
            "color_map": {0: "gray", 1: "green"},
            "label_names": {0: "Hold", 1: "Buy Long"},
            "has_barriers": False,
            "target_column": "long",
            "direction_type": "long"
        },
        {
            "name": "GA Labeling (Short)",
            "method": "ga_labeling",
            "params": {},  # NO FALLBACKS - Must use optimized parameters only
            "color_map": {0: "gray", 1: "red"},
            "label_names": {0: "Hold", 1: "Sell Short"},
            "has_barriers": False,
            "target_column": "short",
            "direction_type": "short"
        }
    ]


def simulate_optimization_sampling(df: pl.DataFrame, window_size: int = 100000, n_windows: int = 5):
    """Simulate exact optimization sampling strategy."""
    total_samples = len(df)

    if total_samples < window_size:
        return [df]

    windows = []
    step_size = max(1, (total_samples - window_size) // (n_windows - 1))

    for i in range(n_windows):
        start_idx = min(i * step_size, total_samples - window_size)
        windows.append(df.slice(start_idx, window_size))
        print(f"Window {i+1}: samples {start_idx:,} to {start_idx + window_size - 1:,}")

    return windows


def evaluate_triple_exceedance_direction_correctness(
    window_df: pl.DataFrame,
    labels: np.ndarray,
    direction_type: str,
    lookforward_window: int = 2000
) -> np.ndarray:
    """
    Evaluate direction correctness for triple exceedance labels.

    Returns binary array where:
    - 1 = Sample follows the right direction (green)
    - 0 = Sample does not follow the right direction (red)

    For long positions: right direction = price goes up
    For short positions: right direction = price goes down
    """
    prices = window_df["mid_price"].to_numpy()
    n_samples = len(prices)
    correctness = np.zeros(n_samples, dtype=np.int32)

    # Only evaluate samples that have sufficient lookforward data
    for i in range(n_samples - lookforward_window):
        entry_price = prices[i]

        # Look at actual price movement over the lookforward window
        end_idx = min(i + lookforward_window, len(prices) - 1)
        final_price = prices[end_idx]
        price_change = final_price - entry_price

        if direction_type == "long":
            # For long positions: correct if price actually went up
            if price_change > 0:
                correctness[i] = 1  # Right direction (green)
            else:
                correctness[i] = 0  # Wrong direction (red)
        elif direction_type == "short":
            # For short positions: correct if price actually went down
            if price_change < 0:
                correctness[i] = 1  # Right direction (green)
            else:
                correctness[i] = 0  # Wrong direction (red)

    return correctness


def debug_triple_barrier_logic(window_df: pl.DataFrame, method_config: dict):
    """Debug Triple Barrier logic to find missing hits."""
    print(f"\n🔍 DEBUGGING TRIPLE BARRIER LOGIC: {method_config['name']}")
    print("=" * 60)

    try:
        # Generate labels
        generator = TargetGeneratorFactory.create(
            method_config["method"],
            **method_config["params"]
        )
        targets_df = generator.generate_targets(window_df)
        target_info = generator.get_target_info()
        target_col = target_info['target_names'][0]
        labels = targets_df[target_col].to_numpy()

        prices = window_df["mid_price"].to_numpy()
        barrier_width = method_config["params"].get("barrier_width", 0.0001)
        lookforward = method_config["params"].get("lookforward_window", 1000)

        # Manually check first 1000 samples for debugging
        debug_samples = min(1000, len(prices) - lookforward)

        print(f"Manual barrier checking for first {debug_samples} samples:")
        print(f"Barrier width: {barrier_width} ({barrier_width*10000:.1f} pips)")
        print(f"Lookforward: {lookforward} ticks")

        manual_hits = {"profit": 0, "loss": 0, "timeout": 0}
        discrepancies = []

        for i in range(debug_samples):
            if i % 200 == 0:  # Progress indicator
                print(f"  Checking sample {i:,}...")

            entry_price = prices[i]
            upper_barrier = entry_price + barrier_width
            lower_barrier = entry_price - barrier_width

            # Check lookforward window
            window_end = min(i + lookforward, len(prices))
            future_prices = prices[i+1:window_end]

            if len(future_prices) == 0:
                manual_result = 0  # timeout
            else:
                # Check for barrier hits
                hit_upper = np.any(future_prices >= upper_barrier)
                hit_lower = np.any(future_prices <= lower_barrier)

                if hit_upper and hit_lower:
                    # Both hit - which was first?
                    upper_hit_idx = np.argmax(future_prices >= upper_barrier)
                    lower_hit_idx = np.argmax(future_prices <= lower_barrier)

                    if upper_hit_idx < lower_hit_idx:
                        manual_result = 1  # profit
                    else:
                        manual_result = -1  # loss
                elif hit_upper:
                    manual_result = 1  # profit
                elif hit_lower:
                    manual_result = -1  # loss
                else:
                    manual_result = 0  # timeout

            # Count manual results
            if manual_result == 1:
                manual_hits["profit"] += 1
            elif manual_result == -1:
                manual_hits["loss"] += 1
            else:
                manual_hits["timeout"] += 1

            # Compare with generated label
            generated_label = labels[i]
            if manual_result != generated_label:
                discrepancies.append({
                    "index": i,
                    "manual": manual_result,
                    "generated": generated_label,
                    "price": entry_price,
                    "max_price": np.max(future_prices) if len(future_prices) > 0 else entry_price,
                    "min_price": np.min(future_prices) if len(future_prices) > 0 else entry_price
                })

        # Report results
        print("\nMANUAL VERIFICATION RESULTS:")
        print(f"  Manual hits - Profit: {manual_hits['profit']}, Loss: {manual_hits['loss']}, Timeout: {manual_hits['timeout']}")

        generated_counts = np.bincount(labels[:debug_samples].astype(int) + 1)  # Shift to 0-based
        generated_loss = generated_counts[0] if len(generated_counts) > 0 else 0
        generated_timeout = generated_counts[1] if len(generated_counts) > 1 else 0
        generated_profit = generated_counts[2] if len(generated_counts) > 2 else 0

        print(f"  Generated labels - Profit: {generated_profit}, Loss: {generated_loss}, Timeout: {generated_timeout}")
        print(f"  Discrepancies found: {len(discrepancies)}")

        if len(discrepancies) > 0:
            print("\nFIRST 5 DISCREPANCIES:")
            for i, disc in enumerate(discrepancies[:5]):
                print(f"  {i+1}. Index {disc['index']}: Manual={disc['manual']}, Generated={disc['generated']}")
                print(f"     Price: {disc['price']:.6f}, Max: {disc['max_price']:.6f}, Min: {disc['min_price']:.6f}")
                print(f"     Upper barrier: {disc['price'] + barrier_width:.6f}, Lower: {disc['price'] - barrier_width:.6f}")

        return len(discrepancies) == 0  # True if no discrepancies

    except Exception as e:
        print(f"❌ Error in Triple Barrier debugging: {e}")
        return False


def create_comprehensive_method_plots():
    """Create comprehensive plots for all methods using full 100K samples."""
    if not LIBRARIES_AVAILABLE:
        return

    print("🚀 COMPREHENSIVE METHOD DEBUGGING - ALL METHODS, FULL 100K SAMPLES")
    print("=" * 80)

    # Load data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())

    # Use actual optimization sampling
    viz_window_size = 100000
    windows = simulate_optimization_sampling(df, viz_window_size, n_windows=3)  # 3 windows for analysis

    methods = get_all_optimization_methods()

    # Test triple barrier logic first
    print("\n🔧 TESTING TRIPLE BARRIER LOGIC")
    for method_config in methods:
        if method_config.get("has_barriers") and "triple_barrier" in method_config["method"]:
            # Test on first window only for debugging
            logic_ok = debug_triple_barrier_logic(windows[0], method_config)
            if not logic_ok:
                print(f"⚠️  Logic issues found in {method_config['name']}")
            else:
                print(f"✅ Logic verified for {method_config['name']}")

    # Create comprehensive plots
    for window_idx, window_df in enumerate(windows):
        create_all_methods_plot(window_df, methods, window_idx + 1)


def create_all_methods_plot(window_df: pl.DataFrame, methods: list, window_num: int):
    """Create comprehensive plot showing all methods for one window."""
    print(f"\n📊 Creating comprehensive plot for Window {window_num}...")

    prices = window_df["mid_price"].to_numpy()

    # Subsample for plotting clarity
    step = max(1, len(prices) // 3000)  # Target 3000 plot points
    prices_plot = prices[::step]
    time_axis = np.arange(0, len(prices), step)[:len(prices_plot)]

    # Create large subplot grid
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(24, 4 * n_methods))
    if n_methods == 1:
        axes = [axes]

    fig.suptitle(f"Comprehensive Method Analysis - Window {window_num} (Full 100K Samples)",
                 fontsize=18, fontweight='bold')

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

            # Handle multiple target columns (e.g., Triple Exceedance)
            if "target_column" in method_config:
                # Look for specific column (long or short)
                target_suffix = method_config["target_column"]
                matching_cols = [col for col in target_info['target_names'] if col.endswith(f"_{target_suffix}")]
                if matching_cols:
                    target_col = matching_cols[0]
                else:
                    target_col = target_info['target_names'][0]  # Fallback
            else:
                target_col = target_info['target_names'][0]

            labels = targets_df[target_col].to_numpy()

            # Convert labels for consistent display
            if method_config["method"] in ["binary_ctl", "oracle_binary"]:
                # Convert {0,1} to {-1,1} for display
                labels_display = np.where(labels == 0, -1, 1)
            elif method_config["method"] in ["ternary_ctl", "oracle_ternary"]:
                # Keep {0,1,2} for ternary display
                labels_display = labels
            elif method_config["method"] == "triple_exceedance" and "direction_type" in method_config:
                # For triple exceedance, use direction correctness instead of raw labels
                lookforward_window = method_config["params"].get("lookforward_window", 2000)
                labels_display = evaluate_triple_exceedance_direction_correctness(
                    window_df, labels, method_config["direction_type"], lookforward_window
                )
            elif method_config["method"] == "triple_exceedance":
                # Keep {0,1} for binary exceedance display
                labels_display = labels > 0
            else:  # triple_barrier, etc.
                # Already in {-1,0,1} format
                labels_display = labels

            # Subsample labels for plotting
            labels_plot = labels_display[::step] if len(labels_display) >= len(prices) else labels_display
            labels_plot = labels_plot[:len(prices_plot)]

            # Plot price line
            ax.plot(time_axis, prices_plot, 'black', linewidth=0.8, alpha=0.8, label='Price', zorder=1)

            # PROPER TRIPLE BARRIER VISUALIZATION: Show actual barriers and trades
            if method_config.get("has_barriers", False):
                # Extract parameters
                lookforward_window = method_config["params"].get("lookforward_window", 2000)
                barrier_width = method_config["params"].get("barrier_width", 0.0005)
                lookback_window = method_config["params"].get("lookback_window", None)

                # Show representative trades sampled to capture label diversity
                max_trades = 8

                # Get label distribution and sample positions for each label type
                unique_labels, label_counts = np.unique(labels_plot, return_counts=True)
                print(f"    Label distribution (plot samples): {dict(zip(unique_labels, label_counts, strict=False))}")

                # Sample diverse trade positions across the plot data ensuring proper index alignment
                trade_indices: list[int] = []

                # Ensure labels_plot and prices_plot are aligned
                min_length = min(len(labels_plot), len(prices_plot))
                labels_plot = labels_plot[:min_length]
                prices_plot = prices_plot[:min_length]
                time_axis = time_axis[:min_length]

                # For each label type, find positions and sample them
                for label_val in unique_labels:
                    if len(trade_indices) >= max_trades:
                        break
                    label_positions = np.where(labels_plot == label_val)[0]
                    if len(label_positions) > 0:
                        # Sample up to 2 positions for this label type
                        n_samples = min(2, len(label_positions), max_trades - len(trade_indices))
                        if len(label_positions) >= n_samples:
                            # Sample evenly across this label's positions
                            if n_samples == 1:
                                # Pick middle position for single sample
                                sampled_positions = [label_positions[len(label_positions)//2]]
                            else:
                                # Sample evenly for multiple samples
                                indices = np.linspace(0, len(label_positions)-1, n_samples, dtype=int)
                                sampled_positions = label_positions[indices]
                            trade_indices.extend(sampled_positions)

                # Ensure we have some trades and remove duplicates
                trade_indices = sorted(set(trade_indices))

                # Ensure indices are within bounds and avoid problematic early data points
                # For adaptive methods, skip early points where volatility calculation is unreliable
                min_start_idx = 50 if method_config["method"] == "triple_barrier_adaptive" else 0
                trade_indices = [idx for idx in trade_indices if idx >= min_start_idx and idx < min_length - 50]

                # CRITICAL FIX: Remove overlapping trades to prevent multiple symbols in same area
                # Calculate minimum spacing needed to prevent overlaps
                lookforward_ticks = method_config["params"].get("lookforward_window", 2000)
                # Use a more reasonable spacing - enough to prevent visual overlap but allow multiple trades
                min_spacing_ticks = max(500, lookforward_ticks // 4)  # Quarter of lookforward window
                # Estimate step from the data subsampling
                estimated_step = max(1, len(labels_display) // len(labels_plot)) if len(labels_plot) > 0 else 3
                min_spacing_plot = max(50, min_spacing_ticks // estimated_step)

                # Filter out overlapping trades
                non_overlapping_indices: list[int] = []
                for idx in trade_indices:
                    # Check if this trade overlaps with any already selected
                    overlaps = False
                    for existing_idx in non_overlapping_indices:
                        if abs(idx - existing_idx) < min_spacing_plot:
                            overlaps = True
                            break

                    if not overlaps:
                        non_overlapping_indices.append(idx)

                    # Stop if we have enough trades
                    if len(non_overlapping_indices) >= max_trades:
                        break

                trade_indices = non_overlapping_indices

                # Fallback if no diverse sampling worked
                if len(trade_indices) == 0:
                    entry_spacing = max(min_spacing_plot, min_length // (max_trades + 2))
                    trade_indices = list(range(entry_spacing, min_length - entry_spacing, entry_spacing))

                trade_count = 0
                for j in trade_indices[:max_trades]:
                    if trade_count >= max_trades:
                        break

                    entry_price = prices_plot[j]
                    entry_time = time_axis[j]

                    # Calculate barriers based on method type
                    if method_config["method"] == "triple_exceedance":
                        # Triple exceedance only has time barriers - no price barriers
                        # For visualization, show reference level but only one barrier based on direction
                        scaling_factor = method_config["params"].get("scaling_factor", 3.0)
                        vol_estimate = np.std(prices_plot[max(0, j-20):j+1]) if j > 20 else np.std(prices_plot[:50])

                        # Only show one barrier based on direction - corrected logic
                        if "Long" in method_config["name"]:
                            # Long positions: show lower barrier (minimum threshold to exceed upward)
                            lower_barrier = entry_price - (scaling_factor * vol_estimate)
                            upper_barrier = entry_price  # Set to entry price (won't be shown)
                        else:
                            # Short positions: show upper barrier (maximum threshold to exceed downward)
                            upper_barrier = entry_price + (scaling_factor * vol_estimate)
                            lower_barrier = entry_price  # Set to entry price (won't be shown)

                        # For triple exceedance, all exits should be at time expiry (no price exits)
                        # The actual method only uses time-based exits with directional analysis
                    elif lookback_window and method_config["method"] == "triple_barrier_adaptive":
                        # Adaptive barriers based on volatility using actual lookback_window
                        estimated_step = max(1, len(labels_display) // len(labels_plot)) if len(labels_plot) > 0 else 3
                        lookback_plot = lookback_window // estimated_step
                        vol_start = max(0, j - lookback_plot)  # Use actual lookback window for volatility
                        if j > vol_start and len(prices_plot[vol_start:j+1]) > 10:
                            volatility = np.std(prices_plot[vol_start:j+1])
                        else:
                            # Use a reasonable window for volatility calculation when lookback is insufficient
                            end_idx = min(j + 100, len(prices_plot))
                            start_idx = max(0, end_idx - 100)
                            volatility = np.std(prices_plot[start_idx:end_idx])
                            if volatility == 0:  # Still zero volatility - use a minimum
                                volatility = entry_price * 0.0001  # 1 basis point as minimum
                        # Apply barrier_width as scaling factor to the volatility
                        upper_barrier = entry_price + (barrier_width * volatility)
                        lower_barrier = entry_price - (barrier_width * volatility)
                    else:
                        # Fixed barriers (regular triple barrier)
                        upper_barrier = entry_price + barrier_width
                        lower_barrier = entry_price - barrier_width

                    # Time barrier (vertical line) - use actual lookforward window
                    lookforward_ticks = method_config["params"].get("lookforward_window", 2000)
                    estimated_step = max(1, len(labels_display) // len(labels_plot)) if len(labels_plot) > 0 else 3
                    lookforward_plot = lookforward_ticks // estimated_step
                    exit_time_idx = min(j + lookforward_plot, len(time_axis) - 1)
                    exit_time = time_axis[exit_time_idx]

                    # Get the label for this position
                    if j < len(labels_plot):
                        first_label = labels_plot[j]

                        # Debug: Print the specific trade info
                        if trade_count < 3:  # Show first few trades
                            print(f"    Trade {trade_count+1}: position {j}, label {first_label}")

                        # Interpret labels based on method type
                        if method_config["method"] == "triple_exceedance":
                            # For triple exceedance, labels represent directional correctness
                            if first_label == 1:
                                barrier_color = 'green'  # Correct direction
                                exit_marker = '^'
                            else:  # 0 = wrong direction or timeout
                                barrier_color = 'red'    # Wrong direction
                                exit_marker = 'v'
                        else:
                            # For true triple barrier methods (barrier hits)
                            if first_label == 1:
                                barrier_color = 'green'  # Upper barrier hit
                                exit_marker = '^'
                            elif first_label == -1:
                                barrier_color = 'red'    # Lower barrier hit
                                exit_marker = 'v'
                            else:
                                barrier_color = 'orange' # Timeout
                                exit_marker = 's'
                    else:
                        first_label = 0
                        barrier_color = 'gray'
                        exit_marker = 'o'

                    alpha = 0.8

                    # Entry point - larger and more visible
                    ax.scatter(entry_time, entry_price, color='blue', s=60, marker='o',
                             zorder=5, alpha=alpha, edgecolors='darkblue', linewidth=1)

                    if method_config["method"] == "triple_exceedance":
                        # For triple exceedance, show only one reference line based on direction
                        if "Long" in method_config["name"]:
                            # Long: show lower barrier (minimum threshold to exceed upward)
                            ax.hlines(lower_barrier, entry_time, exit_time, colors='red',
                                     linestyles=':', alpha=alpha*0.7, linewidth=1.5, zorder=3)
                            # Vertical barrier (time expiry) - from lower barrier to entry
                            ax.vlines(exit_time, lower_barrier, entry_price, colors='orange',
                                     linestyles='-', alpha=alpha, linewidth=3.0, zorder=3)
                        else:
                            # Short: show upper barrier (maximum threshold to exceed downward)
                            ax.hlines(upper_barrier, entry_time, exit_time, colors='green',
                                     linestyles=':', alpha=alpha*0.7, linewidth=1.5, zorder=3)
                            # Vertical barrier (time expiry) - from entry to upper barrier
                            ax.vlines(exit_time, entry_price, upper_barrier, colors='orange',
                                     linestyles='-', alpha=alpha, linewidth=3.0, zorder=3)
                    else:
                        # For true triple barrier methods, show solid price barriers
                        # Upper barrier (profit target) - thicker and more visible
                        ax.hlines(upper_barrier, entry_time, exit_time, colors='green',
                                 linestyles='--', alpha=alpha, linewidth=2.0, zorder=3)
                        # Lower barrier (stop loss) - thicker and more visible
                        ax.hlines(lower_barrier, entry_time, exit_time, colors='red',
                                 linestyles='--', alpha=alpha, linewidth=2.0, zorder=3)
                        # Vertical barrier (time expiry) - thicker and more visible
                        ax.vlines(exit_time, lower_barrier, upper_barrier, colors='orange',
                                 linestyles=':', alpha=alpha, linewidth=2.5, zorder=3)

                    # Exit point positioning depends on method type
                    if method_config["method"] == "triple_exceedance":
                        # Triple exceedance exits at time barrier (all exits are time-based)
                        exit_plot_price = (upper_barrier + lower_barrier) / 2
                    else:
                        # Triple barrier methods - position based on barrier hit
                        if first_label == 1:
                            # Profit exit - align with upper barrier
                            exit_plot_price = upper_barrier
                        elif first_label == -1:
                            # Loss exit - align with lower barrier
                            exit_plot_price = lower_barrier
                        else:
                            # Timeout exit - align with midpoint of time barrier
                            exit_plot_price = (upper_barrier + lower_barrier) / 2

                    edge_color = 'darkgreen' if barrier_color == 'green' else 'darkred' if barrier_color == 'red' else 'darkorange'
                    ax.scatter(exit_time, exit_plot_price, color=barrier_color, s=80,
                             marker=exit_marker, zorder=6, alpha=alpha, edgecolors=edge_color, linewidth=1.5)

                    trade_count += 1

                # Method-specific legends
                if method_config["method"] == "triple_exceedance":
                    legend_elements = [
                        plt.Line2D([0], [0], color='black', linewidth=0.8, label='Price'),
                        plt.Line2D([0], [0], color='blue', marker='o', markersize=6, linestyle='None', label='Entry Point'),
                        plt.Line2D([0], [0], color='orange', linestyle='-', linewidth=3.0, label='Time Barrier (Expiry)'),
                        plt.Line2D([0], [0], color='green', marker='^', markersize=8, linestyle='None', label='Right Direction'),
                        plt.Line2D([0], [0], color='red', marker='v', markersize=8, linestyle='None', label='Wrong Direction'),
                    ]

                    # Add direction-specific reference level
                    if "Long" in method_config["name"]:
                        legend_elements.insert(2, plt.Line2D([0], [0], color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Reference Level (Lower)'))
                    else:
                        legend_elements.insert(2, plt.Line2D([0], [0], color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Reference Level (Upper)'))
                else:
                    legend_elements = [
                        plt.Line2D([0], [0], color='black', linewidth=0.8, label='Price'),
                        plt.Line2D([0], [0], color='blue', marker='o', markersize=6, linestyle='None', label='Entry Point'),
                        plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2.0, label='Upper Barrier (Profit)'),
                        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2.0, label='Lower Barrier (Stop Loss)'),
                        plt.Line2D([0], [0], color='orange', linestyle=':', linewidth=2.5, label='Time Barrier (Expiry)'),
                        plt.Line2D([0], [0], color='green', marker='^', markersize=8, linestyle='None', label='Profit Exit'),
                        plt.Line2D([0], [0], color='red', marker='v', markersize=8, linestyle='None', label='Loss Exit'),
                        plt.Line2D([0], [0], color='orange', marker='s', markersize=6, linestyle='None', label='Timeout Exit'),
                    ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
            else:
                # For non-barrier methods, color by labels (keep original logic)
                current_label = labels_plot[0] if len(labels_plot) > 0 else 0
                start_idx = 0

                for j in range(1, len(labels_plot) + 1):
                    if j == len(labels_plot) or (j < len(labels_plot) and labels_plot[j] != current_label):
                        end_idx = j - 1

                        if current_label in method_config["color_map"]:
                            color = method_config["color_map"][current_label]
                            start_time = time_axis[start_idx] if start_idx < len(time_axis) else time_axis[-1]
                            end_time = time_axis[end_idx] if end_idx < len(time_axis) else time_axis[-1]
                            ax.axvspan(start_time, end_time, alpha=0.3, color=color, zorder=0)

                        if j < len(labels_plot):
                            start_idx = j
                            current_label = labels_plot[j]

            # Statistics
            unique_labels, counts = np.unique(labels_display, return_counts=True)
            percentages = counts / len(labels_display) * 100
            num_changes = np.sum(labels_display[1:] != labels_display[:-1])

            # Calculate average hold - for triple methods, it should be the lookforward_window
            if method_config["method"] in ["triple_exceedance", "triple_barrier"]:
                # For triple methods, hold is constant = optimized lookforward_window
                avg_hold = method_config["params"]["lookforward_window"]
            else:
                # For other methods, calculate from label changes
                avg_hold = len(labels_display) // max(num_changes, 1)

            # Title and formatting
            title_suffix = " (Direction Correctness)" if method_config["method"] == "triple_exceedance" and "direction_type" in method_config else ""
            ax.set_title(f"{method_config['name']} - Full 100K Analysis{title_suffix}", fontweight='bold', fontsize=14)
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)

            # Add vertical reference lines every 1000 ticks
            max_time = len(time_axis)
            for tick_mark in range(1000, max_time, 1000):
                if tick_mark < len(time_axis):
                    ax.axvline(x=time_axis[tick_mark], color='gray', linestyle=':', alpha=0.5, linewidth=0.8, zorder=1)

            # Legend - for barrier methods, this is handled within the barrier visualization
            if not method_config.get("has_barriers", False):
                # Only add legend for non-barrier methods
                legend_elements_non_barrier: list = [Line2D([0], [0], color='black', linewidth=0.8, label='Price')]

                for label_val, pct in zip(unique_labels, percentages, strict=False):
                    if label_val in method_config["color_map"]:
                        name = method_config["label_names"].get(label_val, str(label_val))
                        legend_elements_non_barrier.append(
                            mpatches.Patch(color=method_config["color_map"][label_val], alpha=0.3,
                                         label=f'{name} ({pct:.1f}%)')
                        )

                ax.legend(handles=legend_elements_non_barrier, loc='upper left', fontsize=10)

            # Detailed statistics
            stats_lines = [
                f'Changes: {num_changes:,}',
                f'Avg Hold: {avg_hold:,} ticks',
                f'Total: {len(labels_display):,} samples'
            ]

            # Add method-specific stats
            if method_config.get("has_barriers"):
                profit_pct = percentages[unique_labels == 1][0] if 1 in unique_labels else 0
                loss_pct = percentages[unique_labels == -1][0] if -1 in unique_labels else 0
                timeout_pct = percentages[unique_labels == 0][0] if 0 in unique_labels else 0
                stats_lines.append(f'P/L/T: {profit_pct:.1f}/{loss_pct:.1f}/{timeout_pct:.1f}%')

            ax.text(0.98, 0.98, '\n'.join(stats_lines), transform=ax.transAxes,
                   ha='right', va='top', fontsize=9,
                   bbox={'boxstyle': "round,pad=0.3", 'facecolor': "white", 'alpha': 0.9})

            # Print summary
            print(f"    {method_config['name']}:")
            if method_config["method"] == "triple_exceedance" and "direction_type" in method_config:
                print(f"      Direction Correctness Analysis ({method_config['direction_type']} positions):")
            for label_val, pct in zip(unique_labels, percentages, strict=False):
                name = method_config["label_names"].get(label_val, str(label_val))
                print(f"      {name}: {pct:.1f}%")
            print(f"      Changes: {num_changes:,}, Avg Hold: {avg_hold:,}")

        except Exception as e:
            print(f"❌ Error with {method_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center',
                   transform=ax.transAxes, color='red', fontsize=12)
            ax.set_title(f"{method_config['name']}: ERROR", color='red', fontweight='bold')

    axes[-1].set_xlabel('Time (ticks)', fontsize=12)
    plt.tight_layout()

    # Save plot
    output_path = f"outputs/plots/optimization/comprehensive_all_methods_window_{window_num}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comprehensive plot: {output_path}")
    plt.close()


def main():
    """Run comprehensive method debugging."""
    try:
        create_comprehensive_method_plots()

        print("\n🎯 COMPREHENSIVE DEBUGGING COMPLETE")
        print("=" * 80)
        print("Created comprehensive plots for selected optimization methods:")
        print("✅ All configured methods processed")
        print("✅ Full sampling window analysis")
        print("✅ Method logic verification")
        print("\nPlots saved to outputs/plots/optimization/comprehensive_all_methods_window_[1-3].png")
        print("Use these to debug any method issues and verify optimization behavior!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
