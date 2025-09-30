#!/usr/bin/env python3
"""
Label Set Builder - Build datasets with specific target configurations

This script provides a flexible way to build datasets with custom label configurations
for ML training and research purposes.

Usage:
    python scripts/build_label_set.py --config configs/label_sets/my_config.yaml
    python scripts/build_label_set.py --preset mfe_analysis
    python scripts/build_label_set.py --preset trend_analysis --symbol M6AM4
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from represent.modular_dataset_builder import ModularDatasetBuilder  # noqa: E402
from represent.target_generators.factory import TargetGeneratorFactory  # noqa: E402

# Preset configurations for common use cases
LABEL_SET_PRESETS = {
    "inputs_only": {
        "name": "Clean Inputs Only - No Target Columns",
        "description": "Extract clean market microstructure data without any target columns for parameter optimization",
        "generators": [],  # No generators = no target columns
    },
    "mfe_analysis": {
        "name": "MFE Analysis - Buy/Sell Directional Signals",
        "description": "Maximum Favorable Excursion analysis for both long and short positions",
        "generators": [
            {
                "type": "directional_mfe",
                "lookforward_horizon": 1000,
                "expected_fee_pips": 0.5,
                "winsorize_percentiles": [0.1, 99.9],
                "target_names": ["mfe_buy_1k", "mfe_sell_1k"],
            },
            {
                "type": "directional_mfe",
                "lookforward_horizon": 2000,
                "expected_fee_pips": 0.5,
                "winsorize_percentiles": [0.1, 99.9],
                "target_names": ["mfe_buy_2k", "mfe_sell_2k"],
            },
        ],
        "visualization": True,
    },
    "trend_analysis": {
        "name": "Trend Analysis - Multi-horizon Trend Signals",
        "description": "Comprehensive trend analysis with remaining value tuning",
        "generators": [
            {
                "type": "remaining_value_tuner",
                "lookback_rows": 500,
                "lookforward_input": 2000,
                "lookforward_offset": 100,
                "trend_threshold_bps": 20.0,
                "target_name": "remaining_value_2k",
            },
            {
                "type": "remaining_value_tuner",
                "lookback_rows": 1000,
                "lookforward_input": 4000,
                "lookforward_offset": 200,
                "trend_threshold_bps": 30.0,
                "target_name": "remaining_value_4k",
            },
            {
                "type": "quantile_classification",
                "nbins": 13,
                "lookforward_window": 3000,
                "target_name": "trend_quantile_13",
            },
        ],
        "visualization": True,
    },
    "volatility_analysis": {
        "name": "Volatility Analysis - Risk and Adaptive Returns",
        "description": "Volatility-based targets for risk management and adaptive strategies",
        "generators": [
            {
                "type": "volatility_scaled_returns",
                "volatility_window": 500,
                "vol_multiplier": 2.0,
                "horizon_ticks": 1500,
                "target_name": "vol_scaled_2x_1500",
            },
            {
                "type": "volatility_scaled_returns",
                "volatility_window": 500,
                "vol_multiplier": 3.0,
                "horizon_ticks": 2000,
                "target_name": "vol_scaled_3x_2000",
            },
            {"type": "volatility", "window_size": 1000, "target_name": "rolling_vol_1k"},
        ],
        "visualization": True,
    },
    "returns_analysis": {
        "name": "Returns Analysis - Cumulative and Price Movement",
        "description": "Returns-based analysis for momentum and mean reversion strategies",
        "generators": [
            {"type": "cumulative_returns", "lookforward_samples": 500, "target_name": "cumret_500"},
            {
                "type": "cumulative_returns",
                "lookforward_samples": 1500,
                "target_name": "cumret_1500",
            },
            {
                "type": "cumulative_returns",
                "lookforward_samples": 3000,
                "target_name": "cumret_3000",
            },
            {"type": "price_movement", "lookforward_window": 1000, "target_name": "price_move_1k"},
        ],
        "visualization": True,
    },
    "comprehensive": {
        "name": "Comprehensive Analysis - All Target Types",
        "description": "Complete suite of all available target generators",
        "generators": [
            {
                "type": "quantile_classification",
                "nbins": 13,
                "lookforward_window": 2000,
                "target_name": "quantile_13_2k",
            },
            {
                "type": "directional_mfe",
                "lookforward_horizon": 1000,
                "target_names": ["mfe_buy", "mfe_sell"],
            },
            {
                "type": "remaining_value_tuner",
                "lookforward_input": 3000,
                "target_name": "remaining_value_3k",
            },
            {
                "type": "volatility_scaled_returns",
                "vol_multiplier": 2.5,
                "horizon_ticks": 1500,
                "target_name": "vol_scaled_returns",
            },
            {
                "type": "cumulative_returns",
                "lookforward_samples": 1500,
                "target_name": "cumret_1500",
            },
            {"type": "price_movement", "lookforward_window": 1000, "target_name": "price_movement"},
            {"type": "volatility", "window_size": 1000, "target_name": "volatility"},
        ],
        "visualization": True,
    },
    "log_return_horizons": {
        "name": "Log Return Horizons - Multi-scale Return Analysis",
        "description": "Log return predictions across multiple time horizons (1k-5k ticks)",
        "generators": [
            {
                "type": "log_return_horizons",
                "horizons": [1000, 2000, 3000, 4000, 5000],
                "lookback_window": 1000,
                "target_prefix": "log_return",
            },
            {
                "type": "log_return_horizons",
                "horizons": [500, 1500, 2500],
                "lookback_window": 500,
                "target_prefix": "short_log_return",
            },
        ],
        "visualization": True,
    },
}


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config_template(output_path: Path):
    """Save a template configuration file."""
    template = {
        "name": "Custom Label Set",
        "description": "Description of this label set configuration",
        "generators": [
            {
                "type": "directional_mfe",
                "lookforward_horizon": 1000,
                "expected_fee_pips": 0.5,
                "target_names": ["mfe_buy", "mfe_sell"],
            },
            {
                "type": "quantile_classification",
                "nbins": 13,
                "lookforward_window": 2000,
                "target_name": "quantile_labels",
            },
        ],
        "visualization": True,
        "output": {"base_name": "custom_label_set", "include_timestamp": True},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, indent=2)

    print(f"✅ Template configuration saved to: {output_path}")


def build_label_set(
    data_path: Path, config: dict[str, Any], output_dir: Path, symbol: str = None
) -> Path:
    """Build a label set from configuration."""

    print(f"🎯 Building Label Set: {config['name']}")
    print(f"📝 {config['description']}")
    print("=" * 60)

    # Load data
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"📊 Loading data from: {data_path}")
    df = pl.read_parquet(data_path)

    if symbol and "symbol" in df.columns:
        df = df.filter(pl.col("symbol") == symbol)
        print(f"🎯 Filtered for symbol: {symbol}")

    print(f"✅ Loaded {len(df):,} samples")

    # Create generators from configuration
    generators = []
    for gen_config in config["generators"]:
        gen_type = gen_config.pop("type")
        generator = TargetGeneratorFactory.create(gen_type, **gen_config)
        generators.append(generator)
        gen_config["type"] = gen_type  # Restore for next use

    print(f"🔧 Created {len(generators)} target generators")

    # Build dataset
    builder = ModularDatasetBuilder(generators)
    dataset = builder.build_from_parquet(data_path)

    # Generate output filename
    output_config = config.get("output", {})
    base_name = output_config.get("base_name", "label_set")
    include_timestamp = output_config.get("include_timestamp", True)

    if include_timestamp:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}.parquet"
    else:
        filename = f"{base_name}.parquet"

    if symbol:
        filename = f"{symbol}_{filename}"

    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset
    builder.save_dataset(dataset, output_path)

    # Generate summary
    print("\n📊 LABEL SET SUMMARY")
    print("=" * 40)
    print(f"Output file: {output_path}")
    print(f"Total samples: {len(dataset):,}")

    target_cols = [col for col in dataset.columns if col not in ["mid_price", "ts_event", "symbol"]]
    print(f"Target columns: {len(target_cols)}")

    for col in target_cols:
        values = dataset[col].to_numpy()
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(
                f"  {col}: μ={np.mean(valid_values):.2f}, σ={np.std(valid_values):.2f}, valid={len(valid_values):,}"
            )
        else:
            print(f"  {col}: No valid values")

    # Create visualization if requested
    if config.get("visualization", False):
        print("\n📈 Creating visualization...")
        create_visualization(
            dataset, output_dir / f"{filename.replace('.parquet', '_visualization.png')}"
        )

    return output_path


def create_visualization(dataset: pl.DataFrame, output_path: Path):
    """Create visualization of the label set."""
    import matplotlib.pyplot as plt

    # Get target columns (exclude metadata)
    target_cols = [col for col in dataset.columns if col not in ["mid_price", "ts_event", "symbol"]]
    prices = dataset["mid_price"].to_numpy()

    if len(target_cols) == 0:
        print("⚠️ No target columns to visualize")
        return

    # Create subplots
    n_cols = min(len(target_cols), 4)  # Max 4 columns
    n_rows = (len(target_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=(5 * n_cols, 3 * (n_rows + 1)))
    if n_rows == 0:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot price series at top
    axes[0, 0].plot(prices, "k-", linewidth=1, alpha=0.8)
    axes[0, 0].set_title("Market Price Series", fontweight="bold")
    axes[0, 0].set_ylabel("Price")
    axes[0, 0].grid(True, alpha=0.3)

    # Hide empty price subplots
    for j in range(1, n_cols):
        if j < len(axes[0]):
            axes[0, j].axis("off")

    # Plot each target
    for i, col in enumerate(target_cols):
        row = (i // n_cols) + 1
        col_idx = i % n_cols

        if row >= len(axes):
            break

        ax = axes[row, col_idx]
        values = dataset[col].to_numpy()
        valid_mask = ~np.isnan(values)

        if np.any(valid_mask):
            ax.plot(np.where(valid_mask)[0], values[valid_mask], linewidth=1, alpha=0.7)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

            # Statistics
            valid_values = values[valid_mask]
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            ax.text(
                0.02,
                0.98,
                f"μ={mean_val:.2f}\nσ={std_val:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(target_cols), n_rows * n_cols):
        row = (i // n_cols) + 1
        col_idx = i % n_cols
        if row < len(axes) and col_idx < len(axes[row]):
            axes[row, col_idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build custom label sets for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a preset configuration
  python scripts/build_label_set.py --preset mfe_analysis --symbol M6AM4

  # Use custom configuration file
  python scripts/build_label_set.py --config configs/label_sets/my_config.yaml

  # Generate configuration template
  python scripts/build_label_set.py --template configs/label_sets/template.yaml

  # List available presets
  python scripts/build_label_set.py --list-presets
        """,
    )

    parser.add_argument(
        "--preset",
        choices=LABEL_SET_PRESETS.keys(),
        help="Use a predefined label set configuration",
    )
    parser.add_argument("--config", type=Path, help="Path to custom configuration YAML file")
    parser.add_argument("--template", type=Path, help="Generate a template configuration file")
    parser.add_argument(
        "--list-presets", action="store_true", help="List available preset configurations"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/sample_data.parquet"),
        help="Path to input parquet data file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/danielfisher/data/databento/label_sets"),
        help="Output directory for generated label sets",
    )
    parser.add_argument("--symbol", type=str, help="Filter for specific symbol (optional)")

    args = parser.parse_args()

    # Handle special commands
    if args.list_presets:
        print("\n🎯 Available Label Set Presets:")
        print("=" * 50)
        for key, preset in LABEL_SET_PRESETS.items():
            print(f"\n{key}:")
            print(f"  Name: {preset['name']}")
            print(f"  Description: {preset['description']}")
            print(f"  Generators: {len(preset['generators'])}")
        return

    if args.template:
        save_config_template(args.template)
        return

    # Determine configuration
    if args.preset:
        config = LABEL_SET_PRESETS[args.preset]
        print(f"🎯 Using preset: {args.preset}")
    elif args.config:
        config = load_config(args.config)
        print(f"📄 Using config file: {args.config}")
    else:
        parser.error("Must specify either --preset or --config")

    # Build label set
    try:
        output_path = build_label_set(
            data_path=args.data, config=config, output_dir=args.output, symbol=args.symbol
        )

        print("\n🎉 SUCCESS!")
        print(f"📁 Label set saved to: {output_path}")
        print("🚀 Ready for ML training!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
