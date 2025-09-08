"""
Target-Only Generation API

This module provides high-level API functions for generating standalone target files
that map to input data via keys, dramatically reducing storage requirements.
"""

from pathlib import Path
from typing import Any

import polars as pl

from .modular_dataset_builder import create_modular_builder
from .target_generators.factory import TargetGeneratorFactory


def generate_targets_from_parquet(
    input_path: str | Path,
    output_path: str | Path,
    generator_configs: list[dict[str, Any]],
    symbol: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Generate standalone target file from input parquet data.

    Args:
        input_path: Path to input parquet file with market data
        output_path: Path for output target file
        generator_configs: List of target generator configurations
        symbol: Optional symbol identifier (extracted from filename if not provided)
        verbose: Whether to print progress information

    Returns:
        Dict with generation statistics

    Example:
        generator_configs = [
            {"type": "quantile_classification", "nbins": 13},
            {"type": "directional_mfe", "lookforward_horizon": 3000},
            {"type": "log_return_horizons", "horizons": [1000, 2000, 3000]}
        ]

        stats = generate_targets_from_parquet(
            "symbol_data.parquet",
            "symbol_targets.parquet",
            generator_configs,
            symbol="AUDUSD_M6AM4"
        )
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create modular builder
    builder = create_modular_builder(generator_configs)
    builder.verbose = verbose

    if verbose:
        print(f"🎯 Generating targets from: {input_path.name}")
        print(f"   Output: {output_path.name}")

    # Generate targets
    targets_df = builder.build_targets_from_parquet(input_path, symbol=symbol)

    # Save targets
    stats = builder.save_targets(targets_df, output_path)

    return stats


def generate_targets_from_dataframe(
    input_df: pl.DataFrame,
    generator_configs: list[dict[str, Any]],
    symbol: str | None = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Generate targets from input DataFrame.

    Args:
        input_df: Input DataFrame with market data
        generator_configs: List of target generator configurations
        symbol: Optional symbol identifier
        verbose: Whether to print progress information

    Returns:
        DataFrame with keys and targets (no input data)

    Example:
        generator_configs = [
            {"type": "quantile_classification", "nbins": 13},
            {"type": "volatility", "window_size": 1000}
        ]

        targets_df = generate_targets_from_dataframe(
            market_data_df,
            generator_configs,
            symbol="AUDUSD_M6AM4"
        )
    """
    # Create modular builder
    builder = create_modular_builder(generator_configs)
    builder.verbose = verbose

    if verbose:
        print("🎯 Generating targets from DataFrame")
        print(f"   Input rows: {len(input_df):,}")

    # Generate targets
    targets_df = builder.build_targets(input_df, symbol=symbol)

    return targets_df


def batch_generate_targets(
    input_files: list[str | Path],
    output_dir: str | Path,
    generator_configs: list[dict[str, Any]],
    output_suffix: str = "_targets.parquet",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Generate targets for multiple input files in batch.

    Args:
        input_files: List of input parquet file paths
        output_dir: Directory for output target files
        generator_configs: List of target generator configurations
        output_suffix: Suffix for output filenames
        verbose: Whether to print progress information

    Returns:
        Dict with batch processing statistics

    Example:
        input_files = [
            "AUDUSD_M6AM4_data.parquet",
            "AUDUSD_M6AM5_data.parquet"
        ]

        stats = batch_generate_targets(
            input_files,
            "targets/",
            generator_configs
        )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "total_files": len(input_files),
        "processed": 0,
        "failed": 0,
        "output_files": [],
        "total_size_mb": 0.0,
        "errors": []
    }

    if verbose:
        print(f"🎯 Batch generating targets for {len(input_files)} files")

    for i, input_path in enumerate(input_files):
        input_path = Path(input_path)

        if verbose:
            print(f"\n   📂 {i + 1}/{len(input_files)}: {input_path.name}")

        try:
            # Generate output filename
            output_name = input_path.stem + output_suffix
            output_path = output_dir / output_name

            # Extract symbol from filename
            symbol = input_path.stem.split('_')[0] if '_' in input_path.stem else None

            # Generate targets
            stats = generate_targets_from_parquet(
                input_path,
                output_path,
                generator_configs,
                symbol=symbol,
                verbose=verbose
            )

            results["processed"] += 1
            results["output_files"].append(str(output_path))
            results["total_size_mb"] += stats["file_size_mb"]

        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "file": str(input_path),
                "error": str(e)
            })
            if verbose:
                print(f"   ❌ Error: {e}")

    if verbose:
        print(f"\n✅ Batch complete: {results['processed']} success, {results['failed']} failed")
        print(f"   Total output size: {results['total_size_mb']:.1f} MB")

    return results


def create_target_config_template(
    target_types: list[str],
    classification_bins: int = 13,
    mfe_horizon: int = 3000,
    log_return_horizons: list[int] | None = None
) -> list[dict[str, Any]]:
    """
    Create a template configuration for common target types.

    Args:
        target_types: List of target types to include
        classification_bins: Number of bins for classification targets
        mfe_horizon: Horizon for MFE calculations
        log_return_horizons: List of horizons for log returns

    Returns:
        List of generator configurations

    Example:
        config = create_target_config_template(
            ["classification", "mfe", "log_returns", "volatility"]
        )
    """
    if log_return_horizons is None:
        log_return_horizons = [1000, 2000, 3000, 4000, 5000]

    configs = []

    for target_type in target_types:
        if target_type == "classification":
            configs.append({
                "type": "quantile_classification",
                "nbins": classification_bins,
                "target_name": "classification_label"
            })

        elif target_type == "mfe":
            configs.append({
                "type": "directional_mfe",
                "lookforward_horizon": mfe_horizon,
                "target_names": ("mfe_buy_bps", "mfe_sell_bps")
            })

        elif target_type == "log_returns":
            configs.append({
                "type": "log_return_horizons",
                "horizons": log_return_horizons,
                "target_prefix": "log_return"
            })

        elif target_type == "volatility":
            configs.append({
                "type": "volatility",
                "window_size": 1000,
                "target_name": "volatility_target"
            })

        else:
            available_types = TargetGeneratorFactory.list_available()
            raise ValueError(
                f"Unknown target type '{target_type}'. "
                f"Available types: {list(available_types.keys())}"
            )

    return configs


def load_targets_and_join(
    input_data_path: str | Path,
    targets_path: str | Path,
    join_columns: list[str] | None = None
) -> pl.DataFrame:
    """
    Load input data and targets, then join them for training.

    Args:
        input_data_path: Path to input parquet file
        targets_path: Path to targets parquet file
        join_columns: Columns to join on (defaults to ['row_idx'])

    Returns:
        Combined DataFrame with input data and targets

    Example:
        combined_df = load_targets_and_join(
            "symbol_data.parquet",
            "symbol_targets.parquet"
        )
    """
    if join_columns is None:
        join_columns = ['row_idx']

    # Load data
    input_df = pl.read_parquet(input_data_path)
    targets_df = pl.read_parquet(targets_path)

    # Add row indices to input data if not present
    if 'row_idx' not in input_df.columns:
        input_df = input_df.with_row_index('row_idx')

    # Join on specified columns
    combined_df = input_df.join(targets_df, on=join_columns, how="inner")

    return combined_df
