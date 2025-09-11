"""
Modular Dataset Builder

This module provides a dataset builder that uses pluggable target generators
to create datasets with multiple target types (classification and regression).
"""

from pathlib import Path
from typing import Any

import polars as pl

from .target_generators.base import TargetGenerator
from .target_generators.factory import TargetGeneratorFactory


class ModularDatasetBuilder:
    """
    Dataset builder with pluggable target generation.

    This builder allows combining multiple target generators to create datasets
    with both classification and regression targets in a single pass.
    """

    def __init__(self, target_generators: list[TargetGenerator], verbose: bool = True):
        """
        Initialize modular dataset builder.

        Args:
            target_generators: List of target generators to apply
            verbose: Whether to print progress information
        """
        self.target_generators = target_generators
        self.verbose = verbose
        self._validate_generators()

    def build_targets(self, symbol_df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """
        Build standalone target DataFrame with keys mapping to input data.

        Args:
            symbol_df: Input DataFrame with market data for a single symbol
            symbol: Optional symbol identifier to include in targets

        Returns:
            DataFrame with row keys and all generated targets (no input data)
        """
        if self.verbose:
            print(f"🎯 Building targets with {len(self.target_generators)} target generators")

        target_dfs = []

        for i, generator in enumerate(self.target_generators):
            if self.verbose:
                print(f"   {i + 1}/{len(self.target_generators)}: {generator.__class__.__name__}")

            # Validate required columns
            self._validate_required_columns(symbol_df, generator)

            # Generate targets - now returns DataFrame with keys
            target_df = generator.generate_targets(symbol_df, symbol=symbol)
            target_dfs.append(target_df)

            if self.verbose:
                target_cols = [col for col in target_df.columns if col not in ["row_idx", "symbol", "timestamp"]]
                print(f"      ✅ Generated targets: {target_cols}")

        # Merge all target DataFrames on row_idx
        if target_dfs:
            result_df = target_dfs[0]
            for target_df in target_dfs[1:]:
                # Determine join keys (common key columns)
                join_keys = ["row_idx"]
                for col in ["symbol", "timestamp"]:
                    if col in result_df.columns and col in target_df.columns:
                        join_keys.append(col)

                # Join and then drop duplicate key columns with suffixes
                result_df = result_df.join(
                    target_df,
                    on=join_keys,
                    how="full"  # Use "full" instead of deprecated "outer"
                )

                # Remove duplicate columns created by join (with _right suffix)
                for col in join_keys:
                    right_col = f"{col}_right"
                    if right_col in result_df.columns:
                        result_df = result_df.drop(right_col)

            if self.verbose:
                target_cols = [col for col in result_df.columns if col not in ["row_idx", "symbol", "timestamp"]]
                print(f"   📊 Final targets: {len(result_df)} rows, {len(target_cols)} target columns")

            return result_df
        else:
            # No target generators - return inputs-only dataset with row keys
            if self.verbose:
                print("   📊 Creating inputs-only dataset (no target columns)")
            
            # Create row index column
            result_df = symbol_df.with_row_index("row_idx")
            
            # Add symbol column if provided
            if symbol:
                result_df = result_df.with_columns(pl.lit(symbol).alias("symbol"))
            
            # Add timestamp column if ts_event exists
            if "ts_event" in result_df.columns:
                result_df = result_df.with_columns(pl.col("ts_event").alias("timestamp"))
            
            if self.verbose:
                print(f"   📊 Inputs-only dataset: {len(result_df)} rows, {len(result_df.columns)} columns")
            
            return result_df

    def build_dataset(self, symbol_df: pl.DataFrame) -> pl.DataFrame:
        """
        DEPRECATED: Use build_targets() instead.

        Legacy method for backward compatibility - will be removed.
        """
        import warnings
        warnings.warn(
            "build_dataset() is deprecated. Use build_targets() for target-only generation.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.build_targets(symbol_df)

    def build_targets_from_parquet(self, parquet_path: str | Path, symbol: str | None = None) -> pl.DataFrame:
        """
        Build targets from parquet file.

        Args:
            parquet_path: Path to input parquet file
            symbol: Optional symbol identifier

        Returns:
            DataFrame with generated targets (keys + targets only)
        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        if self.verbose:
            print(f"📂 Loading data from: {parquet_path.name}")

        # Load data
        symbol_df = pl.read_parquet(parquet_path)

        if self.verbose:
            print(f"   📊 Loaded {len(symbol_df):,} rows")

        # Extract symbol from filename if not provided
        if symbol is None:
            symbol = parquet_path.stem.split('_')[0] if '_' in parquet_path.stem else None

        # Build targets
        return self.build_targets(symbol_df, symbol=symbol)

    def build_from_parquet(self, parquet_path: str | Path) -> pl.DataFrame:
        """
        DEPRECATED: Use build_targets_from_parquet() instead.
        Legacy method for backward compatibility - will be removed.
        """
        import warnings
        warnings.warn(
            "build_from_parquet() is deprecated. Use build_targets_from_parquet().",
            DeprecationWarning,
            stacklevel=2
        )
        return self.build_targets_from_parquet(parquet_path)

    def save_targets(
        self, targets_df: pl.DataFrame, output_path: str | Path, include_metadata: bool = True
    ) -> dict[str, Any]:
        """
        Save standalone target DataFrame to parquet file.

        Args:
            targets_df: Target DataFrame with keys and targets
            output_path: Output parquet file path
            include_metadata: Whether to include generator metadata

        Returns:
            Dict with save statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"🎯 Saving targets to: {output_path.name}")

        # Save targets
        targets_df.write_parquet(output_path)

        # Collect statistics
        target_cols = [col for col in targets_df.columns if col not in ["row_idx", "symbol", "timestamp"]]
        stats = {
            "output_path": str(output_path),
            "total_rows": len(targets_df),
            "target_columns": target_cols,
            "total_columns": len(targets_df.columns),
            "file_size_mb": output_path.stat().st_size / 1024 / 1024,
        }

        if include_metadata:
            stats["target_generators"] = [
                generator.get_target_info() for generator in self.target_generators
            ]

        if self.verbose:
            print(f"   ✅ Saved {stats['total_rows']:,} rows, {len(target_cols)} targets, {stats['file_size_mb']:.1f} MB")

        return stats

    def save_dataset(
        self, dataset_df: pl.DataFrame, output_path: str | Path, include_metadata: bool = True
    ) -> dict[str, Any]:
        """
        DEPRECATED: Use save_targets() instead.
        Legacy method for backward compatibility - will be removed.
        """
        import warnings
        warnings.warn(
            "save_dataset() is deprecated. Use save_targets() for target-only files.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.save_targets(dataset_df, output_path, include_metadata)

    def get_builder_info(self) -> dict[str, Any]:
        """
        Get information about this builder configuration.

        Returns:
            Dict with builder metadata
        """
        classification_generators = []
        regression_generators = []

        for generator in self.target_generators:
            info = generator.get_target_info()
            if generator.target_type == "classification":
                classification_generators.append(info)
            else:
                regression_generators.append(info)

        return {
            "total_generators": len(self.target_generators),
            "classification_generators": classification_generators,
            "regression_generators": regression_generators,
            "all_target_names": self._get_all_target_names(),
        }

    def _validate_generators(self) -> None:
        """Validate that all generators are properly configured."""
        # Allow empty generators for inputs-only datasets
        if not self.target_generators:
            if self.verbose:
                print("🔧 No target generators provided - will create inputs-only dataset")
            return

        # Check for duplicate target names
        all_target_names = self._get_all_target_names()
        if len(all_target_names) != len(set(all_target_names)):
            duplicates = [
                name for name in set(all_target_names) if all_target_names.count(name) > 1
            ]
            raise ValueError(f"Duplicate target names found: {duplicates}")

        # Validate each generator
        for generator in self.target_generators:
            if not isinstance(generator, TargetGenerator):
                raise ValueError(
                    f"All generators must implement TargetGenerator interface. "
                    f"Got: {type(generator)}"
                )

    def _validate_required_columns(self, df: pl.DataFrame, generator: TargetGenerator) -> None:
        """Validate that DataFrame has required columns for a generator."""
        missing_columns = set(generator.required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns for {generator.__class__.__name__}: "
                f"{sorted(missing_columns)}"
            )

    def _get_all_target_names(self) -> list[str]:
        """Get all target names from all generators."""
        all_names = []
        for generator in self.target_generators:
            info = generator.get_target_info()
            all_names.extend(info.get("target_names", []))
        return all_names


def create_modular_builder(generator_configs: list[dict[str, Any]]) -> ModularDatasetBuilder:
    """
    Create a modular dataset builder from configuration.

    Args:
        generator_configs: List of generator configurations

    Returns:
        Configured ModularDatasetBuilder

    Example:
        configs = [
            {"type": "quantile_classification", "nbins": 13},
            {"type": "directional_mfe", "lookforward_horizon": 3000},
            {"type": "volatility", "window_size": 1000}
        ]
        builder = create_modular_builder(configs)
    """
    generators = []

    for config in generator_configs:
        generator_type = config.pop("type")
        generator = TargetGeneratorFactory.create(generator_type, **config)
        generators.append(generator)

    return ModularDatasetBuilder(generators)
