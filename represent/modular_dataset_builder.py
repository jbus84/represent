"""
Modular Dataset Builder

This module provides a dataset builder that uses pluggable target generators
to create datasets with multiple target types (classification and regression).
"""

from pathlib import Path
from typing import Any

import numpy as np
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

    def build_dataset(self, symbol_df: pl.DataFrame) -> pl.DataFrame:
        """
        Build dataset with all configured targets.

        Args:
            symbol_df: Input DataFrame with market data for a single symbol

        Returns:
            DataFrame with original data plus all generated targets
        """
        if self.verbose:
            print(f"🔄 Building dataset with {len(self.target_generators)} target generators")

        result_df = symbol_df.clone()

        for i, generator in enumerate(self.target_generators):
            if self.verbose:
                print(f"   {i + 1}/{len(self.target_generators)}: {generator.__class__.__name__}")

            # Validate required columns
            self._validate_required_columns(symbol_df, generator)

            # Generate targets
            targets = generator.generate_targets(symbol_df)

            # Add targets to DataFrame
            for target_name, target_array in targets.items():
                if len(target_array) != len(symbol_df):
                    raise ValueError(
                        f"Target array length ({len(target_array)}) doesn't match "
                        f"DataFrame length ({len(symbol_df)}) for target '{target_name}'"
                    )

                result_df = result_df.with_columns(pl.Series(target_name, target_array))

                if self.verbose:
                    valid_count = np.sum(~np.isnan(target_array))
                    print(f"      ✅ {target_name}: {valid_count:,} valid values")

        if self.verbose:
            print(f"   📊 Final dataset: {len(result_df)} rows, {len(result_df.columns)} columns")

        return result_df

    def build_from_parquet(self, parquet_path: str | Path) -> pl.DataFrame:
        """
        Build dataset from parquet file.

        Args:
            parquet_path: Path to input parquet file

        Returns:
            DataFrame with generated targets
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

        # Build dataset
        return self.build_dataset(symbol_df)

    def save_dataset(
        self, dataset_df: pl.DataFrame, output_path: str | Path, include_metadata: bool = True
    ) -> dict[str, Any]:
        """
        Save dataset to parquet file with optional metadata.

        Args:
            dataset_df: Dataset DataFrame to save
            output_path: Output parquet file path
            include_metadata: Whether to include generator metadata

        Returns:
            Dict with save statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"💾 Saving dataset to: {output_path.name}")

        # Save dataset
        dataset_df.write_parquet(output_path)

        # Collect statistics
        stats = {
            "output_path": str(output_path),
            "total_rows": len(dataset_df),
            "total_columns": len(dataset_df.columns),
            "file_size_mb": output_path.stat().st_size / 1024 / 1024,
        }

        if include_metadata:
            stats["target_generators"] = [
                generator.get_target_info() for generator in self.target_generators
            ]

        if self.verbose:
            print(f"   ✅ Saved {stats['total_rows']:,} rows, {stats['file_size_mb']:.1f} MB")

        return stats

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
        if not self.target_generators:
            raise ValueError("At least one target generator must be provided")

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
