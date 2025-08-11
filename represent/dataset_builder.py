"""
Symbol-Split-Merge Dataset Builder

This module implements the new data processing approach where:
1. Multiple DBN files are split by symbol into intermediate files
2. Each symbol's data is then merged across all files into large parquet datasets

This creates comprehensive datasets per symbol for ML training.
"""

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import databento as db
import numpy as np
import polars as pl

from .config import RepresentConfig
from .global_threshold_calculator import GlobalThresholds


@dataclass
class DatasetBuildConfig:
    """Configuration for the symbol-split-merge dataset building process."""
    currency: str = "AUDUSD"
    features: list[str] | None = None
    min_symbol_samples: int = 10500  # Must be >= lookback_rows + lookforward_input + lookforward_offset
    force_uniform: bool = True
    nbins: int = 13
    global_thresholds: GlobalThresholds | None = None
    intermediate_dir: str | None = None  # If None, uses temp directory
    keep_intermediate: bool = False  # Whether to keep intermediate split files

    def __post_init__(self):
        if self.features is None:
            self.features = ["volume"]

        # Validate that classification method is properly specified
        if not self.force_uniform and self.global_thresholds is None:
            raise ValueError(
                "DatasetBuildConfig requires either force_uniform=True or global_thresholds to be provided. "
                "Fixed threshold fallback has been removed. Please either:\n"
                "  1. Set force_uniform=True for quantile-based classification, or\n"
                "  2. Provide global_thresholds for consistent cross-symbol classification"
            )


class DatasetBuilder:
    """
    Symbol-Split-Merge Dataset Builder.

    This class processes multiple DBN files to create comprehensive symbol datasets:
    1. Split each DBN file by symbol into intermediate files
    2. Merge all instances of each symbol across files into large parquet datasets

    This approach creates comprehensive training datasets per symbol.
    """

    def __init__(
        self,
        config: RepresentConfig,
        dataset_config: DatasetBuildConfig | None = None,
        verbose: bool = True,
    ):
        """
        Initialize the dataset builder.

        Args:
            config: RepresentConfig with currency-specific configuration
            dataset_config: Configuration for dataset building process
            verbose: Whether to print progress information
        """
        self.represent_config = config
        self.dataset_config = dataset_config or DatasetBuildConfig(
            currency=config.currency,
            features=config.features
        )

        # Calculate minimum required samples - only need lookback window size for processing
        required_samples = (
            config.lookback_rows +  # Historical data needed for lookback window
            config.lookforward_input +  # Future data needed for lookforward window
            config.lookforward_offset  # Offset before future window starts
        )

        # Update min_symbol_samples if it's too low
        if self.dataset_config.min_symbol_samples < required_samples:
            if verbose:
                print(f"⚠️  Updating min_symbol_samples from {self.dataset_config.min_symbol_samples:,} to {required_samples:,}")
                print(f"    Required: {config.lookback_rows:,} lookback + {config.lookforward_input:,} lookforward + {config.lookforward_offset:,} offset")
            self.dataset_config.min_symbol_samples = required_samples
        self.verbose = verbose

        # Track symbol data across files
        self.symbol_registry: dict[str, list[Path]] = defaultdict(list)

        if self.verbose:
            print("🏗️  DatasetBuilder initialized")
            print(f"   💱 Currency: {self.dataset_config.currency}")
            print(f"   📊 Features: {self.dataset_config.features}")
            print(f"   🎯 Min samples per symbol: {self.dataset_config.min_symbol_samples}")
            print(f"   ⚖️  Force uniform: {self.dataset_config.force_uniform}")

    def split_dbn_by_symbol(
        self,
        dbn_path: str | Path,
        intermediate_dir: Path
    ) -> dict[str, Path]:
        """
        Split a single DBN file by symbol into intermediate parquet files.

        Args:
            dbn_path: Path to input DBN file
            intermediate_dir: Directory for intermediate symbol files

        Returns:
            Dictionary mapping symbol -> intermediate file path
        """
        if self.verbose:
            print(f"📄 Splitting DBN file: {dbn_path}")

        start_time = time.perf_counter()

        # Load DBN data
        data = db.read_dbn(str(dbn_path))
        df = pl.from_pandas(data.to_df())

        # Get unique symbols
        symbols = df['symbol'].unique().to_list()

        if self.verbose:
            print(f"   📊 Found {len(df):,} rows across {len(symbols)} symbols")

        # Split by symbol and save intermediate files
        symbol_files: dict[str, Path] = {}
        dbn_name = Path(dbn_path).stem

        for symbol in symbols:
            symbol_df = df.filter(pl.col('symbol') == symbol)

            if len(symbol_df) == 0:
                continue

            # Create intermediate file path
            intermediate_file = intermediate_dir / f"{dbn_name}_{symbol}.parquet"

            # Save symbol data
            symbol_df.write_parquet(
                str(intermediate_file),
                compression="snappy",
                statistics=True,
                row_group_size=50000
            )

            symbol_files[symbol] = intermediate_file

            # Register in symbol registry
            self.symbol_registry[symbol].append(intermediate_file)

            if self.verbose:
                print(f"      📁 {symbol}: {len(symbol_df):,} rows → {intermediate_file.name}")

        split_time = time.perf_counter() - start_time

        if self.verbose:
            print(f"   ✅ Split complete in {split_time:.1f}s")

        return symbol_files

    def merge_symbol_data(
        self,
        symbol: str,
        symbol_files: list[Path],
        output_dir: Path
    ) -> Path | None:
        """
        Merge all instances of a symbol across files into a single large parquet dataset.

        Args:
            symbol: Symbol identifier
            symbol_files: List of intermediate files containing this symbol's data
            output_dir: Directory for final dataset files

        Returns:
            Path to merged dataset file, or None if insufficient data
        """
        if self.verbose:
            print(f"🔄 Merging symbol: {symbol}")
            print(f"   📁 Files: {len(symbol_files)}")
            print("   📊 Processing: Lookback → Lookforward → Percentage Change → Classification")
            print("   🎯 Features: Generated on-demand during ML training (no storage overhead)")

        # Load all symbol data from intermediate files
        all_dfs = []
        total_rows = 0

        for file_path in symbol_files:
            if not file_path.exists():
                if self.verbose:
                    print(f"   ⚠️  Missing file: {file_path}")
                continue

            symbol_df = pl.read_parquet(file_path)
            all_dfs.append(symbol_df)
            total_rows += len(symbol_df)

            if self.verbose:
                print(f"      📄 {file_path.name}: {len(symbol_df):,} rows")

        if not all_dfs:
            if self.verbose:
                print(f"   ❌ No data found for {symbol}")
            return None

        # Check if total samples meet minimum threshold
        if total_rows < self.dataset_config.min_symbol_samples:
            if self.verbose:
                print(f"   ❌ Insufficient total samples: {total_rows:,} < {self.dataset_config.min_symbol_samples:,}")
            return None

        # Concatenate all DataFrames
        merged_df = pl.concat(all_dfs, how="vertical")

        # Sort by timestamp to ensure proper order
        merged_df = merged_df.sort('ts_event')

        # Calculate price movements using lookback vs lookforward methodology
        merged_df = self._calculate_price_movements(merged_df)

        # Apply classification
        classified_df = self._apply_classification(merged_df)

        # Filter to processable rows only (no feature generation needed for dataset building)
        final_df = self._filter_processable_rows(classified_df)

        if len(final_df) == 0:
            if self.verbose:
                print(f"   ❌ No processable rows after filtering for {symbol}")
            return None

        # Create output file path
        output_file = output_dir / f"{self.dataset_config.currency}_{symbol}_dataset.parquet"

        # Save merged dataset
        final_df.write_parquet(
            str(output_file),
            compression="snappy",
            statistics=True,
            row_group_size=100000  # Larger row groups for big datasets
        )

        if self.verbose:
            # Show classification distribution
            class_dist = final_df['classification_label'].value_counts().sort('classification_label')
            file_size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"   ✅ Merged {symbol}: {len(final_df):,} samples")
            print(f"      📊 Classes: {class_dist['classification_label'].to_list()}")
            print(f"      📊 Counts: {class_dist['count'].to_list()}")
            print(f"      💾 File: {output_file.name} ({file_size_mb:.1f} MB)")

        return output_file

    def _calculate_price_movements(self, symbol_df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate percentage price movements using lookback vs lookforward methodology.

        Optimized for polars with vectorized operations where possible, falling back
        to numpy for complex window calculations.

        Args:
            symbol_df: DataFrame for a single symbol (sorted by ts_event)

        Returns:
            DataFrame with price_movement column added
        """
        if self.verbose:
            print("   🧮 Calculating price movements using lookback/lookforward methodology")
            print(f"      Lookback: {self.represent_config.lookback_rows} rows")
            print(f"      Lookforward: {self.represent_config.lookforward_input} rows")
            print(f"      Lookforward offset: {self.represent_config.lookforward_offset} rows")
            print("      Processing: Every valid row (no jumping)")

        # Calculate mid prices using polars vectorized operations
        symbol_df = symbol_df.with_columns([
            ((pl.col('ask_px_00') + pl.col('bid_px_00')) / 2).alias('mid_price')
        ])

        # Extract mid prices as numpy array for complex window operations
        mid_prices = symbol_df['mid_price'].to_numpy()
        total_rows = len(mid_prices)

        # Pre-allocate price movements array
        price_movements = np.full(total_rows, np.nan, dtype=np.float64)

        # Calculate valid processing range
        min_required_rows = (
            self.represent_config.lookback_rows +
            self.represent_config.lookforward_input +
            self.represent_config.lookforward_offset
        )

        if total_rows < min_required_rows:
            if self.verbose:
                print(f"   ⚠️  Insufficient data: {total_rows} < {min_required_rows} required")
            return symbol_df.with_columns([
                pl.Series('price_movement', price_movements).cast(pl.Float64)
            ])

        # Calculate price movements using optimized window operations
        valid_indices = []
        movements_calculated = 0

        for stop_row in range(
            self.represent_config.lookback_rows,
            total_rows - (self.represent_config.lookforward_input + self.represent_config.lookforward_offset)
        ):
            # Define window boundaries
            lookback_start = stop_row - self.represent_config.lookback_rows
            lookback_end = stop_row

            target_start_row = stop_row + 1 + self.represent_config.lookforward_offset
            target_stop_row = target_start_row + self.represent_config.lookforward_input

            # Extract window data
            lookback_prices = mid_prices[lookback_start:lookback_end]
            lookforward_prices = mid_prices[target_start_row:target_stop_row]

            # Calculate means (numpy is faster for small arrays)
            lookback_mean = np.mean(lookback_prices)
            lookforward_mean = np.mean(lookforward_prices)

            # Calculate percentage change
            if lookback_mean > 0 and not np.isnan(lookback_mean) and not np.isnan(lookforward_mean):
                price_movements[stop_row] = (lookforward_mean - lookback_mean) / lookback_mean
                valid_indices.append(stop_row)
                movements_calculated += 1

        if self.verbose:
            print(f"   ✅ Calculated {movements_calculated:,} price movements")
            print(f"      Valid range: rows {self.represent_config.lookback_rows} to {total_rows - (self.represent_config.lookforward_input + self.represent_config.lookforward_offset)}")

        # Add price movement column using polars
        return symbol_df.with_columns([
            pl.Series('price_movement', price_movements).cast(pl.Float64)
        ])

    # Feature generation removed - features should be generated on-demand during ML training
    # This eliminates storage overhead and allows flexible feature combinations

    def _apply_classification(self, symbol_df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply classification using first half of symbol data to define bins.

        This method uses the first half of the symbol's data to determine classification
        bins, then applies those bins to the entire dataset. This ensures consistent
        classification boundaries while using symbol-specific distributions.

        Args:
            symbol_df: DataFrame with price_movement column for a single symbol

        Returns:
            DataFrame with classification_label column added
        """
        # Filter valid price movements
        valid_df = symbol_df.filter(
            pl.col('price_movement').is_not_null() &
            pl.col('price_movement').is_finite()
        )

        if len(valid_df) == 0:
            if self.verbose:
                print("   ⚠️  No valid price movements for classification")
            return symbol_df.with_columns(
                pl.lit(None, dtype=pl.Int32).alias('classification_label')
            )

        # Use first half of data to define classification bins
        first_half_size = len(valid_df) // 2

        if first_half_size < self.dataset_config.nbins * 10:  # Minimum samples per bin
            if self.verbose:
                print(f"   ⚠️  Insufficient data for reliable bin calculation: {first_half_size} samples")
            # Fallback to using all data if first half is too small
            training_df = valid_df
        else:
            training_df = valid_df.head(first_half_size)

        training_movements = training_df['price_movement'].to_numpy()
        all_movements = valid_df['price_movement'].to_numpy()

        if self.verbose:
            print(f"   📊 Using {len(training_movements):,} samples to define {self.dataset_config.nbins} bins")
            print(f"   📊 Applying classification to {len(all_movements):,} total samples")

        if self.dataset_config.force_uniform:
            # Quantile-based classification using first half for uniform distribution
            quantiles = np.linspace(0, 1, self.dataset_config.nbins + 1)
            quantile_boundaries = np.quantile(training_movements, quantiles)

            # Ensure unique boundaries
            quantile_boundaries = np.unique(quantile_boundaries)

            # Handle edge case where we don't have enough unique values
            if len(quantile_boundaries) < self.dataset_config.nbins + 1:
                min_val, max_val = training_movements.min(), training_movements.max()
                quantile_boundaries = np.linspace(min_val, max_val, self.dataset_config.nbins + 1)

            # Apply classification to all data using bins from first half
            classification_labels = np.digitize(all_movements, quantile_boundaries[1:-1])
            classification_labels = np.clip(classification_labels, 0, self.dataset_config.nbins - 1)

            if self.verbose:
                print(f"   🎯 Quantile boundaries: {[f'{b:.6f}' for b in quantile_boundaries[:5]]}")
                if len(quantile_boundaries) > 5:
                    print(f"      ... {len(quantile_boundaries)-5} more boundaries")

        elif self.dataset_config.global_thresholds is not None:
            # Global threshold-based classification
            quantile_boundaries = self.dataset_config.global_thresholds.quantile_boundaries
            classification_labels = np.digitize(all_movements, quantile_boundaries[1:-1])
            classification_labels = np.clip(classification_labels, 0, self.dataset_config.nbins - 1)

            if self.verbose:
                print(f"   🌍 Using global thresholds with {len(quantile_boundaries)-1} bins")

        else:
            # No fallback allowed - GlobalThresholds are required
            raise ValueError(
                "DatasetBuilder requires either force_uniform=True or global_thresholds to be provided. "
                "Fixed threshold fallback has been removed. Please either:\n"
                "  1. Set force_uniform=True in DatasetBuildConfig for quantile-based classification, or\n"
                "  2. Provide global_thresholds in DatasetBuildConfig for consistent cross-symbol classification"
            )

        # Create classification series and add to dataframe
        classification_series = pl.Series('classification_label', classification_labels, dtype=pl.Int32)

        # Add classification labels to the valid dataframe
        classified_df = valid_df.with_columns(classification_series)

        if self.verbose:
            # Show classification distribution
            class_dist = classified_df['classification_label'].value_counts().sort('classification_label')
            total_classified = len(classified_df)
            print(f"   📊 Classification distribution ({total_classified:,} samples):")
            for row in class_dist.iter_rows():
                label, count = row
                percentage = (count / total_classified) * 100
                print(f"      Class {label}: {count:,} samples ({percentage:.1f}%)")

        return classified_df

    def _filter_processable_rows(self, symbol_df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter rows that have sufficient data for processing.

        Args:
            symbol_df: DataFrame with classification labels

        Returns:
            Filtered DataFrame with only processable rows
        """
        return symbol_df.filter(
            pl.col('price_movement').is_not_null() &
            pl.col('classification_label').is_not_null()
        )

    def build_datasets_from_dbn_files(
        self,
        dbn_files: Sequence[str | Path],
        output_dir: str | Path,
        intermediate_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Build comprehensive symbol datasets from multiple DBN files.

        This is the main entry point that orchestrates the symbol-split-merge process:
        1. Split each DBN file by symbol into intermediate files
        2. Merge all instances of each symbol across files into large datasets

        Args:
            dbn_files: List of DBN files to process
            output_dir: Directory for final dataset files
            intermediate_dir: Directory for intermediate files (temp if None)

        Returns:
            Processing statistics and results
        """
        if self.verbose:
            print("🏗️  Starting Symbol-Split-Merge Dataset Building")
            print("=" * 70)
            print(f"   📁 Input files: {len(dbn_files)}")
            print(f"   📁 Output directory: {output_dir}")

        start_time = time.perf_counter()

        # Setup directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if intermediate_dir is None:
            intermediate_path = output_path / "intermediate"
        else:
            intermediate_path = Path(intermediate_dir)
        intermediate_path.mkdir(parents=True, exist_ok=True)

        # Phase 1: Split all DBN files by symbol
        if self.verbose:
            print(f"\n🔪 Phase 1: Splitting {len(dbn_files)} DBN files by symbol...")

        split_start = time.perf_counter()
        total_split_files = 0

        for i, dbn_file in enumerate(dbn_files, 1):
            if self.verbose:
                print(f"\n   📄 Processing file {i}/{len(dbn_files)}: {Path(dbn_file).name}")

            try:
                symbol_files = self.split_dbn_by_symbol(dbn_file, intermediate_path)
                total_split_files += len(symbol_files)
            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Failed to process {dbn_file}: {e}")
                continue

        split_time = time.perf_counter() - split_start

        if self.verbose:
            print("\n✅ Phase 1 Complete:")
            print(f"   ⏱️  Split time: {split_time:.1f}s")
            print(f"   📁 Intermediate files: {total_split_files}")
            print(f"   📊 Unique symbols found: {len(self.symbol_registry)}")

        # Phase 2: Merge symbols into comprehensive datasets
        if self.verbose:
            print("\n🔗 Phase 2: Merging symbols into datasets...")

        merge_start = time.perf_counter()
        dataset_files = {}
        total_samples = 0

        for symbol, symbol_file_list in self.symbol_registry.items():
            if self.verbose:
                print(f"\n   🔄 Processing symbol: {symbol}")

            try:
                dataset_file = self.merge_symbol_data(symbol, symbol_file_list, output_path)
                if dataset_file:
                    # Get sample count from the file
                    sample_df = pl.read_parquet(dataset_file, columns=['classification_label'])
                    sample_count = len(sample_df)

                    dataset_files[symbol] = {
                        'file_path': str(dataset_file),
                        'samples': sample_count,
                        'file_size_mb': dataset_file.stat().st_size / 1024 / 1024,
                        'source_files': len(symbol_file_list)
                    }
                    total_samples += sample_count
            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Failed to merge {symbol}: {e}")
                continue

        merge_time = time.perf_counter() - merge_start

        # Clean up intermediate files if requested
        if not self.dataset_config.keep_intermediate:
            if self.verbose:
                print("\n🧹 Cleaning up intermediate files...")

            for intermediate_file in intermediate_path.glob("*.parquet"):
                intermediate_file.unlink()

            # Remove intermediate directory if empty
            try:
                intermediate_path.rmdir()
            except OSError:
                pass  # Directory not empty or doesn't exist

        total_time = time.perf_counter() - start_time

        # Compile results
        results = {
            'input_files': [str(f) for f in dbn_files],
            'output_directory': str(output_path),
            'intermediate_directory': str(intermediate_path),
            'phase_1_stats': {
                'split_time_seconds': split_time,
                'intermediate_files_created': total_split_files,
                'symbols_discovered': len(self.symbol_registry),
            },
            'phase_2_stats': {
                'merge_time_seconds': merge_time,
                'datasets_created': len(dataset_files),
                'total_samples': total_samples,
            },
            'total_processing_time_seconds': total_time,
            'samples_per_second': total_samples / total_time if total_time > 0 else 0,
            'dataset_files': dataset_files,
            'config': {
                'currency': self.dataset_config.currency,
                'features': self.dataset_config.features,
                'min_symbol_samples': self.dataset_config.min_symbol_samples,
                'force_uniform': self.dataset_config.force_uniform,
                'nbins': self.dataset_config.nbins,
                'keep_intermediate': self.dataset_config.keep_intermediate,
            }
        }

        if self.verbose:
            print("\n🎉 DATASET BUILDING COMPLETE!")
            print(f"   📊 Total processing time: {total_time:.1f}s")
            print(f"   📊 Datasets created: {len(dataset_files)}")
            print(f"   📊 Total samples: {total_samples:,}")
            print(f"   📈 Processing rate: {results['samples_per_second']:.0f} samples/sec")
            print(f"   📁 Output directory: {output_path}")
            print("   🚀 Comprehensive symbol datasets ready for ML training!")

        return results


# Convenience function
def build_datasets_from_dbn_files(
    config: RepresentConfig,
    dbn_files: Sequence[str | Path],
    output_dir: str | Path,
    dataset_config: DatasetBuildConfig | None = None,
    intermediate_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Convenience function to build comprehensive symbol datasets from multiple DBN files.

    Args:
        config: RepresentConfig with currency-specific configuration
        dbn_files: List of DBN files to process
        output_dir: Directory for final dataset files
        dataset_config: Configuration for dataset building process
        intermediate_dir: Directory for intermediate files (temp if None)
        verbose: Whether to print progress information

    Returns:
        Processing statistics and results
    """
    builder = DatasetBuilder(
        config=config,
        dataset_config=dataset_config,
        verbose=verbose
    )

    return builder.build_datasets_from_dbn_files(
        dbn_files=dbn_files,
        output_dir=output_dir,
        intermediate_dir=intermediate_dir
    )


def batch_build_datasets_from_directory(
    config: RepresentConfig,
    input_directory: str | Path,
    output_dir: str | Path,
    file_pattern: str = "*.dbn*",
    dataset_config: DatasetBuildConfig | None = None,
    intermediate_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build datasets from all DBN files in a directory.

    Args:
        config: RepresentConfig with currency-specific configuration
        input_directory: Directory containing DBN files
        output_dir: Directory for final dataset files
        file_pattern: Pattern to match DBN files
        dataset_config: Configuration for dataset building process
        intermediate_dir: Directory for intermediate files (temp if None)
        verbose: Whether to print progress information

    Returns:
        Processing statistics and results
    """
    input_path = Path(input_directory)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Find all matching DBN files and sort for consistent processing order
    dbn_files = sorted(input_path.glob(file_pattern))

    if not dbn_files:
        raise ValueError(f"No files found matching pattern '{file_pattern}' in {input_path}")

    if verbose:
        print(f"🔍 Found {len(dbn_files)} DBN files in {input_path}")
        for dbn_file in dbn_files:
            print(f"   📄 {dbn_file.name}")

    return build_datasets_from_dbn_files(
        config=config,
        dbn_files=dbn_files,
        output_dir=output_dir,
        dataset_config=dataset_config,
        intermediate_dir=intermediate_dir,
        verbose=verbose
    )
