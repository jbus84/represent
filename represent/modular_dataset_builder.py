"""
Modular Dataset Builder

This module provides a dataset builder that uses pluggable target generators
to create datasets with multiple target types (classification and regression).
"""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

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

        # Check and rename price column if needed
        if 'price' in symbol_df.columns and 'mid_price' not in symbol_df.columns:
            symbol_df = symbol_df.rename({'price': 'mid_price'})

        if self.verbose:
            print(f"   📊 Loaded {len(symbol_df):,} rows")

        # Extract symbol from filename if not provided
        if symbol is None:
            symbol = parquet_path.stem.split('_')[0] if '_' in parquet_path.stem else None

        # Build targets
        return self.build_targets(symbol_df, symbol=symbol)

    def build_targets_from_parquet_chunked(
        self,
        parquet_path: str | Path,
        symbol: str | None = None,
        chunk_size: int = 500_000
    ) -> pl.DataFrame:
        """
        Build targets from parquet file using chunked processing for memory efficiency.

        Args:
            parquet_path: Path to input parquet file
            symbol: Optional symbol identifier
            chunk_size: Number of samples to process per chunk (default: 500K)

        Returns:
            DataFrame with generated targets (keys + targets only)
        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        # Get dataset info without loading all data
        df_scan = pl.scan_parquet(parquet_path)
        total_rows = df_scan.select(pl.len()).collect().item()
        schema = df_scan.collect_schema()

        if self.verbose:
            print(f"📂 Processing file: {parquet_path.name}")
            print(f"   📊 Dataset has {total_rows:,} samples")
            print(f"   🔄 Processing in chunks of {chunk_size:,}")

        # Extract symbol from filename if not provided
        if symbol is None:
            symbol = parquet_path.stem.split('_')[0]

        # Ensure required columns exist - check for mid_price or price
        has_mid_price = 'mid_price' in schema
        has_price = 'price' in schema

        if not has_mid_price and not has_price:
            raise ValueError(f"Required column 'mid_price' or 'price' not found in {parquet_path}")

        # Use price column name that exists
        price_column = 'mid_price' if has_mid_price else 'price'

        all_target_chunks = []

        # Process each generator
        for i, generator in enumerate(self.target_generators):
            if self.verbose:
                print(f"   🎯 Generator {i+1}/{len(self.target_generators)}: {generator.__class__.__name__}")

            # Check if generator needs lookback data (for adaptive methods)
            lookback_window = getattr(generator, 'lookback_window', 0)
            lookforward_window = getattr(generator, 'lookforward_window', 0)

            # TEMPORARY FIX: Always use standard processing to ensure generator delegation
            # This ensures our metadata implementation is used
            if self.verbose:
                if lookback_window > 0 or lookforward_window > 0:
                    print(f"      🔄 Using standard processing with generator delegation (lookback: {lookback_window:,}, lookforward: {lookforward_window:,})")
                else:
                    print(f"      🔄 Using standard chunking")

            generator_targets = self._process_with_standard_chunks(
                generator, parquet_path, symbol, price_column,
                chunk_size, total_rows
            )

            # Remove duplicates after chunk concatenation
            generator_targets = self._remove_duplicates(
                generator_targets,
                verbose_prefix=f"      "
            )

            all_target_chunks.append(generator_targets)

        # Combine all generators on the same keys
        if len(all_target_chunks) == 1:
            return all_target_chunks[0]

        # Join all target DataFrames on keys
        result_df = all_target_chunks[0]
        for target_df in all_target_chunks[1:]:
            # Join on row keys
            if 'row_idx' in result_df.columns and 'row_idx' in target_df.columns:
                result_df = result_df.join(target_df, on='row_idx', how='left')
            else:
                # Fallback: horizontal concatenation (assuming same order)
                # Drop overlapping keys from subsequent DataFrames
                target_cols_only = target_df.drop(['row_idx', 'timestamp'] if 'timestamp' in target_df.columns else ['row_idx'])
                result_df = pl.concat([result_df, target_cols_only], how="horizontal")

        if self.verbose:
            print(f"   ✅ Generated {len(result_df):,} target rows with {len(result_df.columns)} columns")

        return result_df

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

    def _remove_duplicates(self, df: pl.DataFrame, verbose_prefix: str = "") -> pl.DataFrame:
        """Remove duplicate rows using smart column-based deduplication."""
        before_len = len(df)

        # For target data with fixed row_idx, row_idx should be unique
        # so we primarily check for timestamp-based duplicates
        timestamp_col = "timestamp" if "timestamp" in df.columns else "ts_event"

        # Smart column detection for deduplication (most granular first)
        if "seqnum" in df.columns and timestamp_col in df.columns:
            dedup_subset = [timestamp_col, "seqnum"]
        elif "ts_recv" in df.columns and timestamp_col in df.columns:
            dedup_subset = [timestamp_col, "ts_recv"]
        elif "symbol" in df.columns and timestamp_col in df.columns:
            dedup_subset = [timestamp_col, "symbol"]
        elif timestamp_col in df.columns:
            dedup_subset = [timestamp_col]
        elif "row_idx" in df.columns:
            # If we have row_idx but no timestamps, use row_idx for deduplication
            dedup_subset = ["row_idx"]
        else:
            # Fallback: drop exact duplicate rows
            deduplicated = df.unique(maintain_order=True)
            after_len = len(deduplicated)
            if self.verbose and before_len != after_len:
                print(f"{verbose_prefix}🧹 Removed {before_len - after_len:,} duplicate rows")
            return deduplicated

        deduplicated = df.unique(subset=dedup_subset, maintain_order=True)
        after_len = len(deduplicated)

        if self.verbose and before_len != after_len:
            print(f"{verbose_prefix}🧹 Removed {before_len - after_len:,} duplicate rows using {dedup_subset}")

        return deduplicated

    def _process_with_standard_chunks(
        self,
        generator,
        parquet_path: Path,
        symbol: str,
        price_column: str,
        chunk_size: int,
        total_rows: int
    ) -> pl.DataFrame:
        """Process generator using standard (non-overlapping) chunks."""
        from tqdm import tqdm

        generator_chunks = []

        # Process in chunks with progress bar
        with tqdm(total=total_rows, desc=f"      Processing samples", unit="samples", leave=False) as pbar:
            for offset in range(0, total_rows, chunk_size):
                # Read chunk efficiently using Polars slice
                current_chunk_size = min(chunk_size, total_rows - offset)
                chunk_df = pl.scan_parquet(parquet_path).slice(offset, current_chunk_size).collect()

                # Rename price column to mid_price if needed for compatibility
                if price_column == 'price':
                    chunk_df = chunk_df.rename({'price': 'mid_price'})

                # Generate targets for this chunk
                chunk_targets = generator.generate_targets(chunk_df, symbol=symbol)

                # Fix row_idx to be continuous across chunks
                if 'row_idx' in chunk_targets.columns:
                    chunk_targets = chunk_targets.with_columns(
                        (pl.col('row_idx') + offset).alias('row_idx')
                    )

                # Store the chunk targets
                generator_chunks.append(chunk_targets)

                pbar.update(len(chunk_df))

        # Concatenate all chunks for this generator
        return pl.concat(generator_chunks)

    def _process_with_overlapped_chunks(
        self,
        generator,
        parquet_path: Path,
        symbol: str,
        price_column: str,
        chunk_size: int,
        lookback_window: int,
        total_rows: int
    ) -> pl.DataFrame:
        """Process generator using overlapped chunks to preserve lookback windows."""
        from tqdm import tqdm

        generator_chunks = []
        processed_rows = 0

        # Get lookforward window for proper chunk extension
        lookforward_window = getattr(generator, 'lookforward_window', 0)

        # Process in chunks with overlap for lookback and lookforward data
        with tqdm(total=total_rows, desc=f"      Processing samples", unit="samples", leave=False) as pbar:
            offset = 0

            while offset < total_rows:
                # For first chunk, start from 0
                # For subsequent chunks, include lookback_window overlap
                chunk_start = max(0, offset - lookback_window) if offset > 0 else 0

                # CRITICAL FIX: Extend chunk_end to include lookforward window
                # This ensures the generator has sufficient future data to compute proper labels
                chunk_end_base = min(offset + chunk_size, total_rows)
                chunk_end = min(chunk_end_base + lookforward_window, total_rows)

                # Read chunk with lookback AND lookforward data
                chunk_df = pl.scan_parquet(parquet_path).slice(
                    chunk_start, chunk_end - chunk_start
                ).collect()

                # Rename price column to mid_price if needed for compatibility
                if price_column == 'price':
                    chunk_df = chunk_df.rename({'price': 'mid_price'})

                # Generate targets for this chunk
                chunk_targets = generator.generate_targets(chunk_df, symbol=symbol)

                # Fix row_idx to be continuous across chunks
                if 'row_idx' in chunk_targets.columns:
                    chunk_targets = chunk_targets.with_columns(
                        (pl.col('row_idx') + chunk_start).alias('row_idx')
                    )

                # For overlapped chunks, only keep the new data (not the overlapped part)
                if offset > 0:
                    # Filter to only include rows from the new part of the chunk
                    chunk_targets = chunk_targets.filter(
                        pl.col('row_idx') >= offset
                    )

                # The extended chunk now has sufficient lookforward data
                # Let the generator handle natural boundaries - no additional filtering needed

                # Store the chunk targets
                if len(chunk_targets) > 0:
                    generator_chunks.append(chunk_targets)
                    processed_rows += len(chunk_targets)

                pbar.update(min(chunk_size, chunk_end_base - offset))
                offset = chunk_end_base

        # Concatenate all chunks for this generator
        if generator_chunks:
            return pl.concat(generator_chunks)
        else:
            # Return empty DataFrame with expected schema
            return generator.generate_targets(pl.DataFrame({"mid_price": []}), symbol=symbol)

    def _process_with_two_pass_method(
        self,
        generator,
        parquet_path: Path,
        symbol: str,
        price_column: str,
        chunk_size: int
    ) -> pl.DataFrame:
        """Process generator using memory-efficient two-pass method for adaptive methods."""
        from tqdm import tqdm

        lookback_window = getattr(generator, 'lookback_window', 0)
        lookforward_window = getattr(generator, 'lookforward_window', 0)
        barrier_width = getattr(generator, 'barrier_width', 1.0)

        # Get total rows
        total_rows = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()

        if self.verbose:
            print(f"         📊 Two-pass processing {total_rows:,} samples")

        # PASS 1: Compute volatility for entire dataset
        if self.verbose:
            print(f"         📊 Pass 1: Computing volatility stream")

        volatilities = np.full(total_rows, np.nan, dtype=np.float32)

        with tqdm(total=total_rows, desc="         Computing volatility", unit="samples", leave=False) as pbar:
            offset = 0

            while offset < total_rows:
                # Calculate chunk bounds with lookback overlap
                chunk_start = max(0, offset - lookback_window) if offset > 0 else 0
                chunk_end = min(offset + chunk_size, total_rows)

                # Load chunk
                chunk_df = pl.scan_parquet(parquet_path).slice(
                    chunk_start, chunk_end - chunk_start
                ).collect()

                # Clean and extract prices
                if price_column == 'price':
                    chunk_df = chunk_df.rename({'price': 'mid_price'})

                # Use existing mid_price column instead of calculating from bid/ask
                chunk_clean = chunk_df.filter(
                    pl.col(price_column).is_not_null() &
                    pl.col(price_column).is_finite()
                )

                if len(chunk_clean) == 0:
                    pbar.update(chunk_end - offset)
                    offset = chunk_end
                    continue

                prices = chunk_clean['mid_price'].to_numpy()

                # Compute volatility for this chunk
                for i in range(len(prices)):
                    global_idx = chunk_start + i
                    if global_idx >= total_rows:
                        break

                    if i >= lookback_window:  # Sufficient lookback data
                        vol_window = prices[max(0, i - lookback_window):i + 1]
                        volatilities[global_idx] = np.std(vol_window)

                pbar.update(chunk_end - offset)
                offset = chunk_end

        # PASS 2: Apply triple barrier using pre-computed volatility
        if self.verbose:
            print(f"         🎯 Pass 2: Applying triple barrier labels")

        all_results = []

        with tqdm(total=total_rows, desc="         Applying barriers", unit="samples", leave=False) as pbar:
            offset = 0

            while offset < total_rows:
                # Calculate chunk bounds with lookforward extension
                chunk_start = offset
                chunk_end_base = min(offset + chunk_size, total_rows)
                chunk_end = min(chunk_end_base + lookforward_window, total_rows)

                # Load chunk with lookforward data
                chunk_df = pl.scan_parquet(parquet_path).slice(
                    chunk_start, chunk_end - chunk_start
                ).collect()

                # Clean chunk
                if price_column == 'price':
                    chunk_df = chunk_df.rename({'price': 'mid_price'})

                # Use existing price column instead of calculating from bid/ask
                chunk_clean = chunk_df.filter(
                    pl.col(price_column).is_not_null() &
                    pl.col(price_column).is_finite()
                )

                if len(chunk_clean) == 0:
                    pbar.update(chunk_end_base - offset)
                    offset = chunk_end_base
                    continue

                # Apply triple barrier with pre-computed volatility
                chunk_labels = self._compute_two_pass_labels(
                    chunk_clean, volatilities[chunk_start:chunk_end],
                    lookforward_window, barrier_width, chunk_start
                )

                # Only keep labels for the main chunk (not the lookforward extension)
                main_chunk_size = chunk_end_base - chunk_start
                if len(chunk_labels) > main_chunk_size:
                    chunk_labels = chunk_labels[:main_chunk_size]
                    chunk_clean = chunk_clean[:main_chunk_size]

                # Create result DataFrame
                result_df = self._create_two_pass_result(chunk_clean, chunk_labels, chunk_start, symbol, barrier_width)
                all_results.append(result_df)

                pbar.update(len(chunk_labels))
                offset = chunk_end_base

        # Concatenate all results
        return pl.concat(all_results) if all_results else pl.DataFrame()

    def _compute_two_pass_labels(self, chunk_df: pl.DataFrame, chunk_volatilities: np.ndarray,
                               lookforward_window: int, barrier_width: float, global_offset: int) -> np.ndarray:
        """Compute triple barrier labels using pre-computed volatilities."""

        prices = chunk_df['mid_price'].to_numpy()
        labels = np.zeros(len(prices), dtype=np.int32)

        # Process each position
        for i in range(len(prices)):
            vol_idx = min(i, len(chunk_volatilities) - 1)

            # Skip if insufficient data
            if (i + lookforward_window >= len(prices) or
                vol_idx >= len(chunk_volatilities) or
                np.isnan(chunk_volatilities[vol_idx])):
                continue

            entry_price = prices[i]
            volatility = chunk_volatilities[vol_idx]

            # Calculate barriers using pre-computed volatility
            upper_threshold = entry_price + (volatility * barrier_width)
            lower_threshold = entry_price - (volatility * barrier_width)

            # Look ahead for barrier hits
            future_prices = prices[i + 1:i + 1 + lookforward_window]

            # Find first barrier hit
            upper_hits = np.where(future_prices >= upper_threshold)[0]
            lower_hits = np.where(future_prices <= lower_threshold)[0]

            if len(upper_hits) > 0 and len(lower_hits) > 0:
                # Both barriers hit - use the first one
                if upper_hits[0] < lower_hits[0]:
                    labels[i] = 1  # Long signal
                else:
                    labels[i] = -1  # Short signal
            elif len(upper_hits) > 0:
                labels[i] = 1  # Long signal
            elif len(lower_hits) > 0:
                labels[i] = -1  # Short signal
            else:
                labels[i] = 0  # Timeout

        return labels

    def _create_two_pass_result(self, chunk_df: pl.DataFrame, labels: np.ndarray,
                              global_offset: int, symbol: str, barrier_width: float) -> pl.DataFrame:
        """Create result DataFrame for two-pass processing."""

        # Create row indices
        row_indices = np.arange(global_offset, global_offset + len(labels))

        # Get timestamps
        if 'ts_event' in chunk_df.columns:
            timestamps = chunk_df['ts_event'].to_numpy()[:len(labels)]
        elif 'timestamp' in chunk_df.columns:
            timestamps = chunk_df['timestamp'].to_numpy()[:len(labels)]
        else:
            timestamps = row_indices  # Fallback

        # Create base DataFrame for targets
        result_df = pl.DataFrame({
            'row_idx': row_indices,
            'symbol': [symbol] * len(labels),
            'timestamp': timestamps,
            'adaptive_triple_barrier_label': labels,
            'adaptive_triple_barrier_label_return': np.zeros(len(labels), dtype=np.float32),
            'adaptive_triple_barrier_label_barrier_width': np.full(len(labels), barrier_width)
        })

        return result_df


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
