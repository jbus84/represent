"""
DBN to Parquet Conversion with Classification Labeling

This module provides high-performance conversion from DBN files to labeled parquet datasets
for market depth machine learning workflows.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import databento as db
import polars as pl

from .config import ClassificationConfig, load_currency_config, load_config_from_file
from .constants import MICRO_PIP_SIZE
from .pipeline import MarketDepthProcessor


class DBNToParquetConverter:
    """
    High-performance converter from DBN files to labeled parquet datasets.

    Features:
    - Automatic classification labeling based on price movement
    - Currency-specific configuration support
    - Efficient batch processing for large files
    - Market depth feature extraction and preprocessing
    - PyTorch-compatible output format
    """

    def __init__(
        self,
        classification_config: Optional[Union[Dict[str, Any], ClassificationConfig]] = None,
        currency: Optional[str] = None,
        config_file: Optional[Union[str, Path]] = None,
        batch_size: int = 2000,  # More reasonable batch size for feature extraction
        features: Optional[List[str]] = None,
    ):
        """
        Initialize DBN to Parquet converter.

        Args:
            classification_config: Classification configuration or dict (lowest priority)
            currency: Currency pair (e.g., 'AUDUSD') for predefined configs (medium priority)
            config_file: Path to custom YAML/JSON config file (highest priority)
            batch_size: Number of samples per processing batch
            features: Features to extract ['volume', 'variance', 'trade_counts']

        Configuration Priority (highest to lowest):
            1. config_file (custom YAML/JSON file)
            2. currency (predefined currency-specific config)
            3. classification_config (manual config dict/object)
            4. Default AUDUSD config
        """
        self.batch_size = batch_size
        self.features = features or ["volume"]

        # Load configuration with priority order
        if config_file:
            # Highest priority: custom config file
            currency_config = load_config_from_file(config_file)
            self.classification_config = currency_config.classification
            self.currency = currency_config.currency_pair
            print(f"📋 Loaded config from file: {config_file}")
        elif currency:
            # Medium priority: predefined currency config
            currency_config = load_currency_config(currency)
            self.classification_config = currency_config.classification
            self.currency = currency.upper()
            print(f"📋 Loaded predefined config for: {self.currency}")
        else:
            # Lower priority: manual config or default
            if isinstance(classification_config, ClassificationConfig):
                self.classification_config = classification_config
                self.currency = "CUSTOM"
            elif classification_config is not None:
                self.classification_config = ClassificationConfig(**classification_config)
                self.currency = "CUSTOM"
            else:
                # Lowest priority: default AUDUSD configuration
                default_config = load_currency_config("AUDUSD")
                self.classification_config = default_config.classification
                self.currency = "AUDUSD"
                print("📋 Using default AUDUSD configuration")

        # Initialize market depth processor
        self.processor = MarketDepthProcessor(features=self.features)

    def convert_dbn_to_parquet(
        self,
        dbn_path: Union[str, Path],
        output_path: Union[str, Path],
        symbol_filter: Optional[str] = None,
        chunk_size: int = 100000,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert DBN file to labeled parquet dataset.

        Args:
            dbn_path: Path to input DBN file
            output_path: Path for output parquet file
            symbol_filter: Optional symbol to filter (e.g., 'M6AM4')
            chunk_size: Number of rows to process per chunk
            include_metadata: Include additional metadata columns

        Returns:
            Dict with conversion statistics and metadata
        """
        dbn_path = Path(dbn_path)
        output_path = Path(output_path)

        if not dbn_path.exists():
            raise FileNotFoundError(f"DBN file not found: {dbn_path}")

        print(f"🔄 Converting {dbn_path.name} to labeled parquet dataset...")
        start_time = time.perf_counter()

        # Load DBN data
        print("1. Loading DBN data...")
        store = db.DBNStore.from_file(str(dbn_path))
        df = store.to_df()

        # Convert to polars for performance
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)

        original_rows = len(df)
        print(f"   📊 Loaded {original_rows:,} rows from DBN file")

        # Apply symbol filter if specified
        if symbol_filter:
            df = df.filter(pl.col("symbol") == symbol_filter)
            filtered_rows = len(df)
            print(f"   📊 Filtered to {filtered_rows:,} rows for symbol '{symbol_filter}'")

        # Add metadata columns if requested
        if include_metadata:
            df = self._add_metadata_columns(df, dbn_path)

        # Process data in chunks for memory efficiency
        print("2. Processing market depth features and generating labels...")
        labeled_chunks = []
        total_labeled_samples = 0

        for i in range(0, len(df), chunk_size):
            chunk_end = min(i + chunk_size, len(df))
            chunk = df[i:chunk_end]

            # Generate labeled samples for this chunk
            # Type guard to ensure chunk is DataFrame
            if isinstance(chunk, pl.DataFrame):
                labeled_chunk = self._process_chunk_with_labels(chunk, i)
            else:
                continue

            if labeled_chunk is not None and len(labeled_chunk) > 0:
                labeled_chunks.append(labeled_chunk)
                total_labeled_samples += len(labeled_chunk)

            if (i // chunk_size + 1) % 10 == 0:
                print(f"   📊 Processed {i + len(chunk):,} / {len(df):,} rows...")

        if not labeled_chunks:
            raise ValueError("No labeled samples generated - check data and configuration")

        # Combine all labeled chunks
        print("3. Combining labeled chunks and saving to parquet...")
        final_df = pl.concat(labeled_chunks)

        # Save to parquet with optimal compression
        final_df.write_parquet(
            str(output_path), compression="snappy", statistics=True, row_group_size=50000
        )

        end_time = time.perf_counter()
        conversion_time = end_time - start_time

        # Generate conversion statistics
        stats = {
            "input_file": str(dbn_path),
            "output_file": str(output_path),
            "original_rows": original_rows,
            "labeled_samples": total_labeled_samples,
            "conversion_time_seconds": conversion_time,
            "samples_per_second": total_labeled_samples / conversion_time,
            "features": self.features,
            "classification_config": self.classification_config.model_dump(),
            "currency": getattr(self, "currency", "Unknown"),
            "symbol_filter": symbol_filter,
            "output_file_size_mb": output_path.stat().st_size / 1024 / 1024,
        }

        print("✅ Conversion complete!")
        print(f"   📊 Generated {total_labeled_samples:,} labeled samples")
        print(f"   📊 Processing rate: {stats['samples_per_second']:.1f} samples/second")
        print(f"   📊 Output file: {stats['output_file_size_mb']:.1f}MB")
        print(f"   📊 Saved to: {output_path}")

        return stats

    def _add_metadata_columns(self, df: pl.DataFrame, source_path: Path) -> pl.DataFrame:
        """Add metadata columns for tracking and analysis."""
        return df.with_columns(
            [
                pl.lit(source_path.name).alias("source_file"),
                pl.lit(
                    source_path.stem.split("-")[-1] if "-" in source_path.stem else "unknown"
                ).alias("file_date"),
                pl.col("ts_event").alias("timestamp"),
                pl.col("ts_event").dt.date().alias("date"),
                pl.col("ts_event").dt.hour().alias("hour"),
                pl.int_range(pl.len()).alias("row_id"),
            ]
        )

    def _process_chunk_with_labels(
        self, chunk: pl.DataFrame, chunk_offset: int
    ) -> Optional[pl.DataFrame]:
        """
        Process a chunk of data to generate market depth features and classification labels.

        Args:
            chunk: Data chunk to process
            chunk_offset: Offset of this chunk in the original dataset

        Returns:
            DataFrame with market depth features and classification labels
        """
        if len(chunk) < self.batch_size:
            return None

        labeled_samples = []

        # Generate sliding windows for market depth processing
        lookforward_offset = self.classification_config.lookforward_offset
        lookforward_input = self.classification_config.lookforward_input
        min_lookforward = lookforward_offset + lookforward_input

        # Calculate valid sample positions
        max_start_pos = len(chunk) - self.batch_size - min_lookforward

        if max_start_pos <= 0:
            return None

        # Process samples with stride for efficiency
        stride = max(1, self.batch_size // 10)  # Process every 10th possible sample

        for start_pos in range(0, max_start_pos, stride):
            end_pos = start_pos + self.batch_size
            target_pos = end_pos + lookforward_offset

            if target_pos + lookforward_input >= len(chunk):
                break

            # Extract sample window
            sample_window = chunk[start_pos:end_pos]

            # Generate market depth representation
            try:
                market_depth_tensor = self.processor.process(sample_window)

                # Generate classification label
                classification_label = self._generate_classification_label(
                    chunk, target_pos, lookforward_input
                )

                # Create labeled sample record
                # Convert to numpy array if it's a tensor, otherwise use as-is
                if hasattr(market_depth_tensor, "numpy"):
                    features_array = market_depth_tensor.numpy()  # type: ignore
                else:
                    features_array = market_depth_tensor
                
                # Ensure we have a numpy array
                if not hasattr(features_array, 'tobytes'):
                    import numpy as np
                    features_array = np.array(features_array)

                sample_record = {
                    "market_depth_features": features_array.tobytes(),  # Serialize array
                    "classification_label": classification_label,
                    "feature_shape": str(features_array.shape),
                    "start_timestamp": sample_window["ts_event"].min(),
                    "end_timestamp": sample_window["ts_event"].max(),
                    "target_timestamp": chunk["ts_event"][target_pos],
                    "global_start_idx": chunk_offset + start_pos,
                    "global_end_idx": chunk_offset + end_pos,
                    "sample_id": f"{chunk_offset}_{start_pos}_{end_pos}",
                }

                # Add metadata if available
                if "symbol" in chunk.columns:
                    sample_record["symbol"] = sample_window["symbol"][0]
                if "source_file" in chunk.columns:
                    sample_record["source_file"] = sample_window["source_file"][0]
                if "date" in chunk.columns:
                    sample_record["date"] = sample_window["date"][0]

                labeled_samples.append(sample_record)

            except Exception:
                # Skip samples that fail processing
                continue

        if not labeled_samples:
            return None

        # Convert to DataFrame
        return pl.DataFrame(labeled_samples)

    def _generate_classification_label(
        self, data: pl.DataFrame, target_pos: int, lookforward_window: int
    ) -> int:
        """
        Generate classification label based on price movement in lookforward window.

        Args:
            data: Full data chunk
            target_pos: Position to start measuring price movement
            lookforward_window: Number of ticks to look forward

        Returns:
            Classification label (0=down, 1=neutral, 2=up)
        """
        if target_pos + lookforward_window >= len(data):
            return 1  # Neutral if insufficient data

        # Get mid prices for the lookforward window
        lookforward_data = data[target_pos : target_pos + lookforward_window]

        # Calculate mid price as average of best bid/ask
        start_mid = (lookforward_data["bid_px_00"][0] + lookforward_data["ask_px_00"][0]) / 2
        end_mid = (lookforward_data["bid_px_00"][-1] + lookforward_data["ask_px_00"][-1]) / 2

        # Calculate price movement in micro pips
        price_movement = (end_mid - start_mid) / MICRO_PIP_SIZE

        # Apply classification based on bin thresholds
        return self._classify_price_movement(abs(price_movement))

    def _classify_price_movement(self, abs_movement: float) -> int:
        """
        Classify price movement using bin thresholds.

        Args:
            abs_movement: Absolute price movement in micro pips

        Returns:
            Classification label (0 to nbins-1)
        """
        # Get thresholds for current configuration
        nbins = self.classification_config.nbins
        ticks_per_bin = self.classification_config.ticks_per_bin
        lookforward_input = self.classification_config.lookforward_input

        # Get the appropriate threshold set
        bin_thresholds = (
            self.classification_config.bin_thresholds.get(nbins, {})
            .get(ticks_per_bin, {})
            .get(lookforward_input, {})
        )

        if not bin_thresholds:
            # Fallback: simple 3-class classification
            if abs_movement > 3.0:
                return 2  # Large movement
            elif abs_movement > 1.0:
                return 1  # Medium movement
            else:
                return 0  # Small movement

        # Find the appropriate bin based on thresholds
        thresholds = []
        for i in range(1, (nbins // 2) + 1):
            threshold_key = f"bin_{i}"
            if threshold_key in bin_thresholds:
                thresholds.append(bin_thresholds[threshold_key])

        if not thresholds:
            return nbins // 2  # Return middle bin if no thresholds

        # Classify based on thresholds
        for i, threshold in enumerate(thresholds):
            if abs_movement <= threshold:
                return i

        # If movement exceeds all thresholds, return highest bin
        return nbins - 1


def convert_dbn_file(
    dbn_path: Union[str, Path],
    output_path: Union[str, Path],
    currency: Optional[str] = None,
    config_file: Optional[Union[str, Path]] = None,
    symbol_filter: Optional[str] = None,
    features: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to convert a single DBN file to labeled parquet.

    Args:
        dbn_path: Path to input DBN file
        output_path: Path for output parquet file
        currency: Currency pair for predefined configuration (e.g., 'AUDUSD')
        config_file: Path to custom YAML/JSON configuration file
        symbol_filter: Symbol to filter (e.g., 'M6AM4')
        features: Features to extract ['volume', 'variance', 'trade_counts']
        **kwargs: Additional arguments for converter

    Returns:
        Conversion statistics dictionary

    Examples:
        # Use predefined currency config
        convert_dbn_file('data.dbn', 'output.parquet', currency='AUDUSD')

        # Use custom config file
        convert_dbn_file('data.dbn', 'output.parquet', config_file='my_config.yaml')
    """
    converter = DBNToParquetConverter(
        currency=currency, config_file=config_file, features=features or ["volume"]
    )

    return converter.convert_dbn_to_parquet(
        dbn_path=dbn_path, output_path=output_path, symbol_filter=symbol_filter, **kwargs
    )


def batch_convert_dbn_files(
    input_directory: Union[str, Path],
    output_directory: Union[str, Path],
    currency: Optional[str] = None,
    config_file: Optional[Union[str, Path]] = None,
    pattern: str = "*.dbn*",
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Convert multiple DBN files to labeled parquet datasets.

    Args:
        input_directory: Directory containing DBN files
        output_directory: Directory for output parquet files
        currency: Currency pair for predefined configuration
        config_file: Path to custom YAML/JSON configuration file
        pattern: File pattern to match (default: "*.dbn*")
        **kwargs: Additional arguments for converter

    Returns:
        List of conversion statistics for each file
    """
    input_dir = Path(input_directory)
    output_dir = Path(output_directory)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all DBN files
    dbn_files = list(input_dir.glob(pattern))

    if not dbn_files:
        raise ValueError(f"No DBN files found in {input_dir} matching pattern '{pattern}'")

    print(f"🔄 Found {len(dbn_files)} DBN files to convert")

    results = []

    for dbn_file in dbn_files:
        output_file = output_dir / f"{dbn_file.stem}_labeled.parquet"

        try:
            stats = convert_dbn_file(
                dbn_path=dbn_file,
                output_path=output_file,
                currency=currency,
                config_file=config_file,
                **kwargs,
            )
            results.append(stats)

        except Exception as e:
            print(f"❌ Failed to convert {dbn_file.name}: {e}")
            continue

    print(f"✅ Batch conversion complete! Processed {len(results)} files successfully")

    return results
