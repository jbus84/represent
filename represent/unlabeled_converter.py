"""
Unlabeled DBN to Parquet Conversion with Symbol Grouping

This module provides high-performance conversion from DBN files to unlabeled parquet datasets
grouped by symbol for later post-processing classification.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import databento as db
import polars as pl

from .pipeline import MarketDepthProcessor


class UnlabeledDBNConverter:
    """
    High-performance converter from DBN files to unlabeled parquet datasets grouped by symbol.

    Features:
    - Symbol-based file grouping for targeted analysis
    - No classification overhead during conversion
    - Efficient batch processing for large files
    - Market depth feature extraction and preprocessing
    - Symbol filtering for common symbols only
    """

    def __init__(
        self,
        batch_size: int = 2000,
        features: Optional[List[str]] = None,
        min_symbol_samples: int = 1000,
    ):
        """
        Initialize unlabeled DBN converter.

        Args:
            batch_size: Number of samples per processing batch
            features: Features to extract ['volume', 'variance', 'trade_counts']
            min_symbol_samples: Minimum samples required per symbol to create file
        """
        self.batch_size = batch_size
        self.features = features or ["volume"]
        self.min_symbol_samples = min_symbol_samples

        # Initialize market depth processor
        self.processor = MarketDepthProcessor(features=self.features)

        print("🔧 UnlabeledDBNConverter initialized")
        print(f"   📊 Features: {self.features}")
        print(f"   📊 Min samples per symbol: {self.min_symbol_samples:,}")

    def convert_dbn_to_symbol_parquets(
        self,
        dbn_path: Union[str, Path],
        output_dir: Union[str, Path],
        currency: str = "AUDUSD",
        chunk_size: int = 100000,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert DBN file to symbol-grouped unlabeled parquet datasets.

        Args:
            dbn_path: Path to input DBN file
            output_dir: Directory for output parquet files
            currency: Currency pair for naming convention
            chunk_size: Number of rows to process per chunk
            include_metadata: Include additional metadata columns

        Returns:
            Dict with conversion statistics and metadata
        """
        dbn_path = Path(dbn_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not dbn_path.exists():
            raise FileNotFoundError(f"DBN file not found: {dbn_path}")

        print(f"🔄 Converting {dbn_path.name} to symbol-grouped parquet datasets...")
        start_time = time.perf_counter()

        # Load DBN data
        print("1. Loading and analyzing DBN data...")
        store = db.DBNStore.from_file(str(dbn_path))
        df = store.to_df()

        # Convert to polars for performance
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)

        original_rows = len(df)
        print(f"   📊 Loaded {original_rows:,} rows from DBN file")

        # Analyze symbol distribution
        symbol_counts = df.group_by("symbol").len().sort("len", descending=True)
        print(f"   📊 Found {len(symbol_counts)} unique symbols")

        # Filter to symbols with sufficient data
        valid_symbols = symbol_counts.filter(
            pl.col("len") >= self.min_symbol_samples
        )["symbol"].to_list()

        print(f"   📊 {len(valid_symbols)} symbols have ≥{self.min_symbol_samples:,} samples")
        print(f"   📊 Top symbols: {valid_symbols[:5]}")

        # Filter data to valid symbols only
        df = df.filter(pl.col("symbol").is_in(valid_symbols))
        filtered_rows = len(df)
        print(f"   📊 Filtered to {filtered_rows:,} rows for valid symbols")

        # Add metadata columns if requested
        if include_metadata:
            df = self._add_metadata_columns(df, dbn_path)

        # Process data by symbol
        print("2. Processing market depth features by symbol...")
        symbol_stats = {}
        total_processed_samples = 0

        for symbol in valid_symbols:
            print(f"   🔄 Processing symbol: {symbol}")
            
            symbol_data = df.filter(pl.col("symbol") == symbol)
            symbol_samples = []

            # Process symbol data in chunks
            for i in range(0, len(symbol_data), chunk_size):
                chunk_end = min(i + chunk_size, len(symbol_data))
                chunk = symbol_data[i:chunk_end]
                if not isinstance(chunk, pl.DataFrame):
                    continue

                # Generate market depth samples for this chunk
                chunk_samples = self._process_symbol_chunk(chunk, symbol)
                if chunk_samples is not None and len(chunk_samples) > 0:
                    symbol_samples.append(chunk_samples)

            if symbol_samples:
                # Combine all samples for this symbol
                symbol_df = pl.concat(symbol_samples)
                
                # Save symbol-specific parquet file
                symbol_filename = f"{currency}_{symbol}.parquet"
                symbol_path = output_dir / symbol_filename
                
                symbol_df.write_parquet(
                    str(symbol_path), 
                    compression="snappy", 
                    statistics=True, 
                    row_group_size=50000
                )

                symbol_stats[symbol] = {
                    "samples": len(symbol_df),
                    "file_path": str(symbol_path),
                    "file_size_mb": symbol_path.stat().st_size / 1024 / 1024
                }
                
                total_processed_samples += len(symbol_df)
                print(f"      ✅ Saved {len(symbol_df):,} samples to {symbol_filename}")

        end_time = time.perf_counter()
        conversion_time = end_time - start_time

        # Generate conversion statistics
        stats = {
            "input_file": str(dbn_path),
            "output_directory": str(output_dir),
            "original_rows": original_rows,
            "filtered_rows": filtered_rows,
            "total_processed_samples": total_processed_samples,
            "symbols_processed": len(symbol_stats),
            "conversion_time_seconds": conversion_time,
            "samples_per_second": total_processed_samples / conversion_time if conversion_time > 0 else 0,
            "features": self.features,
            "currency": currency,
            "symbol_stats": symbol_stats,
            "valid_symbols": valid_symbols,
        }

        print("✅ Conversion complete!")
        print(f"   📊 Processed {len(symbol_stats)} symbols")
        print(f"   📊 Generated {total_processed_samples:,} total samples")
        print(f"   📊 Processing rate: {stats['samples_per_second']:.1f} samples/second")
        print(f"   📊 Output directory: {output_dir}")

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

    def _process_symbol_chunk(
        self, chunk: pl.DataFrame, symbol: str
    ) -> Optional[pl.DataFrame]:
        """
        Process a chunk of symbol-specific data to generate market depth features.

        Args:
            chunk: Data chunk for specific symbol
            symbol: Symbol identifier

        Returns:
            DataFrame with market depth features (no classification labels)
        """
        if len(chunk) < self.batch_size:
            return None

        samples = []

        # Generate sliding windows for market depth processing
        max_start_pos = len(chunk) - self.batch_size

        if max_start_pos <= 0:
            return None

        # Process samples with stride for efficiency
        stride = max(1, self.batch_size // 10)

        for start_pos in range(0, max_start_pos, stride):
            end_pos = start_pos + self.batch_size

            # Extract sample window
            sample_window = chunk[start_pos:end_pos]

            # Generate market depth representation
            try:
                market_depth_representation = self.processor.process(sample_window)

                # Convert to numpy array if it's a tensor
                import numpy as np
                if not isinstance(market_depth_representation, np.ndarray):
                    features_array = market_depth_representation.numpy()
                else:
                    features_array = market_depth_representation
                
                # Ensure we have a numpy array
                if not hasattr(features_array, 'tobytes'):
                    features_array = np.array(features_array)

                # Create sample record (no classification label)
                sample_record = {
                    "symbol": symbol,
                    "market_depth_features": features_array.tobytes(),  # Serialized array
                    "feature_shape": str(features_array.shape),
                    "start_timestamp": sample_window["ts_event"].min(),
                    "end_timestamp": sample_window["ts_event"].max(),
                    "global_start_idx": start_pos,
                    "global_end_idx": end_pos,
                    "sample_id": f"{symbol}_{start_pos}_{end_pos}",
                }

                # Add metadata if available
                if "source_file" in chunk.columns:
                    sample_record["source_file"] = sample_window["source_file"][0]
                if "date" in chunk.columns:
                    sample_record["date"] = sample_window["date"][0]
                if "hour" in chunk.columns:
                    sample_record["hour"] = sample_window["hour"][0]

                # Add price information for later classification
                sample_record["start_mid_price"] = (
                    sample_window["bid_px_00"][0] + sample_window["ask_px_00"][0]
                ) / 2
                sample_record["end_mid_price"] = (
                    sample_window["bid_px_00"][-1] + sample_window["ask_px_00"][-1]
                ) / 2

                samples.append(sample_record)

            except Exception:
                # Skip samples that fail processing
                continue

        if not samples:
            return None

        # Convert to DataFrame
        return pl.DataFrame(samples)


def convert_dbn_to_parquet(
    dbn_path: Union[str, Path],
    output_dir: Union[str, Path],
    currency: str = "AUDUSD",
    features: Optional[List[str]] = None,
    group_by_symbol: bool = True,
    min_symbol_samples: int = 1000,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to convert a single DBN file to symbol-grouped parquet.

    Args:
        dbn_path: Path to input DBN file
        output_dir: Directory for output parquet files
        currency: Currency pair for naming convention
        features: Features to extract ['volume', 'variance', 'trade_counts']
        group_by_symbol: Create separate files per symbol
        min_symbol_samples: Minimum samples required per symbol
        **kwargs: Additional arguments for converter

    Returns:
        Conversion statistics dictionary

    Examples:
        # Convert to symbol-grouped parquet files
        stats = convert_dbn_to_parquet(
            'data.dbn', 
            '/data/parquet/', 
            currency='AUDUSD',
            features=['volume', 'variance']
        )
    """
    converter = UnlabeledDBNConverter(
        features=features or ["volume"],
        min_symbol_samples=min_symbol_samples
    )

    return converter.convert_dbn_to_symbol_parquets(
        dbn_path=dbn_path,
        output_dir=output_dir,
        currency=currency,
        **kwargs
    )


def batch_convert_dbn_files(
    input_directory: Union[str, Path],
    output_directory: Union[str, Path],
    currency: str = "AUDUSD",
    features: Optional[List[str]] = None,
    pattern: str = "*.dbn*",
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Convert multiple DBN files to symbol-grouped parquet datasets.

    Args:
        input_directory: Directory containing DBN files
        output_directory: Directory for output parquet files
        currency: Currency pair for naming convention
        features: Features to extract
        pattern: File pattern to match
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
        try:
            stats = convert_dbn_to_parquet(
                dbn_path=dbn_file,
                output_dir=output_dir,
                currency=currency,
                features=features,
                **kwargs,
            )
            results.append(stats)

        except Exception as e:
            print(f"❌ Failed to convert {dbn_file.name}: {e}")
            continue

    print(f"✅ Batch conversion complete! Processed {len(results)} files successfully")

    return results