"""
DBN-to-Classified-Parquet Processor (Streamlined Approach)

This module processes DBN files directly to classified parquet files,
ensuring true uniform distribution and efficient ML training preparation.

Key Features:
- Single-pass processing: DBN → Classified Parquet
- Symbol-level processing with full context
- True uniform distribution using full dataset
- Row filtering for insufficient history/future data
- Pre-computed classification labels for ML training
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import polars as pl
import databento as db
from dataclasses import dataclass

from .config import create_represent_config
from .constants import MICRO_PIP_SIZE
from .global_threshold_calculator import GlobalThresholds


@dataclass
class ClassificationConfig:
    """Configuration for classification processing."""
    currency: str = "AUDUSD"
    features: Optional[List[str]] = None
    input_rows: int = 5000
    lookforward_rows: int = 500
    min_symbol_samples: int = 1000
    force_uniform: bool = True
    nbins: int = 13
    global_thresholds: Optional[GlobalThresholds] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = ["volume"]


class ParquetClassifier:
    """
    DBN-to-Classified-Parquet processor.
    
    This class processes DBN files directly to classified parquet files by symbol, 
    ensuring uniform distribution and efficient ML training preparation.
    """

    def __init__(
        self,
        currency: str = "AUDUSD",
        features: Optional[List[str]] = None,
        input_rows: int = 5000,
        lookforward_rows: int = 500,
        min_symbol_samples: int = 1000,
        force_uniform: bool = True,
        nbins: int = 13,
        global_thresholds: Optional[GlobalThresholds] = None,
        verbose: bool = True,
    ):
        
        """
        Initialize streamlined classifier.

        Args:
            currency: Currency pair for configuration
            features: Features to extract (volume, variance, trade_counts)
            input_rows: Historical rows required for feature generation
            lookforward_rows: Future rows required for classification targets
            min_symbol_samples: Minimum samples required per symbol
            force_uniform: Whether to enforce uniform class distribution
            nbins: Number of classification bins
            global_thresholds: Pre-calculated global thresholds for consistent classification
            verbose: Whether to print progress information
        """
        self.config = ClassificationConfig(
            currency=currency,
            features=features or ["volume"],
            input_rows=input_rows,
            lookforward_rows=lookforward_rows,
            min_symbol_samples=min_symbol_samples,
            force_uniform=force_uniform,
            nbins=nbins,
            global_thresholds=global_thresholds,
        )
        
        self.verbose = verbose
        self.represent_config = create_represent_config(currency)
        
        if self.verbose:
            print("🚀 ParquetClassifier initialized")
            print(f"   💱 Currency: {self.config.currency}")
            print(f"   📊 Features: {self.config.features}")
            print(f"   📈 Input rows: {self.config.input_rows}")
            print(f"   📉 Lookforward rows: {self.config.lookforward_rows}")
            print(f"   🎯 Min samples per symbol: {self.config.min_symbol_samples}")
            print(f"   ⚖️  Force uniform: {self.config.force_uniform}")
            if self.config.global_thresholds:
                print(f"   🌐 Global thresholds: Using pre-calculated from {self.config.global_thresholds.files_analyzed} files")
            else:
                print("   ⚠️  Global thresholds: Using per-symbol calculation (not recommended)")

    def load_dbn_file(self, dbn_path: Union[str, Path]) -> pl.DataFrame:
        """
        Load DBN file into polars DataFrame.

        Args:
            dbn_path: Path to DBN file

        Returns:
            Polars DataFrame with DBN data
        """
        if self.verbose:
            print(f"📄 Loading DBN file: {dbn_path}")
        
        start_time = time.perf_counter()
        
        # Load DBN data
        data = db.read_dbn(str(dbn_path))
        
        # Convert to polars DataFrame
        df = pl.from_pandas(data.to_df())
        
        load_time = time.perf_counter() - start_time
        
        if self.verbose:
            print(f"   ✅ Loaded {len(df):,} rows in {load_time:.1f}s")
            print(f"   📊 Columns: {df.columns}")
            print(f"   📊 Symbols: {df['symbol'].n_unique()} unique")
        
        return df

    def filter_symbols_by_threshold(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, int]]:
        """
        Filter symbols that have sufficient data for processing.

        Args:
            df: Input polars DataFrame

        Returns:
            Tuple of (filtered_df, symbol_counts)
        """
        if self.verbose:
            print("🔍 Filtering symbols by threshold...")
        
        # Calculate minimum required samples
        min_required = self.config.input_rows + self.config.lookforward_rows
        
        # Count samples per symbol
        symbol_counts = df.group_by('symbol').len().sort('len', descending=True)
        
        # Filter symbols with sufficient data
        valid_symbols = symbol_counts.filter(
            pl.col('len') >= min_required
        )['symbol'].to_list()
        
        if self.verbose:
            print(f"   📊 Total symbols: {symbol_counts.height}")
            print(f"   📊 Symbols with ≥{min_required} rows: {len(valid_symbols)}")
            print(f"   📊 Valid symbols: {valid_symbols}")
        
        # Filter DataFrame to valid symbols only
        filtered_df = df.filter(pl.col('symbol').is_in(valid_symbols))
        
        # Convert symbol counts to dict
        symbol_count_dict = {
            row['symbol']: row['len'] 
            for row in symbol_counts.to_dicts()
        }
        
        if self.verbose:
            print(f"   ✅ Filtered to {len(filtered_df):,} rows across {len(valid_symbols)} symbols")
        
        return filtered_df, symbol_count_dict

    def calculate_price_movements(self, symbol_df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate price movements for classification.

        Args:
            symbol_df: DataFrame for a single symbol

        Returns:
            DataFrame with price_movement column added
        """
        # Sort by timestamp to ensure proper order
        symbol_df = symbol_df.sort('ts_event')
        
        # Calculate mid prices from bid/ask
        # Use first level bid/ask prices
        mid_price = (
            (pl.col('ask_px_00') + pl.col('bid_px_00')) / 2
        ).alias('mid_price')
        
        # Add mid price column
        symbol_df = symbol_df.with_columns(mid_price)
        
        # Calculate future mid price (lookforward_rows ahead)
        future_mid_price = (
            pl.col('mid_price')
            .shift(-self.config.lookforward_rows)
            .alias('future_mid_price')
        )
        
        # Add future price column
        symbol_df = symbol_df.with_columns(future_mid_price)
        
        # Calculate price movement in micro pips
        price_movement = (
            (pl.col('future_mid_price') - pl.col('mid_price')) / MICRO_PIP_SIZE
        ).alias('price_movement')
        
        # Add price movement column
        symbol_df = symbol_df.with_columns(price_movement)
        
        return symbol_df

    def apply_quantile_classification(self, symbol_df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply classification using global thresholds or per-symbol quantiles.

        Args:
            symbol_df: DataFrame with price_movement column

        Returns:
            DataFrame with classification_label column added
        """
        # Filter out rows with null price movements (not enough future data)
        valid_df = symbol_df.filter(pl.col('price_movement').is_not_null())
        
        if len(valid_df) == 0:
            # No valid rows, return empty DataFrame with classification column
            return symbol_df.with_columns(
                pl.lit(None, dtype=pl.Int32).alias('classification_label')
            )
        
        # Extract price movements as numpy array
        price_movements = valid_df['price_movement'].to_numpy()
        
        if self.config.global_thresholds is not None:
            # Use pre-calculated global thresholds for consistent classification
            quantile_boundaries = self.config.global_thresholds.quantile_boundaries
            
            # Apply global threshold-based classification
            classification_labels = np.digitize(price_movements, quantile_boundaries[1:-1])
            
            # Ensure labels are in range [0, nbins-1]
            classification_labels = np.clip(classification_labels, 0, self.config.nbins - 1)
            
        elif self.config.force_uniform:
            # Fallback: per-symbol quantile calculation (not recommended for consistency)
            quantiles = np.linspace(0, 1, self.config.nbins + 1)
            quantile_boundaries = np.quantile(price_movements, quantiles)
            
            # Ensure unique boundaries (handle edge cases)
            quantile_boundaries = np.unique(quantile_boundaries)
            
            # If we don't have enough unique values, pad with extremes
            if len(quantile_boundaries) < self.config.nbins + 1:
                min_val, max_val = price_movements.min(), price_movements.max()
                quantile_boundaries = np.linspace(min_val, max_val, self.config.nbins + 1)
            
            # Apply quantile-based classification
            classification_labels = np.digitize(price_movements, quantile_boundaries[1:-1])
            
            # Ensure labels are in range [0, nbins-1]
            classification_labels = np.clip(classification_labels, 0, self.config.nbins - 1)
            
        else:
            # Fixed thresholds fallback
            up_threshold = 5.0  # micro pips
            down_threshold = -5.0  # micro pips
            
            classification_labels = np.where(
                price_movements > up_threshold, 2,
                np.where(price_movements < down_threshold, 0, 1)
            )
        
        # Create mapping back to original DataFrame
        classification_series = pl.Series(classification_labels, dtype=pl.Int32)
        
        # Add classification labels back to valid rows
        classified_df = valid_df.with_columns(
            classification_series.alias('classification_label')
        )
        
        return classified_df

    def filter_processable_rows(self, symbol_df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter rows that have sufficient historical and future data.

        Args:
            symbol_df: DataFrame for a single symbol

        Returns:
            DataFrame with only processable rows
        """
        # Sort by timestamp
        symbol_df = symbol_df.sort('ts_event')
        
        # We need at least input_rows of history and lookforward_rows of future
        # So keep rows from input_rows to (total_rows - lookforward_rows)
        total_rows = len(symbol_df)
        
        if total_rows < self.config.input_rows + self.config.lookforward_rows:
            # Not enough data for any rows
            return symbol_df.head(0)  # Return empty DataFrame with same schema
        
        # Calculate valid row range
        start_idx = self.config.input_rows
        end_idx = total_rows - self.config.lookforward_rows
        
        if start_idx >= end_idx:
            # No valid rows
            return symbol_df.head(0)
        
        # Filter to processable rows
        processable_df = symbol_df.slice(start_idx, end_idx - start_idx)
        
        return processable_df

    def process_symbol(self, symbol: str, symbol_df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        Process a single symbol's data.

        Args:
            symbol: Symbol identifier
            symbol_df: DataFrame for the symbol

        Returns:
            Processed DataFrame with classification labels, or None if insufficient data
        """
        if self.verbose:
            print(f"   🔄 Processing symbol: {symbol}")
        
        # Check if symbol has minimum required samples
        if len(symbol_df) < self.config.min_symbol_samples:
            if self.verbose:
                print(f"      ❌ Insufficient samples: {len(symbol_df)} < {self.config.min_symbol_samples}")
            return None
        
        # Calculate price movements
        symbol_df = self.calculate_price_movements(symbol_df)
        
        # Apply classification
        classified_df = self.apply_quantile_classification(symbol_df)
        
        # Filter to processable rows only
        final_df = self.filter_processable_rows(classified_df)
        
        if len(final_df) == 0:
            if self.verbose:
                print("      ❌ No processable rows after filtering")
            return None
        
        # Filter out rows with null classification (edge cases)
        final_df = final_df.filter(pl.col('classification_label').is_not_null())
        
        if len(final_df) == 0:
            if self.verbose:
                print("      ❌ No valid classifications")
            return None
        
        if self.verbose:
            # Show classification distribution
            class_dist = final_df['classification_label'].value_counts().sort('classification_label')
            print(f"      ✅ Processed {len(final_df):,} samples")
            print(f"      📊 Classes: {class_dist['classification_label'].to_list()}")
            print(f"      📊 Counts: {class_dist['count'].to_list()}")
            print("      📊 Note: Market depth features generated on-demand during loading")
        
        return final_df

    def process_dbn_to_classified_parquets(
        self,
        dbn_path: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Process DBN file directly to classified parquet files by symbol.

        Args:
            dbn_path: Path to input DBN file
            output_dir: Directory for output classified parquet files

        Returns:
            Processing statistics dictionary
        """
        if self.verbose:
            print("🚀 Starting Streamlined DBN-to-Classified-Parquet Processing")
            print("=" * 70)
        
        start_time = time.perf_counter()
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load DBN file
        df = self.load_dbn_file(dbn_path)
        
        # Filter symbols by threshold
        filtered_df, symbol_counts = self.filter_symbols_by_threshold(df)
        
        # Process each symbol
        processed_symbols = {}
        total_classified_samples = 0
        
        if self.verbose:
            print("\n🔄 Processing symbols individually...")
        
        for symbol in filtered_df['symbol'].unique():
            symbol_df = filtered_df.filter(pl.col('symbol') == symbol)
            
            # Process symbol
            processed_df = self.process_symbol(symbol, symbol_df)
            
            if processed_df is not None and len(processed_df) > 0:
                # Save to parquet
                output_file = output_path / f"{self.config.currency}_{symbol}_classified.parquet"
                processed_df.write_parquet(output_file)
                
                processed_symbols[symbol] = {
                    'samples': len(processed_df),
                    'file_path': str(output_file),
                    'file_size_mb': output_file.stat().st_size / 1024 / 1024,
                }
                
                total_classified_samples += len(processed_df)
                
                if self.verbose:
                    print(f"      💾 Saved: {output_file.name} ({processed_symbols[symbol]['file_size_mb']:.1f} MB)")
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Compile results
        results = {
            'input_file': str(dbn_path),
            'output_directory': str(output_path),
            'original_rows': len(df),
            'filtered_rows': len(filtered_df),
            'symbols_processed': len(processed_symbols),
            'total_classified_samples': total_classified_samples,
            'processing_time_seconds': processing_time,
            'samples_per_second': total_classified_samples / processing_time if processing_time > 0 else 0,
            'config': {
                'currency': self.config.currency,
                'features': self.config.features,
                'input_rows': self.config.input_rows,
                'lookforward_rows': self.config.lookforward_rows,
                'min_symbol_samples': self.config.min_symbol_samples,
                'force_uniform': self.config.force_uniform,
                'nbins': self.config.nbins,
            },
            'symbol_stats': processed_symbols,
        }
        
        if self.verbose:
            print("\n✅ STREAMLINED PROCESSING COMPLETE!")
            print(f"   📊 Symbols processed: {len(processed_symbols)}")
            print(f"   📊 Total classified samples: {total_classified_samples:,}")
            print(f"   ⏱️  Processing time: {processing_time:.1f}s")
            print(f"   📈 Processing rate: {results['samples_per_second']:.0f} samples/sec")
            print(f"   📁 Output directory: {output_path}")
            print("   🎯 Files ready for ML training with uniform distribution!")
        
        return results

    def classify_symbol_parquet(
        self,
        parquet_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Classify a single symbol parquet file (compatibility method).
        
        This method provides compatibility with the old API but redirects
        to the streamlined DBN processing approach.
        
        Args:
            parquet_path: Path to input data (treated as DBN path)
            output_path: Directory for output files
            **kwargs: Additional arguments
            
        Returns:
            Processing statistics
        """
        if output_path is None:
            output_path = Path(parquet_path).parent / "classified"
        
        return self.process_dbn_to_classified_parquets(
            dbn_path=parquet_path,
            output_dir=output_path
        )

    def batch_classify_parquets(
        self,
        input_directory: Union[str, Path], 
        output_directory: Union[str, Path],
        pattern: str = "*.dbn*",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch classify multiple files (compatibility method).
        
        Args:
            input_directory: Directory with DBN files
            output_directory: Directory for output files  
            pattern: File pattern to match
            **kwargs: Additional arguments
            
        Returns:
            List of processing statistics
        """
        input_dir = Path(input_directory)
        output_dir = Path(output_directory)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dbn_files = list(input_dir.glob(pattern))
        if not dbn_files:
            raise ValueError(f"No files found matching pattern '{pattern}' in {input_dir}")
        
        results = []
        for dbn_file in dbn_files:
            try:
                stats = self.process_dbn_to_classified_parquets(
                    dbn_path=dbn_file,
                    output_dir=output_dir
                )
                results.append(stats)
            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to process {dbn_file.name}: {e}")
                continue
                
        return results


# Convenience function for direct usage
def process_dbn_to_classified_parquets(
    dbn_path: Union[str, Path],
    output_dir: Union[str, Path],
    currency: str = "AUDUSD",
    features: Optional[List[str]] = None,
    input_rows: int = 5000,
    lookforward_rows: int = 500,
    min_symbol_samples: int = 1000,
    force_uniform: bool = True,
    nbins: int = 13,
    global_thresholds: Optional[GlobalThresholds] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to process DBN file directly to classified parquet files.

    Args:
        dbn_path: Path to input DBN file
        output_dir: Directory for output classified parquet files
        currency: Currency pair for configuration
        features: Features to extract
        input_rows: Historical rows required for feature generation
        lookforward_rows: Future rows required for classification targets
        min_symbol_samples: Minimum samples required per symbol
        force_uniform: Whether to enforce uniform class distribution
        nbins: Number of classification bins
        global_thresholds: Pre-calculated global thresholds for consistent classification
        verbose: Whether to print progress information

    Returns:
        Processing statistics dictionary
    """
    classifier = ParquetClassifier(
        currency=currency,
        features=features,
        input_rows=input_rows,
        lookforward_rows=lookforward_rows,
        min_symbol_samples=min_symbol_samples,
        force_uniform=force_uniform,
        nbins=nbins,
        global_thresholds=global_thresholds,
        verbose=verbose,
    )
    
    return classifier.process_dbn_to_classified_parquets(dbn_path, output_dir)


def classify_parquet_file(
    parquet_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    currency: str = "AUDUSD",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to classify a single file.
    
    Note: This now processes DBN files directly, not parquet files.
    The name is kept for API compatibility.
    
    Args:
        parquet_path: Path to input DBN file
        output_path: Directory for classified output files
        currency: Currency pair for configuration  
        **kwargs: Additional arguments
        
    Returns:
        Classification statistics
    """
    classifier = ParquetClassifier(currency=currency, **kwargs)
    return classifier.classify_symbol_parquet(
        parquet_path=parquet_path,
        output_path=output_path
    )


def batch_classify_parquet_files(
    input_directory: Union[str, Path],
    output_directory: Union[str, Path], 
    currency: str = "AUDUSD",
    pattern: str = "*.dbn*",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to classify multiple files.
    
    Note: This now processes DBN files directly, not parquet files.
    The name is kept for API compatibility.
    
    Args:
        input_directory: Directory with DBN files
        output_directory: Directory for classified output files
        currency: Currency pair for configuration
        pattern: File pattern to match
        **kwargs: Additional arguments
        
    Returns:
        List of classification statistics for each file
    """
    classifier = ParquetClassifier(currency=currency, **kwargs)
    return classifier.batch_classify_parquets(
        input_directory=input_directory,
        output_directory=output_directory,
        pattern=pattern
    )