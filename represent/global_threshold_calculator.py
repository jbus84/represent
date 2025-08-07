"""
Global Threshold Calculator

This module calculates global classification thresholds from a sample of DBN files
to ensure consistent classification across all symbols and files.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import polars as pl
import databento as db
from dataclasses import dataclass

from .constants import MICRO_PIP_SIZE


@dataclass
class GlobalThresholds:
    """Container for global classification thresholds."""
    quantile_boundaries: np.ndarray
    nbins: int
    sample_size: int
    files_analyzed: int
    price_movement_stats: Dict[str, float]


class GlobalThresholdCalculator:
    """
    Calculate global classification thresholds from a sample of DBN files.
    
    This ensures consistent classification thresholds across all symbols and files,
    unlike per-file quantile calculation which creates incomparable classifications.
    """
    
    def __init__(
        self,
        currency: str = "AUDUSD",
        nbins: int = 13,
        lookforward_rows: int = 500,
        sample_fraction: float = 0.5,
        max_samples_per_file: int = 10000,
        verbose: bool = True,
    ):
        """
        Initialize global threshold calculator.
        
        Args:
            currency: Currency pair for configuration
            nbins: Number of classification bins
            lookforward_rows: Future rows required for price movement calculation
            sample_fraction: Fraction of files to use for threshold calculation
            max_samples_per_file: Maximum samples to extract per file (for performance)
            verbose: Whether to print progress information
        """
        self.currency = currency
        self.nbins = nbins
        self.lookforward_rows = lookforward_rows
        self.sample_fraction = sample_fraction
        self.max_samples_per_file = max_samples_per_file
        self.verbose = verbose
        
        if self.verbose:
            print("🌐 GlobalThresholdCalculator initialized")
            print(f"   💱 Currency: {self.currency}")
            print(f"   📊 Bins: {self.nbins}")
            print(f"   📈 Lookforward rows: {self.lookforward_rows}")
            print(f"   🔢 Sample fraction: {self.sample_fraction}")
            print(f"   📏 Max samples per file: {self.max_samples_per_file}")

    def load_dbn_file_sample(self, dbn_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load a sample of price movements from a DBN file.
        
        Args:
            dbn_path: Path to DBN file
            
        Returns:
            Array of price movements in micro pips, or None if file can't be processed
        """
        try:
            if self.verbose:
                print(f"   📄 Loading sample from: {Path(dbn_path).name}")
            
            # Load DBN data
            data = db.read_dbn(str(dbn_path))
            df = pl.from_pandas(data.to_df())
            
            if len(df) < self.lookforward_rows * 2:
                if self.verbose:
                    print(f"      ⚠️  Insufficient data: {len(df)} rows")
                return None
            
            # Calculate mid prices from bid/ask
            mid_price = (
                (pl.col('ask_px_00') + pl.col('bid_px_00')) / 2
            ).alias('mid_price')
            
            df = df.with_columns(mid_price)
            
            # Calculate future mid price (lookforward_rows ahead)
            future_mid_price = (
                pl.col('mid_price')
                .shift(-self.lookforward_rows)
                .alias('future_mid_price')
            )
            
            df = df.with_columns(future_mid_price)
            
            # Calculate price movement in micro pips
            price_movement = (
                (pl.col('future_mid_price') - pl.col('mid_price')) / MICRO_PIP_SIZE
            ).alias('price_movement')
            
            df = df.with_columns(price_movement)
            
            # Filter out rows with null price movements
            valid_df = df.filter(pl.col('price_movement').is_not_null())
            
            if len(valid_df) == 0:
                if self.verbose:
                    print("      ⚠️  No valid price movements")
                return None
            
            # Extract price movements as numpy array
            price_movements = valid_df['price_movement'].to_numpy()
            
            # Sample if too many data points
            if len(price_movements) > self.max_samples_per_file:
                # Use random sampling to get representative sample
                indices = np.random.choice(
                    len(price_movements), 
                    size=self.max_samples_per_file, 
                    replace=False
                )
                price_movements = price_movements[indices]
            
            if self.verbose:
                print(f"      ✅ Extracted {len(price_movements):,} price movements")
            
            return price_movements
            
        except Exception as e:
            if self.verbose:
                print(f"      ❌ Failed to process: {e}")
            return None

    def calculate_global_thresholds(
        self, 
        data_directory: Union[str, Path],
        file_pattern: str = "*.dbn*"
    ) -> GlobalThresholds:
        """
        Calculate global classification thresholds from a sample of DBN files.
        
        Args:
            data_directory: Directory containing DBN files
            file_pattern: Pattern to match DBN files
            
        Returns:
            GlobalThresholds object with quantile boundaries and metadata
        """
        data_dir = Path(data_directory)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find all DBN files
        dbn_files = sorted(list(data_dir.glob(file_pattern)))
        
        if not dbn_files:
            raise ValueError(f"No DBN files found with pattern '{file_pattern}' in {data_dir}")
        
        # Calculate sample size
        num_sample_files = max(1, int(len(dbn_files) * self.sample_fraction))
        sample_files = dbn_files[:num_sample_files]
        
        if self.verbose:
            print("\n🌐 CALCULATING GLOBAL THRESHOLDS")
            print("=" * 60)
            print(f"📁 Data directory: {data_dir}")
            print(f"📊 Total files found: {len(dbn_files)}")
            print(f"🔢 Sample files to analyze: {num_sample_files}")
            print(f"📋 Sample files: {[f.name for f in sample_files]}")
        
        # Collect price movements from sample files
        all_price_movements = []
        files_processed = 0
        
        start_time = time.perf_counter()
        
        for i, dbn_file in enumerate(sample_files):
            if self.verbose:
                print(f"\n🔄 Processing {i+1}/{len(sample_files)}: {dbn_file.name}")
            
            price_movements = self.load_dbn_file_sample(dbn_file)
            
            if price_movements is not None:
                all_price_movements.append(price_movements)
                files_processed += 1
            
        if not all_price_movements:
            raise ValueError("No valid price movements extracted from sample files")
        
        # Combine all price movements
        combined_movements = np.concatenate(all_price_movements)
        
        processing_time = time.perf_counter() - start_time
        
        if self.verbose:
            print("\n📊 THRESHOLD CALCULATION RESULTS")
            print("=" * 40)
            print(f"✅ Files processed: {files_processed}/{len(sample_files)}")
            print(f"📊 Total price movements: {len(combined_movements):,}")
            print(f"⏱️  Processing time: {processing_time:.1f}s")
        
        # Calculate global quantile boundaries
        quantiles = np.linspace(0, 1, self.nbins + 1)
        quantile_boundaries = np.quantile(combined_movements, quantiles)
        
        # Ensure unique boundaries (handle edge cases)
        quantile_boundaries = np.unique(quantile_boundaries)
        
        # If we don't have enough unique values, pad with extremes
        if len(quantile_boundaries) < self.nbins + 1:
            min_val, max_val = combined_movements.min(), combined_movements.max()
            quantile_boundaries = np.linspace(min_val, max_val, self.nbins + 1)
        
        # Calculate price movement statistics
        price_stats = {
            'mean': float(np.mean(combined_movements)),
            'std': float(np.std(combined_movements)),
            'min': float(np.min(combined_movements)),
            'max': float(np.max(combined_movements)),
            'median': float(np.median(combined_movements)),
        }
        
        if self.verbose:
            print("\n📈 PRICE MOVEMENT STATISTICS")
            print("=" * 30)
            print(f"Mean: {price_stats['mean']:.2f} micro pips")
            print(f"Std:  {price_stats['std']:.2f} micro pips")
            print(f"Min:  {price_stats['min']:.2f} micro pips")
            print(f"Max:  {price_stats['max']:.2f} micro pips")
            print(f"Median: {price_stats['median']:.2f} micro pips")
            
            print("\n🎯 GLOBAL QUANTILE BOUNDARIES")
            print("=" * 30)
            for i, boundary in enumerate(quantile_boundaries):
                if i == 0:
                    print(f"Bin {i:2d}: <= {boundary:8.2f} micro pips")
                elif i == len(quantile_boundaries) - 1:
                    continue  # Skip the last boundary as it's just the max
                else:
                    print(f"Bin {i:2d}: <= {boundary:8.2f} micro pips")
        
        global_thresholds = GlobalThresholds(
            quantile_boundaries=quantile_boundaries,
            nbins=self.nbins,
            sample_size=len(combined_movements),
            files_analyzed=files_processed,
            price_movement_stats=price_stats,
        )
        
        if self.verbose:
            print("\n✅ GLOBAL THRESHOLDS CALCULATED SUCCESSFULLY!")
            print("🎯 Ready for consistent classification across all files")
        
        return global_thresholds


def calculate_global_thresholds(
    data_directory: Union[str, Path],
    currency: str = "AUDUSD",
    nbins: int = 13,
    lookforward_rows: int = 500,
    sample_fraction: float = 0.5,
    file_pattern: str = "*.dbn*",
    verbose: bool = True,
) -> GlobalThresholds:
    """
    Convenience function to calculate global thresholds.
    
    Args:
        data_directory: Directory containing DBN files
        currency: Currency pair for configuration
        nbins: Number of classification bins
        lookforward_rows: Future rows required for price movement calculation
        sample_fraction: Fraction of files to use for threshold calculation
        file_pattern: Pattern to match DBN files
        verbose: Whether to print progress information
        
    Returns:
        GlobalThresholds object with quantile boundaries and metadata
        
    Example:
        # Calculate thresholds from first 50% of files
        thresholds = calculate_global_thresholds(
            "/Users/danielfisher/data/databento/AUDUSD-micro",
            currency="AUDUSD",
            sample_fraction=0.5
        )
        
        # Use thresholds for consistent classification
        classifier = ParquetClassifier(
            currency="AUDUSD", 
            global_thresholds=thresholds
        )
    """
    calculator = GlobalThresholdCalculator(
        currency=currency,
        nbins=nbins,
        lookforward_rows=lookforward_rows,
        sample_fraction=sample_fraction,
        verbose=verbose,
    )
    
    return calculator.calculate_global_thresholds(
        data_directory=data_directory,
        file_pattern=file_pattern
    )