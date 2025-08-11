"""
Global Threshold Calculator

This module calculates global classification thresholds from a sample of DBN files
to ensure consistent classification across all symbols and files.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import databento as db
import numpy as np
import polars as pl

from .config import RepresentConfig

# JUMP_SIZE now comes from RepresentConfig


@dataclass
class GlobalThresholds:
    """Container for global classification thresholds."""
    quantile_boundaries: np.ndarray
    nbins: int
    sample_size: int
    files_analyzed: int
    price_movement_stats: dict[str, float]


class GlobalThresholdCalculator:
    """
    Calculate global classification thresholds from a sample of DBN files.

    This ensures consistent classification thresholds across all symbols and files,
    unlike per-file quantile calculation which creates incomparable classifications.
    """

    def __init__(
        self,
        config: RepresentConfig,
        sample_fraction: float = 0.5,
        verbose: bool = True,
    ):
        """
        Initialize global threshold calculator using RepresentConfig.

        Args:
            config: RepresentConfig with currency-specific configuration
            sample_fraction: Fraction of files to use for threshold calculation
            verbose: Whether to print progress information
        """
        self.config = config
        self.currency = config.currency
        self.sample_fraction = sample_fraction
        self.max_samples_per_file = config.max_samples_per_file
        self.verbose = verbose
        self.nbins = self.config.nbins

        if self.verbose:
            print("🌐 GlobalThresholdCalculator initialized")
            print(f"   💱 Currency: {self.currency}")
            print(f"   📊 Bins: {self.nbins}")
            print(f"   📈 Lookforward offset: {self.config.lookforward_offset}")
            print(f"   📉 Lookforward window: {self.config.lookforward_input}")
            print(f"   📏 Total lookforward rows: {self.config.lookforward_input + self.config.lookforward_offset}")
            print(f"   📊 Lookback rows: {self.config.lookback_rows}")
            print(f"   🔢 Sample fraction: {self.sample_fraction}")
            print(f"   📏 Max samples per file: {self.max_samples_per_file}")

    def load_dbn_file_sample(self, dbn_path: str | Path) -> np.ndarray | None:
        """
        Load a sample of price movements from a DBN file using correct lookback vs lookforward methodology.

        Args:
            dbn_path: Path to DBN file

        Returns:
            Array of percentage price movements, or None if file can't be processed
        """
        try:
            if self.verbose:
                print(f"   📄 Loading sample from: {Path(dbn_path).name}")

            # Load DBN data
            data = db.read_dbn(str(dbn_path))
            df = pl.from_pandas(data.to_df())

            # Check if we have sufficient data for the methodology
            min_required_rows = self.config.lookback_rows + self.config.lookforward_input + self.config.lookforward_offset
            if len(df) < min_required_rows:
                if self.verbose:
                    print(f"      ⚠️  Insufficient data: {len(df)} < {min_required_rows} rows")
                return None

            # Filter out invalid/corrupted prices first
            # For AUDUSD, valid prices should be roughly 0.50 to 0.80
            # Anything outside this range is likely corrupted data
            price_filter = (
                (pl.col('bid_px_00') > 0.50) & (pl.col('bid_px_00') < 0.80) &
                (pl.col('ask_px_00') > 0.50) & (pl.col('ask_px_00') < 0.80) &
                (pl.col('bid_px_00') > 0) & (pl.col('ask_px_00') > 0)  # Exclude zeros
            )

            df = df.filter(price_filter)

            if len(df) == 0:
                if self.verbose:
                    print("      ⚠️  No valid prices after filtering")
                return None

            # Calculate mid prices from bid/ask
            df = df.with_columns(
                ((pl.col('ask_px_00') + pl.col('bid_px_00')) / 2).alias('mid_price')
            )

            # Extract mid prices as numpy array for efficient processing
            mid_prices = df['mid_price'].to_numpy()

            # Calculate price movements using correct lookback vs lookforward methodology
            price_movements = []

            # Iterate through valid sample positions using JUMP_SIZE steps
            total_lookforward = self.config.lookforward_input + self.config.lookforward_offset
            for stop_row in range(self.config.lookback_rows, len(mid_prices) - total_lookforward, self.config.jump_size):
                # Define time windows according to the correct methodology
                lookback_start = stop_row - self.config.lookback_rows
                lookback_end = stop_row

                target_start_row = stop_row + 1 + self.config.lookforward_offset
                target_stop_row = stop_row + self.config.lookforward_input

                # Calculate lookback mean (historical average)
                lookback_mean = np.mean(mid_prices[lookback_start:lookback_end])

                # Calculate lookforward mean (future average)
                lookforward_mean = np.mean(mid_prices[target_start_row:target_stop_row])

                # Calculate percentage change: (future - past) / past
                if lookback_mean > 0:  # Avoid division by zero
                    mean_change = (lookforward_mean - lookback_mean) / lookback_mean
                    price_movements.append(mean_change)

            if not price_movements:
                if self.verbose:
                    print("      ⚠️  No valid price movements calculated")
                return None

            price_movements = np.array(price_movements)

            # Filter extreme percentage movements (e.g., beyond ±10% which is unrealistic for AUDUSD)
            valid_mask = np.abs(price_movements) < 0.1  # 10% threshold
            price_movements = price_movements[valid_mask]

            if len(price_movements) == 0:
                if self.verbose:
                    print("      ⚠️  No valid price movements after filtering extremes")
                return None

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
                print(f"      ✅ Extracted {len(price_movements):,} percentage price movements")

            return price_movements

        except Exception as e:
            if self.verbose:
                print(f"      ❌ Failed to process: {e}")
            return None

    def calculate_global_thresholds(
        self,
        data_directory: str | Path,
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
        dbn_files = sorted(data_dir.glob(file_pattern))

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
            print(f"Mean: {price_stats['mean']:.6f} ({price_stats['mean']*100:.4f}%)")
            print(f"Std:  {price_stats['std']:.6f} ({price_stats['std']*100:.4f}%)")
            print(f"Min:  {price_stats['min']:.6f} ({price_stats['min']*100:.4f}%)")
            print(f"Max:  {price_stats['max']:.6f} ({price_stats['max']*100:.4f}%)")
            print(f"Median: {price_stats['median']:.6f} ({price_stats['median']*100:.4f}%)")

            print("\n🎯 GLOBAL QUANTILE BOUNDARIES")
            print("=" * 30)
            for i, boundary in enumerate(quantile_boundaries):
                if i == 0:
                    print(f"Bin {i:2d}: <= {boundary:8.6f} ({boundary*100:+7.4f}%)")
                elif i == len(quantile_boundaries) - 1:
                    continue  # Skip the last boundary as it's just the max
                else:
                    print(f"Bin {i:2d}: <= {boundary:8.6f} ({boundary*100:+7.4f}%)")

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
    config: RepresentConfig,
    data_directory: str | Path,
    sample_fraction: float = 0.5,
    file_pattern: str = "*.dbn*",
    verbose: bool = True,
) -> GlobalThresholds:
    """
    Convenience function to calculate global thresholds using lookback vs lookforward methodology.

    Args:
        config: RepresentConfig with currency-specific configuration
        data_directory: Directory containing DBN files
        sample_fraction: Fraction of files to use for threshold calculation
        file_pattern: Pattern to match DBN files
        verbose: Whether to print progress information

    Returns:
        GlobalThresholds object with percentage-based quantile boundaries and metadata

    Example:
        # Calculate percentage-based thresholds from first 50% of files
        config = create_represent_config("AUDUSD")
        thresholds = calculate_global_thresholds(
            config,
            "/Users/danielfisher/data/databento/AUDUSD-micro",
            sample_fraction=0.5
        )

        # Use thresholds for consistent classification
        classifier = ParquetClassifier(
            currency="AUDUSD",
            global_thresholds=thresholds
        )
    """
    calculator = GlobalThresholdCalculator(
        config=config,
        sample_fraction=sample_fraction,
        verbose=verbose,
    )

    return calculator.calculate_global_thresholds(
        data_directory=data_directory,
        file_pattern=file_pattern
    )
