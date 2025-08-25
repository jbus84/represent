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

from .configs import GlobalThresholdConfig


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
        config: GlobalThresholdConfig | None = None,
        sample_fraction: float = 0.5,
        verbose: bool = True,
        # Legacy support
        legacy_config=None,
    ):
        """
        Initialize global threshold calculator using GlobalThresholdConfig.

        Args:
            config: GlobalThresholdConfig with focused configuration (new preferred way)
            sample_fraction: Fraction of files to use for threshold calculation
            verbose: Whether to print progress information
            legacy_config: Legacy RepresentConfig for backward compatibility
        """
        # Handle legacy usage - legacy_config should now be a tuple from create_represent_config
        if config is None and legacy_config is not None:
            if isinstance(legacy_config, tuple) and len(legacy_config) == 3:
                # New style: tuple of (DatasetBuilderConfig, GlobalThresholdConfig, MarketDepthProcessorConfig)
                legacy_threshold_config = legacy_config[1]  # Use GlobalThresholdConfig
                # Override sample_fraction if provided
                if sample_fraction != 0.5:  # 0.5 is default
                    config = GlobalThresholdConfig(
                        currency=legacy_threshold_config.currency,
                        nbins=legacy_threshold_config.nbins,
                        lookback_rows=legacy_threshold_config.lookback_rows,
                        lookforward_input=legacy_threshold_config.lookforward_input,
                        lookforward_offset=legacy_threshold_config.lookforward_offset,
                        max_samples_per_file=legacy_threshold_config.max_samples_per_file,
                        sample_fraction=sample_fraction,
                        jump_size=legacy_threshold_config.jump_size,
                    )
                else:
                    config = legacy_threshold_config
            else:
                # Very old style - create default config
                config = GlobalThresholdConfig(sample_fraction=sample_fraction)

        # Default config if none provided
        if config is None:
            config = GlobalThresholdConfig(sample_fraction=sample_fraction)
        else:
            # Override sample_fraction if it was explicitly provided and differs from default
            if sample_fraction != 0.5:  # 0.5 is the default value
                config = GlobalThresholdConfig(
                    currency=config.currency,
                    nbins=config.nbins,
                    lookback_rows=config.lookback_rows,
                    lookforward_input=config.lookforward_input,
                    lookforward_offset=config.lookforward_offset,
                    max_samples_per_file=config.max_samples_per_file,
                    sample_fraction=sample_fraction,
                    jump_size=config.jump_size,
                )

        self.config = config
        self.currency = config.currency
        self.sample_fraction = config.sample_fraction
        self.max_samples_per_file = config.max_samples_per_file
        self.verbose = verbose
        self.nbins = config.nbins

        # Access jump_size from config if available, otherwise use default
        self.jump_size = getattr(config, "jump_size", 100)

        if self.verbose:
            print("🌐 GlobalThresholdCalculator initialized")
            print(f"   💱 Currency: {self.currency}")
            print(f"   📊 Bins: {self.nbins}")
            print(f"   📈 Lookforward offset: {self.config.lookforward_offset}")
            print(f"   📉 Lookforward window: {self.config.lookforward_input}")
            print(
                f"   📏 Total lookforward rows: {self.config.lookforward_input + self.config.lookforward_offset}"
            )
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
            min_required_rows = (
                self.config.lookback_rows
                + self.config.lookforward_input
                + self.config.lookforward_offset
            )
            if len(df) < min_required_rows:
                if self.verbose:
                    print(f"      ⚠️  Insufficient data: {len(df)} < {min_required_rows} rows")
                return None

            # Filter out invalid/corrupted prices first
            # For AUDUSD, valid prices should be roughly 0.50 to 0.80
            # Anything outside this range is likely corrupted data
            price_filter = (
                (pl.col("bid_px_00") > 0.50)
                & (pl.col("bid_px_00") < 0.80)
                & (pl.col("ask_px_00") > 0.50)
                & (pl.col("ask_px_00") < 0.80)
                & (pl.col("bid_px_00") > 0)
                & (pl.col("ask_px_00") > 0)  # Exclude zeros
            )

            df = df.filter(price_filter)

            if len(df) == 0:
                if self.verbose:
                    print("      ⚠️  No valid prices after filtering")
                return None

            # Calculate mid prices from bid/ask
            df = df.with_columns(
                ((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid_price")
            )

            # Extract mid prices as numpy array for efficient processing
            mid_prices = df["mid_price"].to_numpy()

            # Calculate price movements using correct lookback vs lookforward methodology
            price_movements = []

            # Iterate through valid sample positions using jump_size steps
            total_lookforward = self.config.lookforward_input + self.config.lookforward_offset
            for stop_row in range(
                self.config.lookback_rows, len(mid_prices) - total_lookforward, self.jump_size
            ):
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
                    len(price_movements), size=self.max_samples_per_file, replace=False
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
        self, data_directory: str | Path, file_pattern: str = "*.dbn*"
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
                print(f"\n🔄 Processing {i + 1}/{len(sample_files)}: {dbn_file.name}")

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
        if self.config.use_heavy_tailed:
            # Use heavy-tailed distribution approach for better class balance
            # This addresses the extreme class concentration problem
            quantile_boundaries = self._calculate_heavy_tailed_boundaries(combined_movements)
        else:
            # Use traditional quantile approach
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
            "mean": float(np.mean(combined_movements)),
            "std": float(np.std(combined_movements)),
            "min": float(np.min(combined_movements)),
            "max": float(np.max(combined_movements)),
            "median": float(np.median(combined_movements)),
        }

        if self.verbose:
            print("\n📈 PRICE MOVEMENT STATISTICS")
            print("=" * 30)
            print(f"Mean: {price_stats['mean']:.6f} ({price_stats['mean'] * 100:.4f}%)")
            print(f"Std:  {price_stats['std']:.6f} ({price_stats['std'] * 100:.4f}%)")
            print(f"Min:  {price_stats['min']:.6f} ({price_stats['min'] * 100:.4f}%)")
            print(f"Max:  {price_stats['max']:.6f} ({price_stats['max'] * 100:.4f}%)")
            print(f"Median: {price_stats['median']:.6f} ({price_stats['median'] * 100:.4f}%)")

            print("\n🎯 GLOBAL QUANTILE BOUNDARIES")
            print("=" * 30)
            for i, boundary in enumerate(quantile_boundaries):
                if i == 0:
                    print(f"Bin {i:2d}: <= {boundary:8.6f} ({boundary * 100:+7.4f}%)")
                elif i == len(quantile_boundaries) - 1:
                    continue  # Skip the last boundary as it's just the max
                else:
                    print(f"Bin {i:2d}: <= {boundary:8.6f} ({boundary * 100:+7.4f}%)")

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

    def _calculate_heavy_tailed_boundaries(self, price_movements: np.ndarray) -> np.ndarray:
        """
        Calculate boundaries using EVT-inspired approach.

        This combines:
        1. Student's t-distribution for overall heavy-tailed modeling
        2. Power-law tail compression to reduce extreme class concentration

        This approach provides the benefits of EVT theory while maintaining stability
        and predictable results for financial returns classification.

        Args:
            price_movements: Array of price movements

        Returns:
            Array of boundary values
        """
        try:
            from scipy import stats

            if self.verbose:
                print("\n📊 EVT-INSPIRED DISTRIBUTION APPROACH:")

            # Step 1: Fit Student's t-distribution
            try:
                df, loc, scale = stats.t.fit(price_movements)
                df = max(2.1, min(30, df))  # Constrain to reasonable range

                if self.verbose:
                    print(f"   Student's t fit: df={df:.2f}, loc={loc:.6f}, scale={scale:.6f}")

                # Use t-distribution if meaningful heavy tails, else normal
                use_t_dist = df < 10

            except Exception:
                # Fallback to normal
                loc = np.mean(price_movements)
                scale = np.std(price_movements)
                use_t_dist = False

                if self.verbose:
                    print(f"   Normal fallback: loc={loc:.6f}, scale={scale:.6f}")

            # Step 2: Generate quantiles with EVT-inspired tail compression
            quantiles = np.linspace(0, 1, self.nbins + 1)
            boundaries = []

            # Tail compression parameters based on EVT theory
            # Financial returns typically show power-law behavior in tails
            tail_compression = 0.75  # Compress tail quantiles by this factor
            center_preservation = 0.4  # Preserve center quantiles (±40% around median)

            if self.verbose:
                print(f"   Tail compression factor: {tail_compression:.2f}")
                print(f"   Center preservation: ±{center_preservation * 100:.0f}% around median")

            for i, q in enumerate(quantiles):
                if i == 0:
                    # Minimum boundary - extend for coverage
                    if use_t_dist:
                        boundary = stats.t.ppf(0.001, df, loc=loc, scale=scale)
                    else:
                        boundary = stats.norm.ppf(0.001, loc=loc, scale=scale)

                    if not np.isfinite(boundary):
                        boundary = price_movements.min() - abs(price_movements.min()) * 0.2

                elif i == len(quantiles) - 1:
                    # Maximum boundary - extend for coverage
                    if use_t_dist:
                        boundary = stats.t.ppf(0.999, df, loc=loc, scale=scale)
                    else:
                        boundary = stats.norm.ppf(0.999, loc=loc, scale=scale)

                    if not np.isfinite(boundary):
                        boundary = price_movements.max() + abs(price_movements.max()) * 0.2

                else:
                    # Internal boundaries with tail compression

                    # Determine if this quantile is in the tails or center
                    distance_from_median = abs(q - 0.5)

                    if distance_from_median > center_preservation:
                        # This is in the tail - apply compression

                        # Calculate compression factor (stronger for more extreme quantiles)
                        tail_strength = (distance_from_median - center_preservation) / (
                            0.5 - center_preservation
                        )
                        compression_factor = 1.0 - (1.0 - tail_compression) * tail_strength

                        # Apply compression by moving quantile toward center
                        if q < 0.5:
                            # Lower tail
                            compressed_q = 0.5 - (0.5 - q) * compression_factor
                        else:
                            # Upper tail
                            compressed_q = 0.5 + (q - 0.5) * compression_factor

                        # Use compressed quantile for boundary
                        if use_t_dist:
                            boundary = stats.t.ppf(compressed_q, df, loc=loc, scale=scale)
                        else:
                            boundary = stats.norm.ppf(compressed_q, loc=loc, scale=scale)

                    else:
                        # This is in the center - use normal quantile
                        if use_t_dist:
                            boundary = stats.t.ppf(q, df, loc=loc, scale=scale)
                        else:
                            boundary = stats.norm.ppf(q, loc=loc, scale=scale)

                    # Fallback for any numerical issues
                    if not np.isfinite(boundary):
                        boundary = np.quantile(price_movements, q)

                boundaries.append(boundary)

            # Step 3: Ensure monotonicity and proper spacing
            boundaries = np.array(sorted(boundaries))

            # Ensure minimum spacing
            min_spacing = (boundaries[-1] - boundaries[0]) / (len(boundaries) * 1000)
            for i in range(1, len(boundaries)):
                if boundaries[i] - boundaries[i - 1] < min_spacing:
                    boundaries[i] = boundaries[i - 1] + min_spacing

            if self.verbose:
                print("\n🎯 EVT-INSPIRED BOUNDARIES vs QUANTILE BOUNDARIES:")
                quantile_boundaries = np.quantile(price_movements, quantiles)
                for i in range(1, len(boundaries) - 1):
                    evt_val = boundaries[i]
                    q_val = quantile_boundaries[i]
                    diff = (evt_val - q_val) / abs(q_val) * 100 if q_val != 0 else 0
                    print(f"   Boundary {i:2d}: EVT={evt_val:8.6f} Q={q_val:8.6f} ({diff:+5.1f}%)")

            return boundaries

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  EVT-inspired fitting failed: {e}")
                print("   📊 Falling back to Student's t approach")

            # Fallback to simpler Student's t approach
            return self._calculate_simple_t_boundaries(price_movements)

    def _calculate_simple_t_boundaries(self, price_movements: np.ndarray) -> np.ndarray:
        """Fallback to simple Student's t distribution (original approach)."""
        try:
            from scipy import stats

            df, loc, scale = stats.t.fit(price_movements)
            df = max(2.1, min(30, df))

            quantiles = np.linspace(0, 1, self.nbins + 1)
            boundaries = stats.t.ppf(quantiles, df, loc=loc, scale=scale)

            # Handle infinities
            if np.any(np.isinf(boundaries)):
                data_range = price_movements.max() - price_movements.min()
                extension = data_range * 0.5
                boundaries[0] = price_movements.min() - extension
                boundaries[-1] = price_movements.max() + extension

            return boundaries

        except Exception:
            # Final fallback to quantiles
            quantiles = np.linspace(0, 1, self.nbins + 1)
            return np.quantile(price_movements, quantiles)


def calculate_global_thresholds(
    config: GlobalThresholdConfig | None = None,
    data_directory: str | Path | None = None,
    sample_fraction: float = 0.5,
    file_pattern: str = "*.dbn*",
    verbose: bool = True,
    # Legacy support
    **kwargs,
) -> GlobalThresholds:
    """
    Convenience function to calculate global thresholds using lookback vs lookforward methodology.

    Args:
        config: GlobalThresholdConfig with focused configuration (preferred)
               or legacy RepresentConfig for backward compatibility
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
    # Handle different config types
    if config is None:
        # Use default config
        calculator = GlobalThresholdCalculator(
            config=None, sample_fraction=sample_fraction, verbose=verbose
        )
    elif isinstance(config, GlobalThresholdConfig):
        # New focused config
        calculator = GlobalThresholdCalculator(
            config=config, sample_fraction=sample_fraction, verbose=verbose
        )
    else:
        # Legacy RepresentConfig
        calculator = GlobalThresholdCalculator(
            config=None, sample_fraction=sample_fraction, verbose=verbose, legacy_config=config
        )

    if data_directory is None:
        raise ValueError("data_directory parameter is required")

    return calculator.calculate_global_thresholds(
        data_directory=data_directory, file_pattern=file_pattern
    )
