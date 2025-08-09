"""
Dynamic Classification Configuration Generator

This module generates optimized classification configurations for any currency
using quantile-based analysis on parquet data, eliminating the need for static
configuration files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import polars as pl
from datetime import datetime

from .config import RepresentConfig

logger = logging.getLogger(__name__)


class ClassificationConfigGenerator:
    """
    Generate optimized classification configurations using quantile-based analysis.
    
    This class analyzes price movement data to automatically determine optimal
    classification thresholds that achieve uniform distribution across all classes.
    """
    
    def __init__(
        self, 
        config: RepresentConfig,
        validation_split: float = 0.3,
        random_seed: int = 42
    ):
        """
        Initialize the classification config generator.
        
        Args:
            config: RepresentConfig with currency-specific configuration
            validation_split: Fraction of data to use for validation (0.0-1.0)
            random_seed: Random seed for reproducible results
        """
        self.config = config
        self.nbins = config.nbins
        self.target_samples = config.target_samples
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.target_percent = 100.0 / self.nbins
        
        logger.info(f"ClassificationConfigGenerator initialized: {config.currency}, {self.nbins} bins, {self.target_samples} min samples")
    
    def extract_price_changes_from_parquet(
        self, 
        parquet_files: Union[str, Path, List[Union[str, Path]]]
    ) -> np.ndarray:
        """
        Extract price changes from parquet files.
        
        Args:
            parquet_files: Single parquet file path or list of paths
            
        Returns:
            Array of price changes
            
        Raises:
            ValueError: If no valid price change data found
            FileNotFoundError: If parquet files don't exist
        """
        if isinstance(parquet_files, (str, Path)):
            parquet_files = [parquet_files]
        
        all_price_changes = []
        
        for parquet_file in parquet_files:
            parquet_path = Path(parquet_file)
            
            if not parquet_path.exists():
                logger.warning(f"Parquet file not found: {parquet_path}")
                continue
                
            try:
                logger.debug(f"Processing parquet file: {parquet_path.name}")
                df = pl.read_parquet(parquet_path)
                
                # Extract price changes from start/end prices
                if 'start_mid_price' in df.columns and 'end_mid_price' in df.columns:
                    start_prices = df['start_mid_price'].to_numpy()
                    end_prices = df['end_mid_price'].to_numpy()
                    price_changes = end_prices - start_prices
                    all_price_changes.extend(price_changes)
                    logger.debug(f"Extracted {len(price_changes)} price changes from {parquet_path.name}")
                    
                elif 'mean_price_change' in df.columns:
                    price_changes = df['mean_price_change'].to_numpy()
                    all_price_changes.extend(price_changes)
                    logger.debug(f"Extracted {len(price_changes)} price changes from {parquet_path.name}")
                    
                else:
                    logger.warning(f"No price change columns found in {parquet_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {parquet_path}: {e}")
                continue
        
        if not all_price_changes:
            raise ValueError("No price change data found in any parquet files")
        
        price_changes = np.array(all_price_changes)
        logger.info(f"Total extracted price changes: {len(price_changes):,}")
        
        return price_changes
    
    def calculate_quantile_thresholds(self, price_changes: np.ndarray) -> Dict[str, Union[List[float], Dict, str]]:
        """
        Calculate quantile-based classification thresholds.
        
        Args:
            price_changes: Array of price changes
            
        Returns:
            Dictionary containing thresholds and metadata
        """
        if len(price_changes) < self.target_samples:
            logger.warning(f"Only {len(price_changes)} samples available, less than target {self.target_samples}")
        
        # Sort data for quantile calculation
        sorted_data = np.sort(price_changes)
        n_samples = len(sorted_data)
        
        # Calculate exact quantile positions for uniform distribution
        quantile_positions = []
        for i in range(1, self.nbins):
            quantile = i / self.nbins
            quantile_positions.append(quantile)
        
        # Get threshold values at quantile positions
        thresholds = []
        for q in quantile_positions:
            idx = int(q * n_samples)
            if idx >= n_samples:
                idx = n_samples - 1
            thresholds.append(float(sorted_data[idx]))
        
        # Calculate statistics
        data_stats = {
            'count': n_samples,
            'mean': float(price_changes.mean()),
            'std': float(price_changes.std()),
            'min': float(price_changes.min()),
            'max': float(price_changes.max()),
            'p01': float(np.percentile(price_changes, 1)),
            'p05': float(np.percentile(price_changes, 5)),
            'p10': float(np.percentile(price_changes, 10)),
            'p25': float(np.percentile(price_changes, 25)),
            'p50': float(np.percentile(price_changes, 50)),
            'p75': float(np.percentile(price_changes, 75)),
            'p90': float(np.percentile(price_changes, 90)),
            'p95': float(np.percentile(price_changes, 95)),
            'p99': float(np.percentile(price_changes, 99)),
        }
        
        # Extract exactly 6 positive thresholds for compatibility with ClassificationConfig
        middle_idx = len(thresholds) // 2
        positive_thresholds: Dict[str, float] = {}
        
        # For 13-bin config, we need exactly 6 positive thresholds (bin_1 through bin_6)
        # Take thresholds from the second half of sorted thresholds
        positive_indices = list(range(middle_idx + 1, len(thresholds)))
        
        # If we have more than 6, take evenly spaced ones
        if len(positive_indices) > 6:
            step = len(positive_indices) / 6
            selected_indices = [positive_indices[int(i * step)] for i in range(6)]
        else:
            selected_indices = positive_indices
        
        # Ensure we have exactly 6 thresholds
        while len(selected_indices) < 6:
            # Duplicate the last threshold if needed
            selected_indices.append(selected_indices[-1] if selected_indices else middle_idx)
        
        for i in range(6):
            idx = selected_indices[i] if i < len(selected_indices) else selected_indices[-1]
            positive_thresholds[f"bin_{i + 1}"] = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        
        logger.info(f"Generated {len(thresholds)} quantile thresholds")
        for i, threshold in enumerate(thresholds):
            pips = threshold / self.config.micro_pip_size
            logger.debug(f"  Threshold {i+1}: {threshold:.6f} ({pips:.2f} pips)")
        
        return {
            'all_thresholds': thresholds,
            'positive_thresholds': positive_thresholds,
            'quantile_positions': quantile_positions,
            'data_stats': data_stats,
            'method': 'quantile_based'
        }
    
    def validate_thresholds(
        self, 
        thresholds: List[float], 
        validation_data: np.ndarray
    ) -> Dict[str, Union[float, str, List[float]]]:
        """
        Validate classification thresholds on validation data.
        
        Args:
            thresholds: List of classification thresholds
            validation_data: Validation price changes
            
        Returns:
            Validation metrics dictionary
        """
        if len(validation_data) == 0:
            logger.warning("No validation data available")
            return {
                'validation_samples': 0,
                'max_deviation': float('inf'),
                'avg_deviation': float('inf'),
                'quality': 'NO_DATA',
                'distribution': []
            }
        
        # Classify validation data
        labels = np.zeros(len(validation_data), dtype=int)
        
        for i, value in enumerate(validation_data):
            bin_idx = 0
            for threshold in thresholds:
                if value <= threshold:
                    break
                bin_idx += 1
            labels[i] = min(bin_idx, self.nbins - 1)
        
        # Calculate distribution
        distribution: List[float] = []
        for class_idx in range(self.nbins):
            count = np.sum(labels == class_idx)
            percentage = (count / len(labels)) * 100.0
            distribution.append(percentage)
        
        # Calculate quality metrics
        deviations = [abs(d - self.target_percent) for d in distribution]
        max_deviation = max(deviations) if deviations else float('inf')
        avg_deviation = float(np.mean(deviations)) if deviations else float('inf')
        
        # Determine quality rating
        if max_deviation < 1.0:
            quality = "EXCELLENT"
        elif max_deviation < 2.0:
            quality = "GOOD"
        elif max_deviation < 3.0:
            quality = "ACCEPTABLE"
        else:
            quality = "NEEDS_IMPROVEMENT"
        
        logger.info(f"Validation results: {quality}, max_dev={max_deviation:.1f}%, avg_dev={avg_deviation:.1f}%")
        
        return {
            'validation_samples': len(validation_data),
            'max_deviation': max_deviation,
            'avg_deviation': avg_deviation, 
            'quality': quality,
            'distribution': distribution,
            'deviations': deviations
        }
    
    def generate_classification_config(
        self, 
        parquet_files: Union[str, Path, List[Union[str, Path]]],
        currency: str = "AUDUSD",
        lookforward_input: int = 5000,
        lookback_rows: int = 5000,
        lookforward_offset: int = 500
    ) -> Tuple[RepresentConfig, Dict]:
        """
        Generate optimized classification configuration from parquet data.
        
        Args:
            parquet_files: Parquet file(s) containing price data
            currency: Currency pair name for metadata
            lookforward_input: Lookforward window size
            lookback_rows: Lookback window size  
            lookforward_offset: Lookforward offset
            
        Returns:
            Tuple of (RepresentConfig, validation_metrics)
            
        Raises:
            ValueError: If insufficient data or generation fails
        """
        logger.info(f"Generating classification config for {currency}")
        
        # Extract price changes
        price_changes = self.extract_price_changes_from_parquet(parquet_files)
        
        if len(price_changes) < self.target_samples:
            logger.warning(f"Only {len(price_changes)} samples, less than recommended {self.target_samples}")
        
        # Split data for training/validation
        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(price_changes))
        split_idx = int(len(price_changes) * (1 - self.validation_split))
        
        training_data = price_changes[indices[:split_idx]]
        validation_data = price_changes[indices[split_idx:]] if self.validation_split > 0 else np.array([])
        
        logger.info(f"Split data: {len(training_data)} training, {len(validation_data)} validation")
        
        # Calculate quantile thresholds
        threshold_info = self.calculate_quantile_thresholds(training_data)
        
        all_thresholds = threshold_info.get('all_thresholds')
        if not isinstance(all_thresholds, list):
            raise TypeError(f"Expected 'all_thresholds' to be a list, but got {type(all_thresholds)}")

        # Validate thresholds
        validation_metrics = self.validate_thresholds(
            all_thresholds, 
            validation_data
        ) if len(validation_data) > 0 else {}
        
        positive_thresholds = threshold_info.get('positive_thresholds')
        if not isinstance(positive_thresholds, dict):
            raise TypeError(f"Expected 'positive_thresholds' to be a dict, but got {type(positive_thresholds)}")

        # Create RepresentConfig with simplified parameters
        config = RepresentConfig(
            currency=currency,
            nbins=self.nbins,
            lookforward_input=lookforward_input,
            lookback_rows=lookback_rows,
            lookforward_offset=lookforward_offset,
        )
        
        # Note: We can't add custom fields to Pydantic models, so we'll include
        # the quantile thresholds in the metrics return instead
        
        # Add validation metrics to return
        full_metrics = {
            'threshold_info': threshold_info,
            'validation_metrics': validation_metrics,
            'generation_metadata': {
                'currency': currency,
                'generation_date': datetime.now().isoformat(),
                'training_samples': len(training_data),
                'validation_samples': len(validation_data),
                'total_samples': len(price_changes),
                'method': 'quantile_based_dynamic'
            }
        }
        
        logger.info(f"Generated config for {currency}: {validation_metrics.get('quality', 'UNKNOWN')} quality")
        return config, full_metrics


def generate_classification_config_from_parquet(
    config: RepresentConfig,
    parquet_files: Union[str, Path, List[Union[str, Path]]],
    **kwargs
) -> Tuple[RepresentConfig, Dict]:
    """
    Convenience function to generate classification config from parquet data.
    
    Args:
        config: RepresentConfig with currency-specific configuration
        parquet_files: Parquet file(s) containing price data
        **kwargs: Additional parameters for ClassificationConfigGenerator
        
    Returns:
        Tuple of (RepresentConfig, validation_metrics)
    """
    generator = ClassificationConfigGenerator(config=config, **kwargs)
    return generator.generate_classification_config(parquet_files, config.currency)


def classify_with_generated_config(
    price_changes: np.ndarray, 
    quantile_thresholds: List[float],
    nbins: int = 13
) -> np.ndarray:
    """
    Classify price changes using quantile thresholds.
    
    Args:
        price_changes: Array of price changes to classify
        quantile_thresholds: List of quantile-based thresholds
        nbins: Number of classification bins
        
    Returns:
        Array of classification labels
    """
    labels = np.zeros(len(price_changes), dtype=int)
    
    for i, value in enumerate(price_changes):
        bin_idx = 0
        for threshold in quantile_thresholds:
            if value <= threshold:
                break
            bin_idx += 1
        labels[i] = min(bin_idx, nbins - 1)
    
    return labels