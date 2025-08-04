"""
High-level API for the represent package.

This module provides simplified, user-friendly functions for common workflows
in market depth machine learning pipelines.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from .converter import convert_dbn_file, batch_convert_dbn_files, DBNToParquetConverter
from .lazy_dataloader import LazyParquetDataset, LazyParquetDataLoader
from .dataloader import create_market_depth_dataloader
from .config import load_currency_config, load_config_from_file


class RepresentAPI:
    """
    High-level API class for represent package workflows.

    This provides a clean, object-oriented interface for common tasks:
    - Converting DBN files to training datasets
    - Loading and configuring dataloaders
    - Managing currency-specific configurations
    """

    def __init__(self):
        """Initialize the API."""
        self._available_currencies = ["AUDUSD", "GBPUSD", "EURJPY", "EURUSD", "USDJPY"]

    def convert_dbn_to_training_data(
        self,
        dbn_path: Union[str, Path],
        output_path: Union[str, Path],
        currency: Optional[str] = None,
        config_file: Optional[Union[str, Path]] = None,
        features: Optional[List[str]] = None,
        symbol_filter: Optional[str] = None,
        chunk_size: int = 100000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert DBN file to labeled parquet training dataset.

        Args:
            dbn_path: Path to input DBN file
            output_path: Path for output parquet file
            currency: Currency pair for predefined configuration
            config_file: Path to custom YAML/JSON configuration
            features: Features to extract ['volume', 'variance', 'trade_counts']
            symbol_filter: Filter by symbol (e.g., 'M6AM4')
            chunk_size: Processing chunk size
            verbose: Print progress information

        Returns:
            Dictionary with conversion statistics

        Examples:
            api = RepresentAPI()

            # Convert with predefined currency config
            stats = api.convert_dbn_to_training_data('data.dbn', 'output.parquet', currency='AUDUSD')

            # Convert with custom config and multi-features
            stats = api.convert_dbn_to_training_data(
                'data.dbn',
                'output.parquet',
                config_file='my_config.yaml',
                features=['volume', 'variance']
            )
        """
        if not verbose:
            # Temporarily suppress print statements in converter
            import builtins

            original_print = builtins.print
            builtins.print = lambda *args, **kwargs: None

        try:
            stats = convert_dbn_file(
                dbn_path=dbn_path,
                output_path=output_path,
                currency=currency,
                config_file=config_file,
                features=features or ["volume"],
                symbol_filter=symbol_filter,
                chunk_size=chunk_size,
                include_metadata=True,
            )
            return stats
        finally:
            if not verbose:
                builtins.print = original_print

    def batch_convert_dbn_directory(
        self,
        input_directory: Union[str, Path],
        output_directory: Union[str, Path],
        currency: Optional[str] = None,
        config_file: Optional[Union[str, Path]] = None,
        pattern: str = "*.dbn*",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Batch convert all DBN files in a directory.

        Args:
            input_directory: Directory containing DBN files
            output_directory: Directory for output parquet files
            currency: Currency pair for predefined configuration
            config_file: Path to custom configuration file
            pattern: File pattern to match
            **kwargs: Additional conversion parameters

        Returns:
            List of conversion statistics for each file
        """
        return batch_convert_dbn_files(
            input_directory=input_directory,
            output_directory=output_directory,
            currency=currency,
            config_file=config_file,
            pattern=pattern,
            **kwargs,
        )

    def create_dataloader(
        self,
        parquet_path: Union[str, Path],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        sample_fraction: float = 1.0,
        cache_size: int = 1000,
    ) -> LazyParquetDataLoader:
        """
        Create a PyTorch-compatible dataloader from labeled parquet file.

        Args:
            parquet_path: Path to labeled parquet file
            batch_size: Batch size for training
            shuffle: Whether to shuffle samples
            num_workers: Number of worker processes
            sample_fraction: Fraction of dataset to use
            cache_size: Number of samples to cache

        Returns:
            LazyParquetDataLoader instance

        Examples:
            api = RepresentAPI()
            dataloader = api.create_dataloader('labeled_data.parquet', batch_size=64)

            for features, labels in dataloader:
                # features: (64, 402, 500) market depth tensors
                # labels: (64,) classification targets
                pass
        """
        return create_market_depth_dataloader(
            parquet_path=parquet_path,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sample_fraction=sample_fraction,
            cache_size=cache_size,
        )

    def load_dataset(
        self,
        parquet_path: Union[str, Path],
        batch_size: int = 32,
        shuffle: bool = True,
        sample_fraction: float = 1.0,
        cache_size: int = 1000,
    ) -> LazyParquetDataset:
        """
        Load a dataset from labeled parquet file.

        Args:
            parquet_path: Path to labeled parquet file
            batch_size: Batch size for compatibility
            shuffle: Whether to shuffle sample indices
            sample_fraction: Fraction of dataset to use
            cache_size: Number of samples to cache

        Returns:
            LazyParquetDataset instance
        """
        return LazyParquetDataset(
            parquet_path=parquet_path,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_fraction=sample_fraction,
            cache_size=cache_size,
        )

    def get_currency_config(self, currency: str):
        """
        Get configuration for a specific currency pair.

        Args:
            currency: Currency pair identifier

        Returns:
            CurrencyConfig object
        """
        return load_currency_config(currency)

    def load_custom_config(self, config_path: Union[str, Path]):
        """
        Load configuration from custom YAML/JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            CurrencyConfig object
        """
        return load_config_from_file(config_path)

    def create_converter(
        self,
        currency: Optional[str] = None,
        config_file: Optional[Union[str, Path]] = None,
        features: Optional[List[str]] = None,
        batch_size: int = 2000,
    ) -> DBNToParquetConverter:
        """
        Create a DBN to Parquet converter with specified configuration.

        Args:
            currency: Currency pair for predefined config
            config_file: Path to custom configuration file
            features: Features to extract
            batch_size: Processing batch size

        Returns:
            DBNToParquetConverter instance
        """
        return DBNToParquetConverter(
            currency=currency,
            config_file=config_file,
            features=features or ["volume"],
            batch_size=batch_size,
        )

    def list_available_currencies(self) -> List[str]:
        """
        List available predefined currency configurations.

        Returns:
            List of available currency pairs
        """
        return self._available_currencies.copy()

    def get_package_info(self) -> Dict[str, Any]:
        """
        Get information about the represent package.

        Returns:
            Dictionary with package information
        """
        from . import __version__

        return {
            "version": __version__,
            "architecture": "DBN → Parquet → PyTorch",
            "available_currencies": self.list_available_currencies(),
            "supported_features": ["volume", "variance", "trade_counts"],
            "tensor_shape": "(402, 500) - market depth representation",
        }


# Create a default API instance for convenience
api = RepresentAPI()


# Convenience functions that use the default API instance
def convert_to_training_data(*args, **kwargs):
    """Convenience function for converting DBN to training data."""
    return api.convert_dbn_to_training_data(*args, **kwargs)


def create_training_dataloader(*args, **kwargs):
    """Convenience function for creating training dataloader."""
    return api.create_dataloader(*args, **kwargs)


def load_training_dataset(*args, **kwargs):
    """Convenience function for loading training dataset."""
    return api.load_dataset(*args, **kwargs)
