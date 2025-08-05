"""
High-level API for the represent package.

This module provides simplified, user-friendly functions for common workflows
in market depth machine learning pipelines.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from .converter import convert_dbn_file, batch_convert_dbn_files, DBNToParquetConverter
from .unlabeled_converter import convert_dbn_to_parquet, batch_convert_dbn_files as batch_convert_unlabeled
from .parquet_classifier import classify_parquet_file, batch_classify_parquet_files
from .lazy_dataloader import LazyParquetDataset, LazyParquetDataLoader
from .dataloader import create_market_depth_dataloader
from .config import load_currency_config, load_config_from_file
from .classification_config_generator import (
    ClassificationConfigGenerator, 
    generate_classification_config_from_parquet,
    classify_with_generated_config
)


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

    def convert_dbn_to_unlabeled_parquet(
        self,
        dbn_path: Union[str, Path],
        output_dir: Union[str, Path],
        currency: str = "AUDUSD",
        features: Optional[List[str]] = None,
        min_symbol_samples: int = 1000,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Stage 1: Convert DBN file to unlabeled symbol-grouped parquet datasets.

        Args:
            dbn_path: Path to input DBN file
            output_dir: Directory for output parquet files
            currency: Currency pair for naming convention
            features: Features to extract ['volume', 'variance', 'trade_counts']
            min_symbol_samples: Minimum samples required per symbol
            **kwargs: Additional arguments for converter

        Returns:
            Conversion statistics dictionary

        Examples:
            api = RepresentAPI()
            
            # Convert to symbol-grouped parquet files
            stats = api.convert_dbn_to_unlabeled_parquet(
                'data.dbn', 
                '/data/unlabeled/', 
                currency='AUDUSD',
                features=['volume', 'variance']
            )
        """
        return convert_dbn_to_parquet(
            dbn_path=dbn_path,
            output_dir=output_dir,
            currency=currency,
            features=features,
            min_symbol_samples=min_symbol_samples,
            **kwargs,
        )

    def batch_convert_dbn_to_unlabeled_parquet(
        self,
        input_directory: Union[str, Path],
        output_directory: Union[str, Path],
        currency: str = "AUDUSD",
        features: Optional[List[str]] = None,
        pattern: str = "*.dbn*",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Batch convert multiple DBN files to unlabeled parquet datasets.

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
        return batch_convert_unlabeled(
            input_directory=input_directory,
            output_directory=output_directory,
            currency=currency,
            features=features,
            pattern=pattern,
            **kwargs,
        )

    def classify_symbol_parquet(
        self,
        parquet_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        currency: str = "AUDUSD",
        force_uniform: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Stage 2: Apply uniform classification to unlabeled symbol parquet file.

        Args:
            parquet_path: Path to unlabeled parquet file
            output_path: Path for classified output file
            currency: Currency pair for configuration
            force_uniform: Apply optimization for uniform distribution
            **kwargs: Additional arguments for classifier

        Returns:
            Classification statistics dictionary

        Examples:
            api = RepresentAPI()
            
            # Classify single symbol parquet file
            stats = api.classify_symbol_parquet(
                '/data/unlabeled/AUDUSD_M6AM4.parquet',
                '/data/classified/AUDUSD_M6AM4_classified.parquet',
                currency='AUDUSD'
            )
        """
        return classify_parquet_file(
            parquet_path=parquet_path,
            output_path=output_path,
            currency=currency,
            force_uniform=force_uniform,
            **kwargs,
        )

    def batch_classify_symbol_parquets(
        self,
        input_directory: Union[str, Path],
        output_directory: Union[str, Path],
        currency: str = "AUDUSD",
        pattern: str = "*_*.parquet",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Batch apply classification to multiple symbol parquet files.

        Args:
            input_directory: Directory containing unlabeled parquet files
            output_directory: Directory for classified parquet files
            currency: Currency pair for configuration
            pattern: File pattern to match
            **kwargs: Additional arguments for classifier

        Returns:
            List of classification statistics for each file

        Examples:
            api = RepresentAPI()
            
            # Classify all symbol parquet files in directory
            results = api.batch_classify_symbol_parquets(
                '/data/unlabeled/',
                '/data/classified/',
                currency='AUDUSD'
            )
        """
        return batch_classify_parquet_files(
            input_directory=input_directory,
            output_directory=output_directory,
            currency=currency,
            pattern=pattern,
            **kwargs,
        )

    def create_ml_dataloader(
        self,
        parquet_path: Union[str, Path],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        sample_fraction: float = 1.0,
        cache_size: int = 1000,
    ) -> LazyParquetDataLoader:
        """
        Stage 3: Create ML training dataloader from classified parquet file.

        Args:
            parquet_path: Path to classified parquet file
            batch_size: Batch size for training
            shuffle: Whether to shuffle samples
            num_workers: Number of worker processes
            sample_fraction: Fraction of dataset to use
            cache_size: Number of samples to cache

        Returns:
            LazyParquetDataLoader instance

        Examples:
            api = RepresentAPI()
            dataloader = api.create_ml_dataloader(
                '/data/classified/AUDUSD_M6AM4_classified.parquet', 
                batch_size=64
            )

            for features, labels in dataloader:
                # features: (64, [N_features,] 402, 500) market depth tensors
                # labels: (64,) classification targets 0-12
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

    def run_complete_pipeline(
        self,
        dbn_path: Union[str, Path],
        output_base_dir: Union[str, Path],
        currency: str = "AUDUSD",
        features: Optional[List[str]] = None,
        min_symbol_samples: int = 1000,
        force_uniform: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete 3-stage pipeline from DBN to ML-ready dataset.

        Args:
            dbn_path: Path to input DBN file
            output_base_dir: Base directory for all outputs
            currency: Currency pair for configuration
            features: Features to extract
            min_symbol_samples: Minimum samples per symbol
            force_uniform: Apply uniform distribution optimization
            verbose: Print progress information

        Returns:
            Complete pipeline statistics

        Examples:
            api = RepresentAPI()
            
            # Run complete pipeline
            results = api.run_complete_pipeline(
                'data.dbn',
                '/data/pipeline_output/',
                currency='AUDUSD',
                features=['volume', 'variance']
            )
        """
        output_base = Path(output_base_dir)
        unlabeled_dir = output_base / "unlabeled"
        classified_dir = output_base / "classified"
        
        if verbose:
            print(f"🚀 Running Complete 3-Stage Pipeline")
            print(f"   📁 Output base: {output_base}")
            print(f"   💱 Currency: {currency}")
            print(f"   📊 Features: {features or ['volume']}")

        # Stage 1: DBN to unlabeled parquet
        if verbose:
            print("\n🔄 Stage 1: DBN → Unlabeled Parquet...")
        
        stage_1_stats = self.convert_dbn_to_unlabeled_parquet(
            dbn_path=dbn_path,
            output_dir=unlabeled_dir,
            currency=currency,
            features=features,
            min_symbol_samples=min_symbol_samples,
        )

        # Stage 2: Classify parquet files
        if verbose:
            print("\n🔄 Stage 2: Post-Processing Classification...")
        
        stage_2_stats = self.batch_classify_symbol_parquets(
            input_directory=unlabeled_dir,
            output_directory=classified_dir,
            currency=currency,
            force_uniform=force_uniform,
        )

        pipeline_results = {
            "pipeline_version": "v2.0.0 - 3-Stage Architecture",
            "input_file": str(dbn_path),
            "output_base_directory": str(output_base),
            "currency": currency,
            "features": features or ["volume"],
            "stage_1_stats": stage_1_stats,
            "stage_2_stats": stage_2_stats,
            "unlabeled_directory": str(unlabeled_dir),
            "classified_directory": str(classified_dir),
            "total_symbols": stage_1_stats.get("symbols_processed", 0),
            "total_samples": stage_1_stats.get("total_processed_samples", 0),
            "classified_files": len(stage_2_stats),
        }

        if verbose:
            print(f"\n✅ Complete Pipeline Finished!")
            print(f"   📊 Symbols processed: {pipeline_results['total_symbols']}")
            print(f"   📊 Total samples: {pipeline_results['total_samples']:,}")
            print(f"   📊 Classified files: {pipeline_results['classified_files']}")
            print(f"   📁 Classified data: {classified_dir}")
            print(f"   🚀 Ready for ML training!")

        return pipeline_results

    def list_available_currencies(self) -> List[str]:
        """
        List available predefined currency configurations.

        Returns:
            List of available currency pairs
        """
        return self._available_currencies.copy()

    def generate_classification_config(
        self,
        parquet_files: Union[str, Path, List[Union[str, Path]]],
        currency: str,
        nbins: int = 13,
        target_samples: int = 1000,
        validation_split: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate optimized classification configuration from parquet data.
        
        This method dynamically creates classification configurations using
        quantile-based analysis, eliminating the need for static config files.

        Args:
            parquet_files: Parquet file(s) containing price data
            currency: Currency pair name (e.g., "AUDUSD")
            nbins: Number of classification bins (default: 13)
            target_samples: Minimum samples required for reliable config
            validation_split: Fraction of data for validation (0.0-1.0)
            **kwargs: Additional parameters for config generation

        Returns:
            Dictionary containing generated config and validation metrics

        Example:
            >>> api = RepresentAPI()
            >>> result = api.generate_classification_config(
            ...     parquet_files="/path/to/audusd_data.parquet",
            ...     currency="AUDUSD"
            ... )
            >>> print(f"Quality: {result['metrics']['validation_metrics']['quality']}")
            >>> config = result['config']
        """
        config, metrics = generate_classification_config_from_parquet(
            parquet_files=parquet_files,
            currency=currency,
            nbins=nbins,
            target_samples=target_samples,
            validation_split=validation_split,
            **kwargs
        )
        
        return {
            'config': config,
            'metrics': metrics,
            'currency': currency,
            'generation_method': 'quantile_based_dynamic'
        }

    def convert_with_dynamic_classification(
        self,
        dbn_path: Union[str, Path],
        output_path: Union[str, Path],
        parquet_files_for_config: Union[str, Path, List[Union[str, Path]]],
        currency: str,
        features: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert DBN file using dynamically generated classification config.
        
        This method first generates an optimized classification configuration
        from existing parquet data, then uses it to convert the DBN file.

        Args:
            dbn_path: Path to input DBN file
            output_path: Path for output classified parquet file
            parquet_files_for_config: Existing parquet files to analyze for config
            currency: Currency pair name
            features: List of features to extract
            **kwargs: Additional conversion parameters

        Returns:
            Dictionary with conversion results and config generation metrics

        Example:
            >>> api = RepresentAPI()
            >>> result = api.convert_with_dynamic_classification(
            ...     dbn_path="/path/to/new_data.dbn",
            ...     output_path="/path/to/classified_output.parquet", 
            ...     parquet_files_for_config="/path/to/training_data.parquet",
            ...     currency="AUDUSD"
            ... )
            >>> print(f"Config quality: {result['config_metrics']['validation_metrics']['quality']}")
        """
        # Generate dynamic classification config
        config_result = self.generate_classification_config(
            parquet_files=parquet_files_for_config,
            currency=currency,
            **kwargs
        )
        
        # Convert DBN file using generated config
        conversion_result = convert_dbn_file(
            dbn_path=dbn_path,
            output_path=output_path,
            currency=currency,
            config=config_result['config'],
            features=features or ['volume'],
            **kwargs
        )
        
        return {
            'conversion_result': conversion_result,
            'config_metrics': config_result['metrics'],
            'generated_config': config_result['config'],
            'currency': currency,
            'method': 'dynamic_classification'
        }

    def get_package_info(self) -> Dict[str, Any]:
        """
        Get information about the represent package.

        Returns:
            Dictionary with package information
        """
        from . import __version__

        return {
            "version": __version__,
            "architecture": "DBN → Unlabeled Parquet → Classification → PyTorch (v2.0.0+)",
            "available_currencies": self.list_available_currencies(),
            "supported_features": ["volume", "variance", "trade_counts"],
            "tensor_shape": "(402, 500) - market depth representation",
            "pipeline_stages": ["Raw Data Processing", "Post-Processing Classification", "ML Training"],
            "dynamic_features": ["Quantile-based Config Generation", "Automatic Threshold Optimization"],
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
