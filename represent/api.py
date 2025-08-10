"""
High-level API for the represent package.

This module provides simplified, user-friendly functions for the
symbol-split-merge architecture (v5.0.0).
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from .dataset_builder import (
    DatasetBuilder, 
    DatasetBuildConfig,
    build_datasets_from_dbn_files,
    batch_build_datasets_from_directory
)
from .config import RepresentConfig
from .classification_config_generator import (
    generate_classification_config_from_parquet
)
from .global_threshold_calculator import GlobalThresholds, calculate_global_thresholds


class RepresentAPI:
    """
    High-level API class for represent package symbol-split-merge workflows.

    This provides a clean, object-oriented interface for creating comprehensive
    symbol datasets from multiple DBN files.
    """

    def __init__(self):
        """Initialize the API."""
        self._available_currencies = ["AUDUSD", "GBPUSD", "EURJPY", "EURUSD", "USDJPY"]

    def build_comprehensive_symbol_datasets(
        self,
        config: RepresentConfig,
        dbn_files: List[Union[str, Path]],
        output_dir: Union[str, Path],
        dataset_config: Optional[DatasetBuildConfig] = None,
        intermediate_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Build comprehensive symbol datasets from multiple DBN files.
        
        This is the primary workflow that implements symbol-split-merge architecture:
        1. Split each DBN file by symbol into intermediate files
        2. Merge all instances of each symbol across files into comprehensive datasets
        
        Args:
            config: RepresentConfig with currency-specific configuration
            dbn_files: List of DBN files to process
            output_dir: Directory for final comprehensive symbol datasets
            dataset_config: Configuration for dataset building process
            intermediate_dir: Directory for intermediate split files (temp if None)
            verbose: Whether to print progress information
            
        Returns:
            Processing statistics and comprehensive dataset information
            
        Examples:
            api = RepresentAPI()
            config = create_represent_config("AUDUSD", features=['volume', 'variance'])
            
            # Build comprehensive datasets from multiple DBN files
            results = api.build_comprehensive_symbol_datasets(
                config=config,
                dbn_files=[
                    'data/AUDUSD-20240101.dbn.zst',
                    'data/AUDUSD-20240102.dbn.zst',
                    'data/AUDUSD-20240103.dbn.zst'
                ],
                output_dir='/data/symbol_datasets/'
            )
            
            # Results contain comprehensive symbol datasets ready for ML training
            print(f"Created {results['phase_2_stats']['datasets_created']} comprehensive datasets")
            print(f"Total samples: {results['phase_2_stats']['total_samples']:,}")
        """
        if dataset_config is None:
            dataset_config = DatasetBuildConfig(
                currency=config.currency,
                features=config.features
            )
        
        return build_datasets_from_dbn_files(
            config=config,
            dbn_files=dbn_files,
            output_dir=output_dir,
            dataset_config=dataset_config,
            intermediate_dir=intermediate_dir,
            verbose=verbose
        )
    
    def build_datasets_from_directory(
        self,
        config: RepresentConfig,
        input_directory: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.dbn*",
        dataset_config: Optional[DatasetBuildConfig] = None,
        intermediate_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Build comprehensive symbol datasets from all DBN files in a directory.
        
        Args:
            config: RepresentConfig with currency-specific configuration
            input_directory: Directory containing DBN files
            output_dir: Directory for final comprehensive symbol datasets
            file_pattern: Pattern to match DBN files (default: "*.dbn*")
            dataset_config: Configuration for dataset building process
            intermediate_dir: Directory for intermediate split files (temp if None)
            verbose: Whether to print progress information
            
        Returns:
            Processing statistics and comprehensive dataset information
            
        Examples:
            api = RepresentAPI()
            config = create_represent_config("AUDUSD", features=['volume'])
            
            # Build datasets from all DBN files in directory
            results = api.build_datasets_from_directory(
                config=config,
                input_directory='data/audusd_dbn_files/',
                output_dir='/data/symbol_datasets/'
            )
            
            print(f"Processed {len(results['input_files'])} DBN files")
            print(f"Created {results['phase_2_stats']['datasets_created']} symbol datasets")
        """
        if dataset_config is None:
            dataset_config = DatasetBuildConfig(
                currency=config.currency,
                features=config.features
            )
        
        return batch_build_datasets_from_directory(
            config=config,
            input_directory=input_directory,
            output_dir=output_dir,
            file_pattern=file_pattern,
            dataset_config=dataset_config,
            intermediate_dir=intermediate_dir,
            verbose=verbose
        )
    
    def create_dataset_builder(
        self,
        config: RepresentConfig,
        dataset_config: Optional[DatasetBuildConfig] = None,
        verbose: bool = True,
    ) -> DatasetBuilder:
        """
        Create a DatasetBuilder instance for advanced symbol-split-merge processing.
        
        Args:
            config: RepresentConfig with currency-specific configuration
            dataset_config: Configuration for dataset building process
            verbose: Whether to print progress information
            
        Returns:
            DatasetBuilder instance for advanced processing
            
        Examples:
            api = RepresentAPI()
            config = create_represent_config("AUDUSD", features=['volume'])
            
            # Create builder for custom processing
            builder = api.create_dataset_builder(
                config=config,
                dataset_config=DatasetBuildConfig(min_symbol_samples=5000)
            )
            
            # Use builder for multiple operations
            for batch_of_files in dbn_file_batches:
                results = builder.build_datasets_from_dbn_files(
                    batch_of_files, 
                    f'/data/batch_{i}_datasets/'
                )
        """
        if dataset_config is None:
            dataset_config = DatasetBuildConfig(
                currency=config.currency,
                features=config.features
            )
            
        return DatasetBuilder(
            config=config,
            dataset_config=dataset_config,
            verbose=verbose
        )

    def list_available_currencies(self) -> List[str]:
        """
        List available predefined currency configurations.

        Returns:
            List of available currency pairs
        """
        return self._available_currencies.copy()

    def generate_classification_config(
        self,
        config: RepresentConfig,
        parquet_files: Union[str, Path, List[Union[str, Path]]],
        validation_split: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate optimized classification configuration from parquet data.
        
        This method dynamically creates classification configurations using
        quantile-based analysis, eliminating the need for static config files.

        Args:
            config: RepresentConfig with currency-specific configuration
            parquet_files: Parquet file(s) containing price data
            validation_split: Fraction of data for validation (0.0-1.0)
            **kwargs: Additional parameters for config generation

        Returns:
            Dictionary containing generated config and validation metrics

        Example:
            >>> api = RepresentAPI()
            >>> config = create_represent_config("AUDUSD")
            >>> result = api.generate_classification_config(
            ...     config,
            ...     parquet_files="/path/to/audusd_data.parquet"
            ... )
            >>> print(f"Quality: {result['metrics']['validation_metrics']['quality']}")
        """
        generated_config, metrics = generate_classification_config_from_parquet(
            config=config,
            parquet_files=parquet_files,
            validation_split=validation_split,
            **kwargs
        )
        
        return {
            'config': generated_config,
            'metrics': metrics,
            'currency': config.currency,
            'generation_method': 'quantile_based_dynamic'
        }

    def calculate_global_thresholds(
        self,
        config: RepresentConfig,
        data_directory: Union[str, Path],
        sample_fraction: float = 0.5,
        file_pattern: str = "*.dbn*",
        verbose: bool = True,
    ) -> GlobalThresholds:
        """
        Calculate global classification thresholds from a sample of DBN files.
        
        This ensures consistent classification thresholds across all symbols and files,
        preventing the problems of per-file quantile calculation.
        
        Args:
            config: RepresentConfig with currency-specific configuration
            data_directory: Directory containing DBN files
            sample_fraction: Fraction of files to use for threshold calculation
            file_pattern: Pattern to match DBN files
            verbose: Whether to print progress information
            
        Returns:
            GlobalThresholds object with quantile boundaries and metadata
            
        Example:
            api = RepresentAPI()
            config = create_represent_config("AUDUSD")
            
            # Calculate global thresholds from first 50% of files
            thresholds = api.calculate_global_thresholds(
                config,
                "/data/databento/AUDUSD-micro",
                sample_fraction=0.5
            )
            
            # Use thresholds for consistent classification
            results = api.build_comprehensive_symbol_datasets(
                config,
                dbn_files=["data.dbn"],
                output_dir="/data/datasets/",
                dataset_config=DatasetBuildConfig(global_thresholds=thresholds)
            )
        """
        return calculate_global_thresholds(
            config=config,
            data_directory=data_directory,
            sample_fraction=sample_fraction,
            file_pattern=file_pattern,
            verbose=verbose,
        )

    def get_package_info(self) -> Dict[str, Any]:
        """
        Get information about the represent package.

        Returns:
            Dictionary with package information
        """
        from . import __version__

        return {
            "version": __version__,
            "architecture": "Symbol-Split-Merge (v5.0.0+)",
            "available_currencies": self.list_available_currencies(),
            "supported_features": ["volume", "variance", "trade_counts"],
            "tensor_shape": "(402, 500) - market depth representation",
            "pipeline_approach": "Symbol-Split-Merge: Split DBN Files by Symbol → Merge Symbol Data Across Files → Comprehensive Symbol Datasets → ML Training",
            "pipeline_approaches": {
                "symbol_split_merge": "Symbol-Split-Merge Dataset Building (Primary approach v5.0.0+)",
                "first_half_training": "Per-symbol classification using first-half data for bin definition",
                "comprehensive_datasets": "Multi-DBN symbol datasets for comprehensive ML training",
                "uniform_distribution": "Quantile-based uniform class distribution per symbol",
                "classic_3_stage": "Legacy 3-stage approach (deprecated in v5.0.0+)",
                "streamlined_2_stage": "Legacy 2-stage approach (replaced by symbol-split-merge)"
            },
            "key_features": [
                "Symbol-Split-Merge Dataset Building",
                "Comprehensive Multi-File Symbol Datasets", 
                "Per-Symbol First-Half Classification",
                "Quantile-based Config Generation", 
                "Automatic Threshold Optimization",
                "Two-Phase Processing with Cleanup",
                "Performance-Optimized ML Training Datasets"
            ],
            "dynamic_features": [
                "Per-symbol first-half training classification",
                "Symbol-specific quantile boundaries",
                "Uniform distribution enforcement",
                "Multi-DBN comprehensive datasets"
            ],
        }


# Create a default API instance for convenience
api = RepresentAPI()


# Convenience functions that use the default API instance
def load_training_dataset(*args, **kwargs):
    """
    DataLoader functionality has been moved to ML training repositories.
    
    See examples/symbol_split_merge_demo.py for comprehensive dataset creation,
    then implement custom dataloaders in your ML training repository.
    """
    raise NotImplementedError(
        "DataLoader functionality has been moved out of the represent package. "
        "Use build_datasets_from_dbn_files() to create comprehensive symbol datasets, "
        "then implement custom dataloaders in your ML training repository. "
        "See examples/symbol_split_merge_demo.py for complete workflow."
    )