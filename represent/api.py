"""
High-level API for the represent package.

This module provides simplified, user-friendly functions for common workflows
in market depth machine learning pipelines.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from .unlabeled_converter import convert_dbn_to_parquet, batch_convert_dbn_files as batch_convert_unlabeled
from .parquet_classifier import classify_parquet_file, batch_classify_parquet_files
from .config import RepresentConfig
from .classification_config_generator import (
    generate_classification_config_from_parquet
)
from .parquet_classifier import ParquetClassifier, process_dbn_to_classified_parquets
from .global_threshold_calculator import GlobalThresholds, calculate_global_thresholds


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







    def convert_dbn_to_unlabeled_parquet(
        self,
        config: RepresentConfig,
        dbn_path: Union[str, Path],
        output_dir: Union[str, Path],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Stage 1: Convert DBN file to unlabeled symbol-grouped parquet datasets.

        Args:
            config: RepresentConfig with currency-specific configuration
            dbn_path: Path to input DBN file
            output_dir: Directory for output parquet files
            **kwargs: Additional arguments for converter

        Returns:
            Conversion statistics dictionary

        Examples:
            api = RepresentAPI()
            config = create_represent_config("AUDUSD", features=['volume', 'variance'])
            
            # Convert to symbol-grouped parquet files
            stats = api.convert_dbn_to_unlabeled_parquet(
                config,
                'data.dbn', 
                '/data/unlabeled/'
            )
        """
        return convert_dbn_to_parquet(
            config=config,
            dbn_path=dbn_path,
            output_dir=output_dir,
            **kwargs,
        )

    def batch_convert_dbn_to_unlabeled_parquet(
        self,
        config: RepresentConfig,
        input_directory: Union[str, Path],
        output_directory: Union[str, Path],
        pattern: str = "*.dbn*",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Batch convert multiple DBN files to unlabeled parquet datasets.

        Args:
            config: RepresentConfig with currency-specific configuration
            input_directory: Directory containing DBN files
            output_directory: Directory for output parquet files
            pattern: File pattern to match
            **kwargs: Additional arguments for converter

        Returns:
            List of conversion statistics for each file
        """
        return batch_convert_unlabeled(
            config=config,
            input_directory=input_directory,
            output_directory=output_directory,
            pattern=pattern,
            **kwargs,
        )

    def classify_symbol_parquet(
        self,
        config: RepresentConfig,
        parquet_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        force_uniform: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Stage 2: Apply uniform classification to unlabeled symbol parquet file.

        Args:
            config: RepresentConfig with currency-specific configuration
            parquet_path: Path to unlabeled parquet file
            output_path: Path for classified output file
            force_uniform: Apply optimization for uniform distribution
            **kwargs: Additional arguments for classifier

        Returns:
            Classification statistics dictionary

        Examples:
            api = RepresentAPI()
            config = create_represent_config("AUDUSD")
            
            # Classify single symbol parquet file
            stats = api.classify_symbol_parquet(
                config,
                '/data/unlabeled/AUDUSD_M6AM4.parquet',
                '/data/classified/AUDUSD_M6AM4_classified.parquet'
            )
        """
        return classify_parquet_file(
            config=config,
            parquet_path=parquet_path,
            output_path=output_path,
            force_uniform=force_uniform,
            **kwargs,
        )

    def batch_classify_symbol_parquets(
        self,
        config: RepresentConfig,
        input_directory: Union[str, Path],
        output_directory: Union[str, Path],
        pattern: str = "*_*.parquet",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Batch apply classification to multiple symbol parquet files.

        Args:
            config: RepresentConfig with currency-specific configuration
            input_directory: Directory containing unlabeled parquet files
            output_directory: Directory for classified parquet files
            pattern: File pattern to match
            **kwargs: Additional arguments for classifier

        Returns:
            List of classification statistics for each file

        Examples:
            api = RepresentAPI()
            config = create_represent_config("AUDUSD")
            
            # Classify all symbol parquet files in directory
            results = api.batch_classify_symbol_parquets(
                config,
                '/data/unlabeled/',
                '/data/classified/'
            )
        """
        return batch_classify_parquet_files(
            config=config,
            input_directory=input_directory,
            output_directory=output_directory,
            pattern=pattern,
            **kwargs,
        )


    def run_complete_pipeline(
        self,
        config: RepresentConfig,
        dbn_path: Union[str, Path],
        output_base_dir: Union[str, Path],
        force_uniform: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete 3-stage pipeline from DBN to ML-ready dataset.

        Args:
            config: RepresentConfig with currency-specific configuration
            dbn_path: Path to input DBN file
            output_base_dir: Base directory for all outputs
            force_uniform: Apply uniform distribution optimization
            verbose: Print progress information

        Returns:
            Complete pipeline statistics

        Examples:
            api = RepresentAPI()
            config = create_represent_config("AUDUSD", features=['volume', 'variance'])
            
            # Run complete pipeline
            results = api.run_complete_pipeline(
                config,
                'data.dbn',
                '/data/pipeline_output/'
            )
        """
        output_base = Path(output_base_dir)
        unlabeled_dir = output_base / "unlabeled"
        classified_dir = output_base / "classified"
        
        if verbose:
            print("🚀 Running Complete 3-Stage Pipeline")
            print(f"   📁 Output base: {output_base}")
            print(f"   💱 Currency: {config.currency}")
            print(f"   📊 Features: {config.features}")

        # Stage 1: DBN to unlabeled parquet
        if verbose:
            print("\n🔄 Stage 1: DBN → Unlabeled Parquet...")
        
        stage_1_stats = self.convert_dbn_to_unlabeled_parquet(
            config=config,
            dbn_path=dbn_path,
            output_dir=unlabeled_dir,
        )

        # Stage 2: Classify parquet files
        if verbose:
            print("\n🔄 Stage 2: Post-Processing Classification...")
        
        stage_2_stats = self.batch_classify_symbol_parquets(
            config=config,
            input_directory=unlabeled_dir,
            output_directory=classified_dir,
            force_uniform=force_uniform,
        )

        pipeline_results = {
            "pipeline_version": "v2.0.0 - 3-Stage Architecture",
            "input_file": str(dbn_path),
            "output_base_directory": str(output_base),
            "currency": config.currency,
            "features": config.features,
            "stage_1_stats": stage_1_stats,
            "stage_2_stats": stage_2_stats,
            "unlabeled_directory": str(unlabeled_dir),
            "classified_directory": str(classified_dir),
            "total_symbols": stage_1_stats.get("symbols_processed", 0),
            "total_samples": stage_1_stats.get("total_processed_samples", 0),
            "classified_files": len(stage_2_stats),
        }

        if verbose:
            print("\n✅ Complete Pipeline Finished!")
            print(f"   📊 Symbols processed: {pipeline_results['total_symbols']}")
            print(f"   📊 Total samples: {pipeline_results['total_samples']:,}")
            print(f"   📁 Classified data: {classified_dir}")
            print("   🚀 Ready for ML training!")

        return pipeline_results

    def process_dbn_to_classified_parquets(
        self,
        config: RepresentConfig,
        dbn_path: Union[str, Path],
        output_dir: Union[str, Path],
        force_uniform: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Streamlined approach: Process DBN file directly to classified parquet files.
        
        This is the new streamlined method that eliminates intermediate files by
        processing DBN data directly to classified parquet files, one per symbol.
        
        Args:
            config: RepresentConfig with currency-specific configuration
            dbn_path: Path to input DBN file
            output_dir: Directory for output classified parquet files
            force_uniform: Whether to enforce uniform class distribution
            verbose: Whether to print progress information
            
        Returns:
            Processing statistics dictionary
            
        Examples:
            api = RepresentAPI()
            config = create_represent_config("AUDUSD", features=['volume', 'variance'])
            
            # Streamlined processing: DBN → Classified Parquet (one step)
            results = api.process_dbn_to_classified_parquets(
                config,
                'market_data.dbn',
                '/data/classified/',
                force_uniform=True
            )
            
            # Files are immediately ready for ML training
            print(f"Generated {results['symbols_processed']} classified files")
        """
        return process_dbn_to_classified_parquets(
            config=config,
            dbn_path=dbn_path,
            output_dir=output_dir,
            force_uniform=force_uniform,
            verbose=verbose,
        )

    def create_parquet_classifier(
        self,
        config: RepresentConfig,
        force_uniform: bool = True,
        verbose: bool = True,
    ) -> ParquetClassifier:
        """
        Create a DBN-to-parquet classifier instance.
        
        Args:
            config: RepresentConfig with currency-specific configuration
            force_uniform: Whether to enforce uniform class distribution
            verbose: Whether to print progress information
            
        Returns:
            ParquetClassifier instance
            
        Examples:
            api = RepresentAPI()
            config = create_represent_config("AUDUSD", features=['volume'])
            
            # Create classifier with custom settings
            classifier = api.create_parquet_classifier(
                config,
                force_uniform=True
            )
            
            # Process multiple files
            for dbn_file in dbn_files:
                results = classifier.process_dbn_to_classified_parquets(
                    dbn_file, 
                    '/data/classified/'
                )
        """
        return ParquetClassifier(
            config=config,
            force_uniform=force_uniform,
            verbose=verbose,
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
                "/Users/danielfisher/data/databento/AUDUSD-micro",
                sample_fraction=0.5
            )
            
            # Use thresholds for consistent classification
            results = api.process_dbn_to_classified_parquets(
                config,
                "data.dbn",
                "/data/classified/",
                global_thresholds=thresholds
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
            "architecture": "Multi-approach: Classic 3-stage + Streamlined 2-stage (v4.0.0+)",
            "available_currencies": self.list_available_currencies(),
            "supported_features": ["volume", "variance", "trade_counts"],
            "tensor_shape": "(402, 500) - market depth representation",
            "pipeline_approaches": {
                "classic_3_stage": ["DBN → Unlabeled Parquet", "Post-Processing Classification", "ML Training"],
                "streamlined_2_stage": ["DBN → Classified Parquet (Direct)", "ML Training"]
            },
            "dynamic_features": [
                "Quantile-based Config Generation", 
                "Automatic Threshold Optimization",
                "Streamlined Direct Processing",
                "On-demand Feature Generation"
            ],
        }


# Create a default API instance for convenience
api = RepresentAPI()


# Convenience functions that use the default API instance
def load_training_dataset(config: RepresentConfig, *args, **kwargs):
    """Deprecated: Dataloader functionality moved to ML training repos."""
    raise NotImplementedError(
        "Dataloader functionality has been moved out of the represent package. "
        "Please see DATALOADER_MIGRATION_GUIDE.md for instructions on rebuilding "
        "the dataloader in your ML training repository."
    )
