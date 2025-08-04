#!/usr/bin/env python3
"""
Comprehensive API Usage Examples for the represent package.

This file demonstrates various ways to use the represent package API
for DBN to parquet conversion and PyTorch training workflows.
"""

import sys
from pathlib import Path

# Add represent to path for examples
sys.path.insert(0, str(Path(__file__).parent.parent))

import represent


def example_1_simple_conversion():
    """Example 1: Simple DBN to parquet conversion with predefined currency."""
    print("📊 Example 1: Simple Conversion with Predefined Currency")
    print("-" * 60)

    # Simple one-line conversion
    try:
        stats = represent.convert_to_training_data(
            dbn_path="data/market_data.dbn",
            output_path="output/audusd_training.parquet",
            currency="AUDUSD",
        )
        print(f"✅ Converted {stats['labeled_samples']:,} samples")
        print(f"✅ Output file: {stats['output_file_size_mb']:.1f}MB")
    except FileNotFoundError:
        print("ℹ️  Skipping - no DBN file found (example only)")


def example_2_custom_config():
    """Example 2: Conversion with custom YAML configuration."""
    print("\n🔧 Example 2: Custom Configuration")
    print("-" * 40)

    # Create RepresentAPI instance for more control
    api = represent.RepresentAPI()

    # Load custom configuration
    try:
        custom_config = api.load_custom_config("examples/custom_config_example.yaml")
        print(f"✅ Loaded custom config: {custom_config.currency_pair}")
        print(f"   Lookforward window: {custom_config.classification.lookforward_input}")
        print(f"   Classification bins: {custom_config.classification.nbins}")

        # Convert with custom config
        stats = api.convert_dbn_to_training_data(
            dbn_path="data/scalping_data.dbn",
            output_path="output/custom_training.parquet",
            config_file="examples/custom_config_example.yaml",
            features=["volume", "variance"],
            verbose=False,  # Suppress progress output
        )
        print(f"✅ Custom conversion completed: {stats['labeled_samples']:,} samples")

    except FileNotFoundError:
        print("ℹ️  Skipping - no DBN file found (example only)")


def example_3_batch_processing():
    """Example 3: Batch processing directory of DBN files."""
    print("\n📁 Example 3: Batch Processing")
    print("-" * 35)

    api = represent.RepresentAPI()

    try:
        results = api.batch_convert_dbn_directory(
            input_directory="data/raw_dbn/",
            output_directory="output/labeled_parquet/",
            currency="GBPUSD",
            pattern="*.dbn*",
        )

        total_samples = sum(r["labeled_samples"] for r in results)
        print(f"✅ Batch processed {len(results)} files")
        print(f"✅ Total samples: {total_samples:,}")

    except FileNotFoundError:
        print("ℹ️  Skipping - no input directory found (example only)")


def example_4_training_pipeline():
    """Example 4: Complete training pipeline setup."""
    print("\n🎯 Example 4: Training Pipeline Setup")
    print("-" * 42)

    # Use existing test data if available
    parquet_file = "test_conversion/labeled_dataset.parquet"

    if Path(parquet_file).exists():
        print("Using existing test data for demonstration...")

        # Create dataloader for training
        dataloader = represent.create_training_dataloader(
            parquet_path=parquet_file, batch_size=16, shuffle=True, sample_fraction=0.5
        )

        print(f"✅ Created dataloader: {len(dataloader)} batches")

        # Simulate training loop
        batch_count = 0
        for features, labels in dataloader:
            batch_count += 1
            print(f"   Batch {batch_count}: features={features.shape}, labels={labels.shape}")

            # Simulate model forward pass
            # batch_predictions = labels  # Mock predictions (would be used in real training)
            print(f"   Processed batch with {features.shape[0]} samples")

            if batch_count >= 3:  # Just show first 3 batches
                break

        print("✅ Training loop simulation completed")

    else:
        print("ℹ️  No test data available - would create dataloader for training")


def example_5_api_exploration():
    """Example 5: Exploring API capabilities."""
    print("\n🔍 Example 5: API Exploration")
    print("-" * 32)

    api = represent.RepresentAPI()

    # Get package information
    info = api.get_package_info()
    print(f"📦 Package version: {info['version']}")
    print(f"🏗️  Architecture: {info['architecture']}")
    print(f"💰 Available currencies: {', '.join(info['available_currencies'])}")
    print(f"📊 Supported features: {', '.join(info['supported_features'])}")
    print(f"🎯 Tensor shape: {info['tensor_shape']}")

    # Compare currency configurations
    print("\n📊 Currency Configuration Comparison:")
    for currency in ["AUDUSD", "GBPUSD", "EURJPY"]:
        config = api.get_currency_config(currency)
        print(
            f"   {currency}: lookforward={config.classification.lookforward_input}, "
            f"micro_pip={config.classification.micro_pip_size}"
        )


def example_6_advanced_usage():
    """Example 6: Advanced API usage patterns."""
    print("\n⚡ Example 6: Advanced Usage Patterns")
    print("-" * 42)

    # Create converter with specific settings
    converter = represent.RepresentAPI().create_converter(
        currency="AUDUSD", features=["volume", "variance", "trade_counts"], batch_size=5000
    )

    print(f"🔧 Created converter: {converter.currency}")
    print(f"   Features: {converter.features}")
    print(f"   Batch size: {converter.batch_size}")
    print(f"   Classification bins: {converter.classification_config.nbins}")

    # Access low-level components if needed
    print("\n🔧 Low-level component access:")
    print(f"   DBNToParquetConverter: {represent.DBNToParquetConverter}")
    print(f"   LazyParquetDataset: {represent.LazyParquetDataset}")
    print(f"   Configuration loader: {represent.load_currency_config}")

    # Show constants and utilities
    print("\n📏 Constants:")
    print(f"   Price levels: {represent.PRICE_LEVELS}")
    print(f"   Time bins: {represent.TIME_BINS}")
    print(f"   Micro pip size: {represent.MICRO_PIP_SIZE}")


def main():
    """Run all API usage examples."""
    print("🚀 REPRESENT PACKAGE API EXAMPLES")
    print("=" * 50)
    print("Demonstrating various ways to use the represent package")
    print("for market depth machine learning workflows.")
    print("=" * 50)

    # Run all examples
    example_1_simple_conversion()
    example_2_custom_config()
    example_3_batch_processing()
    example_4_training_pipeline()
    example_5_api_exploration()
    example_6_advanced_usage()

    print("\n" + "=" * 50)
    print("🎉 API Examples Complete!")
    print("\n📚 Key Takeaways:")
    print("✅ Simple one-line conversions: represent.convert_to_training_data()")
    print("✅ Advanced control: represent.RepresentAPI() class")
    print("✅ Custom configurations: Load YAML/JSON config files")
    print("✅ Batch processing: Convert entire directories")
    print("✅ Training integration: PyTorch-compatible dataloaders")
    print("✅ Low-level access: All components available if needed")


if __name__ == "__main__":
    main()
