#!/usr/bin/env python3
"""
Comprehensive API Usage Examples for represent v3.0.0.

This file demonstrates the clean 3-stage pipeline:
1. DBN → Unlabeled Parquet (Symbol-Grouped)
2. Dynamic Classification (Uniform Distribution)
3. ML Training (Memory-Efficient)
"""

import sys
from pathlib import Path

# Add represent to path for examples
sys.path.insert(0, str(Path(__file__).parent.parent))

import represent


def example_1_complete_pipeline():
    """Example 1: Complete 3-stage pipeline from DBN to ML training."""
    print("🚀 Example 1: Complete 3-Stage Pipeline")
    print("="*60)

    # Use high-level API for complete workflow
    api = represent.RepresentAPI()
    
    try:
        # Run complete pipeline: DBN → Unlabeled → Classified → ML Ready
        results = api.run_complete_pipeline(
            dbn_path="data/AUDUSD-20240101.dbn.zst",
            output_base_dir="data/pipeline_output/",
            currency="AUDUSD",
            features=['volume', 'variance'],
            min_symbol_samples=5000,
            force_uniform=True
        )
        
        print(f"✅ Pipeline complete!")
        print(f"   📊 Symbols processed: {results['total_symbols']}")
        print(f"   📊 Total samples: {results['total_samples']:,}")
        print(f"   📁 Classified data: {results['classified_directory']}")
        
    except FileNotFoundError:
        print("ℹ️  Skipping - no DBN file found (example only)")
        print("   Replace with your actual DBN file path")


def example_2_stage_by_stage():
    """Example 2: Manual stage-by-stage processing for fine control."""
    print("\n🔄 Example 2: Stage-by-Stage Processing")
    print("="*50)
    
    # Stage 1: DBN to unlabeled parquet (symbol-grouped)
    print("🔄 Stage 1: Converting DBN to unlabeled parquet...")
    try:
        conversion_stats = represent.convert_dbn_to_parquet(
            dbn_path="data/market_data.dbn.zst",
            output_dir="data/unlabeled/",
            currency="AUDUSD",
            features=['volume', 'variance', 'trade_counts'],  # All features
            min_symbol_samples=1000
        )
        
        print(f"✅ Stage 1 complete:")
        print(f"   📊 Symbols: {conversion_stats['symbols_processed']}")
        print(f"   📊 Samples: {conversion_stats['total_processed_samples']:,}")
        
        # Stage 2: Dynamic classification with uniform distribution
        print("\n🔄 Stage 2: Applying dynamic classification...")
        classification_stats = represent.classify_parquet_file(
            parquet_path="data/unlabeled/AUDUSD_M6AM4.parquet",
            output_path="data/classified/AUDUSD_M6AM4_classified.parquet",
            currency="AUDUSD",
            force_uniform=True  # Guaranteed uniform distribution
        )
        
        print(f"✅ Stage 2 complete:")
        print(f"   📊 Quality: {classification_stats.get('uniform_quality', 'N/A')}")
        print(f"   📊 Distribution: Uniform (7.69% per class)")
        
        # Stage 3: ML training dataloader
        print("\n🔄 Stage 3: Creating ML training dataloader...")
        dataloader = represent.create_parquet_dataloader(
            parquet_path="data/classified/AUDUSD_M6AM4_classified.parquet",
            batch_size=32,
            shuffle=True,
            sample_fraction=0.2  # Use 20% for quick iteration
        )
        
        print(f"✅ Stage 3 complete: Dataloader ready for training")
        
        # Demonstrate training loop
        print("\n🔄 Demo training loop...")
        for i, (features, labels) in enumerate(dataloader):
            print(f"   Batch {i+1}: Features {features.shape}, Labels {labels.shape}")
            if i >= 2:  # Just show first few batches
                break
                
    except FileNotFoundError:
        print("ℹ️  Skipping - no DBN file found (example only)")


def example_3_dynamic_classification():
    """Example 3: Dynamic classification config generation."""
    print("\n⚡ Example 3: Dynamic Classification Configuration")
    print("="*55)
    
    api = represent.RepresentAPI()
    
    # Generate classification config from existing data
    try:
        config_result = api.generate_classification_config(
            parquet_files="data/unlabeled/AUDUSD_M6AM4.parquet",
            currency="AUDUSD",
            nbins=13,
            target_samples=5000
        )
        
        print(f"✅ Dynamic config generated:")
        print(f"   🎯 Quality: {config_result['metrics']['validation_metrics']['quality']:.1%}")
        print(f"   📊 Method: {config_result['generation_method']}")
        print(f"   💱 Currency: {config_result['currency']}")
        
        # Show some config details
        config = config_result['config']
        print(f"   🔧 Bins: {config.classification.nbins}")
        print(f"   🔧 Lookforward: {config.classification.lookforward_input}")
        
    except (FileNotFoundError, ValueError) as e:
        print("ℹ️  Skipping - no parquet file found (run Stage 1 first)")
        print(f"   💡 This example would generate optimal classification thresholds from your data")


def example_4_batch_processing():
    """Example 4: Batch processing multiple files."""
    print("\n📦 Example 4: Batch Processing")
    print("="*35)
    
    # Batch convert multiple DBN files
    try:
        unlabeled_results = represent.batch_convert_unlabeled(
            input_directory="data/dbn_files/",
            output_directory="data/batch_unlabeled/",
            currency="AUDUSD",
            features=['volume', 'variance'],
            pattern="*.dbn*"
        )
        
        print(f"✅ Batch conversion complete:")
        print(f"   📊 Files processed: {len(unlabeled_results)}")
        
        # Batch classify the results
        classified_results = represent.batch_classify_parquet_files(
            input_directory="data/batch_unlabeled/",
            output_directory="data/batch_classified/",
            currency="AUDUSD",
            force_uniform=True
        )
        
        print(f"✅ Batch classification complete:")
        print(f"   📊 Files classified: {len(classified_results)}")
        
    except (FileNotFoundError, ValueError) as e:
        print("ℹ️  Skipping - no input directory found (example only)")
        print(f"   💡 This would process multiple DBN files in one command")


def example_5_currency_configurations():
    """Example 5: Currency-specific configurations."""
    print("\n💱 Example 5: Currency Configurations")
    print("="*40)
    
    # Show available currencies
    api = represent.RepresentAPI()
    currencies = api.list_available_currencies()
    print(f"📋 Available currencies: {currencies}")
    
    # Show currency-specific details
    for currency in ["AUDUSD", "USDJPY", "EURJPY"][:3]:  # Show first 3
        try:
            config = represent.load_currency_config(currency)
            print(f"\n💱 {currency}:")
            print(f"   📊 Classification bins: {config.classification.nbins}")
            print(f"   📏 Pip size: {config.classification.true_pip_size}")
            print(f"   🎯 Lookforward: {config.classification.lookforward_input}")
            
        except Exception as e:
            print(f"   ❌ Error loading {currency}: {e}")


def example_6_feature_combinations():
    """Example 6: Different feature combinations."""
    print("\n🎯 Example 6: Feature Combinations")
    print("="*38)
    
    feature_combinations = [
        ['volume'],
        ['volume', 'variance'], 
        ['volume', 'variance', 'trade_counts']
    ]
    
    for features in feature_combinations:
        print(f"\n🔧 Features: {features}")
        
        # Show expected output shape
        from represent import get_output_shape
        shape = get_output_shape(features)
        print(f"   📐 Output shape: {shape}")
        print(f"   📊 Dimensions: {'2D' if len(shape) == 2 else '3D'} tensor")


def example_7_memory_optimization():
    """Example 7: Memory optimization strategies."""
    print("\n💾 Example 7: Memory Optimization")
    print("="*37)
    
    # Different memory strategies
    strategies = [
        {"sample_fraction": 0.1, "cache_size": 500, "description": "Quick iteration"},
        {"sample_fraction": 0.5, "cache_size": 1000, "description": "Balanced training"},
        {"sample_fraction": 1.0, "cache_size": 2000, "description": "Full dataset"}
    ]
    
    for strategy in strategies:
        print(f"\n📊 Strategy: {strategy['description']}")
        print(f"   🎯 Sample fraction: {strategy['sample_fraction']:.0%}")
        print(f"   💾 Cache size: {strategy['cache_size']} samples")
        
        # This would create the dataloader with these settings
        print(f"   🔧 Usage: create_parquet_dataloader(..., sample_fraction={strategy['sample_fraction']}, cache_size={strategy['cache_size']})")


def main():
    """Run all examples."""
    print("🎉 Represent v3.0.0 API Usage Examples")
    print("="*45)
    
    # Get package info
    api = represent.RepresentAPI()
    info = api.get_package_info()
    print(f"📦 Version: {info['version']}")
    print(f"🏗️  Architecture: {info['architecture']}")
    print(f"🎯 Features: {info['supported_features']}")
    
    # Run examples
    example_1_complete_pipeline()
    example_2_stage_by_stage()
    example_3_dynamic_classification()
    example_4_batch_processing()
    example_5_currency_configurations()
    example_6_feature_combinations()
    example_7_memory_optimization()
    
    print("\n" + "="*60)
    print("🎉 All examples completed!")
    print("💡 Tip: Replace example paths with your actual data files")
    print("📚 See README.md for more details")


if __name__ == "__main__":
    main()