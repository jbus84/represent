#!/usr/bin/env python3
"""
Dynamic Classification Configuration Demo

This demo shows how to use the new dynamic classification config generation
to automatically optimize classification thresholds from parquet data.
"""

import sys
from pathlib import Path

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from represent import (
        ClassificationConfigGenerator,
        generate_classification_config_from_parquet,
        RepresentAPI
    )
    print("✅ Successfully imported represent with dynamic config support")
except ImportError as e:
    print(f"❌ Error importing represent package: {e}")
    sys.exit(1)


def demo_basic_config_generation():
    """Demo basic dynamic config generation from parquet data."""
    print("\n🎯 Demo 1: Basic Dynamic Config Generation")
    print("=" * 60)
    
    # Find available parquet files
    parquet_dirs = [
        Path("data/pipeline_outputs/classified_parquet"),
        Path("data/pipeline_outputs/working")
    ]
    
    parquet_files = []
    for parquet_dir in parquet_dirs:
        if parquet_dir.exists():
            parquet_files.extend(list(parquet_dir.glob("*.parquet")))
    
    if not parquet_files:
        print("❌ No parquet files found for demo")
        return
    
    print(f"📊 Using parquet files: {[f.name for f in parquet_files]}")
    
    try:
        # Generate config using convenience function
        config, metrics = generate_classification_config_from_parquet(
            parquet_files=parquet_files,
            currency="AUDUSD",
            nbins=13,
            target_samples=500
        )
        
        print("✅ Generated config for AUDUSD")
        print(f"   📊 Training samples: {metrics['generation_metadata']['training_samples']:,}")
        print(f"   📊 Validation samples: {metrics['generation_metadata']['validation_samples']:,}")
        print(f"   📊 Quality: {metrics['validation_metrics']['quality']}")
        print(f"   📊 Max deviation: {metrics['validation_metrics']['max_deviation']:.1f}%")
        print(f"   📊 Avg deviation: {metrics['validation_metrics']['avg_deviation']:.1f}%")
        
        # Show some thresholds
        thresholds = metrics['threshold_info']['all_thresholds']
        print(f"   📊 Generated {len(thresholds)} thresholds:")
        for i, threshold in enumerate(thresholds[:5]):  # Show first 5
            pips = threshold / 0.00001  # MICRO_PIP_SIZE
            print(f"      Threshold {i+1}: {threshold:.6f} ({pips:.2f} pips)")
        
        return config, metrics
        
    except Exception as e:
        print(f"❌ Error generating config: {e}")
        return None, None


def demo_advanced_config_generator():
    """Demo advanced config generation with custom parameters."""
    print("\n🎯 Demo 2: Advanced Config Generator")
    print("=" * 60)
    
    # Find parquet files
    parquet_files = list(Path("data/pipeline_outputs/classified_parquet").glob("*.parquet"))
    working_files = list(Path("data/pipeline_outputs/working").glob("*.parquet"))
    parquet_files.extend(working_files)
    
    if not parquet_files:
        print("❌ No parquet files found for demo")
        return
    
    try:
        # Create advanced config generator
        generator = ClassificationConfigGenerator(
            nbins=13,
            target_samples=800,
            validation_split=0.25,  # Use 25% for validation
            random_seed=42
        )
        
        print(f"📊 Using {len(parquet_files)} parquet files")
        print("📊 Configuration: 13 bins, 25% validation split")
        
        # Generate config with detailed metrics
        config, full_metrics = generator.generate_classification_config(
            parquet_files=parquet_files,
            currency="AUDUSD",
            lookforward_input=5000,
            lookback_rows=5000,
            lookforward_offset=500
        )
        
        print("✅ Advanced config generated successfully")
        
        # Show detailed results
        val_metrics = full_metrics['validation_metrics']
        threshold_info = full_metrics['threshold_info']
        
        print("   📊 Data Quality:")
        print(f"      Training samples: {full_metrics['generation_metadata']['training_samples']:,}")
        print(f"      Validation samples: {full_metrics['generation_metadata']['validation_samples']:,}")
        print(f"      Total samples: {full_metrics['generation_metadata']['total_samples']:,}")
        
        print("   📊 Classification Quality:")
        print(f"      Overall quality: {val_metrics['quality']}")
        print(f"      Max deviation: {val_metrics['max_deviation']:.2f}%")
        print(f"      Avg deviation: {val_metrics['avg_deviation']:.2f}%")
        
        print("   📊 Data Statistics:")
        stats = threshold_info['data_stats']
        print(f"      Mean price change: {stats['mean']:.6f}")
        print(f"      Std deviation: {stats['std']:.6f}")
        print(f"      Range: {stats['min']:.6f} to {stats['max']:.6f}")
        
        # Show distribution quality
        distribution = val_metrics['distribution']
        target_percent = 100.0 / 13
        print(f"   📊 Class Distribution (target: {target_percent:.1f}% each):")
        for i, percentage in enumerate(distribution):
            deviation = abs(percentage - target_percent)
            status = "✅" if deviation < 2.0 else "⚠️" if deviation < 3.0 else "❌"
            print(f"      Class {i:2d}: {percentage:5.1f}% {status}")
        
        return config, full_metrics
        
    except Exception as e:
        print(f"❌ Error in advanced config generation: {e}")
        return None, None


def demo_api_integration():
    """Demo integration with RepresentAPI."""
    print("\n🎯 Demo 3: RepresentAPI Integration")
    print("=" * 60)
    
    # Find parquet files
    parquet_files = list(Path("data/pipeline_outputs/classified_parquet").glob("*.parquet"))
    working_files = list(Path("data/pipeline_outputs/working").glob("*.parquet"))
    parquet_files.extend(working_files)
    
    if not parquet_files:
        print("❌ No parquet files found for demo")
        return
    
    try:
        # Use RepresentAPI for config generation
        api = RepresentAPI()
        
        result = api.generate_classification_config(
            parquet_files=parquet_files,
            currency="AUDUSD",
            nbins=13,
            target_samples=500,
            validation_split=0.3
        )
        
        print("✅ Generated config via RepresentAPI")
        print(f"   📊 Currency: {result['currency']}")
        print(f"   📊 Method: {result['generation_method']}")
        
        metrics = result['metrics']
        val_metrics = metrics['validation_metrics']
        
        print("   📊 Quality Assessment:")
        print(f"      Overall quality: {val_metrics['quality']}")
        print(f"      Max deviation: {val_metrics['max_deviation']:.2f}%")
        print(f"      Avg deviation: {val_metrics['avg_deviation']:.2f}%")
        print(f"      Training samples: {metrics['generation_metadata']['training_samples']:,}")
        
        # Show package info with dynamic features
        package_info = api.get_package_info()
        print("   📊 Package Info:")
        print(f"      Version: {package_info['version']}")
        print(f"      Architecture: {package_info['architecture']}")
        print(f"      Dynamic features: {package_info['dynamic_features']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in API integration demo: {e}")
        return None


def demo_config_comparison():
    """Demo comparing static vs dynamic configs."""
    print("\n🎯 Demo 4: Static vs Dynamic Config Comparison")
    print("=" * 60)
    
    try:
        # Load static config (if available)
        from represent import load_currency_config
        static_config = load_currency_config("AUDUSD")
        
        print("📊 Static Config (AUDUSD):")
        print(f"   Bins: {static_config.classification.nbins}")
        
        # Get threshold from bin_thresholds structure
        bin_thresholds = static_config.classification.bin_thresholds
        if static_config.classification.nbins in bin_thresholds:
            nested_thresholds = bin_thresholds[static_config.classification.nbins]
            if 100 in nested_thresholds and 5000 in nested_thresholds[100]:
                thresholds = nested_thresholds[100][5000]
                max_threshold = max(thresholds.values()) if thresholds else 0.0
                print(f"   Max threshold: {max_threshold:.6f}")
                print(f"   Threshold count: {len(thresholds)}")
        
        print("   Method: Static/Predefined")
        
    except Exception as e:
        print(f"⚠️  Static config not available: {e}")
        static_config = None
    
    # Generate dynamic config
    parquet_files = list(Path("data/pipeline_outputs/classified_parquet").glob("*.parquet"))
    working_files = list(Path("data/pipeline_outputs/working").glob("*.parquet"))
    parquet_files.extend(working_files)
    
    if parquet_files:
        try:
            config, metrics = generate_classification_config_from_parquet(
                parquet_files=parquet_files,
                currency="AUDUSD"
            )
            
            print("📊 Dynamic Config (Generated):")
            print(f"   Bins: {config.nbins}")
            
            # Get max threshold from quantile thresholds in metrics
            if 'threshold_info' in metrics and 'all_thresholds' in metrics['threshold_info']:
                quantile_thresholds = metrics['threshold_info']['all_thresholds']
                max_threshold = max(quantile_thresholds) if quantile_thresholds else 0.0
                min_threshold = min(quantile_thresholds) if quantile_thresholds else 0.0
                print(f"   Threshold range: {min_threshold:.6f} to {max_threshold:.6f}")
                print(f"   Threshold count: {len(quantile_thresholds)}")
            
            print("   Method: Quantile-based Dynamic")
            print(f"   Quality: {metrics['validation_metrics']['quality']}")
            print(f"   Deviation: {metrics['validation_metrics']['avg_deviation']:.2f}%")
            
            if static_config:
                print("📊 Comparison:")
                print("   Dynamic config optimized for actual data distribution")
                print("   Static config uses predefined values")
                print(f"   Dynamic config provides {metrics['validation_metrics']['avg_deviation']:.1f}% average deviation")
            
        except Exception as e:
            print(f"❌ Error generating dynamic config: {e}")
    
    else:
        print("❌ No parquet files available for dynamic config generation")


def main():
    """Run all dynamic config demos."""
    print("🚀 Dynamic Classification Configuration Demos")
    print("=" * 70)
    print("This demo shows the new dynamic config generation capabilities")
    print("that eliminate the need for static currency configuration files.")
    
    # Run demos
    config1, metrics1 = demo_basic_config_generation()
    config2, metrics2 = demo_advanced_config_generator() 
    demo_api_integration()
    demo_config_comparison()
    
    print("\n" + "=" * 70)
    print("🎉 Dynamic Configuration Demo Complete!")
    print("=" * 70)
    
    print("Key Benefits of Dynamic Configuration:")
    print("✅ No need for static currency config files")
    print("✅ Automatic optimization based on actual data")
    print("✅ Quantile-based uniform distribution")
    print("✅ Real-time quality assessment")
    print("✅ Easy integration with existing workflows")
    
    if config1 and metrics1:
        print(f"\nBest Result: {metrics1['validation_metrics']['quality']} quality")
        print(f"Average deviation: {metrics1['validation_metrics']['avg_deviation']:.2f}%")
        print("Ready for production use! 🚀")


if __name__ == "__main__":
    main()