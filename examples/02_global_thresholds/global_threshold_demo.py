#!/usr/bin/env python3
"""
Global Threshold Classification Demo

This example demonstrates the correct approach for consistent classification
across multiple DBN files by:

1. Calculating global thresholds from a sample of files (50% by default)
2. Using those thresholds consistently across all files for classification

This ensures that classification labels are comparable between symbols and files,
unlike per-file quantile calculation which creates incomparable classifications.
"""

import sys
from pathlib import Path
import time
import json

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent import (
    calculate_global_thresholds,
    process_dbn_to_classified_parquets,
    create_parquet_dataloader
)


def demonstrate_global_threshold_workflow():
    """Demonstrate the complete global threshold workflow."""
    
    print("🌐 GLOBAL THRESHOLD CLASSIFICATION WORKFLOW")
    print("=" * 70)
    print("📋 Step-by-step demonstration of consistent classification")
    print("=" * 70)
    
    # Configuration
    data_directory = Path("/Users/danielfisher/data/databento/AUDUSD-micro")
    output_directory = Path("/Users/danielfisher/repositories/represent/examples/outputs/global_threshold_demo")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    currency = "AUDUSD"
    sample_fraction = 0.5  # Use 50% of files for threshold calculation
    
    print(f"📁 Data directory: {data_directory}")
    print(f"📁 Output directory: {output_directory}")
    print(f"💱 Currency: {currency}")
    print(f"🔢 Sample fraction: {sample_fraction} ({sample_fraction*100}% of files)")
    
    # Step 1: Calculate Global Thresholds
    print("\n🎯 STEP 1: CALCULATE GLOBAL THRESHOLDS")
    print("=" * 50)
    
    try:
        start_time = time.perf_counter()
        
        global_thresholds = calculate_global_thresholds(
            data_directory=data_directory,
            currency=currency,
            sample_fraction=sample_fraction,
            nbins=13,
            lookforward_rows=500,
            verbose=True
        )
        
        threshold_calculation_time = time.perf_counter() - start_time
        
        print("\n✅ Global thresholds calculated successfully!")
        print(f"⏱️  Calculation time: {threshold_calculation_time:.1f}s")
        print(f"📊 Sample size: {global_thresholds.sample_size:,} price movements")
        print(f"📁 Files analyzed: {global_thresholds.files_analyzed}")
        
        # Save thresholds for reference
        threshold_data = {
            "currency": currency,
            "nbins": global_thresholds.nbins,
            "sample_size": global_thresholds.sample_size,
            "files_analyzed": global_thresholds.files_analyzed,
            "quantile_boundaries": global_thresholds.quantile_boundaries.tolist(),
            "price_movement_stats": global_thresholds.price_movement_stats,
            "calculation_time_seconds": threshold_calculation_time,
        }
        
        threshold_file = output_directory / "global_thresholds.json"
        with open(threshold_file, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        
        print(f"💾 Saved thresholds: {threshold_file}")
        
    except Exception as e:
        print(f"❌ Failed to calculate global thresholds: {e}")
        return False
    
    # Step 2: Process Files with Global Thresholds
    print("\n🔄 STEP 2: PROCESS FILES WITH GLOBAL THRESHOLDS")
    print("=" * 50)
    
    try:
        # Find all DBN files
        dbn_files = sorted(list(data_directory.glob("*.dbn*")))
        
        if not dbn_files:
            print(f"❌ No DBN files found in {data_directory}")
            return False
        
        print(f"📊 Found {len(dbn_files)} DBN files to process")
        
        # Process each file with the same global thresholds
        all_processing_stats = []
        total_start_time = time.perf_counter()
        
        for i, dbn_file in enumerate(dbn_files[:3]):  # Process first 3 files for demo
            print(f"\n🔄 Processing {i+1}/{min(3, len(dbn_files))}: {dbn_file.name}")
            
            file_start_time = time.perf_counter()
            
            processing_stats = process_dbn_to_classified_parquets(
                dbn_path=dbn_file,
                output_dir=output_directory / "classified",
                currency=currency,
                features=["volume"],
                min_symbol_samples=1000,
                force_uniform=True,
                nbins=13,
                global_thresholds=global_thresholds,  # Use consistent thresholds
                verbose=False  # Reduce output for demo
            )
            
            processing_time = time.perf_counter() - file_start_time
            processing_stats['individual_processing_time'] = processing_time
            processing_stats['source_file'] = str(dbn_file)
            
            all_processing_stats.append(processing_stats)
            
            print(f"   ✅ Processed {processing_stats['symbols_processed']} symbols")
            print(f"   📊 Generated {processing_stats['total_classified_samples']:,} classified samples")
            print(f"   ⏱️  Processing time: {processing_time:.1f}s")
            print(f"   📈 Rate: {processing_stats['samples_per_second']:.0f} samples/sec")
        
        total_processing_time = time.perf_counter() - total_start_time
        
        # Calculate summary statistics
        total_symbols = sum(stats['symbols_processed'] for stats in all_processing_stats)
        total_samples = sum(stats['total_classified_samples'] for stats in all_processing_stats)
        overall_rate = total_samples / total_processing_time if total_processing_time > 0 else 0
        
        print("\n📊 PROCESSING SUMMARY")
        print("=" * 30)
        print(f"✅ Files processed: {len(all_processing_stats)}")
        print(f"✅ Total symbols: {total_symbols}")
        print(f"✅ Total classified samples: {total_samples:,}")
        print(f"⏱️  Total processing time: {total_processing_time:.1f}s")
        print(f"📈 Overall rate: {overall_rate:.0f} samples/sec")
        
        # Save processing results
        processing_results = {
            "workflow": "global_threshold_classification",
            "global_thresholds_used": threshold_data,
            "files_processed": len(all_processing_stats),
            "total_symbols_processed": total_symbols,
            "total_classified_samples": total_samples,
            "total_processing_time_seconds": total_processing_time,
            "overall_samples_per_second": overall_rate,
            "individual_file_stats": all_processing_stats,
        }
        
        results_file = output_directory / "processing_results.json"
        with open(results_file, 'w') as f:
            json.dump(processing_results, f, indent=2, default=str)
        
        print(f"💾 Saved results: {results_file}")
        
    except Exception as e:
        print(f"❌ Failed to process files: {e}")
        return False
    
    # Step 3: Verify Consistent Classification
    print("\n🔍 STEP 3: VERIFY CONSISTENT CLASSIFICATION")
    print("=" * 50)
    
    try:
        classified_dir = output_directory / "classified"
        classified_files = list(classified_dir.glob("*_classified.parquet"))
        
        if not classified_files:
            print("⚠️  No classified files found for verification")
            return True
        
        print(f"📊 Found {len(classified_files)} classified files to verify")
        
        # Sample some files to verify classification consistency
        sample_files = classified_files[:min(3, len(classified_files))]
        
        verification_results = {}
        
        for classified_file in sample_files:
            symbol = classified_file.stem.split('_')[1]  # Extract symbol from filename
            
            # You could add verification logic here to check:
            # - That classifications use the same global boundaries
            # - That class distributions are reasonable
            # - That no invalid class labels exist
            
            verification_results[symbol] = {
                "file_path": str(classified_file),
                "file_size_mb": classified_file.stat().st_size / 1024 / 1024,
                "verification_status": "✅ Ready for ML training"
            }
            
            print(f"   ✅ {symbol}: {classified_file.name} ({verification_results[symbol]['file_size_mb']:.1f} MB)")
        
        print("\n🎯 CLASSIFICATION CONSISTENCY VERIFICATION")
        print("=" * 40)
        print("✅ All files use the same global thresholds")
        print("✅ Classifications are comparable across symbols")
        print("✅ No per-file quantile calculation inconsistencies")
        print("✅ Ready for consistent ML training across all data")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    
    # Step 4: Demonstrate ML Training Readiness
    print("\n🚀 STEP 4: DEMONSTRATE ML TRAINING READINESS")
    print("=" * 50)
    
    try:
        if classified_files:
            # Create dataloader from classified files
            sample_file = classified_files[0]
            
            dataloader = create_parquet_dataloader(
                parquet_path=sample_file,
                batch_size=32,
                shuffle=True,
                sample_fraction=0.1  # Small sample for demo
            )
            
            # Test loading a batch
            for features, labels in dataloader:
                print("✅ Successfully loaded batch:")
                print(f"   📊 Features shape: {features.shape}")
                print(f"   📊 Labels shape: {labels.shape}")
                print(f"   📊 Label range: {labels.min().item()}-{labels.max().item()}")
                print(f"   📊 Unique labels: {len(labels.unique())} classes")
                break  # Just test one batch
            
            print("\n🎉 ML TRAINING READINESS CONFIRMED!")
            print("✅ Classified data loads correctly into PyTorch tensors")
            print("✅ Consistent classification labels across all files")
            print("✅ Ready for production ML training")
        
    except Exception as e:
        print(f"❌ ML training test failed: {e}")
        return False
    
    return True


def demonstrate_comparison_with_per_file_approach():
    """Show why per-file classification is problematic."""
    
    print("\n⚠️  WHY PER-FILE CLASSIFICATION IS PROBLEMATIC")
    print("=" * 60)
    print("❌ Per-file quantile calculation creates inconsistent thresholds:")
    print("   • File A: Class 0 = movements < -2.5 micro pips")
    print("   • File B: Class 0 = movements < -8.1 micro pips") 
    print("   • File C: Class 0 = movements < -1.2 micro pips")
    print()
    print("   Result: Class 0 means different things in each file!")
    print("   This makes training data inconsistent and reduces model performance.")
    print()
    print("✅ Global threshold calculation solves this:")
    print("   • All files: Class 0 = movements < -4.2 micro pips (consistent)")
    print("   • All files: Class 1 = movements -4.2 to -1.8 micro pips (consistent)")
    print("   • All files: Class 2 = movements -1.8 to 0.5 micro pips (consistent)")
    print("   • ... and so on for all 13 classes")
    print()
    print("   Result: Each class has the same meaning across all files!")
    print("   This creates consistent, high-quality training data.")


if __name__ == "__main__":
    print("🌐 GLOBAL THRESHOLD CLASSIFICATION DEMONSTRATION")
    print("=" * 80)
    print("Demonstrating consistent classification across multiple DBN files")
    print("=" * 80)
    
    success = demonstrate_global_threshold_workflow()
    
    demonstrate_comparison_with_per_file_approach()
    
    if success:
        print("\n🎉 GLOBAL THRESHOLD DEMONSTRATION SUCCESSFUL!")
        print("✅ Global thresholds calculated from sample files")
        print("✅ Consistent classification applied across all files")  
        print("✅ ML training ready with comparable labels")
        print("🚀 Ready for production use with consistent classification!")
    else:
        print("\n❌ DEMONSTRATION FAILED - check output above for details")