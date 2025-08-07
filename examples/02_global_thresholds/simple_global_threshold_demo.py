#!/usr/bin/env python3
"""
Simple Global Threshold Demo

A focused demonstration of the global threshold approach using the AUDUSD
market data at /Users/danielfisher/data/databento/AUDUSD-micro

This example shows:
1. How to calculate global thresholds from sample files
2. How to apply them consistently to other files
3. Why this approach is superior to per-file quantiles
"""

import sys
from pathlib import Path
import time
import json

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from represent import (
    calculate_global_thresholds,
    process_dbn_to_classified_parquets
)


def simple_global_threshold_demo():
    """Simple demonstration of global threshold workflow."""
    
    print("🌐 SIMPLE GLOBAL THRESHOLD DEMONSTRATION")
    print("=" * 60)
    
    # Data source
    data_directory = Path("/Users/danielfisher/data/databento/AUDUSD-micro")
    output_directory = Path("/Users/danielfisher/repositories/represent/examples/classification_analysis/outputs/simple_demo")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Data source: {data_directory}")
    print(f"📁 Output: {output_directory}")
    
    if not data_directory.exists():
        print(f"❌ Data directory not found: {data_directory}")
        print("   Please ensure AUDUSD market data is available")
        return False
    
    # Find DBN files
    dbn_files = sorted([f for f in data_directory.glob("*.dbn*") if f.is_file()])
    
    if not dbn_files:
        print("❌ No DBN files found")
        return False
    
    print(f"📊 Found {len(dbn_files)} DBN files")
    print("🎯 Will use first 10 files for threshold calculation")
    
    # Step 1: Calculate Global Thresholds
    print("\\n🎯 STEP 1: Calculate Global Thresholds")
    print("-" * 40)
    
    try:
        start_time = time.perf_counter()
        
        # Use first 10 files (adjust sample fraction accordingly)
        sample_fraction = min(10 / len(dbn_files), 1.0)
        
        global_thresholds = calculate_global_thresholds(
            data_directory=data_directory,
            currency="AUDUSD",
            nbins=13,
            sample_fraction=sample_fraction,
            max_samples_per_file=5000,  # Smaller for quick demo
            verbose=True
        )
        
        threshold_time = time.perf_counter() - start_time
        
        print(f"\\n✅ Thresholds calculated in {threshold_time:.1f}s")
        print(f"📊 Based on {global_thresholds.sample_size:,} price movements")
        print(f"📁 From {global_thresholds.files_analyzed} files")
        
        # Show some threshold boundaries
        boundaries = global_thresholds.quantile_boundaries
        print("\\n🎯 Sample Global Thresholds:")
        for i in range(min(5, len(boundaries)-1)):
            print(f"   Class {i}: movements ≤ {boundaries[i]:.2f} micro pips")
        print("   ... (13 classes total)")
        
        # Save for reuse
        threshold_data = {
            "currency": "AUDUSD",
            "nbins": global_thresholds.nbins,
            "boundaries": boundaries.tolist(),
            "sample_size": global_thresholds.sample_size,
            "files_analyzed": global_thresholds.files_analyzed,
        }
        
        threshold_file = output_directory / "simple_thresholds.json"
        with open(threshold_file, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        
        print(f"💾 Saved: {threshold_file}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Step 2: Apply to New Files
    print("\\n🔄 STEP 2: Apply to New Files")
    print("-" * 40)
    
    try:
        # Use files NOT in threshold calculation
        start_idx = int(len(dbn_files) * sample_fraction)
        demo_files = dbn_files[start_idx:start_idx+2]  # Process 2 files
        
        if not demo_files:
            demo_files = dbn_files[-2:]  # Fallback to last 2 files
        
        print(f"🔄 Processing {len(demo_files)} files with global thresholds...")
        
        total_samples = 0
        total_symbols = 0
        
        for i, dbn_file in enumerate(demo_files):
            print(f"\\n   File {i+1}: {dbn_file.name}")
            
            results = process_dbn_to_classified_parquets(
                dbn_path=dbn_file,
                output_dir=output_directory / "classified",
                currency="AUDUSD",
                global_thresholds=global_thresholds,  # 🎯 Global thresholds!
                features=["volume"],
                min_symbol_samples=500,
                verbose=False
            )
            
            total_samples += results['total_classified_samples']
            total_symbols += results['symbols_processed']
            
            print(f"      ✅ {results['symbols_processed']} symbols, {results['total_classified_samples']:,} samples")
        
        print("\\n✅ Processing complete!")
        print(f"   📊 Total: {total_symbols} symbols, {total_samples:,} samples")
        print("   🎯 All used IDENTICAL thresholds for consistency")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False
    
    # Step 3: Show the Benefit
    print("\\n🎯 THE KEY BENEFIT")
    print("-" * 40)
    print("✅ GLOBAL APPROACH (what we just did):")
    print("   • Same price movement = Same label across ALL files")
    print("   • -5.0 μpip movement always gets the same classification")
    print("   • Consistent, comparable training data")
    print("\\n❌ PER-FILE APPROACH (the problem we solved):")
    print("   • Same price movement = Different labels in different files")
    print("   • -5.0 μpip movement could be Class 0, 1, or 2 depending on file")
    print("   • Inconsistent, incomparable training data")
    print("\\n🚀 Result: Better ML model performance with consistent data!")
    
    return True


if __name__ == "__main__":
    success = simple_global_threshold_demo()
    
    if success:
        print("\\n🎉 SIMPLE DEMONSTRATION SUCCESSFUL!")
        print("🚀 Ready to use global thresholds for consistent classification!")
    else:
        print("\\n❌ DEMONSTRATION FAILED")
        print("Please check the error messages above")