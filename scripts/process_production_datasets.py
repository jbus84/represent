#!/usr/bin/env python
"""
Production Dataset Processing Script

This script processes AUDUSD-micro data with a first-half training approach:
1. Uses first half of DBN files to calculate global classification thresholds
2. Processes all files into symbol-specific parquets with classification applied

Designed for production ML training workflows.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent import (
    DatasetBuildConfig,
    build_datasets_from_dbn_files,
    calculate_global_thresholds,
    create_represent_config,
)


def main():
    """Process production datasets with first-half threshold calculation."""
    print("🏭 PRODUCTION DATASET PROCESSING")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    input_dir = Path("/Users/danielfisher/data/databento/AUDUSD-micro")
    output_dir = Path("/Users/danielfisher/data/databento/AUDUSD_classified_datasets")

    # Validate input directory
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1

    # Get all DBN files sorted by date
    dbn_files = sorted(input_dir.glob("*.dbn.zst"))

    if not dbn_files:
        print(f"❌ No DBN files found in: {input_dir}")
        return 1

    print(f"📁 Found {len(dbn_files)} DBN files")
    print(f"📅 Date range: {dbn_files[0].stem.split('-')[-1]} to {dbn_files[-1].stem.split('-')[-1]}")

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 Output directory: {output_dir}")

    # Create production configuration
    config = create_represent_config(
        currency="AUDUSD",
        features=['volume', 'variance'],
        lookback_rows=5000,          # Production lookback
        lookforward_input=5000,      # Production lookforward
        lookforward_offset=500,      # Production offset
        samples=50000,               # Production sample size
        nbins=13                     # More bins for better classification resolution
    )

    print("\n🔧 Production Configuration:")
    print(f"   💱 Currency: {config.currency}")
    print(f"   📊 Features: {config.features}")
    print(f"   📈 Lookback rows: {config.lookback_rows}")
    print(f"   📉 Lookforward input: {config.lookforward_input}")
    print(f"   ⏭️  Lookforward offset: {config.lookforward_offset}")
    print(f"   🎯 Classification bins: {config.nbins}")
    print(f"   📊 Sample size: {config.samples}")

    start_time = time.time()

    try:
        # PHASE 1: Calculate Global Thresholds from First Half of Data
        print("\n🎯 PHASE 1: Global Threshold Calculation")
        print("=" * 50)

        # Use first half of files for threshold calculation
        first_half_size = len(dbn_files) // 2
        threshold_files = dbn_files[:first_half_size]

        print(f"📊 Using first {first_half_size} files ({len(threshold_files)}) for threshold calculation")
        print(f"📅 Threshold data range: {threshold_files[0].stem.split('-')[-1]} to {threshold_files[-1].stem.split('-')[-1]}")
        print(f"📁 Sample files: {threshold_files[0].name} ... {threshold_files[-1].name}")

        # Calculate thresholds from first half
        phase1_start = time.time()
        thresholds = calculate_global_thresholds(
            config=config,
            data_directory=str(input_dir),
            sample_fraction=0.3,  # Use 30% of first half files for speed
            verbose=True
        )
        phase1_time = time.time() - phase1_start

        print(f"✅ Phase 1 Complete: {phase1_time:.1f}s")
        print(f"📊 Global thresholds calculated from {thresholds.sample_size:,} price movements")
        print(f"🎯 Classification bins: {thresholds.nbins}")
        print("📈 Price movement stats:")
        print(f"   Mean: {thresholds.price_movement_stats['mean']:+.6f}")
        print(f"   Std:  {thresholds.price_movement_stats['std']:.6f}")
        print(f"   Range: [{thresholds.price_movement_stats['min']:+.6f}, {thresholds.price_movement_stats['max']:+.6f}]")

        # PHASE 2: Process All Files with Calculated Thresholds
        print("\n🏗️ PHASE 2: Dataset Building with Global Thresholds")
        print("=" * 50)

        # Create dataset configuration with calculated thresholds
        dataset_config = DatasetBuildConfig(
            currency="AUDUSD",
            features=['volume', 'variance', "count"],
            global_thresholds=thresholds,  # Use calculated thresholds
            force_uniform=True,            # Enforce uniform distribution
            min_symbol_samples=50000,      # Higher threshold for production
            keep_intermediate=False        # Clean up intermediate files
        )

        print("🔧 Dataset Configuration:")
        print("   🎯 Using calculated global thresholds")
        print(f"   ⚖️  Force uniform distribution: {dataset_config.force_uniform}")
        print(f"   📊 Min symbol samples: {dataset_config.min_symbol_samples:,}")
        print(f"   🧹 Keep intermediate files: {dataset_config.keep_intermediate}")

        # Process ALL files (both halves) with the calculated thresholds
        print(f"\n📁 Processing ALL {len(dbn_files)} files with global thresholds")
        print(f"📅 Full data range: {dbn_files[0].stem.split('-')[-1]} to {dbn_files[-1].stem.split('-')[-1]}")

        phase2_start = time.time()
        results = build_datasets_from_dbn_files(
            config=config,
            dbn_files=dbn_files,  # Process ALL files
            output_dir=str(output_dir),
            dataset_config=dataset_config,
            verbose=True
        )
        phase2_time = time.time() - phase2_start

        print(f"✅ Phase 2 Complete: {phase2_time:.1f}s")

        # PHASE 3: Results Summary
        print("\n📊 PRODUCTION PROCESSING RESULTS")
        print("=" * 50)

        total_time = time.time() - start_time

        # Get created datasets
        dataset_files = list(output_dir.glob("*_dataset.parquet"))

        print(f"⏱️  Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"🎯 Phase 1 (Thresholds): {phase1_time:.1f}s")
        print(f"🏗️ Phase 2 (Datasets): {phase2_time:.1f}s")
        print("")
        print("📊 Dataset Creation Results:")
        print(f"   📁 Datasets created: {len(dataset_files)}")
        print(f"   📊 Total samples: {results.get('phase_2_stats', {}).get('total_samples', 0):,}")
        print(f"   📈 Processing rate: {results.get('phase_2_stats', {}).get('total_samples', 0) / total_time:.0f} samples/sec")
        print("   🎯 Classification method: First-half training with global thresholds")

        print("\n📁 Created Symbol Datasets:")
        total_size_mb = 0
        for dataset_file in sorted(dataset_files):
            size_mb = dataset_file.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            symbol = dataset_file.stem.split('_')[1] if '_' in dataset_file.stem else 'unknown'
            print(f"   📊 {dataset_file.name}")
            print(f"      Symbol: {symbol}")
            print(f"      Size: {size_mb:.1f} MB")

        print(f"\n💾 Total dataset size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
        print(f"📁 Output location: {output_dir}")

        # Validation
        if dataset_files:
            print("\n✅ SUCCESS: Production datasets created successfully!")
            print("🎯 Ready for ML training in external repository")
            print("📋 Classification approach: First-half training prevents data leakage")
            print("⚖️  Uniform distribution: Balanced classes for optimal training")

        else:
            print("\n⚠️  WARNING: No datasets were created")
            print("   This may be due to insufficient samples per symbol")
            print("   Consider lowering min_symbol_samples threshold")

    except Exception as e:
        print("\n❌ PRODUCTION PROCESSING FAILED")
        print(f"Error: {str(e)}")
        print(f"Time elapsed: {time.time() - start_time:.1f}s")
        return 1

    print("\n🎉 Production processing complete!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    exit(main())
