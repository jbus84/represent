#!/usr/bin/env python3
"""
Test script to process real DBN files using the symbol-split-merge architecture.
"""

from pathlib import Path

from represent import DatasetBuildConfig, build_datasets_from_dbn_files
from represent.configs import create_represent_config


def main():
    """Test processing real DBN files."""
    print("🧪 Testing DBN Processing with Symbol-Split-Merge Architecture")
    print("=" * 70)

    # Set up paths
    data_dir = Path("data")
    output_dir = Path("test_output") / "dbn_processing_test"

    # Find available DBN files
    dbn_files = list(data_dir.glob("*.dbn*"))

    if not dbn_files:
        print("❌ No DBN files found in data/ directory")
        return

    print(f"📁 Found {len(dbn_files)} DBN files:")
    for file in dbn_files:
        file_size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   📄 {file.name} ({file_size_mb:.1f} MB)")

    # Create configuration - use smaller parameters for testing
    configs = create_represent_config(
        currency="AUDUSD",  # Assuming these are AUDUSD files
        features=["volume"],
        samples=25000,      # Minimum allowed
        lookback_rows=1000, # Smaller for testing
        lookforward_input=1000,
        lookforward_offset=100,
        jump_size=50       # Smaller jump for more data points
    )
    dataset_cfg, threshold_cfg, processor_cfg = configs

    # Create dataset configuration
    dataset_config = DatasetBuildConfig(
        currency="AUDUSD",
        min_symbol_samples=5000,  # Lower threshold for testing
        force_uniform=True,
        keep_intermediate=True    # Keep intermediate files for inspection
    )

    print("\n⚙️  Configuration:")
    print(f"   💱 Currency: {dataset_cfg.currency}")
    print(f"   📊 Features: {processor_cfg.features}")
    print(f"   📏 Samples: {processor_cfg.samples:,}")
    print(f"   📈 Lookback rows: {dataset_cfg.lookback_rows:,}")
    print(f"   📉 Lookforward input: {dataset_cfg.lookforward_input:,}")
    print(f"   ⏭️  Lookforward offset: {dataset_cfg.lookforward_offset:,}")
    print(f"   🦘 Jump size: {threshold_cfg.jump_size}")
    print(f"   🎯 Min samples per symbol: {dataset_config.min_symbol_samples:,}")

    try:
        # Process the DBN files
        print("\n🚀 Starting DBN Processing...")

        results = build_datasets_from_dbn_files(
            config=dataset_cfg,
            dbn_files=dbn_files,
            output_dir=output_dir,
            dataset_config=dataset_config,
            verbose=True
        )

        print("\n✅ Processing Complete!")
        print("📊 Results Summary:")
        print(f"   📁 Input files: {len(results['input_files'])}")
        print(f"   📊 Phase 1 - Symbols discovered: {results['phase_1_stats']['symbols_discovered']}")
        print(f"   📊 Phase 1 - Intermediate files: {results['phase_1_stats']['intermediate_files_created']}")
        print(f"   📊 Phase 1 - Processing time: {results['phase_1_stats']['split_time_seconds']:.1f}s")
        print(f"   📊 Phase 2 - Datasets created: {results['phase_2_stats']['datasets_created']}")
        print(f"   📊 Phase 2 - Total samples: {results['phase_2_stats']['total_samples']:,}")
        print(f"   📊 Phase 2 - Processing time: {results['phase_2_stats']['merge_time_seconds']:.1f}s")
        print(f"   ⏱️  Total processing time: {results['total_processing_time_seconds']:.1f}s")
        print(f"   📈 Processing rate: {results['samples_per_second']:.0f} samples/sec")

        if results['dataset_files']:
            print("\n📁 Dataset Files Created:")
            for _symbol, info in results['dataset_files'].items():
                print(f"   📄 {Path(info['file_path']).name}")
                print(f"      🔢 Samples: {info['samples']:,}")
                print(f"      📦 Size: {info['file_size_mb']:.1f} MB")
                print(f"      📁 Source files: {info['source_files']}")
        else:
            print("\n⚠️  No dataset files were created (possibly due to insufficient samples)")

        print(f"\n📁 Output directory: {output_dir.absolute()}")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
