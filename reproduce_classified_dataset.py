#!/usr/bin/env python3
"""
Reproduce Classified Dataset Script

This script reproduces the AUDUSD_M6AM4_dataset.parquet using the latest
symbol-split-merge architecture to demonstrate the new workflow.
"""

import sys
from pathlib import Path

# Add represent to path
sys.path.insert(0, str(Path(__file__).parent))

from represent import DatasetBuildConfig, build_datasets_from_dbn_files, create_represent_config


def reproduce_classified_dataset():
    """Reproduce the AUDUSD classified dataset using symbol-split-merge approach."""

    print("🔄 REPRODUCING CLASSIFIED DATASET")
    print("=" * 50)
    print("Target: AUDUSD_M6AM4_dataset.parquet")
    print("Workflow: Symbol-Split-Merge Architecture")
    print()

    # Create represent configuration
    print("📝 Creating configuration...")
    config = create_represent_config(
        currency="AUDUSD",
        features=["volume"],  # Using single feature as in original
        lookback_rows=5000,
        lookforward_input=5000,
        lookforward_offset=500,
        jump_size=100,
        nbins=13,
    )
    dataset_cfg, threshold_cfg, processor_cfg = config
    print(f"   Currency: {dataset_cfg.currency}")
    print(f"   Features: {processor_cfg.features}")
    print(f"   Classification bins: {threshold_cfg.nbins}")
    print()

    # Create dataset building configuration
    dataset_config = DatasetBuildConfig(
        currency="AUDUSD",
        min_symbol_samples=60500,  # samples + lookback + lookforward + offset
        force_uniform=True,  # Guarantee uniform class distribution
        keep_intermediate=False,  # Clean up intermediate split files
    )
    print("⚙️  Dataset build configuration:")
    print(f"   Min symbol samples: {dataset_config.min_symbol_samples:,}")
    print(f"   Force uniform distribution: {dataset_config.force_uniform}")
    print()

    # Select first 10 DBN files for reproduction
    dbn_files = [
        "/Users/danielfisher/data/databento/AUDUSD-micro/glbx-mdp3-20240403.mbp-10.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/glbx-mdp3-20240404.mbp-10.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/glbx-mdp3-20240405.mbp-10.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/glbx-mdp3-20240407.mbp-10.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/glbx-mdp3-20240408.mbp-10.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/glbx-mdp3-20240409.mbp-10.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/glbx-mdp3-20240410.mbp-10.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/glbx-mdp3-20240411.mbp-10.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/glbx-mdp3-20240412.mbp-10.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/glbx-mdp3-20240414.mbp-10.dbn.zst",
    ]

    # Verify files exist
    print("📂 Verifying DBN files...")
    existing_files = []
    for dbn_file in dbn_files:
        if Path(dbn_file).exists():
            existing_files.append(dbn_file)
            print(f"   ✅ {Path(dbn_file).name}")
        else:
            print(f"   ❌ {Path(dbn_file).name} (not found)")

    if not existing_files:
        print("\n❌ No DBN files found! Please check the data directory.")
        return False

    print(f"\n📊 Processing {len(existing_files)} DBN files...")
    print()

    # Run symbol-split-merge dataset building
    output_dir = "/Users/danielfisher/data/databento/AUDUSD_classified_datasets_new/"
    Path(output_dir).mkdir(exist_ok=True)

    try:
        print("🚀 Starting Symbol-Split-Merge dataset building...")
        results = build_datasets_from_dbn_files(
            config=config,
            dbn_files=existing_files,
            output_dir=output_dir,
            dataset_config=dataset_config,
            verbose=True,
        )

        print("\n✅ DATASET REPRODUCTION COMPLETE!")
        print("=" * 50)

        # Display results
        if "phase_1_stats" in results:
            print(f"Phase 1 (Split): {results['phase_1_stats']['files_processed']} files processed")
            print(
                f"                 {results['phase_1_stats']['total_symbols_found']} total symbols found"
            )

        if "phase_2_stats" in results:
            print(
                f"Phase 2 (Merge): {results['phase_2_stats']['datasets_created']} symbol datasets created"
            )
            print(f"                 {results['phase_2_stats']['total_samples']:,} total samples")

        print(f"\n📁 Output directory: {output_dir}")

        # List created datasets
        output_path = Path(output_dir)
        if output_path.exists():
            datasets = list(output_path.glob("*.parquet"))
            print(f"\n📋 Created datasets ({len(datasets)}):")
            for dataset in sorted(datasets):
                size_mb = dataset.stat().st_size / (1024 * 1024)
                print(f"   {dataset.name} ({size_mb:.1f} MB)")

        print(
            "\n🎉 SUCCESS: Classified dataset reproduced using new symbol-split-merge architecture!"
        )
        return True

    except Exception as e:
        print(f"\n❌ REPRODUCTION FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = reproduce_classified_dataset()
    if success:
        print("\n📈 The reproduced dataset can now be used for ML training!")
        print("   Use polars.read_parquet() to load the datasets in your ML pipeline.")
    else:
        print("\n💡 Check the error messages above and verify:")
        print("   1. DBN files exist in the specified directory")
        print("   2. Sufficient disk space for output datasets")
        print("   3. All dependencies are properly installed")
