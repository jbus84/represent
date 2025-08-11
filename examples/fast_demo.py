#!/usr/bin/env python3
"""
Fast Demo - Minimal Processing Time
===================================

A super-fast demonstration that processes real DBN data in seconds to show:
1. Symbol-split-merge architecture
2. Uniform classification
3. Feature generation
4. ML-ready dataset creation

Uses minimal parameters for maximum speed while still demonstrating all core functionality.
"""

import sys
import time
from pathlib import Path

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent import (
    DatasetBuildConfig,
    build_datasets_from_dbn_files,
    create_represent_config,
)


def main():
    """
    Fast demo using real DBN data with minimal processing time.
    """
    print("⚡ FAST DEMO - Symbol-Split-Merge Architecture")
    print("=" * 60)
    print("Demonstrating:")
    print("  • Real DBN file processing")
    print("  • Symbol-split-merge workflow")
    print("  • Uniform classification")
    print("  • Feature generation")
    print("  • ML-ready datasets")
    print()

    # Check for DBN files
    data_dir = Path("data")
    dbn_files = list(data_dir.glob("*.dbn*"))

    if not dbn_files:
        print("❌ No DBN files found in data/ directory")
        print("   Please place .dbn or .dbn.zst files in the data/ directory")
        return

    print(f"📁 Found {len(dbn_files)} DBN files: {[f.name for f in dbn_files]}")
    print(f"📄 Using first file for fast demo: {dbn_files[0].name}")
    print()

    # Fast configuration - minimal parameters for speed
    print("⚙️  Configuration (optimized for speed):")
    config = create_represent_config(
        currency="AUDUSD",
        features=['volume'],        # Single feature for speed
        samples=25000,              # Minimum allowed samples
        lookback_rows=200,          # Minimal lookback
        lookforward_input=200,      # Minimal lookforward
        lookforward_offset=50,      # Small offset
        jump_size=25               # Larger jump for faster processing
    )

    dataset_config = DatasetBuildConfig(
        currency="AUDUSD",
        features=['volume'],
        min_symbol_samples=1000,    # Low threshold for demo
        force_uniform=True,
        keep_intermediate=True      # Keep files for inspection
    )

    print(f"   💱 Currency: {config.currency}")
    print(f"   📊 Features: {config.features}")
    print(f"   🎯 Samples: {config.samples:,}")
    print(f"   🔄 Lookback/Forward: {config.lookback_rows}/{config.lookforward_input}")
    print(f"   ⚡ Jump size: {config.jump_size} (faster processing)")
    print()

    # Process single file for speed
    print("🔄 Processing with Symbol-Split-Merge...")
    start_time = time.perf_counter()

    results = build_datasets_from_dbn_files(
        config=config,
        dbn_files=[dbn_files[0]],   # Use only first file for speed
        output_dir="examples/fast_demo_output",
        dataset_config=dataset_config,
        verbose=True
    )

    processing_time = time.perf_counter() - start_time

    print()
    print("🎉 FAST DEMO RESULTS:")
    print("=" * 40)
    print(f"⏱️  Processing time: {processing_time:.1f}s")
    print(f"📊 Datasets created: {results['phase_2_stats']['datasets_created']}")
    print(f"📈 Total ML samples: {results['phase_2_stats']['total_samples']:,}")
    print(f"🚀 Processing rate: {results['samples_per_second']:.0f} samples/sec")
    print("📁 Output directory: examples/fast_demo_output/")

    # Show datasets created
    output_dir = Path("examples/fast_demo_output")
    dataset_files = list(output_dir.glob("*_dataset.parquet"))

    print()
    print("📋 Generated Datasets:")
    for dataset_file in dataset_files:
        # Quick analysis of generated dataset
        try:
            import polars as pl
            df = pl.read_parquet(str(dataset_file))

            print(f"   📄 {dataset_file.name}")
            print(f"      📊 Samples: {len(df):,}")
            print(f"      💾 Size: {dataset_file.stat().st_size / 1024 / 1024:.1f} MB")

            # Check classification distribution
            if 'classification_label' in df.columns:
                class_counts = df['classification_label'].value_counts().sort('classification_label')
                total_samples = len(df)
                uniform_check = []

                for row in class_counts.iter_rows():
                    label, count = row
                    percentage = (count / total_samples) * 100
                    uniform_check.append(percentage)

                # Check uniformity (should be close to 7.69% each for 13 bins)
                import numpy as np
                std_dev = np.std(uniform_check)
                uniformity_ratio = std_dev / np.mean(uniform_check)
                uniform_achieved = uniformity_ratio < 0.15

                print(f"      ⚖️  Uniform distribution: {'✅ Yes' if uniform_achieved else '❌ No'} (ratio: {uniformity_ratio:.3f})")

            # Check features (now generated on-demand, not stored)
            feature_cols = [col for col in df.columns if col.endswith('_representation')]
            if feature_cols:
                print(f"      🎨 Features: {len(feature_cols)} ({', '.join([col.replace('_representation', '') for col in feature_cols])})")
            else:
                print("      🎨 Features: Generated on-demand (no storage overhead)")

                # Create feature visualization
                try:
                    import matplotlib.pyplot as plt
                    import numpy as np

                    for feature_col in feature_cols:
                        feature_name = feature_col.replace('_representation', '')
                        feature_data = df[feature_col][0]  # First sample

                        if feature_data is not None and len(feature_data) > 0:
                            feature_array = np.array(feature_data)

                            # Create visualization
                            fig, ax = plt.subplots(figsize=(12, 6))
                            im = ax.imshow(feature_array.reshape(-1, feature_array.shape[0] if len(feature_array.shape) == 1 else feature_array.shape[1]),
                                          cmap='viridis', aspect='auto')
                            ax.set_title(f'{dataset_file.stem} - {feature_name.title()} Feature Representation', fontsize=14, fontweight='bold')
                            ax.set_xlabel('Time Bins')
                            ax.set_ylabel('Price Levels (Ask=Top, Bid=Bottom)')

                            # Add colorbar
                            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                            cbar.set_label(f'{feature_name.title()} Intensity')

                            plt.tight_layout()

                            # Save plot
                            plot_path = output_dir / f"{dataset_file.stem}_{feature_name}_visualization.png"
                            plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
                            plt.close(fig)

                            print(f"      📊 Visualization saved: {plot_path.name}")

                except Exception as e:
                    print(f"      ⚠️  Visualization failed: {e}")

        except Exception as e:
            print(f"   📄 {dataset_file.name} (analysis failed: {e})")

    print()
    print("🚀 SUCCESS! Fast demo completed in seconds")
    print("💡 Next steps:")
    print("   • Inspect datasets in examples/fast_demo_output/ directory")
    print("   • Try: make process-dbn for balanced processing")
    print("   • Try: make process-dbn-production for high-quality datasets")
    print("   • Use datasets for ML training (see README.md)")


if __name__ == "__main__":
    main()
