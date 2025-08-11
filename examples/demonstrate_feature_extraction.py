#!/usr/bin/env python
"""
Demonstrate Feature Extraction and Visualization

This script provides a fast demonstration of generating and visualizing
all available features using existing processed datasets.
"""
# Add represent package to path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from represent import (
    create_represent_config,
    process_market_data,
)


def demonstrate_feature_extraction():
    """
    Demonstrates on-the-fly feature generation from existing processed datasets.
    """
    print("\n--- ON-THE-FLY Feature Extraction Demonstration ---")

    # Use existing processed datasets from examples
    output_dir = Path("examples/feature_demonstration_output")
    output_dir.mkdir(exist_ok=True)

    # Look for existing dataset files
    demo_datasets = list(Path("examples").glob("**/AUDUSD_M6AM4_dataset_cleaned.parquet"))

    if not demo_datasets:
        # Fallback to any cleaned dataset
        demo_datasets = list(Path("examples").glob("**/*_cleaned.parquet"))

    if not demo_datasets:
        print("❌ No processed datasets found in examples/ directory")
        print("   Please run: make process-dbn-demo first")
        return

    dataset_file = demo_datasets[0]
    print(f"📁 Using existing dataset: {dataset_file}")

    # Load the cleaned dataset
    df = pl.read_parquet(dataset_file)
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🔍 Available market data columns: {len([col for col in df.columns if col.startswith(('bid_', 'ask_'))])}")

    # Define features to demonstrate
    features = ['volume', 'variance', 'trade_counts']

    print("\n🎯 Demonstrating ON-THE-FLY feature generation:")
    print(f"   Features: {features}")
    print("   Key advantage: No storage overhead, generated when needed")

    # Take sufficient samples for feature generation (minimum 500 required)
    num_samples = min(1000, len(df))  # Use 1000 samples for fast demo
    sample_data = df.head(num_samples)
    print(f"📊 Using {num_samples} samples for feature generation")

    # Create config for all features together (proper approach)
    print(f"\n📊 Generating all features together: {features}")

    try:
        feature_config = create_represent_config(
            currency="AUDUSD",
            samples=25000,             # Minimum allowed by validation
            lookback_rows=200,         # Minimal for speed
            lookforward_input=200,     # Minimal for speed
            lookforward_offset=50,     # Small offset
            jump_size=50,              # Optimal jump size
            ticks_per_bin=50           # Set to 50 to get 500 time bins (25000/50=500)
        )

        # Generate multi-feature tensor on-the-fly (pass features as separate parameter)
        multi_feature_tensor = process_market_data(sample_data, feature_config, features=features)

        if multi_feature_tensor is not None and multi_feature_tensor.size > 0:
            print(f"   ✅ Success! Generated multi-feature tensor: {multi_feature_tensor.shape}")

            feature_arrays = {}

            # Extract individual features from the multi-feature tensor
            for i, feature in enumerate(features):
                print(f"\n📊 Extracting '{feature}' feature from multi-tensor...")

                # Extract feature from multi-feature tensor (correct approach from main branch)
                if len(multi_feature_tensor.shape) == 3:
                    feature_array = multi_feature_tensor[i]  # Extract feature i from tensor
                elif len(multi_feature_tensor.shape) == 2 and len(features) == 1:
                    feature_array = multi_feature_tensor  # Single feature case
                else:
                    print(f"   ❌ Unexpected tensor shape: {multi_feature_tensor.shape}")
                    continue

                print(f"   ✅ Extracted {feature} tensor")
                print(f"      📏 Shape: {feature_array.shape}")
                print(f"      📊 Value range: [{feature_array.min():.6f}, {feature_array.max():.6f}]")

                # Add detailed signed data verification like the main branch examples
                neg_count = np.sum(feature_array < 0)
                pos_count = np.sum(feature_array > 0)
                zero_count = np.sum(feature_array == 0)
                print(f"      📊 {feature} signed distribution: neg={neg_count:,}, pos={pos_count:,}, zero={zero_count:,}")
                print(f"      🎯 Non-zero elements: {np.count_nonzero(feature_array):,}/{feature_array.size:,} ({100*np.count_nonzero(feature_array)/feature_array.size:.1f}%)")

                # Create high-quality visualization with RdBu colormap
                fig, ax = plt.subplots(figsize=(14, 8))
                im = ax.imshow(feature_array, cmap='RdBu', aspect='auto', interpolation='bilinear')
                ax.set_title(f"Feature: '{feature.upper()}' - Generated On-The-Fly", fontsize=18, fontweight='bold')
                ax.set_xlabel(f"Time Bins ({feature_array.shape[1]} total)", fontsize=14)
                ax.set_ylabel("Price Levels (402 total: 200 Ask + 2 Mid + 200 Bid)", fontsize=14)

                # Add grid for better visualization
                ax.set_xticks(range(0, feature_array.shape[1], max(1, feature_array.shape[1]//5)))
                ax.set_yticks(range(0, 402, 100))
                ax.grid(True, alpha=0.3, linestyle='--')

                # Enhanced colorbar with RdBu theme
                cbar = fig.colorbar(im, shrink=0.8)
                cbar.set_label(f"Normalized {feature.title()} Value", fontsize=12, fontweight='bold')

                plt.tight_layout()

                # Save individual visualization
                plot_path = output_dir / f"{feature}_feature_plot.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)

                # Store for RGB combination
                feature_arrays[feature] = feature_array

                print(f"      📊 Saved visualization: {plot_path}")

        else:
            print("   ❌ Failed to generate multi-feature tensor (insufficient data or processing error)")

    except Exception as e:
        print(f"   ❌ Multi-feature generation error: {str(e)}")
        return

    # Create RGB combination visualization if we have multiple features
    if 'feature_arrays' in locals() and len(feature_arrays) >= 3:
        print("\n🌈 Creating RGB Feature Combination...")

        try:
            # Get the three features for RGB channels
            feature_list = list(feature_arrays.keys())
            red_feature = feature_arrays[feature_list[0]]    # volume -> Red
            green_feature = feature_arrays[feature_list[1]]  # variance -> Green
            blue_feature = feature_arrays[feature_list[2]]   # trade_counts -> Blue

            # Normalize each feature to [0, 1] range for RGB
            def normalize_for_rgb(arr):
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max > arr_min:
                    return (arr - arr_min) / (arr_max - arr_min)
                else:
                    return np.zeros_like(arr)

            red_norm = normalize_for_rgb(red_feature)
            green_norm = normalize_for_rgb(green_feature)
            blue_norm = normalize_for_rgb(blue_feature)

            # Create RGB array
            rgb_array = np.stack([red_norm, green_norm, blue_norm], axis=-1)

            # Create RGB combination plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # RGB visualization
            ax1.imshow(rgb_array, aspect='auto', interpolation='bilinear')
            ax1.set_title("RGB Feature Combination\n(Red=Volume, Green=Variance, Blue=Trade Counts)",
                         fontsize=16, fontweight='bold')
            ax1.set_xlabel(f"Time Bins ({rgb_array.shape[1]} total)", fontsize=14)
            ax1.set_ylabel("Price Levels (402 total: 200 Ask + 2 Mid + 200 Bid)", fontsize=14)
            ax1.set_xticks(range(0, rgb_array.shape[1], max(1, rgb_array.shape[1]//5)))
            ax1.set_yticks(range(0, 402, 100))
            ax1.grid(True, alpha=0.3, linestyle='--')

            # Individual channel intensities
            channel_data = np.stack([red_norm.mean(axis=0), green_norm.mean(axis=0), blue_norm.mean(axis=0)])
            im2 = ax2.imshow(channel_data, cmap='RdBu', aspect='auto', interpolation='bilinear')
            ax2.set_title("Average Channel Intensities Over Time", fontsize=16, fontweight='bold')
            ax2.set_xlabel(f"Time Bins ({rgb_array.shape[1]} total)", fontsize=14)
            ax2.set_ylabel("RGB Channels", fontsize=14)
            ax2.set_yticks([0, 1, 2])
            ax2.set_yticklabels(['Volume (Red)', 'Variance (Green)', 'Trade Counts (Blue)'])

            # Add colorbar
            cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label("Normalized Intensity", fontsize=12, fontweight='bold')

            plt.tight_layout()

            # Save RGB combination
            rgb_path = output_dir / "rgb_feature_combination.png"
            fig.savefig(rgb_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"   ✅ RGB combination created with shape: {rgb_array.shape}")
            print(f"   📊 Saved RGB visualization: {rgb_path}")
            print(f"   🎨 Channels: Red={feature_list[0]}, Green={feature_list[1]}, Blue={feature_list[2]}")

        except Exception as e:
            print(f"   ❌ RGB combination failed: {str(e)[:100]}")

    elif 'feature_arrays' in locals():
        print(f"\n⚠️  RGB combination requires 3 features, only {len(feature_arrays)} generated successfully")

    print("\n🎉 ON-THE-FLY Feature Generation Demo Complete!")
    print(f"   📁 Visualizations saved to: {output_dir}")
    print("   💡 Key Benefits of On-The-Fly Generation:")
    print("      • 70%+ storage space savings (no pre-stored tensors)")
    print("      • Fresh features generated for each training batch")
    print("      • Flexible feature combinations at runtime")
    print("      • Memory efficient for large datasets")
    print("      • RGB combinations possible for multi-channel analysis")


def main():
    """Main function to run the feature extraction demonstration."""
    demonstrate_feature_extraction()


if __name__ == "__main__":
    main()
