#!/usr/bin/env python
"""
Quick Classification Check

Quick analysis of classification results to identify issues.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def quick_check_dataset(dataset_path):
    """Quick check of a single dataset."""
    print(f"\n📊 QUICK CHECK: {dataset_path.name}")
    print("=" * 50)

    # Load dataset
    df = pl.read_parquet(dataset_path)
    print(f"📁 Dataset shape: {df.shape}")
    print(f"📅 Columns: {len(df.columns)}")

    # Check classification column
    if 'classification_label' not in df.columns:
        print("❌ No classification_label column!")
        return

    labels = df['classification_label'].to_numpy()

    print("\n🎯 CLASSIFICATION DISTRIBUTION:")
    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"   Classes found: {unique_labels}")
    print("   Expected: 0-12 (13 classes)")
    print(f"   Total samples: {len(labels):,}")

    for label, count in zip(unique_labels, counts, strict=False):
        percentage = (count / len(labels)) * 100
        print(f"   Class {label:2d}: {count:7,} samples ({percentage:5.1f}%)")

    # Check for obvious issues
    print("\n⚠️  ISSUE CHECK:")

    # Issue 1: Wrong number of classes
    if len(unique_labels) != 13:
        print(f"   ❌ Expected 13 classes, got {len(unique_labels)}")

        missing = set(range(13)) - set(unique_labels)
        if missing:
            print(f"   ❌ Missing classes: {sorted(missing)}")

        extra = set(unique_labels) - set(range(13))
        if extra:
            print(f"   ❌ Unexpected classes: {sorted(extra)}")
    else:
        print("   ✅ Correct number of classes (13)")

    # Issue 2: Extreme imbalance
    if len(counts) > 1:
        min_count, max_count = min(counts), max(counts)
        imbalance_ratio = max_count / min_count
        print(f"   Class imbalance ratio: {imbalance_ratio:.2f} (min={min_count:,}, max={max_count:,})")

        if imbalance_ratio > 5.0:
            print("   ❌ Severe class imbalance (ratio > 5.0)")
        elif imbalance_ratio > 2.0:
            print("   ⚠️  Moderate class imbalance (ratio > 2.0)")
        else:
            print("   ✅ Reasonable class balance")

    # Issue 3: Sample mid prices
    print("\n📈 SAMPLE PRICE DATA:")
    if 'bid_px_00' in df.columns and 'ask_px_00' in df.columns:
        sample_size = min(1000, len(df))
        sample_df = df.head(sample_size)

        bid_prices = sample_df['bid_px_00'].to_numpy()
        ask_prices = sample_df['ask_px_00'].to_numpy()
        mid_prices = (bid_prices + ask_prices) / 2.0

        print(f"   Sample size: {sample_size:,}")
        print(f"   Mid price range: {np.min(mid_prices):.5f} to {np.max(mid_prices):.5f}")
        print(f"   Mid price mean: {np.mean(mid_prices):.5f}")
        print(f"   Mid price std: {np.std(mid_prices):.5f}")

        # Calculate simple price movements (adjacent rows)
        if len(mid_prices) > 1:
            simple_movements = np.diff(mid_prices) / mid_prices[:-1]
            print(f"   Adjacent movements mean: {np.mean(simple_movements):+.6f} ({np.mean(simple_movements)*100:+.4f}%)")
            print(f"   Adjacent movements std: {np.std(simple_movements):.6f} ({np.std(simple_movements)*100:.4f}%)")

            large_moves = np.abs(simple_movements) > 0.001  # 0.1%
            if np.any(large_moves):
                print(f"   ⚠️  Large movements (>0.1%): {np.sum(large_moves)} out of {len(simple_movements)}")

    return {
        'name': dataset_path.name,
        'samples': len(labels),
        'unique_classes': len(unique_labels),
        'classes': unique_labels.tolist(),
        'imbalance_ratio': max(counts) / min(counts) if len(counts) > 1 else 1.0
    }


def main():
    """Quick analysis of classification results."""
    print("🔍 QUICK CLASSIFICATION CHECK")
    print("=" * 50)

    datasets_dir = Path("/Users/danielfisher/data/databento/AUDUSD_classified_datasets")
    dataset_files = sorted(datasets_dir.glob("*_dataset.parquet"))

    print(f"📁 Found {len(dataset_files)} datasets")

    results = []
    for dataset_file in dataset_files:
        try:
            result = quick_check_dataset(dataset_file)
            results.append(result)
        except Exception as e:
            print(f"❌ Error checking {dataset_file.name}: {e}")

    # Summary
    print("\n📋 SUMMARY")
    print("=" * 50)

    total_samples = sum(r['samples'] for r in results)
    print(f"📊 Total samples across all datasets: {total_samples:,}")

    # Check class consistency
    all_classes = set()
    for result in results:
        all_classes.update(result['classes'])

    print(f"🎯 All classes found across datasets: {sorted(all_classes)}")

    # Check for problematic datasets
    problems = []
    for result in results:
        if result['unique_classes'] != 13:
            problems.append(f"{result['name']}: {result['unique_classes']} classes (expected 13)")
        if result['imbalance_ratio'] > 3.0:
            problems.append(f"{result['name']}: high imbalance ratio {result['imbalance_ratio']:.1f}")

    if problems:
        print("\n⚠️  ISSUES DETECTED:")
        for problem in problems:
            print(f"   {problem}")
    else:
        print("\n✅ No obvious issues detected")


if __name__ == "__main__":
    exit(main())

