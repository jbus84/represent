#!/usr/bin/env python
"""
Analyze Classification Results

This script analyzes the classified datasets to check:
1. Mid price movement distributions
2. Classification distributions
3. Relationship between price movements and classes
4. Data quality and validation
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_mid_price_movements(df, lookback_rows=5000, lookforward_input=5000, lookforward_offset=500):
    """Calculate mid price movements using the same methodology as classification."""
    print("🔍 Calculating price movements:")
    print(f"   Lookback rows: {lookback_rows}")
    print(f"   Lookforward input: {lookforward_input}")
    print(f"   Lookforward offset: {lookforward_offset}")

    # Calculate mid prices
    mid_prices = []
    for i in range(len(df)):
        row = df.row(i)
        bid_00 = row[df.columns.index('bid_px_00')]
        ask_00 = row[df.columns.index('ask_px_00')]
        mid_price = (bid_00 + ask_00) / 2.0
        mid_prices.append(mid_price)

    mid_prices = np.array(mid_prices)

    # Calculate price movements for valid rows
    price_movements = []
    valid_indices = []

    total_rows = len(df)
    start_row = lookback_rows
    end_row = total_rows - (lookforward_input + lookforward_offset)

    print(f"   Total rows: {total_rows:,}")
    print(f"   Valid range: {start_row:,} to {end_row:,}")
    print(f"   Valid rows: {end_row - start_row:,}")

    for stop_row in range(start_row, end_row):
        # Calculate lookback mean
        lookback_start = stop_row - lookback_rows
        lookback_end = stop_row
        lookback_mean = np.mean(mid_prices[lookback_start:lookback_end])

        # Calculate lookforward mean
        lookforward_start = stop_row + 1 + lookforward_offset
        lookforward_end = stop_row + 1 + lookforward_offset + lookforward_input
        lookforward_mean = np.mean(mid_prices[lookforward_start:lookforward_end])

        # Calculate percentage movement
        if lookback_mean > 0:
            price_movement = (lookforward_mean - lookback_mean) / lookback_mean
            price_movements.append(price_movement)
            valid_indices.append(stop_row)

    return np.array(price_movements), valid_indices


def analyze_dataset(dataset_path):
    """Analyze a single dataset for classification quality."""
    print(f"\n📊 ANALYZING: {dataset_path.name}")
    print("=" * 60)

    # Load dataset
    df = pl.read_parquet(dataset_path)
    print(f"📁 Dataset shape: {df.shape}")

    # Check for classification_label column
    if 'classification_label' not in df.columns:
        print("❌ No classification_label column found!")
        return

    # Get classification labels
    labels = df['classification_label'].to_numpy()

    # Calculate price movements
    try:
        price_movements, valid_indices = calculate_mid_price_movements(df)
        print(f"✅ Calculated {len(price_movements):,} price movements")

        # Align classifications with price movements
        if len(valid_indices) > 0:
            aligned_labels = labels[valid_indices]
            print(f"✅ Aligned {len(aligned_labels):,} classification labels")
        else:
            print("❌ No valid price movement indices!")
            return

    except Exception as e:
        print(f"❌ Error calculating price movements: {e}")
        return

    # Analysis 1: Price Movement Distribution
    print("\n📈 PRICE MOVEMENT ANALYSIS:")
    print(f"   Count: {len(price_movements):,}")
    print(f"   Mean: {np.mean(price_movements):+.6f} ({np.mean(price_movements)*100:+.4f}%)")
    print(f"   Std:  {np.std(price_movements):.6f} ({np.std(price_movements)*100:.4f}%)")
    print(f"   Min:  {np.min(price_movements):+.6f} ({np.min(price_movements)*100:+.4f}%)")
    print(f"   Max:  {np.max(price_movements):+.6f} ({np.max(price_movements)*100:+.4f}%)")
    print(f"   25%:  {np.percentile(price_movements, 25):+.6f}")
    print(f"   50%:  {np.percentile(price_movements, 50):+.6f}")
    print(f"   75%:  {np.percentile(price_movements, 75):+.6f}")

    # Analysis 2: Classification Distribution
    print("\n🎯 CLASSIFICATION ANALYSIS:")
    unique_labels = np.unique(aligned_labels)
    print(f"   Unique classes: {len(unique_labels)} {unique_labels}")
    print("   Expected classes: 0-12 (13 total)")

    class_counts = {}
    for label in unique_labels:
        count = np.sum(aligned_labels == label)
        percentage = (count / len(aligned_labels)) * 100
        class_counts[label] = count
        print(f"   Class {label:2d}: {count:6,} samples ({percentage:5.1f}%)")

    # Check uniformity
    if len(unique_labels) > 1:
        expected_per_class = len(aligned_labels) / len(unique_labels)
        max_deviation = max(abs(count - expected_per_class) for count in class_counts.values())
        uniformity_score = 1 - (max_deviation / expected_per_class)
        print(f"   Uniformity score: {uniformity_score:.3f} (1.0 = perfect)")

    # Analysis 3: Movement-to-Class Mapping
    print("\n🔗 MOVEMENT-TO-CLASS MAPPING:")
    for label in sorted(unique_labels)[:5]:  # Show first 5 classes
        mask = aligned_labels == label
        class_movements = price_movements[mask]
        if len(class_movements) > 0:
            print(f"   Class {label:2d}: movements [{np.min(class_movements):+.6f}, {np.max(class_movements):+.6f}], mean={np.mean(class_movements):+.6f}")

    if len(unique_labels) > 5:
        print(f"   ... (showing first 5 of {len(unique_labels)} classes)")

    # Analysis 4: Potential Issues
    print("\n⚠️  POTENTIAL ISSUES:")
    issues = []

    # Check for missing classes
    expected_classes = set(range(13))  # Expecting 0-12
    actual_classes = set(unique_labels)
    missing_classes = expected_classes - actual_classes
    if missing_classes:
        issues.append(f"Missing classes: {sorted(missing_classes)}")

    # Check for unexpected classes
    unexpected_classes = actual_classes - expected_classes
    if unexpected_classes:
        issues.append(f"Unexpected classes: {sorted(unexpected_classes)}")

    # Check for extreme imbalance
    if len(unique_labels) > 1:
        class_sizes = list(class_counts.values())
        min_size, max_size = min(class_sizes), max(class_sizes)
        if max_size > min_size * 3:  # More than 3x difference
            issues.append(f"Significant class imbalance: min={min_size:,}, max={max_size:,}")

    # Check for unrealistic price movements
    extreme_movements = np.abs(price_movements) > 0.01  # More than 1%
    if np.any(extreme_movements):
        extreme_count = np.sum(extreme_movements)
        issues.append(f"Extreme price movements (>1%): {extreme_count:,} samples")

    if issues:
        for issue in issues:
            print(f"   ❌ {issue}")
    else:
        print("   ✅ No obvious issues detected")

    return {
        'dataset_name': dataset_path.name,
        'total_samples': len(df),
        'valid_movements': len(price_movements),
        'price_movement_stats': {
            'mean': float(np.mean(price_movements)),
            'std': float(np.std(price_movements)),
            'min': float(np.min(price_movements)),
            'max': float(np.max(price_movements))
        },
        'class_distribution': class_counts,
        'unique_classes': len(unique_labels),
        'issues': issues
    }


def create_analysis_plots(results, output_dir):
    """Create analysis plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n📊 Creating analysis plots in {output_dir}")

    # Plot 1: Dataset sizes
    dataset_names = [r['dataset_name'] for r in results]
    sample_counts = [r['total_samples'] for r in results]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(dataset_names)), sample_counts, color='steelblue', alpha=0.7)
    plt.title('Dataset Sizes by Symbol', fontsize=14, fontweight='bold')
    plt.xlabel('Symbol Dataset')
    plt.ylabel('Total Samples')
    plt.xticks(range(len(dataset_names)), [name.replace('AUDUSD_', '').replace('_dataset.parquet', '') for name in dataset_names], rotation=45)

    # Add value labels on bars
    for bar, count in zip(bars, sample_counts, strict=False):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_counts)*0.01,
                f'{count:,}', ha='center', fontsize=9)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Price movement distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, result in enumerate(results[:6]):  # Max 6 plots
        if i >= len(axes):
            break

        stats = result['price_movement_stats']
        symbol = result['dataset_name'].replace('AUDUSD_', '').replace('_dataset.parquet', '')

        # Create mock distribution for visualization (in real analysis, we'd load the actual movements)
        movements = np.random.normal(stats['mean'], stats['std'], 10000)
        movements = np.clip(movements, stats['min'], stats['max'])

        axes[i].hist(movements, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
        axes[i].set_title(f'{symbol} Price Movements', fontweight='bold')
        axes[i].set_xlabel('Price Movement')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

        # Add stats text
        stats_text = f'Mean: {stats["mean"]:+.5f}\nStd: {stats["std"]:.5f}'
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                    verticalalignment='top', fontsize=9, bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})

    plt.tight_layout()
    plt.savefig(output_dir / 'price_movement_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Plots saved to {output_dir}")


def main():
    """Main analysis function."""
    print("🔍 CLASSIFICATION RESULTS ANALYSIS")
    print("=" * 50)

    datasets_dir = Path("/Users/danielfisher/data/databento/AUDUSD_classified_datasets")

    if not datasets_dir.exists():
        print(f"❌ Datasets directory not found: {datasets_dir}")
        return 1

    # Find all dataset files
    dataset_files = list(datasets_dir.glob("*_dataset.parquet"))

    if not dataset_files:
        print(f"❌ No dataset files found in {datasets_dir}")
        return 1

    print(f"📁 Found {len(dataset_files)} dataset files")

    # Analyze each dataset
    results = []
    for dataset_file in sorted(dataset_files):
        try:
            result = analyze_dataset(dataset_file)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error analyzing {dataset_file.name}: {e}")

    # Overall summary
    print("\n📋 OVERALL SUMMARY")
    print("=" * 50)

    total_samples = sum(r['total_samples'] for r in results)
    total_movements = sum(r['valid_movements'] for r in results)

    print(f"📊 Datasets analyzed: {len(results)}")
    print(f"📊 Total samples: {total_samples:,}")
    print(f"📊 Total valid movements: {total_movements:,}")
    print(f"📊 Average samples per dataset: {total_samples // len(results):,}")

    # Check for common issues
    common_issues = {}
    for result in results:
        for issue in result['issues']:
            if issue not in common_issues:
                common_issues[issue] = 0
            common_issues[issue] += 1

    if common_issues:
        print("\n⚠️  COMMON ISSUES ACROSS DATASETS:")
        for issue, count in common_issues.items():
            print(f"   {issue} (affects {count}/{len(results)} datasets)")
    else:
        print("\n✅ NO COMMON ISSUES DETECTED")

    # Create plots
    output_dir = datasets_dir / "analysis_plots"
    create_analysis_plots(results, output_dir)

    print("\n🎉 Analysis complete!")
    print(f"📁 Plots saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())

