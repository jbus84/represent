#!/usr/bin/env python3
"""
Streamlined Classification Analysis Demo

Demonstrates the streamlined DBN-to-classified-parquet approach with:
- Direct DBN processing to classified parquet files by symbol
- True uniform distribution using quantile-based classification
- Complete statistical analysis and visualization
- Production-ready single-pass processing
"""

import sys
from pathlib import Path
import time
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy import stats
import json
import shutil
from typing import Dict

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from represent.parquet_classifier import ParquetClassifier
import gc


def process_dbn_data_streamlined(
    data_dir: Path,
    currency: str = "AUDUSD"
) -> tuple[np.ndarray, Dict]:
    """Process DBN data using streamlined single-pass approach."""
    
    print("🔄 Processing DBN data with streamlined approach...")
    
    # Find DBN files
    dbn_files = sorted([f for f in data_dir.glob("*.dbn.zst") if f.is_file()])
    
    if not dbn_files:
        raise FileNotFoundError(f"No DBN files found in {data_dir}")
    
    print(f"📊 Found {len(dbn_files)} DBN files")
    for i, file_path in enumerate(dbn_files):
        size_mb = file_path.stat().st_size / 1024 / 1024
        print(f"   {i+1}. {file_path.name} ({size_mb:.1f} MB)")
    
    # Create output directory
    classified_dir = data_dir / "streamlined_classified"
    classified_dir.mkdir(exist_ok=True)
    
    # Create streamlined classifier with optimized settings
    classifier = ParquetClassifier(
        currency=currency,
        features=["volume"],           # Single feature for reliability
        input_rows=5000,              # Historical data required
        lookforward_rows=500,         # Future data required
        min_symbol_samples=2000,      # Higher quality threshold
        force_uniform=True,           # Ensure uniform distribution
        nbins=13,                     # Standard 13-class classification
        verbose=False                 # Reduce output for cleaner demo
    )
    
    all_labels = []
    processing_stats = {
        "files_processed": 0,
        "total_samples": 0,
        "symbols_processed": 0,
        "processing_time": 0,
    }
    
    start_time = time.perf_counter()
    
    for i, dbn_file in enumerate(dbn_files):
        print(f"\n🚀 Processing {i+1}/{len(dbn_files)}: {dbn_file.name}")
        
        try:
            # Single-pass DBN to classified parquet processing
            stats_result = classifier.process_dbn_to_classified_parquets(
                dbn_path=dbn_file,
                output_dir=classified_dir
            )
            
            processing_stats["files_processed"] += 1
            processing_stats["total_samples"] += stats_result["total_classified_samples"]
            processing_stats["symbols_processed"] += stats_result["symbols_processed"]
            
            print(f"   ✅ Generated {stats_result['symbols_processed']} symbol files")
            print(f"   📊 Classified {stats_result['total_classified_samples']:,} samples")
            
            # Collect labels from all generated files
            for symbol, symbol_stats in stats_result['symbol_stats'].items():
                classified_path = Path(symbol_stats['file_path'])
                if classified_path.exists():
                    df = pl.read_parquet(classified_path)
                    if 'classification_label' in df.columns:
                        labels = df['classification_label'].to_numpy()
                        all_labels.extend(labels)
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    end_time = time.perf_counter()
    processing_stats["processing_time"] = end_time - start_time
    
    # Memory cleanup
    import gc
    gc.collect()
    
    if not all_labels:
        raise ValueError("No classification results generated")
    
    classification_array = np.array(all_labels)
    
    print("✅ Processing complete!")
    print(f"   📊 Total samples: {len(classification_array):,}")
    print(f"   ⏱️  Processing time: {processing_stats['processing_time']:.1f}s")
    
    return classification_array, processing_stats


def calculate_statistics(labels: np.ndarray) -> Dict:
    """Calculate comprehensive statistics."""
    
    # Basic stats
    total_samples = len(labels)
    mean_label = float(np.mean(labels))
    std_label = float(np.std(labels))
    median_label = float(np.median(labels))
    min_label = int(np.min(labels))
    max_label = int(np.max(labels))
    
    # Distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    percentages = (counts / total_samples) * 100
    
    distribution = {}
    for label, count, percentage in zip(unique_labels, counts, percentages):
        distribution[int(label)] = {
            "count": int(count),
            "percentage": float(percentage)
        }
    
    # Normality test
    try:
        test_sample = labels[:5000] if len(labels) > 5000 else labels
        shapiro_stat, shapiro_p = stats.shapiro(test_sample)
        is_normal = shapiro_p > 0.05
    except Exception:
        shapiro_stat, shapiro_p = 0.0, 0.0
        is_normal = False
    
    # Uniformity analysis
    expected_percentage = 100.0 / 13
    deviations = []
    for i in range(13):
        actual_percentage = distribution.get(i, {"percentage": 0.0})["percentage"]
        deviation = abs(actual_percentage - expected_percentage)
        deviations.append(deviation)
    
    max_deviation = max(deviations)
    avg_deviation = float(np.mean(deviations))
    
    # Quality assessment
    if max_deviation < 2.0:
        quality = "EXCELLENT"
    elif max_deviation < 3.0:
        quality = "GOOD"
    elif max_deviation < 5.0:
        quality = "ACCEPTABLE"
    else:
        quality = "NEEDS_IMPROVEMENT"
    
    return {
        "basic_stats": {
            "total_samples": total_samples,
            "mean": mean_label,
            "std": std_label,
            "median": median_label,
            "min": min_label,
            "max": max_label,
            "range": f"{min_label} - {max_label}"
        },
        "distribution": distribution,
        "normality": {
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "is_normal": is_normal
        },
        "uniformity": {
            "expected_percentage": expected_percentage,
            "max_deviation": max_deviation,
            "avg_deviation": avg_deviation,
            "quality_assessment": quality
        }
    }


def create_analysis_plot(labels: np.ndarray, stats_dict: Dict, processing_stats: Dict, output_dir: Path) -> str:
    """Create comprehensive analysis plot."""
    
    print("📊 Creating analysis plot...")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main histogram
    ax1 = fig.add_subplot(gs[0, :2])
    n_bins = len(np.unique(labels))
    ax1.hist(labels, bins=n_bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add normal fit
    mean_val = stats_dict["basic_stats"]["mean"]
    std_val = stats_dict["basic_stats"]["std"]
    x_range = np.linspace(labels.min(), labels.max(), 100)
    y_normal = stats.norm.pdf(x_range, mean_val, std_val)
    ax1.plot(x_range, y_normal, 'r-', linewidth=2, label=f'Normal(μ={mean_val:.2f}, σ={std_val:.2f})')
    
    ax1.set_xlabel('Classification Label')
    ax1.set_ylabel('Density')
    ax1.set_title('Classification Distribution Analysis (13 classes)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    ax2 = fig.add_subplot(gs[0, 2])
    stats.probplot(labels, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot\n(Normality Test)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax3 = fig.add_subplot(gs[1, 0])
    box_plot = ax3.boxplot(labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    ax3.set_ylabel('Classification Label')
    ax3.set_title('Box Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. Frequency bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    all_labels = list(range(13))
    frequencies = []
    percentages_list = []
    
    for label in all_labels:
        if label in stats_dict["distribution"]:
            freq = stats_dict["distribution"][label]["count"]
            perc = stats_dict["distribution"][label]["percentage"]
        else:
            freq = 0
            perc = 0.0
        frequencies.append(freq)
        percentages_list.append(perc)
    
    bars = ax4.bar(all_labels, frequencies, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.set_xlabel('Classification Label')
    ax4.set_ylabel('Count')
    ax4.set_title('Label Frequencies')
    ax4.set_xticks(all_labels)
    ax4.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, percentage in zip(bars, percentages_list):
        if percentage > 0:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 5. Cumulative distribution
    ax5 = fig.add_subplot(gs[1, 2])
    sorted_labels = np.sort(labels)
    y_values = np.arange(1, len(sorted_labels) + 1) / len(sorted_labels)
    ax5.plot(sorted_labels, y_values, 'b-', linewidth=2, label='Empirical CDF')
    
    normal_cdf = stats.norm.cdf(sorted_labels, mean_val, std_val)
    ax5.plot(sorted_labels, normal_cdf, 'r--', linewidth=2, label='Normal CDF')
    
    ax5.set_xlabel('Classification Label')
    ax5.set_ylabel('Cumulative Probability')
    ax5.set_title('Cumulative Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Time series
    ax6 = fig.add_subplot(gs[2, 0])
    sample_size = min(500, len(labels))
    sample_indices = np.arange(sample_size)
    sample_labels = labels[:sample_size]
    
    ax6.plot(sample_indices, sample_labels, 'b-', alpha=0.7, linewidth=0.8)
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('Classification Label')
    ax6.set_title(f'Time Series Sample\n({sample_size} points)')
    ax6.grid(True, alpha=0.3)
    
    # 7. Statistics summary
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.axis('off')
    
    summary_text = f"""Statistical Summary

Total Samples: {stats_dict["basic_stats"]["total_samples"]:,}
Mean: {stats_dict["basic_stats"]["mean"]:.2f}
Std Dev: {stats_dict["basic_stats"]["std"]:.2f}
Median: {stats_dict["basic_stats"]["median"]:.2f}
Range: {stats_dict["basic_stats"]["range"]}

Normality Test:
Statistic: {stats_dict["normality"]["shapiro_stat"]:.2f}
P-Value: {stats_dict["normality"]["shapiro_p"]:.4f}
Normal: {'✓' if stats_dict["normality"]["is_normal"] else '✗'}

STREAMLINED PROCESSING:
✅ Direct DBN Processing
✅ Single-Pass Approach
✅ Per-Symbol Classification
✅ Uniform Distribution
"""
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            fontfamily='monospace')
    
    # 8. Distribution comparison
    ax8 = fig.add_subplot(gs[2, 2])
    
    # KDE if enough unique values
    if len(np.unique(labels)) > 3:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(labels)
        x_kde = np.linspace(labels.min()-1, labels.max()+1, 200)
        kde_values = kde(x_kde)
        ax8.plot(x_kde, kde_values, 'b-', linewidth=2, label='Actual (KDE)')
        
        # Normal overlay
        normal_values = stats.norm.pdf(x_kde, mean_val, std_val)
        ax8.plot(x_kde, normal_values, 'r--', linewidth=2, label='Normal Fit')
    
    # Uniform reference
    ax8.axhline(1.0/13, color='green', linestyle=':', linewidth=2, label='Uniform')
    
    ax8.set_xlabel('Classification Label')
    ax8.set_ylabel('Density')
    ax8.set_title('Distribution Comparison')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Main title
    quality = stats_dict["uniformity"]["quality_assessment"]
    deviation = stats_dict["uniformity"]["avg_deviation"]
    processing_rate = processing_stats["total_samples"] / processing_stats["processing_time"]
    
    fig.suptitle(f'Comprehensive Classification Analysis - Streamlined Implementation\n'
                f'Quality: {quality} (Avg Deviation: {deviation:.1f}%) | '
                f'Samples: {stats_dict["basic_stats"]["total_samples"]:,} | '
                f'Rate: {processing_rate:.0f} samples/sec',
                fontsize=14, fontweight='bold')
    
    # Save
    plot_path = output_dir / "streamlined_comprehensive_classification_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved plot: {plot_path}")
    return str(plot_path)


def create_summary_report(stats_dict: Dict, processing_stats: Dict, output_dir: Path) -> str:
    """Create summary report."""
    
    print("📄 Creating summary report...")
    
    processing_rate = processing_stats["total_samples"] / processing_stats["processing_time"]
    
    report_content = f"""# Streamlined Classification Analysis Summary

## Key Results

✅ **Streamlined Implementation**: Direct DBN-to-classified-parquet processing.

✅ **Distribution Analysis**: Classification targets analyzed with {stats_dict["basic_stats"]["total_samples"]:,} samples.

## Statistical Summary

- **Total Classifications**: {stats_dict["basic_stats"]["total_samples"]:,}
- **Mean Label**: {stats_dict["basic_stats"]["mean"]:.2f}
- **Standard Deviation**: {stats_dict["basic_stats"]["std"]:.2f}
- **Label Range**: {stats_dict["basic_stats"]["range"]}
- **Normality Test P-value**: {stats_dict["normality"]["shapiro_p"]:.4f}

## Streamlined Processing Results

- **Files Processed**: {processing_stats["files_processed"]}
- **Symbols Processed**: {processing_stats["symbols_processed"]}
- **Processing Rate**: {processing_rate:.0f} samples/sec
- **Total Processing Time**: {processing_stats["processing_time"]:.1f} seconds

### Streamlined Settings Used:
- ✅ **Processing**: Direct DBN → Classified Parquet
- ✅ **Per-Symbol**: Individual symbol processing  
- ✅ **Uniform Distribution**: Quantile-based classification
- ✅ **Features**: Volume (single feature for reliability)
- ✅ **Quality Threshold**: Min symbol samples enforced
- ✅ **Memory Efficient**: No intermediate files

## Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|"""

    # Add distribution table
    for label in range(13):
        if label in stats_dict["distribution"]:
            count = stats_dict["distribution"][label]["count"]
            percentage = stats_dict["distribution"][label]["percentage"]
            report_content += f"\n| {label} | {count} | {percentage:.1f}% |"
        else:
            report_content += f"\n| {label} | 0 | 0.0% |"

    # Quality assessment
    quality = stats_dict["uniformity"]["quality_assessment"]
    max_dev = stats_dict["uniformity"]["max_deviation"]
    avg_dev = stats_dict["uniformity"]["avg_deviation"]
    
    status_icon = "✅" if quality in ["EXCELLENT", "GOOD"] else "⚠️" if quality == "ACCEPTABLE" else "❌"

    report_content += f"""

## Validation Status

{status_icon} **{quality}**: Distribution shows {avg_dev:.1f}% average deviation from uniform (max: {max_dev:.1f}%).

## Uniformity Analysis

- **Expected Percentage**: {stats_dict["uniformity"]["expected_percentage"]:.2f}% (uniform distribution)
- **Max Deviation**: {max_dev:.2f}%
- **Average Deviation**: {avg_dev:.2f}%
- **Quality Assessment**: {quality}

## Visualization

![Streamlined Classification Analysis](streamlined_comprehensive_classification_analysis.png)

---

*Report generated by Streamlined Classification Analysis Demo with direct DBN processing*
"""

    # Save report
    report_path = output_dir / "streamlined_classification_validation_summary.md"
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"   ✅ Saved report: {report_path}")
    return str(report_path)


def cleanup_resources(temp_dirs=None, verbose=True):
    """Clean up temporary directories and free memory."""
    if verbose:
        print("🧹 Cleaning up resources...")
    
    # Clean up temporary directories
    if temp_dirs:
        for temp_dir in temp_dirs:
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    if verbose:
                        print(f"   ✅ Cleaned up: {temp_dir}")
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Could not clean {temp_dir}: {e}")
    
    # Force garbage collection
    gc.collect()
    
    if verbose:
        print("   ✅ Memory cleanup complete")


def main():
    """Run streamlined classification analysis with proper resource cleanup."""
    
    print("🚀 STREAMLINED CLASSIFICATION ANALYSIS")
    print("=" * 60)
    print("🎯 Streamlined production data processing with uniform distribution")
    print("=" * 60)
    
    # Setup paths
    data_dir = Path("/Users/danielfisher/repositories/represent/data")
    output_dir = Path("/Users/danielfisher/repositories/represent/examples/classification_analysis/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Process data
        labels, processing_stats = process_dbn_data_streamlined(data_dir, "AUDUSD")
        
        # Calculate statistics
        stats_dict = calculate_statistics(labels)
        
        # Create visualization
        plot_path = create_analysis_plot(labels, stats_dict, processing_stats, output_dir)
        
        # Create report
        report_path = create_summary_report(stats_dict, processing_stats, output_dir)
        
        # Save JSON results
        results = {
            "analysis_type": "streamlined_classification_analysis",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_stats": processing_stats,
            "statistical_analysis": stats_dict,
        }
        
        json_path = output_dir / "streamlined_detailed_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Final summary
        processing_rate = processing_stats["total_samples"] / processing_stats["processing_time"]
        
        print("\n" + "=" * 60)
        print("🎉 STREAMLINED ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"📊 Total samples: {len(labels):,}")
        print(f"📊 Processing rate: {processing_rate:.0f} samples/sec")
        print(f"📊 Quality: {stats_dict['uniformity']['quality_assessment']}")
        print(f"📊 Avg deviation: {stats_dict['uniformity']['avg_deviation']:.2f}%")
        print()
        print("📁 Output files:")
        print(f"   📊 Plot: {Path(plot_path).name}")
        print(f"   📄 Report: {Path(report_path).name}")  
        print(f"   📋 JSON: {Path(json_path).name}")
        print()
        print("🚀 Ready for production with streamlined processing!")
        
        # Clean up resources before returning
        cleanup_resources(verbose=True)
        
        return {
            "status": "success",
            "total_samples": len(labels),
            "quality": stats_dict['uniformity']['quality_assessment'],
            "processing_rate": processing_rate,
            "uniform_distribution": True,
            "force_uniform_enabled": True,
        }
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up resources even on failure
        cleanup_resources(verbose=True)
        
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    results = main()