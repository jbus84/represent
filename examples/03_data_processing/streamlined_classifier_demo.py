#!/usr/bin/env python3
"""
Streamlined DBN Classifier Demo

This example demonstrates the new streamlined approach that processes DBN files
directly to classified parquet files with uniform distribution, eliminating
intermediate files and ensuring optimal ML training data.

Key Features:
- Single DBN file → Multiple classified parquet files by symbol
- True uniform distribution using quantile-based classification
- No intermediate parquet files needed
- Automatic row filtering for insufficient history/future data
- Direct symbol-tagged output files ready for ML training
"""

import sys
from pathlib import Path
import time
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from typing import Dict, List
import json

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from represent.parquet_classifier import ParquetClassifier, process_dbn_to_classified_parquets


def analyze_classification_uniformity(classified_files: List[Path]) -> Dict:
    """Analyze uniformity across all classified parquet files."""
    print("\n📊 Analyzing classification uniformity across symbols...")
    
    all_labels = []
    symbol_stats = {}
    
    for parquet_file in classified_files:
        if not parquet_file.exists():
            continue
            
        # Load classified data
        df = pl.read_parquet(parquet_file)
        if 'classification_label' not in df.columns:
            continue
            
        labels = df['classification_label'].to_numpy()
        symbol = parquet_file.stem.split('_')[1]  # Extract symbol from filename
        
        # Per-symbol analysis
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = (counts / len(labels)) * 100
        
        symbol_stats[symbol] = {
            'samples': len(labels),
            'distribution': {int(label): {'count': int(count), 'percentage': float(perc)} 
                           for label, count, perc in zip(unique_labels, counts, percentages)},
            'mean': float(np.mean(labels)),
            'std': float(np.std(labels)),
        }
        
        all_labels.extend(labels)
        
        print(f"   📈 {symbol}: {len(labels):,} samples, classes: {list(unique_labels)}")
    
    if not all_labels:
        return {'error': 'No valid classification data found'}
    
    # Overall analysis
    all_labels = np.array(all_labels)
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    percentages = (counts / len(all_labels)) * 100
    
    overall_distribution = {
        int(label): {'count': int(count), 'percentage': float(perc)} 
        for label, count, perc in zip(unique_labels, counts, percentages)
    }
    
    # Uniformity assessment
    expected_percentage = 100.0 / 13  # 13 classes
    deviations = []
    for i in range(13):
        actual_percentage = overall_distribution.get(i, {'percentage': 0.0})['percentage']
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
        'total_samples': len(all_labels),
        'symbols_analyzed': len(symbol_stats),
        'overall_distribution': overall_distribution,
        'symbol_stats': symbol_stats,
        'uniformity_analysis': {
            'expected_percentage': expected_percentage,
            'max_deviation': max_deviation,
            'avg_deviation': avg_deviation,
            'quality_assessment': quality,
            'deviations_by_class': deviations,
        },
        'basic_stats': {
            'mean': float(np.mean(all_labels)),
            'std': float(np.std(all_labels)),
            'min': int(np.min(all_labels)),
            'max': int(np.max(all_labels)),
        }
    }


def create_streamlined_analysis_plot(analysis_results: Dict, output_dir: Path) -> str:
    """Create comprehensive analysis plot for streamlined classifier results."""
    print("📊 Creating streamlined analysis visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall Distribution Histogram
    labels = list(range(13))
    percentages = [
        analysis_results['overall_distribution'].get(i, {'percentage': 0.0})['percentage']
        for i in labels
    ]
    
    bars = ax1.bar(labels, percentages, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axhline(y=100.0/13, color='red', linestyle='--', linewidth=2, label='Uniform Target (7.69%)')
    ax1.set_xlabel('Classification Label')
    ax1.set_ylabel('Percentage')
    ax1.set_title('Overall Classification Distribution\n(Streamlined Approach)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(labels)
    
    # Add percentage labels on bars
    for bar, percentage in zip(bars, percentages):
        if percentage > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Deviation from Uniform
    deviations = analysis_results['uniformity_analysis']['deviations_by_class']
    ax2.bar(labels, deviations, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axhline(y=2.0, color='green', linestyle=':', label='Excellent (<2%)')
    ax2.axhline(y=3.0, color='orange', linestyle=':', label='Good (<3%)')
    ax2.axhline(y=5.0, color='red', linestyle=':', label='Acceptable (<5%)')
    ax2.set_xlabel('Classification Label')
    ax2.set_ylabel('Deviation from Uniform (%)')
    ax2.set_title('Deviation from Uniform Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(labels)
    
    # 3. Per-Symbol Sample Counts
    symbol_names = list(analysis_results['symbol_stats'].keys())
    sample_counts = [analysis_results['symbol_stats'][symbol]['samples'] for symbol in symbol_names]
    
    ax3.bar(range(len(symbol_names)), sample_counts, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Symbol')
    ax3.set_ylabel('Sample Count')
    ax3.set_title('Samples per Symbol')
    ax3.set_xticks(range(len(symbol_names)))
    ax3.set_xticklabels(symbol_names, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add count labels
    for i, count in enumerate(sample_counts):
        ax3.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    # 4. Summary Statistics Box
    ax4.axis('off')
    
    quality = analysis_results['uniformity_analysis']['quality_assessment']
    max_dev = analysis_results['uniformity_analysis']['max_deviation']
    avg_dev = analysis_results['uniformity_analysis']['avg_deviation']
    total_samples = analysis_results['total_samples']
    symbols_count = analysis_results['symbols_analyzed']
    
    summary_text = f"""STREAMLINED CLASSIFICATION RESULTS
    
📊 Total Samples: {total_samples:,}
📊 Symbols Processed: {symbols_count}
📊 Classes Generated: 13 (0-12)

UNIFORMITY ANALYSIS:
🎯 Target: 7.69% per class
📈 Max Deviation: {max_dev:.1f}%
📈 Avg Deviation: {avg_dev:.1f}%
📈 Quality: {quality}

STREAMLINED BENEFITS:
✅ Direct DBN → Classified Parquet
✅ No Intermediate Files
✅ True Uniform Distribution
✅ Symbol-Level Processing
✅ Automatic Row Filtering
✅ ML-Ready Output Format

STATUS: {"🟢 READY" if quality in ["EXCELLENT", "GOOD"] else "🟡 REVIEW" if quality == "ACCEPTABLE" else "🔴 IMPROVE"}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            fontfamily='monospace')
    
    # Main title
    fig.suptitle(f'Streamlined DBN Classifier Analysis\n'
                f'Quality: {quality} | Symbols: {symbols_count} | '
                f'Samples: {total_samples:,}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "streamlined_classifier_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved analysis plot: {plot_path}")
    return str(plot_path)


def demonstrate_streamlined_classifier():
    """Demonstrate the new streamlined DBN classifier approach."""
    
    print("🚀 STREAMLINED DBN CLASSIFIER DEMONSTRATION")
    print("=" * 70)
    print("📋 Single DBN → Multiple Classified Parquet Files by Symbol")
    print("=" * 70)
    
    # Setup paths
    data_dir = Path("/Users/danielfisher/repositories/represent/data")
    output_dir = Path("/Users/danielfisher/repositories/represent/examples/classification_analysis/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find DBN files
    dbn_files = sorted([f for f in data_dir.glob("*.dbn.zst") if f.is_file()])
    
    if not dbn_files:
        print(f"❌ No DBN files found in {data_dir}")
        return False
    
    print(f"📊 Found {len(dbn_files)} DBN files:")
    for i, dbn_file in enumerate(dbn_files):
        size_mb = dbn_file.stat().st_size / 1024 / 1024
        print(f"   {i+1}. {dbn_file.name} ({size_mb:.1f} MB)")
    
    # Use first DBN file for demonstration
    dbn_file = dbn_files[0]
    
    print(f"\n🎯 Processing: {dbn_file.name}")
    print("=" * 50)
    
    # Create streamlined classifier with optimized settings
    classifier = ParquetClassifier(
        currency="AUDUSD",
        features=["volume"],  # Start with single feature
        input_rows=5000,      # Historical data required
        lookforward_rows=500, # Future data required  
        min_symbol_samples=50, # Lower threshold based on available data
        force_uniform=True,   # Ensure uniform distribution
        nbins=13,            # 13-class classification
        verbose=True,        # Show detailed progress
    )
    
    # Process DBN file directly to classified parquet files
    start_time = time.perf_counter()
    
    try:
        processing_stats = classifier.process_dbn_to_classified_parquets(
            dbn_path=dbn_file,
            output_dir=output_dir / "streamlined_classified",
        )
        
        processing_time = time.perf_counter() - start_time
        
        print("\n✅ STREAMLINED PROCESSING COMPLETE!")
        print("=" * 50)
        print(f"📊 Files processed: 1 DBN → {processing_stats['symbols_processed']} parquet files")
        print(f"📊 Total samples: {processing_stats['total_classified_samples']:,}")
        print(f"📊 Processing rate: {processing_stats['samples_per_second']:.0f} samples/sec")
        print(f"📊 Processing time: {processing_time:.1f}s")
        
        # Find generated classified files
        classified_dir = output_dir / "streamlined_classified"
        classified_files = sorted(classified_dir.glob("*_classified.parquet"))
        
        if not classified_files:
            print("❌ No classified files generated")
            return False
            
        print("\n📁 Generated classified files:")
        for i, file_path in enumerate(classified_files):
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"   {i+1}. {file_path.name} ({size_mb:.1f} MB)")
        
        # Analyze classification uniformity
        analysis_results = analyze_classification_uniformity(classified_files)
        
        if 'error' in analysis_results:
            print(f"❌ Analysis failed: {analysis_results['error']}")
            return False
        
        # Create visualization
        plot_path = create_streamlined_analysis_plot(analysis_results, output_dir)
        
        # Save detailed results
        results = {
            "approach": "streamlined_dbn_classifier",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": str(dbn_file),
            "processing_stats": processing_stats,
            "analysis_results": analysis_results,
            "classified_files": [str(f) for f in classified_files],
            "plot_path": plot_path,
        }
        
        json_path = output_dir / "streamlined_classifier_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Final summary
        quality = analysis_results['uniformity_analysis']['quality_assessment']
        avg_deviation = analysis_results['uniformity_analysis']['avg_deviation']
        
        print("\n🎉 STREAMLINED CLASSIFIER DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print(f"🎯 Quality Assessment: {quality}")
        print(f"🎯 Average Deviation: {avg_deviation:.1f}% from uniform")
        print(f"🎯 Symbols Processed: {analysis_results['symbols_analyzed']}")
        print(f"🎯 Total Samples: {analysis_results['total_samples']:,}")
        print("\n📁 Output Files:")
        print(f"   📊 Analysis Plot: {Path(plot_path).name}")
        print(f"   📋 Results JSON: {json_path.name}")
        print(f"   📁 Classified Data: {classified_dir.name}/")
        print("\n🚀 Classified parquet files ready for ML training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_convenience_function():
    """Demonstrate the convenience function approach."""
    
    print("\n" + "=" * 70)
    print("🔧 CONVENIENCE FUNCTION DEMONSTRATION")
    print("=" * 70)
    
    # Setup paths
    data_dir = Path("/Users/danielfisher/repositories/represent/data")
    output_dir = Path("/Users/danielfisher/repositories/represent/examples/classification_analysis/outputs")
    
    # Find DBN files
    dbn_files = sorted([f for f in data_dir.glob("*.dbn.zst") if f.is_file()])
    if not dbn_files:
        print("❌ No DBN files found")
        return False
    
    dbn_file = dbn_files[0]
    
    print(f"📄 Using convenience function with: {dbn_file.name}")
    
    try:
        # Use convenience function
        results = process_dbn_to_classified_parquets(
            dbn_path=dbn_file,
            output_dir=output_dir / "convenience_classified",
            currency="AUDUSD",
            features=["volume"],
            input_rows=5000,
            lookforward_rows=500,
            min_symbol_samples=50,
            force_uniform=True,
            nbins=13,
            verbose=True,
        )
        
        print("\n✅ Convenience function completed!")
        print(f"📊 Symbols: {results['symbols_processed']}")
        print(f"📊 Samples: {results['total_classified_samples']:,}")
        print(f"📊 Rate: {results['samples_per_second']:.0f} samples/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience function failed: {e}")
        return False


if __name__ == "__main__":
    success1 = demonstrate_streamlined_classifier()
    success2 = demonstrate_convenience_function()
    
    if success1 and success2:
        print("\n🎉 ALL DEMONSTRATIONS SUCCESSFUL!")
        print("✅ Streamlined classifier approach validated")
        print("✅ Convenience function approach validated") 
        print("🚀 Ready for production use!")
    else:
        print("\n⚠️  Some demonstrations failed - check output above")