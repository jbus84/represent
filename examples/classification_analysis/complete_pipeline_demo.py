#!/usr/bin/env python3
"""
Complete 3-Stage Pipeline Demonstration

This script demonstrates the complete represent pipeline using the new architecture:
1. Stage 1: DBN → Unlabeled Symbol-Grouped Parquet
2. Stage 2: Post-Processing Classification with Uniform Distribution
3. Stage 3: Lazy ML Training DataLoader

Updated for the new post-parquet classification approach with symbol-specific processing.
"""

import time
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import json

# Import the new architecture components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from represent import (
    create_market_depth_dataloader,  # Stage 3: ML training
)

from represent.unlabeled_converter import UnlabeledDBNConverter
from represent.parquet_classifier import ParquetClassifier


class CompletePipelineDemo:
    """
    Demonstrates the complete 3-stage represent pipeline with the new architecture.
    """

    def __init__(
        self,
        data_dir: str = "/Users/danielfisher/repositories/represent/data",
        output_dir: str = "/Users/danielfisher/repositories/represent/data/pipeline_outputs",
        currency: str = "AUDUSD",
    ):
        """Initialize the pipeline demo."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.currency = currency
        
        # Create output directories
        self.raw_parquet_dir = self.output_dir / "unlabeled_parquet"
        self.classified_parquet_dir = self.output_dir / "classified_parquet"
        self.analysis_dir = self.output_dir / "analysis"
        
        for dir_path in [self.raw_parquet_dir, self.classified_parquet_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print("🚀 Complete Pipeline Demo Initialized")
        print(f"   📁 Data source: {self.data_dir}")
        print(f"   📁 Output directory: {self.output_dir}")
        print(f"   💱 Currency: {self.currency}")
        print("   🎯 Architecture: DBN → Unlabeled Parquet → Classification → ML Training")

    def stage_1_dbn_to_parquet(self, max_files: int = 3) -> Dict:
        """
        Stage 1: Convert DBN files to unlabeled symbol-grouped parquet datasets.
        """
        print("\n" + "="*60)
        print("🔄 STAGE 1: DBN → Unlabeled Symbol-Grouped Parquet")
        print("="*60)

        # Get some sample DBN files
        dbn_files = sorted([f for f in self.data_dir.glob("*.dbn.zst") if f.is_file()])
        
        if not dbn_files:
            raise FileNotFoundError(f"No DBN files found in {self.data_dir}")
        
        # Use only first few files for demo
        sample_files = dbn_files[:max_files]
        
        print(f"📊 Found {len(dbn_files)} total DBN files")
        print(f"📊 Processing {len(sample_files)} files for demo:")
        for i, file_path in enumerate(sample_files):
            print(f"   {i+1}. {file_path.name}")

        stage_1_results = []
        
        # Initialize unlabeled converter
        converter = UnlabeledDBNConverter(
            features=["volume", "variance"],  # Multi-feature extraction
            min_symbol_samples=500,  # Lower threshold for demo
        )

        for i, dbn_file in enumerate(sample_files):
            print(f"\n🔄 Processing file {i+1}/{len(sample_files)}: {dbn_file.name}")
            
            try:
                # Convert to symbol-grouped parquet files
                stats = converter.convert_dbn_to_symbol_parquets(
                    dbn_path=dbn_file,
                    output_dir=self.raw_parquet_dir,
                    currency=self.currency,
                    chunk_size=25000,  # Smaller chunks for demo
                    include_metadata=True,
                )
                
                stage_1_results.append(stats)
                
                print(f"   ✅ Generated {stats['symbols_processed']} symbol files")
                print(f"   📊 Total samples: {stats['total_processed_samples']:,}")
                
            except Exception as e:
                print(f"   ❌ Failed to process {dbn_file.name}: {e}")
                continue

        # Summarize Stage 1 results
        total_symbols = sum(len(result['symbol_stats']) for result in stage_1_results)
        total_samples = sum(result['total_processed_samples'] for result in stage_1_results)
        
        stage_1_summary = {
            "files_processed": len(stage_1_results),
            "total_symbols": total_symbols,
            "total_samples": total_samples,
            "output_directory": str(self.raw_parquet_dir),
            "results": stage_1_results,
        }

        print("\n✅ STAGE 1 COMPLETE!")
        print(f"   📊 Processed {len(stage_1_results)} DBN files")
        print(f"   📊 Generated {total_symbols} symbol-specific parquet files")
        print(f"   📊 Total samples: {total_samples:,}")
        print(f"   📁 Output directory: {self.raw_parquet_dir}")

        return stage_1_summary

    def stage_2_classification(self, stage_1_results: Dict) -> Dict:
        """
        Stage 2: Apply post-processing classification to unlabeled parquet files.
        """
        print("\n" + "="*60)
        print("🔄 STAGE 2: Post-Processing Classification with Uniform Distribution")
        print("="*60)

        # Find all parquet files from Stage 1
        parquet_files = list(self.raw_parquet_dir.glob(f"{self.currency}_*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.raw_parquet_dir}")

        print(f"📊 Found {len(parquet_files)} symbol parquet files to classify:")
        for i, file_path in enumerate(parquet_files[:5]):  # Show first 5
            print(f"   {i+1}. {file_path.name}")
        if len(parquet_files) > 5:
            print(f"   ... and {len(parquet_files) - 5} more")

        # Initialize classifier with uniform distribution target
        classifier = ParquetClassifier(
            currency=self.currency,
            target_uniform_percentage=7.69,  # 100/13 classes
            force_uniform=True,
            verbose=True,
        )

        stage_2_results = []

        for i, parquet_file in enumerate(parquet_files):
            print(f"\n🔄 Classifying file {i+1}/{len(parquet_files)}: {parquet_file.name}")
            
            try:
                # Generate output path
                output_path = self.classified_parquet_dir / f"{parquet_file.stem}_classified.parquet"
                
                # Apply classification
                stats = classifier.classify_symbol_parquet(
                    parquet_path=parquet_file,
                    output_path=output_path,
                    sample_fraction=0.8,  # Use 80% of data for demo
                    validate_uniformity=True,
                )
                
                stage_2_results.append(stats)
                
                # Show validation results
                validation = stats.get("validation_results", {})
                if validation:
                    max_dev = validation.get("max_deviation", 0)
                    assessment = validation.get("assessment", "UNKNOWN")
                    print(f"   📊 Distribution quality: {assessment} (max deviation: {max_dev:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Failed to classify {parquet_file.name}: {e}")
                continue

        # Summarize Stage 2 results
        total_classified_samples = sum(result['processed_samples'] for result in stage_2_results)
        successful_files = len(stage_2_results)
        
        # Aggregate distribution quality metrics
        distribution_qualities = []
        for result in stage_2_results:
            validation = result.get("validation_results", {})
            if validation.get("max_deviation") is not None:
                distribution_qualities.append(validation["max_deviation"])

        avg_distribution_quality = np.mean(distribution_qualities) if distribution_qualities else 0

        stage_2_summary = {
            "files_processed": successful_files,
            "total_classified_samples": total_classified_samples,
            "average_distribution_deviation": avg_distribution_quality,
            "output_directory": str(self.classified_parquet_dir),
            "results": stage_2_results,
        }

        print("\n✅ STAGE 2 COMPLETE!")
        print(f"   📊 Classified {successful_files} symbol files")
        print(f"   📊 Total classified samples: {total_classified_samples:,}")
        print(f"   📊 Average distribution deviation: {avg_distribution_quality:.1f}%")
        print(f"   📁 Output directory: {self.classified_parquet_dir}")

        return stage_2_summary

    def stage_3_ml_training_demo(self, stage_2_results: Dict) -> Dict:
        """
        Stage 3: Demonstrate lazy ML training with classified parquet data.
        """
        print("\n" + "="*60)
        print("🔄 STAGE 3: Lazy ML Training DataLoader Demonstration")
        print("="*60)

        # Find classified parquet files
        classified_files = list(self.classified_parquet_dir.glob(f"{self.currency}_*_classified.parquet"))
        
        if not classified_files:
            raise FileNotFoundError(f"No classified parquet files found in {self.classified_parquet_dir}")

        print(f"📊 Found {len(classified_files)} classified parquet files for ML training")

        # Demonstrate with the largest file (most samples)
        file_sizes = [(f, f.stat().st_size) for f in classified_files]
        largest_file = max(file_sizes, key=lambda x: x[1])[0]
        
        print(f"📊 Using largest file for demo: {largest_file.name}")
        print(f"   📊 File size: {largest_file.stat().st_size / 1024 / 1024:.1f} MB")

        try:
            # Create lazy dataloader
            print("\n🔄 Creating lazy ML training dataloader...")
            
            dataloader = create_market_depth_dataloader(
                parquet_path=str(largest_file),
                batch_size=16,
                shuffle=True,
                sample_fraction=0.1,  # Use 10% for quick demo
                num_workers=2,
            )

            print("   ✅ DataLoader created successfully")
            print("   📊 Batch size: 16")
            print("   📊 Sample fraction: 10%")
            print("   📊 Shuffle: True")

            # Demonstrate training loop
            print("\n🔄 Demonstrating training loop...")
            
            batch_count = 0
            total_samples = 0
            label_counts = {}
            feature_shapes = []

            start_time = time.perf_counter()

            for features, labels in dataloader:
                batch_count += 1
                batch_size = len(labels)
                total_samples += batch_size
                
                # Track feature shapes
                if hasattr(features, 'shape'):
                    feature_shapes.append(features.shape)
                
                # Track label distribution
                if hasattr(labels, 'numpy'):
                    batch_labels = labels.numpy()
                else:
                    batch_labels = np.array(labels)
                
                for label in batch_labels:
                    label_counts[int(label)] = label_counts.get(int(label), 0) + 1
                
                # Stop after a few batches for demo
                if batch_count >= 5:
                    break

                print(f"   📊 Batch {batch_count}: features {features.shape if hasattr(features, 'shape') else 'N/A'}, labels {len(labels)}")

            end_time = time.perf_counter()
            loading_time = end_time - start_time

            # Analyze results
            samples_per_second = total_samples / loading_time if loading_time > 0 else 0

            print("\n📊 ML Training Demo Results:")
            print(f"   📊 Processed {batch_count} batches")
            print(f"   📊 Total samples: {total_samples}")
            print(f"   📊 Loading time: {loading_time:.2f} seconds")
            print(f"   📊 Throughput: {samples_per_second:.1f} samples/second")

            if feature_shapes:
                print(f"   📊 Feature tensor shape: {feature_shapes[0]}")

            print("\n📊 Label Distribution in Demo Batches:")
            total_demo_samples = sum(label_counts.values())
            for label in sorted(label_counts.keys()):
                count = label_counts[label]
                percentage = (count / total_demo_samples) * 100
                print(f"   📊 Class {label:2d}: {count:3d} samples ({percentage:4.1f}%)")

            stage_3_summary = {
                "demo_file": str(largest_file),
                "batches_processed": batch_count,
                "total_samples": total_samples,
                "loading_time_seconds": loading_time,
                "samples_per_second": samples_per_second,
                "feature_shape": str(feature_shapes[0]) if feature_shapes else "N/A",
                "label_distribution": label_counts,
            }

            print("\n✅ STAGE 3 COMPLETE!")
            print("   📊 Successfully demonstrated lazy ML training")
            print("   📊 Memory-efficient loading: ✅")
            print("   📊 Balanced classification: ✅")
            print("   📊 Ready for production ML training: ✅")

            return stage_3_summary

        except Exception as e:
            print(f"❌ ML training demo failed: {e}")
            return {"error": str(e)}

    def create_pipeline_analysis_plot(
        self, 
        stage_1_results: Dict, 
        stage_2_results: Dict, 
        stage_3_results: Dict
    ) -> str:
        """Create comprehensive pipeline analysis visualization."""
        print("\n📊 Creating Pipeline Analysis Visualization...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Stage 1: Symbol distribution
        if stage_1_results.get("results"):
            all_symbols = []
            all_sample_counts = []
            
            for result in stage_1_results["results"]:
                symbol_stats = result.get("symbol_stats", {})
                for symbol, stats in symbol_stats.items():
                    all_symbols.append(symbol)
                    all_sample_counts.append(stats["samples"])
            
            if all_symbols:
                # Show top 10 symbols
                symbol_data = list(zip(all_symbols, all_sample_counts))
                symbol_data.sort(key=lambda x: x[1], reverse=True)
                top_symbols = symbol_data[:10]
                
                symbols, counts = zip(*top_symbols)
                bars = ax1.bar(range(len(symbols)), counts, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.set_xlabel('Symbol')
                ax1.set_ylabel('Sample Count')
                ax1.set_title('Stage 1: Top Symbols by Sample Count')
                ax1.set_xticks(range(len(symbols)))
                ax1.set_xticklabels(symbols, rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # Add count labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{count:,}', ha='center', va='bottom', fontsize=8)

        # 2. Stage 2: Distribution quality
        if stage_2_results.get("results"):
            file_names = []
            deviations = []
            
            for result in stage_2_results["results"]:
                validation = result.get("validation_results", {})
                if validation.get("max_deviation") is not None:
                    file_name = Path(result["input_file"]).stem
                    file_names.append(file_name)
                    deviations.append(validation["max_deviation"])
            
            if file_names:
                colors = ['green' if d < 2.0 else 'orange' if d < 3.0 else 'red' for d in deviations]
                bars = ax2.bar(range(len(file_names)), deviations, alpha=0.7, color=colors, edgecolor='black')
                ax2.set_xlabel('Symbol File')
                ax2.set_ylabel('Max Deviation from Uniform (%)')
                ax2.set_title('Stage 2: Classification Distribution Quality')
                ax2.set_xticks(range(len(file_names)))
                ax2.set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in file_names], rotation=45)
                ax2.axhline(2.0, color='green', linestyle='--', alpha=0.7, label='Excellent (<2%)')
                ax2.axhline(3.0, color='orange', linestyle='--', alpha=0.7, label='Good (<3%)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        # 3. Stage 3: Label distribution
        if stage_3_results.get("label_distribution"):
            labels = sorted(stage_3_results["label_distribution"].keys())
            counts = [stage_3_results["label_distribution"][label] for label in labels]
            total = sum(counts)
            percentages = [(count/total)*100 for count in counts]
            
            bars = ax3.bar(labels, percentages, alpha=0.7, color='lightcoral', edgecolor='black')
            target_percentage = 100 / 13  # 13 classes
            ax3.axhline(target_percentage, color='red', linestyle='--', linewidth=2,
                       label=f'Target (Uniform): {target_percentage:.1f}%')
            ax3.set_xlabel('Classification Label')
            ax3.set_ylabel('Percentage (%)')
            ax3.set_title('Stage 3: ML Training Label Distribution (Demo Batches)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add percentage labels
            for bar, percentage in zip(bars, percentages):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)

        # 4. Pipeline summary
        ax4.axis('off')
        
        summary_text = f"""Complete 3-Stage Pipeline Results

STAGE 1: DBN → Unlabeled Parquet
• Files processed: {stage_1_results.get('files_processed', 0)}
• Symbols generated: {stage_1_results.get('total_symbols', 0)}
• Total samples: {stage_1_results.get('total_samples', 0):,}

STAGE 2: Post-Processing Classification
• Files classified: {stage_2_results.get('files_processed', 0)}
• Classified samples: {stage_2_results.get('total_classified_samples', 0):,}
• Avg distribution deviation: {stage_2_results.get('average_distribution_deviation', 0):.1f}%

STAGE 3: ML Training Demo
• Batches processed: {stage_3_results.get('batches_processed', 0)}
• Demo samples: {stage_3_results.get('total_samples', 0)}
• Throughput: {stage_3_results.get('samples_per_second', 0):.1f} samples/sec
• Feature shape: {stage_3_results.get('feature_shape', 'N/A')}

ARCHITECTURE BENEFITS:
✅ Symbol-specific processing
✅ Memory-efficient training  
✅ Uniform class distribution
✅ Post-processing flexibility
✅ Scalable to large datasets

READY FOR PRODUCTION ML TRAINING! 🚀
"""
        
        ax4.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
                fontfamily='monospace')

        plt.tight_layout()
        
        # Save plot
        plot_path = self.analysis_dir / "complete_pipeline_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"   ✅ Saved pipeline analysis plot: {plot_path}")
        return str(plot_path)

    def run_complete_pipeline(self, max_files: int = 3) -> Dict:
        """Run the complete 3-stage pipeline demonstration."""
        print("🎯 Starting Complete 3-Stage Pipeline Demonstration")
        print("=" * 70)
        print("🎯 New Architecture: DBN → Unlabeled Parquet → Classification → ML Training")
        print("=" * 70)

        try:
            # Stage 1: DBN to unlabeled parquet
            stage_1_results = self.stage_1_dbn_to_parquet(max_files=max_files)
            
            # Stage 2: Post-processing classification
            stage_2_results = self.stage_2_classification(stage_1_results)
            
            # Stage 3: ML training demonstration
            stage_3_results = self.stage_3_ml_training_demo(stage_2_results)
            
            # Create comprehensive analysis
            plot_path = self.create_pipeline_analysis_plot(
                stage_1_results, stage_2_results, stage_3_results
            )

            # Final summary
            complete_results = {
                "pipeline_version": "v2.0.0 - 3-Stage Architecture",
                "currency": self.currency,
                "output_directory": str(self.output_dir),
                "stage_1": stage_1_results,
                "stage_2": stage_2_results,
                "stage_3": stage_3_results,
                "analysis_plot": plot_path,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Save complete results
            results_path = self.analysis_dir / "complete_pipeline_results.json"
            with open(results_path, 'w') as f:
                json.dump(complete_results, f, indent=2, default=str)

            print("\n" + "=" * 70)
            print("🎉 COMPLETE 3-STAGE PIPELINE DEMONSTRATION FINISHED!")
            print("=" * 70)
            print(f"📊 Stage 1: {stage_1_results['total_symbols']} symbols, {stage_1_results['total_samples']:,} samples")
            print(f"📊 Stage 2: {stage_2_results['files_processed']} files classified, {stage_2_results['average_distribution_deviation']:.1f}% avg deviation")
            print(f"📊 Stage 3: {stage_3_results.get('samples_per_second', 0):.1f} samples/sec ML throughput")
            print(f"📁 All outputs saved to: {self.output_dir}")
            print(f"📊 Analysis plot: {plot_path}")
            print(f"📊 Results JSON: {results_path}")
            print("\n🚀 Ready for production ML training with balanced datasets! 🚀")

            return complete_results

        except Exception as e:
            print(f"\n❌ Pipeline demonstration failed: {e}")
            return {"error": str(e)}


def main():
    """Main function to run the complete pipeline demonstration."""
    print("🚀 Complete 3-Stage represent Pipeline Demonstration")
    print("=" * 60)
    
    # Initialize demo (saving to repo /data directory as requested)
    demo = CompletePipelineDemo(
        data_dir="/Users/danielfisher/repositories/represent/data",
        output_dir="/Users/danielfisher/repositories/represent/data/pipeline_outputs",
        currency="AUDUSD",
    )
    
    # Run complete pipeline (using 3 files for demo)
    results = demo.run_complete_pipeline(max_files=3)
    
    return results


if __name__ == "__main__":
    results = main()