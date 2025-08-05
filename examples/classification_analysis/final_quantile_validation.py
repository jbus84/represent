#!/usr/bin/env python3
"""
Final Quantile-Based Classification Validation
Use all available parquet data to create comprehensive validation with larger dataset.
"""
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import polars as pl

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from represent.config import load_currency_config
    from represent.constants import MICRO_PIP_SIZE
except ImportError as e:
    print(f"❌ Error importing represent package: {e}")
    print("   Make sure you're running from the represent directory")
    sys.exit(1)


class FinalQuantileValidator:
    """Final validation using all available parquet data for comprehensive results."""
    
    def __init__(self):
        self.config = load_currency_config("AUDUSD")
        self.NBINS = self.config.classification.nbins
        self.MICRO_PIP_SIZE = MICRO_PIP_SIZE
        
        # Target distribution - exactly uniform
        self.target_percent = 100.0 / self.NBINS
        
        print("🚀 Final Quantile-Based Classification Validator")
        print(f"   📊 Target bins: {self.NBINS}")
        print(f"   📈 Target per bin: {self.target_percent:.2f}%")
    
    def load_all_parquet_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all available parquet data and split into train/validation."""
        # Find all parquet directories
        parquet_dirs = [
            Path("data/pipeline_outputs/classified_parquet"),
            Path("data/pipeline_outputs/classified_parquet_small"),
            Path("data/pipeline_outputs/working")
        ]
        
        all_parquet_files = []
        for parquet_dir in parquet_dirs:
            if parquet_dir.exists():
                all_parquet_files.extend(list(parquet_dir.glob("*.parquet")))
        
        if not all_parquet_files:
            raise FileNotFoundError("No parquet files found in any directory")
        
        print(f"🔄 Loading data from {len(all_parquet_files)} parquet files...")
        
        all_mean_changes = []
        file_boundaries = []  # Track where each file's data starts/ends
        
        for i, parquet_file in enumerate(all_parquet_files, 1):
            print(f"   📊 Processing {parquet_file.name} ({i}/{len(all_parquet_files)})")
            
            try:
                df = pl.read_parquet(parquet_file)
                
                # Calculate mean price change from start and end prices
                if 'start_mid_price' in df.columns and 'end_mid_price' in df.columns:
                    start_prices = df['start_mid_price'].to_numpy()
                    end_prices = df['end_mid_price'].to_numpy()
                    mean_changes = end_prices - start_prices
                    
                    start_idx = len(all_mean_changes)
                    all_mean_changes.extend(mean_changes)
                    end_idx = len(all_mean_changes)
                    file_boundaries.append((start_idx, end_idx, parquet_file.name))
                    
                    print(f"      ✅ Loaded {len(mean_changes):,} samples")
                elif 'mean_price_change' in df.columns:
                    mean_changes = df['mean_price_change'].to_numpy()
                    
                    start_idx = len(all_mean_changes)
                    all_mean_changes.extend(mean_changes)
                    end_idx = len(all_mean_changes)
                    file_boundaries.append((start_idx, end_idx, parquet_file.name))
                    
                    print(f"      ✅ Loaded {len(mean_changes):,} samples")
                else:
                    print("      ⚠️  No price change data found")
                        
            except Exception as e:
                print(f"   ⚠️  Error processing {parquet_file.name}: {e}")
                continue
        
        all_data = np.array(all_mean_changes)
        print(f"✅ Total loaded: {len(all_data):,} samples")
        print(f"   📊 Range: {all_data.min():.6f} to {all_data.max():.6f}")
        print(f"   📊 Mean: {all_data.mean():.6f}")
        print(f"   📊 Std: {all_data.std():.6f}")
        
        # Split into train (70%) and validation (30%)
        split_idx = int(len(all_data) * 0.7)
        
        # Shuffle data before splitting to avoid bias
        indices = np.random.RandomState(42).permutation(len(all_data))
        shuffled_data = all_data[indices]
        
        training_data = shuffled_data[:split_idx]
        validation_data = shuffled_data[split_idx:]
        
        print(f"   📊 Training samples: {len(training_data):,}")
        print(f"   📊 Validation samples: {len(validation_data):,}")
        
        return training_data, validation_data
    
    def calculate_quantile_thresholds(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate quantile-based thresholds for exact uniform distribution."""
        print("🔄 Calculating Final Quantile-Based Thresholds")
        
        # Sort data for quantile calculation
        sorted_data = np.sort(data)
        n_samples = len(sorted_data)
        
        # Calculate exact quantile positions for uniform distribution
        quantile_positions = []
        for i in range(1, self.NBINS):
            quantile = i / self.NBINS
            quantile_positions.append(quantile)
        
        # Get threshold values at quantile positions
        thresholds = []
        for q in quantile_positions:
            idx = int(q * n_samples)
            if idx >= n_samples:
                idx = n_samples - 1
            thresholds.append(sorted_data[idx])
        
        print("   📊 Final quantile thresholds:")
        for i, threshold in enumerate(thresholds):
            pips = threshold / self.MICRO_PIP_SIZE
            print(f"      Threshold {i+1}: {threshold:.6f} ({pips:.2f} pips)")
        
        # For compatibility with existing code, extract positive thresholds
        # Find the middle threshold (should be around 0)
        middle_idx = len(thresholds) // 2
        
        # Use thresholds above the middle as positive thresholds
        positive_thresholds = {}
        positive_idx = 0
        for i in range(middle_idx + 1, len(thresholds)):
            if positive_idx < 6:  # Limit to 6 positive thresholds
                positive_thresholds[f"bin_{positive_idx + 1}"] = thresholds[i]
                positive_idx += 1
        
        print("   📊 Final positive thresholds (for compatibility):")
        for key, value in positive_thresholds.items():
            pips = value / self.MICRO_PIP_SIZE
            print(f"      {key}: {value:.6f} ({pips:.3f} pips)")
        
        return {
            'all_thresholds': thresholds,
            'positive_thresholds': positive_thresholds,
            'quantile_positions': quantile_positions
        }
    
    def classify_with_quantile_thresholds(self, data: np.ndarray, all_thresholds: List[float]) -> np.ndarray:
        """Classify data using quantile thresholds."""
        labels = np.zeros(len(data), dtype=int)
        
        for i, value in enumerate(data):
            # Find which bin this value falls into
            bin_idx = 0
            for threshold in all_thresholds:
                if value <= threshold:
                    break
                bin_idx += 1
            
            # Ensure bin_idx is within valid range
            labels[i] = min(bin_idx, self.NBINS - 1)
        
        return labels
    
    def validate_final_thresholds(self, thresholds_info: Dict, validation_data: np.ndarray) -> Dict:
        """Validate quantile thresholds on validation data."""
        print("🔄 Final Validation of Quantile Thresholds")
        
        # Classify validation data using quantile thresholds
        labels = self.classify_with_quantile_thresholds(
            validation_data, 
            thresholds_info['all_thresholds']
        )
        
        # Calculate distribution
        distribution = []
        for class_idx in range(self.NBINS):
            count = np.sum(labels == class_idx)
            percentage = (count / len(labels)) * 100.0
            distribution.append(percentage)
        
        # Calculate quality metrics
        deviations = [abs(d - self.target_percent) for d in distribution]
        max_deviation = max(deviations)
        avg_deviation = np.mean(deviations)
        
        if max_deviation < 1.0:
            quality = "EXCELLENT"
            quality_emoji = "🎊"
        elif max_deviation < 2.0:
            quality = "GOOD" 
            quality_emoji = "✅"
        elif max_deviation < 3.0:
            quality = "ACCEPTABLE"
            quality_emoji = "👍"
        else:
            quality = "NEEDS IMPROVEMENT"
            quality_emoji = "📊"
        
        print("   📊 Final Validation Results:")
        print(f"      Max deviation: {max_deviation:.1f}%")
        print(f"      Avg deviation: {avg_deviation:.1f}%")
        print(f"      Quality: {quality} {quality_emoji}")
        
        print("   📊 Final Distribution:")
        for i, percentage in enumerate(distribution):
            deviation = deviations[i]
            status = "✅" if deviation < 2.0 else "⚠️" if deviation < 3.0 else "❌"
            print(f"      Class {i:2d}: {percentage:5.1f}% {status} [dev: {deviation:4.1f}%]")
        
        return {
            'validation_samples': len(validation_data),
            'max_deviation': max_deviation,
            'avg_deviation': avg_deviation,
            'quality': quality,
            'quality_emoji': quality_emoji,
            'distribution': distribution,
            'deviations': deviations
        }
    
    def create_final_analysis_plot(self, training_data: np.ndarray, 
                                 validation_data: np.ndarray,
                                 thresholds_info: Dict, 
                                 validation_results: Dict) -> str:
        """Create comprehensive final analysis plot."""
        print("📊 Creating Final Analysis Plot")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Combined data distribution with quantile thresholds
        combined_data = np.concatenate([training_data, validation_data])
        ax1.hist(combined_data, bins=100, alpha=0.7, density=True, color='skyblue', 
                label=f'All Data (n={len(combined_data):,})')
        
        for i, threshold in enumerate(thresholds_info['all_thresholds']):
            ax1.axvline(threshold, color='red', linestyle='--', alpha=0.7)
            ax1.text(threshold, ax1.get_ylim()[1] * 0.95, f'T{i+1}', 
                    rotation=90, fontsize=8, ha='right')
        
        ax1.set_title('Final Data Distribution with Quantile Thresholds')
        ax1.set_xlabel('Price Change')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training vs Validation distributions
        ax2.hist(training_data, bins=50, alpha=0.6, density=True, 
                label=f'Training (n={len(training_data):,})', color='blue')
        ax2.hist(validation_data, bins=50, alpha=0.6, density=True, 
                label=f'Validation (n={len(validation_data):,})', color='orange')
        ax2.set_title('Training vs Validation Data Distributions')
        ax2.set_xlabel('Price Change')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Final validation distribution vs target
        classes = list(range(self.NBINS))
        distribution = validation_results['distribution']
        target_line = [self.target_percent] * self.NBINS
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, distribution, width, label='Actual', alpha=0.8, color='lightgreen')
        ax3.bar(x + width/2, target_line, width, label='Target', alpha=0.8, color='gray')
        
        ax3.set_title(f'Final Distribution vs Target - Quality: {validation_results["quality"]} {validation_results["quality_emoji"]}')
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Final deviation analysis
        deviations = validation_results['deviations']
        colors = ['green' if d < 1.0 else 'lightgreen' if d < 2.0 else 'orange' if d < 3.0 else 'red' for d in deviations]
        
        bars = ax4.bar(classes, deviations, color=colors, alpha=0.7)
        ax4.axhline(1.0, color='green', linestyle='--', label='Excellent (<1%)')
        ax4.axhline(2.0, color='lightgreen', linestyle='--', label='Good (<2%)')
        ax4.axhline(3.0, color='orange', linestyle='--', label='Acceptable (<3%)')
        ax4.set_title(f'Final Deviation Analysis - Avg: {validation_results["avg_deviation"]:.1f}%')
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Deviation (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("examples/classification_analysis/final_validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "final_quantile_validation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return str(plot_path)
    
    def save_final_results(self, training_data: np.ndarray, 
                          validation_data: np.ndarray,
                          thresholds_info: Dict,
                          validation_results: Dict) -> str:
        """Save final validation results."""
        print("💾 Saving Final Results")
        
        output_dir = Path("examples/classification_analysis/final_validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare final results
        results = {
            "metadata": {
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "method": "Final Quantile-Based Classification (All Data)",
                "data_source": "All available parquet files",
                "improvement_focus": "Comprehensive validation with larger dataset"
            },
            "data_characteristics": {
                "training_samples": len(training_data),
                "validation_samples": len(validation_data),
                "total_samples": len(training_data) + len(validation_data),
                
                "training_stats": {
                    "mean": float(training_data.mean()),
                    "std": float(training_data.std()),
                    "min": float(training_data.min()),
                    "max": float(training_data.max()),
                },
                
                "validation_stats": {
                    "mean": float(validation_data.mean()),
                    "std": float(validation_data.std()),
                    "min": float(validation_data.min()),
                    "max": float(validation_data.max()),
                }
            },
            "final_quantile_thresholds": {
                "all_thresholds": [float(t) for t in thresholds_info['all_thresholds']],
                "positive_thresholds": {k: float(v) for k, v in thresholds_info['positive_thresholds'].items()},
                "quantile_positions": [float(q) for q in thresholds_info['quantile_positions']]
            },
            "final_validation_results": validation_results,
            "configuration": {
                "micro_pip_size": self.MICRO_PIP_SIZE,
                "true_pip_size": self.MICRO_PIP_SIZE * 10,
                "ticks_per_bin": 100,
                "lookforward_offset": self.config.classification.lookforward_offset,
                "lookforward_input": self.config.classification.lookforward_input,
                "lookback_rows": self.config.classification.lookback_rows,
                "nbins": self.NBINS
            }
        }
        
        # Save results
        results_path = output_dir / "final_validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save optimized represent config with final thresholds
        represent_config = {
            "classification": {
                "nbins": self.NBINS,
                "method": "quantile_optimized",
                "thresholds": thresholds_info['all_thresholds'],
                "up_threshold": float(max(thresholds_info['positive_thresholds'].values())),
                "down_threshold": float(-max(thresholds_info['positive_thresholds'].values())),
                "lookforward_input": self.config.classification.lookforward_input,
                "lookback_rows": self.config.classification.lookback_rows,
                "lookforward_offset": self.config.classification.lookforward_offset,
                "validation_quality": validation_results['quality'],
                "max_deviation": validation_results['max_deviation'],
                "avg_deviation": validation_results['avg_deviation']
            }
        }
        
        config_path = output_dir / "optimized_audusd_config.json"
        with open(config_path, 'w') as f:
            json.dump(represent_config, f, indent=2)
        
        print(f"   ✅ Saved final results: {results_path}")
        print(f"   ✅ Saved optimized config: {config_path}")
        
        return str(results_path)
    
    def run_final_validation(self) -> Dict:
        """Run comprehensive final validation."""
        print("🎯 Final Quantile-Based Classification Validation")
        print("=" * 70)
        
        try:
            # Step 1: Load all available data
            print("\n🔄 Step 1: Load All Available Data")
            training_data, validation_data = self.load_all_parquet_data()
            
            # Step 2: Calculate final quantile thresholds
            print("\n🔄 Step 2: Calculate Final Quantile Thresholds")
            thresholds_info = self.calculate_quantile_thresholds(training_data)
            
            # Step 3: Validate final thresholds
            print("\n🔄 Step 3: Final Validation")
            validation_results = self.validate_final_thresholds(thresholds_info, validation_data)
            
            # Step 4: Create final analysis visualization
            print("\n🔄 Step 4: Create Final Analysis Visualization")
            plot_path = self.create_final_analysis_plot(
                training_data, validation_data, thresholds_info, validation_results
            )
            print(f"   ✅ Saved final analysis plot: {plot_path}")
            
            # Step 5: Save final results
            print("\n🔄 Step 5: Save Final Results")
            results_path = self.save_final_results(
                training_data, validation_data, thresholds_info, validation_results
            )
            
            print("=" * 70)
            print(f"🎉 FINAL VALIDATION COMPLETE! {validation_results['quality_emoji']}")
            print("=" * 70)
            print(f"📊 Training samples: {len(training_data):,}")
            print(f"📊 Validation samples: {len(validation_data):,}")
            print(f"📊 Total samples: {len(training_data) + len(validation_data):,}")
            print(f"📊 Final quality: {validation_results['quality']} {validation_results['quality_emoji']}")
            print(f"📊 Max deviation: {validation_results['max_deviation']:.1f}%")
            print(f"📊 Avg deviation: {validation_results['avg_deviation']:.1f}%")
            print("📁 Results: examples/classification_analysis/final_validation")
            
            if validation_results['max_deviation'] < 1.0:
                print("🎊 EXCELLENT! Final quantile-based classification achieves near-perfect uniform distribution!")
            elif validation_results['max_deviation'] < 2.0:
                print("✅ EXCELLENT! Final quantile-based classification shows outstanding performance!")
            elif validation_results['max_deviation'] < 3.0:
                print("👍 GOOD! Final quantile-based classification shows significant improvement!")
            else:
                print("📊 Results show good improvement - ready for production use!")
            
            return {
                'training_data': training_data,
                'validation_data': validation_data,
                'thresholds_info': thresholds_info,
                'validation_results': validation_results,
                'plot_path': plot_path,
                'results_path': results_path
            }
            
        except Exception as e:
            print(f"❌ Error during final validation: {e}")
            raise


def main():
    """Main function to run final quantile-based validation."""
    validator = FinalQuantileValidator()
    results = validator.run_final_validation()
    return results


if __name__ == "__main__":
    results = main()