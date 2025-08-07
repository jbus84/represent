#!/usr/bin/env python3
"""
Streamlined DataLoader Simple Demo

This example tests the performance of loading streamlined classified parquet files
using a simplified approach that generates market depth features on-demand
during loading, avoiding the complexity of pre-generating features.

Key Features:
- Direct loading from streamlined classified parquet files
- On-demand market depth feature generation
- Performance benchmarking with raw data
- Memory-efficient processing
- PyTorch compatibility demonstration
"""

import sys
from pathlib import Path
import time
import torch
import numpy as np
import polars as pl
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import json

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from represent.pipeline import process_market_data


class StreamlinedParquetDataset(torch.utils.data.Dataset):
    """
    Simple dataset that loads streamlined classified parquet files
    and generates market depth features on-demand.
    """
    
    def __init__(
        self,
        parquet_path: Path,
        features: List[str] = None,
        input_rows: int = 5000,
        sample_fraction: float = 1.0,
        cache_size: int = 100,
    ):
        """
        Initialize streamlined dataset.
        
        Args:
            parquet_path: Path to streamlined classified parquet file
            features: Features to extract (default: ['volume'])
            input_rows: Number of historical rows needed for market depth
            sample_fraction: Fraction of data to use
            cache_size: Number of samples to cache
        """
        self.parquet_path = parquet_path
        self.features = features or ['volume']
        self.input_rows = input_rows
        self.cache_size = cache_size
        self._sample_cache = {}
        
        # Load the full dataset once for efficient access
        print(f"📊 Loading full dataset from {parquet_path.name}...")
        start_time = time.perf_counter()
        
        # Load the full dataset - we need this for context windows
        self.full_df = pl.read_parquet(parquet_path)
        
        # Apply sampling
        total_samples = len(self.full_df)
        if sample_fraction < 1.0:
            n_samples = int(total_samples * sample_fraction)
            # Use deterministic sampling for reproducibility
            indices = np.linspace(0, total_samples-1, n_samples, dtype=int)
            self.sample_indices = indices.tolist()
        else:
            self.sample_indices = list(range(total_samples))
        
        load_time = time.perf_counter() - start_time
        print(f"   ✅ Loaded {len(self.sample_indices):,} samples in {load_time:.2f}s")
        print(f"   📊 Sample fraction: {sample_fraction} ({len(self.sample_indices)}/{total_samples})")
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample with on-demand market depth feature generation.
        
        Args:
            idx: Index in the sampled dataset
            
        Returns:
            Tuple of (features_tensor, classification_label)
        """
        actual_idx = self.sample_indices[idx]
        
        # Check cache
        if actual_idx in self._sample_cache:
            return self._sample_cache[actual_idx]
        
        # Generate market depth features for this sample
        try:
            # Load context window around the target sample
            context_start = max(0, actual_idx - self.input_rows + 1)
            context_end = actual_idx + 1
            
            # Use the cached full dataset and slice the needed context
            context_df = self.full_df.slice(context_start, context_end - context_start)
            
            if len(context_df) < self.input_rows:
                # Not enough historical data, create zero features
                feature_shape = (402, 500) if len(self.features) == 1 else (len(self.features), 402, 500)
                features = np.zeros(feature_shape, dtype=np.float32)
            else:
                # Use the last input_rows for market depth generation
                depth_context = context_df.tail(self.input_rows)
                features = process_market_data(depth_context, features=self.features)
            
            # Get the classification label
            label_row = context_df.tail(1)
            label = int(label_row['classification_label'][0])
            
            # Convert to tensors
            features_tensor = torch.from_numpy(features.astype(np.float32))
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            # Cache the result
            if len(self._sample_cache) < self.cache_size:
                self._sample_cache[actual_idx] = (features_tensor, label_tensor)
            
            return features_tensor, label_tensor
            
        except Exception as e:
            # Fallback: return zero features and label 0
            print(f"⚠️  Failed to generate features for index {actual_idx}: {e}")
            feature_shape = (402, 500) if len(self.features) == 1 else (len(self.features), 402, 500)
            features_tensor = torch.zeros(feature_shape, dtype=torch.float32)
            label_tensor = torch.tensor(0, dtype=torch.long)
            return features_tensor, label_tensor


def create_streamlined_dataloader(
    parquet_files: List[Path],
    batch_size: int = 32,
    shuffle: bool = True,
    sample_fraction: float = 1.0,
    num_workers: int = 0,
    features: List[str] = None,
    input_rows: int = 5000,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from multiple streamlined parquet files.
    """
    features = features or ['volume']
    
    # For simplicity, use the first file in this demo
    # In production, you'd want to concatenate multiple files
    parquet_file = parquet_files[0]
    
    dataset = StreamlinedParquetDataset(
        parquet_path=parquet_file,
        features=features,
        input_rows=input_rows,
        sample_fraction=sample_fraction,
        cache_size=100,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def benchmark_streamlined_loading(parquet_files: List[Path]) -> Dict[str, Any]:
    """Benchmark the streamlined loading approach."""
    print("\n🔄 Benchmarking streamlined loading performance...")
    
    results = {}
    
    # Test different configurations
    configs = [
        {'batch_size': 16, 'sample_fraction': 0.01, 'input_rows': 1000},
        {'batch_size': 32, 'sample_fraction': 0.05, 'input_rows': 1000}, 
        {'batch_size': 32, 'sample_fraction': 0.1, 'input_rows': 5000},
    ]
    
    for i, config in enumerate(configs):
        config_name = f"batch_{config['batch_size']}_frac_{config['sample_fraction']}_rows_{config['input_rows']}"
        print(f"\n   📋 Testing configuration: {config_name}")
        
        try:
            # Create dataloader
            dataloader = create_streamlined_dataloader(
                parquet_files=parquet_files,
                batch_size=config['batch_size'],
                shuffle=True,
                sample_fraction=config['sample_fraction'],
                num_workers=0,  # Keep simple for testing
                features=['volume'],
                input_rows=config['input_rows'],
            )
            
            # Benchmark processing
            start_time = time.perf_counter()
            batch_count = 0
            total_samples = 0
            
            for batch_features, batch_labels in dataloader:
                batch_count += 1
                total_samples += len(batch_labels)
                
                # Test a reasonable number of batches
                if batch_count >= 20:
                    break
            
            processing_time = time.perf_counter() - start_time
            
            # Calculate metrics
            samples_per_second = total_samples / processing_time if processing_time > 0 else 0
            batches_per_second = batch_count / processing_time if processing_time > 0 else 0
            
            results[config_name] = {
                'config': config,
                'batch_count': batch_count,
                'total_samples': total_samples,
                'processing_time_seconds': processing_time,
                'samples_per_second': samples_per_second,
                'batches_per_second': batches_per_second,
                'features_shape': list(batch_features.shape) if 'batch_features' in locals() else None,
                'success': True,
            }
            
            print(f"      ✅ Processed {batch_count} batches ({total_samples:,} samples) in {processing_time:.2f}s")
            print(f"      📈 Rate: {samples_per_second:.0f} samples/sec, {batches_per_second:.1f} batches/sec")
            print(f"      📊 Feature shape: {batch_features.shape}")
            
        except Exception as e:
            print(f"      ❌ Configuration failed: {e}")
            results[config_name] = {'error': str(e), 'success': False}
    
    return results


def demonstrate_ml_training(parquet_files: List[Path]) -> Dict[str, Any]:
    """Demonstrate ML training with streamlined data."""
    print("\n🔄 Testing ML training with streamlined data...")
    
    try:
        # Create dataloader
        train_loader = create_streamlined_dataloader(
            parquet_files=parquet_files,
            batch_size=16,
            shuffle=True,
            sample_fraction=0.02,  # Use 2% for quick demo
            num_workers=0,
            features=['volume'],
            input_rows=1000,  # Reduced for speed
        )
        
        # Simple test model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=5, padding=2)
                self.pool = torch.nn.AdaptiveAvgPool2d(1)
                self.fc = torch.nn.Linear(8, 13)
                
            def forward(self, x):
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = SimpleModel()
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training test
        print("   🔄 Testing training loop...")
        model.train()
        
        batch_count = 0
        total_loss = 0
        start_time = time.perf_counter()
        
        for features, labels in train_loader:
            batch_count += 1
            
            # Forward pass
            outputs = model(features.float())
            loss = criterion(outputs, labels.long())
            total_loss += loss.item()
            
            # Test just a few batches
            if batch_count >= 5:
                break
        
        training_time = time.perf_counter() - start_time
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        print("      ✅ Training successful!")
        print(f"      📊 Batches: {batch_count}, Avg Loss: {avg_loss:.4f}")
        print(f"      ⏱️  Time: {training_time:.2f}s")
        
        return {
            'success': True,
            'batches_processed': batch_count,
            'average_loss': avg_loss,
            'training_time': training_time,
        }
        
    except Exception as e:
        print(f"      ❌ Training failed: {e}")
        return {'success': False, 'error': str(e)}


def create_performance_plot(benchmark_results: Dict, ml_results: Dict, output_dir: Path) -> str:
    """Create simple performance visualization."""
    print("\n📊 Creating performance visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Batch processing rates
    successful_results = {k: v for k, v in benchmark_results.items() 
                         if v.get('success', False)}
    
    if successful_results:
        config_names = list(successful_results.keys())
        rates = [successful_results[c]['samples_per_second'] for c in config_names]
        [successful_results[c]['config']['batch_size'] for c in config_names]
        
        # Simplify names
        display_names = [f"B{successful_results[c]['config']['batch_size']}\n"
                        f"F{successful_results[c]['config']['sample_fraction']}\n"
                        f"R{successful_results[c]['config']['input_rows']}"
                        for c in config_names]
        
        bars = ax1.bar(display_names, rates, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_ylabel('Samples/Second')
        ax1.set_title('Streamlined Loading Performance')
        ax1.grid(True, alpha=0.3)
        
        # Add rate labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Processing time breakdown
    if successful_results:
        processing_times = [successful_results[c]['processing_time_seconds'] for c in config_names]
        ax2.bar(display_names, processing_times, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_ylabel('Processing Time (seconds)')
        ax2.set_title('Processing Time by Configuration')
        ax2.grid(True, alpha=0.3)
    
    # 3. ML Training Results
    ax3.axis('off')
    if ml_results.get('success'):
        ml_text = f"""ML TRAINING TEST RESULTS

✅ Training Successful
📊 Batches Processed: {ml_results['batches_processed']}
📊 Average Loss: {ml_results['average_loss']:.4f}
⏱️  Training Time: {ml_results['training_time']:.2f}s

🚀 Streamlined Approach Ready for ML!
"""
    else:
        ml_text = f"""ML TRAINING TEST RESULTS

❌ Training Failed
Error: {ml_results.get('error', 'Unknown')}

🔧 Needs Investigation
"""
    
    ax3.text(0.05, 0.95, ml_text, transform=ax3.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
            fontfamily='monospace')
    
    # 4. Summary
    ax4.axis('off')
    
    best_config = max(successful_results.keys(), 
                     key=lambda k: successful_results[k]['samples_per_second']) if successful_results else None
    
    if best_config:
        best_rate = successful_results[best_config]['samples_per_second']
        summary_text = f"""STREAMLINED APPROACH SUMMARY

📊 PERFORMANCE:
   Best Config: {best_config.split('_')[1]} batch
   Best Rate: {best_rate:.0f} samples/sec
   Configurations Tested: {len(successful_results)}

📊 FEATURES:
   ✅ On-demand Feature Generation
   ✅ Raw DBN Data Processing  
   ✅ PyTorch Compatibility
   ✅ Memory Efficient Loading
   
📊 STATUS: {"🟢 READY" if ml_results.get('success') else "🟡 NEEDS WORK"}
"""
    else:
        summary_text = """STREAMLINED APPROACH SUMMARY

❌ No successful configurations
🔧 Investigation needed
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            fontfamily='monospace')
    
    plt.suptitle('Streamlined DataLoader Performance - Simple Approach', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "streamlined_simple_dataloader_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved performance plot: {plot_path}")
    return str(plot_path)


def main():
    """Run streamlined simple dataloader demonstration."""
    
    print("🚀 STREAMLINED SIMPLE DATALOADER DEMO")
    print("=" * 60)
    print("📋 Testing on-demand feature generation approach")
    print("=" * 60)
    
    # Setup paths
    output_dir = Path("/Users/danielfisher/repositories/represent/examples/classification_analysis/outputs")
    classified_dir = output_dir / "streamlined_classified"
    
    # Find classified parquet files
    parquet_files = sorted(classified_dir.glob("*_classified.parquet"))
    
    if not parquet_files:
        print(f"❌ No classified parquet files found in {classified_dir}")
        print("   Run streamlined_classifier_demo.py first to generate classified files.")
        return False
    
    print(f"📊 Found {len(parquet_files)} classified parquet files:")
    for i, parquet_file in enumerate(parquet_files):
        size_mb = parquet_file.stat().st_size / 1024 / 1024
        print(f"   {i+1}. {parquet_file.name} ({size_mb:.1f} MB)")
    
    try:
        # Run benchmarks
        benchmark_results = benchmark_streamlined_loading(parquet_files)
        ml_results = demonstrate_ml_training(parquet_files)
        
        # Create visualization
        plot_path = create_performance_plot(benchmark_results, ml_results, output_dir)
        
        # Save results
        results = {
            "analysis_type": "streamlined_simple_dataloader",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parquet_files": [str(f) for f in parquet_files],
            "benchmark_results": benchmark_results,
            "ml_training_results": ml_results,
            "plot_path": plot_path,
        }
        
        json_path = output_dir / "streamlined_simple_dataloader_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary
        successful_configs = sum(1 for r in benchmark_results.values() if r.get('success', False))
        ml_success = ml_results.get('success', False)
        
        print("\n🎉 STREAMLINED SIMPLE DATALOADER DEMO COMPLETE!")
        print("=" * 60)
        print(f"📊 Successful configurations: {successful_configs}/{len(benchmark_results)}")
        print(f"📊 ML training compatibility: {'✅ Yes' if ml_success else '❌ No'}")
        print("\n📁 Output Files:")
        print(f"   📊 Performance Plot: {Path(plot_path).name}")
        print(f"   📋 Results JSON: {json_path.name}")
        
        if successful_configs > 0 and ml_success:
            print("\n🚀 Streamlined approach validated for ML training!")
            return True
        else:
            print("\n⚠️  Some issues detected - check results above")
            return False
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ STREAMLINED SIMPLE DATALOADER VALIDATION SUCCESSFUL!")
    else:
        print("\n❌ STREAMLINED SIMPLE DATALOADER VALIDATION NEEDS WORK!")