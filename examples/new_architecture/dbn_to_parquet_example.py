#!/usr/bin/env python3
"""
DBN to Parquet Conversion Example

This script demonstrates the new represent architecture:
1. Convert DBN files to labeled parquet datasets
2. Use lazy loading parquet dataloader for ML training
3. Pre-computed classification labels for efficient training

Features demonstrated:
- Currency-specific classification configuration
- High-performance DBN to parquet conversion
- Automatic labeling based on price movements
- Memory-efficient lazy loading from parquet
- PyTorch-compatible training workflows
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from represent.converter import convert_dbn_file, batch_convert_dbn_files
from represent.dataloader import create_market_depth_dataloader


def demonstrate_dbn_conversion():
    """Demonstrate DBN to parquet conversion with labeling."""
    print("=" * 70)
    print("🔄 DBN TO PARQUET CONVERSION DEMONSTRATION")
    print("=" * 70)
    
    # Check if we have DBN data available
    data_dir = Path("data")
    dbn_files = list(data_dir.glob("*.dbn*")) if data_dir.exists() else []
    
    if not dbn_files:
        print("❌ No DBN files found in data/ directory")
        print("Please ensure you have DBN files available for conversion")
        return None
    
    print(f"📊 Found {len(dbn_files)} DBN files for conversion")
    
    # Convert first DBN file for demonstration
    input_file = dbn_files[0]
    output_file = Path("data") / f"{input_file.stem}_labeled.parquet"
    
    print(f"\n1. Converting {input_file.name} to labeled parquet...")
    
    try:
        stats = convert_dbn_file(
            dbn_path=input_file,
            output_path=output_file,
            currency='AUDUSD',  # Use AUDUSD configuration
            symbol_filter='M6AM4',  # Filter to specific symbol
            features=['volume', 'variance', 'trade_counts'],  # Include all features
            chunk_size=50000  # Process in 50K row chunks
        )
        
        print("\n📊 CONVERSION STATISTICS:")
        print(f"   Original rows: {stats['original_rows']:,}")
        print(f"   Labeled samples: {stats['labeled_samples']:,}")
        print(f"   Conversion time: {stats['conversion_time_seconds']:.1f}s")
        print(f"   Processing rate: {stats['samples_per_second']:.1f} samples/sec")
        print(f"   Output file size: {stats['output_file_size_mb']:.1f}MB")
        print(f"   Features: {stats['features']}")
        print(f"   Currency config: {stats['currency']}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None


def demonstrate_lazy_dataloader(parquet_file: Path):
    """Demonstrate lazy loading parquet dataloader."""
    print("\n" + "=" * 70)
    print("📚 LAZY PARQUET DATALOADER DEMONSTRATION")
    print("=" * 70)
    
    if not parquet_file or not parquet_file.exists():
        print("❌ No parquet file available for dataloader demonstration")
        return None
    
    print(f"1. Creating lazy dataloader for {parquet_file.name}...")
    
    try:
        # Create dataloader with lazy loading
        dataloader = create_market_depth_dataloader(
            parquet_path=parquet_file,
            batch_size=16,
            shuffle=True,
            num_workers=0,  # Single process for demo
            sample_fraction=0.1  # Use 10% of data for quick demo
        )
        
        # Get dataset information
        dataset_info = dataloader.get_dataset_info()
        
        print("✅ Dataloader created successfully!")
        print("\n📊 DATASET INFORMATION:")
        print(f"   File: {dataset_info['parquet_file']}")
        print(f"   File size: {dataset_info['file_size_mb']:.1f}MB")
        print(f"   Total samples: {dataset_info['total_samples']:,}")
        print(f"   Active samples: {dataset_info['active_samples']:,}")
        print(f"   Sample fraction: {dataset_info['sample_fraction']:.1%}")
        print(f"   Unique symbols: {dataset_info['unique_symbols']}")
        print(f"   Time range: {dataset_info['min_timestamp']} to {dataset_info['max_timestamp']}")
        print(f"   Label distribution: {dataset_info['label_distribution']}")
        
        return dataloader
        
    except Exception as e:
        print(f"❌ Dataloader creation failed: {e}")
        return None


def demonstrate_training_workflow(dataloader):
    """Demonstrate ML training workflow with lazy dataloader."""
    print("\n" + "=" * 70)
    print("🤖 ML TRAINING WORKFLOW DEMONSTRATION")
    print("=" * 70)
    
    if dataloader is None:
        print("❌ No dataloader available for training demonstration")
        return
    
    print("1. Setting up CNN model for market depth classification...")
    
    # Get first batch to determine tensor shape
    first_batch = next(iter(dataloader))
    features, labels = first_batch
    
    input_shape = features.shape[1:]  # Remove batch dimension
    num_classes = len(torch.unique(labels))
    
    print(f"   Input shape: {input_shape}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Batch size: {features.shape[0]}")
    
    # Create CNN model
    if len(input_shape) == 2:  # 2D input (height, width)
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    elif len(input_shape) == 3:  # 3D input (features, height, width)
        model = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    else:
        print(f"❌ Unsupported input shape: {input_shape}")
        return
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("   ✅ Model and optimizer configured")
    
    # Training loop demonstration
    print("\n2. Running training demonstration...")
    
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_times = []
    
    for batch_idx, (batch_features, batch_labels) in enumerate(dataloader):
        if batch_idx >= 5:  # Demo with just 5 batches
            break
        
        batch_start = time.perf_counter()
        
        # Add channel dimension if needed for 2D input
        if len(input_shape) == 2:
            batch_features = batch_features.unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        _, predicted = torch.max(outputs.data, 1)
        batch_correct = (predicted == batch_labels).sum().item()
        batch_samples = batch_labels.size(0)
        
        total_loss += loss.item()
        total_correct += batch_correct
        total_samples += batch_samples
        
        batch_time = (time.perf_counter() - batch_start) * 1000
        batch_times.append(batch_time)
        
        print(f"   Batch {batch_idx + 1}: Loss={loss.item():.4f}, "
              f"Acc={batch_correct/batch_samples:.3f}, Time={batch_time:.1f}ms")
    
    # Training statistics
    avg_loss = total_loss / (batch_idx + 1)
    avg_accuracy = total_correct / total_samples
    avg_batch_time = sum(batch_times) / len(batch_times)
    
    print("\n📊 TRAINING RESULTS:")
    print(f"   Average loss: {avg_loss:.4f}")
    print(f"   Average accuracy: {avg_accuracy:.3f}")
    print(f"   Average batch time: {avg_batch_time:.1f}ms")
    print(f"   Samples processed: {total_samples}")
    print(f"   Training throughput: {1000/avg_batch_time:.1f} batches/second")
    
    # Performance assessment
    if avg_batch_time < 100:
        print("   ✅ EXCELLENT - Fast training suitable for large datasets")
    elif avg_batch_time < 500:
        print("   ⚡ GOOD - Reasonable training speed")
    else:
        print("   ⚠️  SLOW - Consider optimizations for large-scale training")


def demonstrate_batch_conversion():
    """Demonstrate batch conversion of multiple DBN files."""
    print("\n" + "=" * 70)
    print("📦 BATCH DBN CONVERSION DEMONSTRATION")
    print("=" * 70)
    
    data_dir = Path("data")
    output_dir = Path("data/labeled_parquet")
    
    if not data_dir.exists():
        print("❌ No data directory found")
        return
    
    dbn_files = list(data_dir.glob("*.dbn*"))
    
    if len(dbn_files) < 2:
        print("⚠️  Need at least 2 DBN files for batch conversion demo")
        print("   Skipping batch conversion demonstration")
        return
    
    print(f"📊 Found {len(dbn_files)} DBN files for batch conversion")
    print(f"1. Converting files to {output_dir}...")
    
    try:
        results = batch_convert_dbn_files(
            input_directory=data_dir,
            output_directory=output_dir,
            currency='AUDUSD',
            symbol_filter='M6AM4',
            features=['volume'],  # Use single feature for speed
            chunk_size=25000  # Smaller chunks for demo
        )
        
        print("\n✅ Batch conversion complete!")
        print("📊 BATCH STATISTICS:")
        
        total_samples = sum(r['labeled_samples'] for r in results)
        total_time = sum(r['conversion_time_seconds'] for r in results)
        total_size = sum(r['output_file_size_mb'] for r in results)
        
        print(f"   Files processed: {len(results)}")
        print(f"   Total labeled samples: {total_samples:,}")
        print(f"   Total processing time: {total_time:.1f}s")
        print(f"   Total output size: {total_size:.1f}MB")
        print(f"   Average rate: {total_samples/total_time:.1f} samples/sec")
        
        return output_dir
        
    except Exception as e:
        print(f"❌ Batch conversion failed: {e}")
        return None


def main():
    """Main demonstration function."""
    print("🚀 NEW REPRESENT ARCHITECTURE DEMONSTRATION")
    print("=" * 80)
    print("This demonstrates the new DBN→Parquet→ML pipeline:")
    print("1. Convert DBN files to labeled parquet datasets")
    print("2. Use lazy loading for memory-efficient training")
    print("3. Pre-computed labels for fast ML workflows")
    print("=" * 80)
    
    # Step 1: DBN to Parquet conversion
    parquet_file = demonstrate_dbn_conversion()
    
    # Step 2: Lazy dataloader demonstration
    dataloader = demonstrate_lazy_dataloader(parquet_file)
    
    # Step 3: ML training workflow
    demonstrate_training_workflow(dataloader)
    
    # Step 4: Batch conversion (if multiple files available)
    demonstrate_batch_conversion()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 NEW ARCHITECTURE DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("Key advantages of the new architecture:")
    print("   ✅ Pre-computed labels eliminate runtime classification overhead")
    print("   ✅ Parquet format provides optimal compression and query performance")
    print("   ✅ Lazy loading enables training on datasets larger than memory")
    print("   ✅ Currency-specific configurations optimize for different markets")
    print("   ✅ Batch processing scales to multiple files efficiently")
    print("   ✅ PyTorch-native tensors for seamless ML integration")
    print("=" * 80)
    print("\nWorkflow summary:")
    print("1. convert_dbn_file() → Creates labeled parquet dataset")
    print("2. create_market_depth_dataloader() → Lazy ML-ready dataloader")
    print("3. Standard PyTorch training loop → High-performance ML training")
    print("=" * 80)


if __name__ == "__main__":
    main()