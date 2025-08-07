#!/usr/bin/env python3
"""
Complete 3-Stage Pipeline Example (v3.0.0)

This script demonstrates the clean represent v3.0.0 architecture:
1. DBN → Unlabeled Parquet (Symbol-Grouped)
2. Dynamic Classification (Uniform Distribution) 
3. ML Training (Memory-Efficient)

Features demonstrated:
- Clean separation of concerns across 3 stages
- Dynamic classification with guaranteed uniform distribution
- Symbol-grouped processing for targeted analysis
- Memory-efficient lazy loading from parquet
- PyTorch-compatible training workflows with optimal class balance
"""

import time
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from represent import (
    convert_dbn_to_parquet,
    classify_parquet_file, 
    create_parquet_dataloader,
    RepresentAPI
)


def stage_1_dbn_to_unlabeled():
    """Stage 1: Convert DBN to unlabeled symbol-grouped parquet files."""
    print("=" * 70)
    print("🔄 STAGE 1: DBN → UNLABELED PARQUET (SYMBOL-GROUPED)")
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
    output_dir = Path("data/unlabeled")

    print(f"\n1. Converting {input_file.name} to symbol-grouped unlabeled parquet...")
    
    start_time = time.perf_counter()
    
    # Stage 1: DBN to unlabeled parquet with symbol grouping
    conversion_stats = convert_dbn_to_parquet(
        dbn_path=input_file,
        output_dir=output_dir,
        currency="AUDUSD",
        features=['volume', 'variance'],  # Multi-feature extraction
        min_symbol_samples=1000           # Only symbols with sufficient data
    )
    
    end_time = time.perf_counter()
    
    print(f"✅ Stage 1 Complete! ({end_time - start_time:.1f}s)")
    print(f"   📊 Symbols processed: {conversion_stats['symbols_processed']}")
    print(f"   📊 Total samples: {conversion_stats['total_processed_samples']:,}")
    print(f"   📊 Processing rate: {conversion_stats['samples_per_second']:.1f} samples/sec")
    print(f"   📁 Output directory: {output_dir}")
    
    # List generated files
    parquet_files = list(output_dir.glob("*.parquet"))
    print(f"   📊 Generated {len(parquet_files)} symbol files:")
    for pf in parquet_files[:3]:  # Show first 3
        size_mb = pf.stat().st_size / 1024 / 1024
        print(f"      - {pf.name} ({size_mb:.1f}MB)")
    if len(parquet_files) > 3:
        print(f"      ... and {len(parquet_files) - 3} more files")
    
    return parquet_files[0] if parquet_files else None


def stage_2_dynamic_classification(unlabeled_file):
    """Stage 2: Apply dynamic classification with uniform distribution."""
    print("\n" + "=" * 70)
    print("⚡ STAGE 2: DYNAMIC CLASSIFICATION (UNIFORM DISTRIBUTION)")
    print("=" * 70)
    
    if not unlabeled_file:
        print("❌ No unlabeled file from Stage 1")
        return None
    
    # Output classified file
    classified_dir = Path("data/classified")
    classified_dir.mkdir(exist_ok=True)
    classified_file = classified_dir / f"{unlabeled_file.stem}_classified.parquet"
    
    print(f"📊 Applying dynamic classification to: {unlabeled_file.name}")
    
    start_time = time.perf_counter()
    
    # Stage 2: Dynamic classification with guaranteed uniform distribution
    classification_stats = classify_parquet_file(
        parquet_path=unlabeled_file,
        output_path=classified_file,
        currency="AUDUSD",
        force_uniform=True  # Guaranteed uniform distribution
    )
    
    end_time = time.perf_counter()
    
    print(f"✅ Stage 2 Complete! ({end_time - start_time:.1f}s)")
    print(f"   🎯 Classification quality: {classification_stats.get('uniform_quality', 'High')}")
    print("   📊 Distribution: Uniform (7.69% per class)")
    print("   📊 Classes: 0-12 (13-bin classification)")
    print(f"   📁 Output file: {classified_file}")
    
    # Show file size
    size_mb = classified_file.stat().st_size / 1024 / 1024
    print(f"   📊 File size: {size_mb:.1f}MB")
    
    return classified_file


def stage_3_ml_training(classified_file):
    """Stage 3: Memory-efficient ML training with optimal class balance."""
    print("\n" + "=" * 70)
    print("🧠 STAGE 3: ML TRAINING (MEMORY-EFFICIENT)")
    print("=" * 70)
    
    if not classified_file:
        print("❌ No classified file from Stage 2")
        return
    
    print(f"🔄 Creating ML training dataloader from: {classified_file.name}")
    
    # Stage 3: Create memory-efficient dataloader
    dataloader = create_parquet_dataloader(
        parquet_path=classified_file,
        batch_size=32,
        shuffle=True,
        sample_fraction=0.2,  # Use 20% for quick demo
        cache_size=1000       # Optimize for memory
    )
    
    print("✅ Dataloader created with guaranteed uniform distribution")
    print("   🎯 Batch size: 32")
    print("   🔀 Shuffled: Yes")
    print("   📊 Sample fraction: 20% (for quick demo)")
    print("   💾 Cache size: 1000 samples")
    
    # Create simple CNN model for demo
    print("\n🏗️  Creating PyTorch model...")
    model = nn.Sequential(
        nn.Conv2d(2, 32, 3),                   # 2 features: volume + variance
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 13)                      # 13-class uniform classification
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("   🧠 Model: CNN (2 features → 13 classes)")
    print("   🎯 Optimizer: Adam")
    print("   📊 Loss: CrossEntropyLoss")
    
    # Demonstrate training loop
    print("\n🔄 Running demo training loop...")
    
    model.train()
    total_batches = 0
    total_loss = 0.0
    
    start_time = time.perf_counter()
    
    for epoch in range(2):  # Quick demo with 2 epochs
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (features, labels) in enumerate(dataloader):
            # features: (32, 2, 402, 500) for volume+variance
            # labels: (32,) with uniform distribution (7.69% each class 0-12)
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            total_batches += 1
            
            if batch_idx == 0:  # Show first batch info
                print(f"   📊 Epoch {epoch+1}, Batch {batch_idx+1}:")
                print(f"      Features: {features.shape}")
                print(f"      Labels: {labels.shape}")
                print(f"      Loss: {loss.item():.4f}")
            
            if batch_idx >= 4:  # Limit batches for demo
                break
        
        avg_epoch_loss = epoch_loss / batch_count
        total_loss += epoch_loss
        print(f"   ✅ Epoch {epoch+1} complete - Avg loss: {avg_epoch_loss:.4f}")
    
    end_time = time.perf_counter()
    training_time = end_time - start_time
    
    avg_loss = total_loss / total_batches
    samples_per_sec = (total_batches * 32) / training_time
    
    print("\n✅ Training Demo Complete!")
    print(f"   ⏱️  Training time: {training_time:.2f}s")
    print(f"   📊 Total batches: {total_batches}")
    print(f"   📊 Average loss: {avg_loss:.4f}")
    print(f"   ⚡ Throughput: {samples_per_sec:.1f} samples/sec")
    print("   🎯 Class balance: Guaranteed uniform (optimal for ML)")


def demonstrate_high_level_api():
    """Demonstrate high-level API for complete pipeline."""
    print("\n" + "=" * 70)
    print("🚀 HIGH-LEVEL API: COMPLETE PIPELINE")
    print("=" * 70)
    
    # Use RepresentAPI for complete workflow
    api = RepresentAPI()
    
    print("📦 Package Information:")
    info = api.get_package_info()
    print(f"   📦 Version: {info['version']}")
    print(f"   🏗️  Architecture: {info['architecture']}")
    print(f"   🎯 Features: {info['supported_features']}")
    
    print("\n🔄 High-level API can run complete pipeline in one call:")
    print("   api.run_complete_pipeline(")
    print("       dbn_path='data.dbn',")
    print("       output_base_dir='/data/pipeline/',")
    print("       currency='AUDUSD',")
    print("       features=['volume', 'variance'],")
    print("       force_uniform=True")
    print("   )")
    
    print("\n💡 This automatically runs all 3 stages:")
    print("   1. DBN → Unlabeled Parquet (Symbol-Grouped)")
    print("   2. Dynamic Classification (Uniform Distribution)")
    print("   3. ML-Ready Classified Data")


def main():
    """Run complete 3-stage pipeline demonstration."""
    print("🎉 REPRESENT v3.0.0 - COMPLETE 3-STAGE PIPELINE DEMO")
    print("=" * 70)
    
    # Stage 1: DBN to unlabeled parquet
    unlabeled_file = stage_1_dbn_to_unlabeled()
    
    if unlabeled_file:
        # Stage 2: Dynamic classification
        classified_file = stage_2_dynamic_classification(unlabeled_file)
        
        if classified_file:
            # Stage 3: ML training
            stage_3_ml_training(classified_file)
    
    # Show high-level API
    demonstrate_high_level_api()
    
    print("\n" + "=" * 70)
    print("🎉 COMPLETE 3-STAGE PIPELINE DEMONSTRATION FINISHED!")
    print("=" * 70)
    print("🎯 Key Benefits of v3.0.0:")
    print("   ✅ Clean separation of concerns")
    print("   ✅ Guaranteed uniform class distribution")
    print("   ✅ Symbol-specific processing")
    print("   ✅ Dynamic configuration (no static files)")
    print("   ✅ Memory-efficient lazy loading")
    print("   ✅ Optimal for ML training")


if __name__ == "__main__":
    main()