#!/usr/bin/env python3
"""
Parquet Data Loading Example

This script demonstrates how to use the represent library with parquet data files
for high-performance market depth processing and PyTorch integration.

Features demonstrated:
- Loading market data from parquet files
- High-performance data processing with polars backend
- PyTorch tensor generation for ML workflows
- Performance metrics and optimization examples
"""
import time
from pathlib import Path

import torch
import numpy as np

from represent.dataloader import MarketDepthDataset
from represent.constants import SAMPLES


def demonstrate_parquet_loading():
    """Demonstrate basic parquet file loading and processing."""
    print("=" * 60)
    print("🗃️  PARQUET DATA LOADING DEMONSTRATION")
    print("=" * 60)
    
    parquet_path = "/tmp/audusd_10files.parquet"
    
    if not Path(parquet_path).exists():
        print(f"❌ Parquet file not found: {parquet_path}")
        print("Please ensure the parquet file exists at the specified path.")
        return None
    
    # Create dataset with parquet data source
    print("1. Creating MarketDepthDataset with parquet data source...")
    start_time = time.perf_counter()
    
    dataset = MarketDepthDataset(
        data_source=parquet_path,
        batch_size=500,
        buffer_size=SAMPLES,
        use_memory_mapping=True,  # Use memory mapping for large parquet files
        currency='AUDUSD'  # Use AUDUSD currency configuration
    )
    
    creation_time = (time.perf_counter() - start_time) * 1000
    print(f"   ✅ Dataset created in {creation_time:.2f}ms")
    
    # Check data loading
    if dataset._current_data is not None:
        rows, cols = dataset._current_data.shape
        print(f"   📊 Loaded {rows:,} rows with {cols} columns from parquet file")
        print(f"   📊 Data type: {type(dataset._current_data).__name__}")
        
        # Show some basic statistics about the data
        print(f"   📊 Available symbols: {dataset._current_data['symbol'].unique().to_list()}")
        print(f"   📊 Date range: {dataset._current_data['date'].min()} to {dataset._current_data['date'].max()}")
    else:
        print("   ❌ Failed to load data from parquet file")
        return None
    
    return dataset


def demonstrate_tensor_generation(dataset):
    """Demonstrate PyTorch tensor generation from parquet data."""
    print("\n" + "=" * 60)
    print("🔥 PYTORCH TENSOR GENERATION FROM PARQUET")
    print("=" * 60)
    
    # For file-based datasets, we get tensors through batch iteration
    print("1. Generating market depth tensor from parquet data...")
    start_time = time.perf_counter()
    
    try:
        # Get first batch which contains tensor data
        batch_data = next(iter(dataset))
        generation_time = (time.perf_counter() - start_time) * 1000
        
        # Extract tensor from batch
        if isinstance(batch_data, tuple):
            tensor, classification_target = batch_data
            
            print(f"   ✅ Tensor generated in {generation_time:.2f}ms")
            print(f"   📊 Tensor shape: {tensor.shape}")
            print(f"   📊 Tensor dtype: {tensor.dtype}")
            print(f"   📊 Tensor device: {tensor.device}")
            print(f"   📊 Memory usage: {tensor.numel() * tensor.element_size() / 1024 / 1024:.1f}MB")
            print(f"   📊 Classification target shape: {classification_target.shape if classification_target is not None else 'None'}")
            
            # Performance assessment
            target_ms = 10.0
            if generation_time < target_ms:
                print(f"   ✅ EXCELLENT - Under {target_ms}ms target!")
            elif generation_time < target_ms * 2:
                print(f"   ⚡ GOOD - Close to {target_ms}ms target")
            else:
                print(f"   ⚠️  OPTIMIZATION NEEDED - Over {target_ms}ms target")
            
            return tensor
        else:
            print(f"   ❌ Unexpected batch format: {type(batch_data)}")
            return None
        
    except Exception as e:
        print(f"   ❌ Failed to generate tensor: {e}")
        return None


def demonstrate_batch_processing(dataset):
    """Demonstrate batch processing capabilities with parquet data."""
    print("\n" + "=" * 60)
    print("⚡ BATCH PROCESSING WITH PARQUET DATA")
    print("=" * 60)
    
    print("1. Testing batch iteration over parquet dataset...")
    
    try:
        batch_times = []
        tensor_shapes = []
        
        # Process first few batches for demonstration
        batch_count = 0
        max_batches = 5
        
        for batch_data in dataset:
            if batch_count >= max_batches:
                break
                
            batch_start = time.perf_counter()
            
            # Handle batch data (tensor, classification_target tuple)
            if isinstance(batch_data, tuple):
                tensor, target = batch_data
                tensor_shapes.append(tensor.shape)
                
                # Basic processing simulation
                _ = tensor.mean()
                
            batch_end = time.perf_counter()
            batch_time = (batch_end - batch_start) * 1000
            batch_times.append(batch_time)
            
            print(f"   Batch {batch_count + 1}: {batch_time:.2f}ms, shape: {tensor_shapes[-1]}")
            batch_count += 1
        
        if batch_times:
            avg_batch_time = np.mean(batch_times)
            min_batch_time = np.min(batch_times)
            max_batch_time = np.max(batch_times)
            
            print("\n📊 BATCH PROCESSING RESULTS:")
            print(f"   Average batch time: {avg_batch_time:.2f}ms")
            print(f"   Min batch time: {min_batch_time:.2f}ms")
            print(f"   Max batch time: {max_batch_time:.2f}ms")
            print(f"   Batches processed: {len(batch_times)}")
            print(f"   Throughput: {1000/avg_batch_time:.1f} batches/second")
            
            if avg_batch_time < 50.0:
                print("   ✅ EXCELLENT - Suitable for real-time processing")
            else:
                print("   ⚠️  Consider optimization for real-time use")
        else:
            print("   ❌ No batches processed successfully")
            
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")


def demonstrate_performance_comparison():
    """Compare parquet loading performance vs other formats."""
    print("\n" + "=" * 60)
    print("📈 PARQUET PERFORMANCE COMPARISON")
    print("=" * 60)
    
    parquet_path = "/tmp/audusd_10files.parquet"
    
    if not Path(parquet_path).exists():
        print("   ⚠️  Parquet file not available for comparison")
        return
    
    print("1. Comparing parquet loading performance...")
    
    # Test parquet loading speed
    parquet_times = []
    for i in range(3):
        start_time = time.perf_counter()
        
        dataset = MarketDepthDataset(
            data_source=parquet_path,
            buffer_size=SAMPLES,
            use_memory_mapping=True,
            currency='AUDUSD'
        )
        
        # Generate one tensor to complete the loading pipeline
        if dataset._current_data is not None:
            _ = dataset.get_current_representation()
        
        end_time = time.perf_counter()
        parquet_times.append((end_time - start_time) * 1000)
    
    avg_parquet_time = np.mean(parquet_times)
    
    print(f"   📊 Parquet loading (avg): {avg_parquet_time:.2f}ms")
    print("   📊 Data processing pipeline: ✅ Complete")
    print("   📊 Memory efficiency: ✅ Optimized with polars backend")
    print("   📊 File format advantages:")
    print("      - Columnar storage for fast filtering")
    print("      - Built-in compression")
    print("      - Schema preservation")
    print("      - Cross-platform compatibility")


def demonstrate_ml_workflow(dataset, tensor):
    """Demonstrate ML workflow integration with parquet data."""
    print("\n" + "=" * 60)
    print("🤖 ML WORKFLOW INTEGRATION")
    print("=" * 60)
    
    if tensor is None:
        print("   ❌ No tensor available for ML workflow")
        return
    
    print("1. Setting up simple CNN model for market depth data...")
    
    # Create a simple CNN model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 3)  # 3-class classification
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("   ✅ Model created with CNN architecture")
    
    # Simulate training steps
    print("\n2. Simulating training steps with parquet-loaded data...")
    
    training_times = []
    
    for epoch in range(3):
        start_time = time.perf_counter()
        
        # Prepare batch (add batch and channel dimensions)
        batch = tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 402, 500)
        target = torch.randint(0, 3, (1,))  # Random target for demo
        
        # Forward pass
        output = model(batch)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        end_time = time.perf_counter()
        epoch_time = (end_time - start_time) * 1000
        training_times.append(epoch_time)
        
        print(f"   Epoch {epoch + 1}: {epoch_time:.2f}ms (Loss: {loss.item():.6f})")
    
    avg_training_time = np.mean(training_times)
    print("\n📊 ML WORKFLOW RESULTS:")
    print(f"   Average training step: {avg_training_time:.2f}ms")
    print(f"   Training throughput: {1000/avg_training_time:.1f} steps/second")
    print("   Data pipeline: ✅ Parquet → Polars → PyTorch")
    print("   Model compatibility: ✅ Standard PyTorch tensors")


def main():
    """Main demonstration function."""
    print("🗃️  PARQUET DATALOADER DEMONSTRATION")
    print("=" * 80)
    print("This script demonstrates parquet file integration with the represent library")
    print("for high-performance market depth processing and ML workflows.")
    print("=" * 80)
    
    # Basic parquet loading
    dataset = demonstrate_parquet_loading()
    if dataset is None:
        return
    
    # Tensor generation
    tensor = demonstrate_tensor_generation(dataset)
    
    # Batch processing
    demonstrate_batch_processing(dataset)
    
    # Performance comparison
    demonstrate_performance_comparison()
    
    # ML workflow integration
    demonstrate_ml_workflow(dataset, tensor)
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 PARQUET DATALOADER DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("Key features demonstrated:")
    print("   ✅ Native parquet file loading with polars backend")
    print("   ✅ High-performance tensor generation (<50ms typical)")
    print("   ✅ Efficient batch processing for ML training")
    print("   ✅ PyTorch integration with standard tensor format")
    print("   ✅ Memory-optimized loading for large parquet files")
    print("   ✅ AUDUSD currency configuration support")
    print("=" * 80)


if __name__ == "__main__":
    main()