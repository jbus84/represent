#!/usr/bin/env python3
"""
Production DataLoader Example

This script demonstrates how to use the HighPerformanceDataLoader for 
production environments where AsyncDataLoader threading may cause issues.

The HighPerformanceDataLoader is a simpler, synchronous alternative that:
- Uses PyTorch's built-in DataLoader (battle-tested)
- No complex threading or background processes
- Reliable resource management
- Production-ready stability
"""

import time
import numpy as np
import polars as pl

from represent import (
    MarketDepthDataset, 
    HighPerformanceDataLoader,
    create_high_performance_dataloader
)
from represent.constants import (
    SAMPLES, ASK_PRICE_COLUMNS, BID_PRICE_COLUMNS, 
    ASK_VOL_COLUMNS, BID_VOL_COLUMNS
)


def generate_sample_data(num_rows: int = SAMPLES + 10000) -> pl.DataFrame:
    """Generate sample market data for testing."""
    print(f"📊 Generating {num_rows:,} rows of sample market data...")
    
    # Base price around 1.2500 for AUD/USD
    base_price = 1.25000
    
    # Generate time series 
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.00005, num_rows)
    mid_prices = base_price + np.cumsum(price_changes)
    spreads = np.random.uniform(0.00001, 0.00003, num_rows)
    
    # Create minimal required columns
    data = {
        'ts_event': np.arange(num_rows) * 1000000,
        'ts_recv': np.arange(num_rows) * 1000000 + 100,
        'rtype': np.ones(num_rows, dtype=np.int32),
        'publisher_id': np.ones(num_rows, dtype=np.int32),
        'symbol': ['AUDUSD'] * num_rows,
    }
    
    # Add price columns (just first few levels)
    for i, col_name in enumerate(ASK_PRICE_COLUMNS):
        level_offset = (i + 1) * 0.00001
        data[col_name] = mid_prices + spreads/2 + level_offset
    
    for i, col_name in enumerate(BID_PRICE_COLUMNS):
        level_offset = (i + 1) * 0.00001
        data[col_name] = mid_prices - spreads/2 - level_offset
    
    # Add volume columns
    for i, col_name in enumerate(ASK_VOL_COLUMNS):
        base_volume = np.random.uniform(1000000, 5000000, num_rows)
        level_multiplier = 1.0 / (i + 1)
        data[col_name] = base_volume * level_multiplier
    
    for i, col_name in enumerate(BID_VOL_COLUMNS):
        base_volume = np.random.uniform(1000000, 5000000, num_rows)
        level_multiplier = 1.0 / (i + 1)
        data[col_name] = base_volume * level_multiplier
    
    return pl.DataFrame(data)


def demo_high_performance_dataloader():
    """Demonstrate HighPerformanceDataLoader usage."""
    print("🏭 Production DataLoader Demo")
    print("=" * 50)
    
    # Generate sample data
    sample_data = generate_sample_data()
    print(f"✅ Generated {len(sample_data):,} rows of sample data")
    
    try:
        # Create dataset with currency configuration
        dataset = MarketDepthDataset(
            data_source=sample_data,
            currency='AUDUSD',
            features=['volume'],
            batch_size=500
        )
        
        # Adjust for test data
        dataset.classification_config.lookforward_input = 2000
        dataset.sampling_config.coverage_percentage = 0.3
        dataset._analyze_and_select_end_ticks()
        
        print("📊 Dataset created:")
        print(f"  - Currency: {dataset.currency}")
        print(f"  - Features: {dataset.features}")
        print(f"  - Output shape: {dataset.output_shape}")
        print(f"  - Available batches: {len(dataset)}")
        
        if len(dataset) == 0:
            print("❌ No batches available - insufficient data")
            return
        
        # Method 1: Direct instantiation (single-threaded for stability)
        print("\n🔧 Method 1: Direct HighPerformanceDataLoader")
        dataloader1 = HighPerformanceDataLoader(
            dataset=dataset,
            batch_size=8,
            num_workers=0,  # Single-threaded for production stability
            pin_memory=False  # Disabled for production safety
        )
        
        print(f"  - Batch size: {dataloader1.batch_size}")
        print(f"  - Total batches in dataloader: {len(dataloader1)}")
        
        # Test first few batches
        batch_count = 0
        start_time = time.perf_counter()
        
        for batch in dataloader1:
            batch_time = (time.perf_counter() - start_time) * 1000
            features, targets = batch
            print(f"  - Batch {batch_count + 1}: {features.shape} -> {targets.shape} ({batch_time:.2f}ms)")
            
            batch_count += 1
            if batch_count >= 3:  # Just test first 3 batches
                break
            start_time = time.perf_counter()
        
        # Method 2: Factory function
        print("\n🏗️  Method 2: Factory Function")
        dataloader2 = create_high_performance_dataloader(
            dataset=dataset,
            batch_size=4,
            num_workers=0,  # Single-threaded for maximum compatibility
            pin_memory=False
        )
        
        print(f"  - Batch size: {dataloader2.batch_size}")
        print(f"  - Total batches in dataloader: {len(dataloader2)}")
        
        # Test performance
        batch_count = 0
        total_time = 0
        
        for batch in dataloader2:
            start_time = time.perf_counter()
            features, targets = batch
            batch_time = (time.perf_counter() - start_time) * 1000
            total_time += batch_time
            
            print(f"  - Batch {batch_count + 1}: {features.shape} -> {targets.shape} ({batch_time:.2f}ms)")
            
            batch_count += 1
            if batch_count >= 3:  # Just test first 3 batches
                break
        
        avg_time = total_time / batch_count if batch_count > 0 else 0
        print(f"  - Average batch time: {avg_time:.2f}ms")
        
        print("\n✅ HighPerformanceDataLoader working correctly!")
        print("\n🎯 Production Benefits:")
        print("  ✓ Simple, synchronous operation")
        print("  ✓ No complex threading issues")
        print("  ✓ Uses PyTorch's proven DataLoader")
        print("  ✓ Configurable workers and memory pinning")
        print("  ✓ Reliable resource management")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


def demo_production_patterns():
    """Show common production patterns."""
    print("\n\n🏭 Production Usage Patterns")
    print("=" * 50)
    
    sample_data = generate_sample_data(SAMPLES + 5000)
    
    # Pattern 1: Single-threaded (maximum compatibility)
    print("\n1️⃣  Single-threaded (Maximum Compatibility)")
    dataset = MarketDepthDataset(
        data_source=sample_data,
        currency='EURUSD',
        features=['volume']
    )
    dataset.classification_config.lookforward_input = 1500
    dataset.sampling_config.coverage_percentage = 0.2
    dataset._analyze_and_select_end_ticks()
    
    if len(dataset) > 0:
        single_thread_loader = create_high_performance_dataloader(
            dataset=dataset,
            batch_size=16,
            num_workers=0,  # Single-threaded
            pin_memory=False
        )
        print(f"  ✅ Single-threaded loader: {len(single_thread_loader)} batches ready")
    
    # Pattern 2: Single-threaded large batches (recommended)
    print("\n2️⃣  Single-threaded Large Batches (Recommended)")
    if len(dataset) > 0:
        large_batch_loader = create_high_performance_dataloader(
            dataset=dataset,
            batch_size=32,  # Larger batches for efficiency
            num_workers=0,  # Keep single-threaded for stability
            pin_memory=False  # Safe for production
        )
        print(f"  ✅ Large batch loader: {len(large_batch_loader)} batches ready")
    
    # Pattern 3: GPU-ready (when using GPU in production)
    print("\n3️⃣  GPU-ready (When Using GPU)")
    if len(dataset) > 0:
        gpu_loader = create_high_performance_dataloader(
            dataset=dataset,
            batch_size=16,  # Moderate batch size
            num_workers=0,  # Still single-threaded for stability
            pin_memory=True  # Enable only when actually using GPU
        )
        print(f"  ✅ GPU-ready loader: {len(gpu_loader)} batches ready")
    
    print("\n💡 Production Recommendations:")
    print("  - Use single-threaded (num_workers=0) to avoid pickle/threading issues")
    print("  - Increase batch_size rather than workers for better performance")
    print("  - Enable pin_memory=True only when actually using GPU in production")
    print("  - Start with pin_memory=False for maximum compatibility")
    print("  - Monitor memory usage with large batch sizes")


def main():
    """Run the production dataloader demonstration."""
    try:
        demo_high_performance_dataloader()
        demo_production_patterns()
        
        print("\n" + "=" * 50)
        print("✅ Production DataLoader Demo Complete!")
        print("\n📚 Key Takeaways:")
        print("  1. HighPerformanceDataLoader is production-ready")
        print("  2. No threading complexity like AsyncDataLoader")
        print("  3. Uses PyTorch's battle-tested DataLoader underneath")
        print("  4. Configurable for different production needs")
        print("  5. Simple factory function for easy setup")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()