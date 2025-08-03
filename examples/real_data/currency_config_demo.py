#!/usr/bin/env python3
"""
Demonstration of currency configuration usage in PyTorch DataLoader.

This script shows how to:
1. Create currency-specific configurations
2. Use currency configurations with the MarketDepthDataset 
3. Compare different currency settings
4. Test with sample data
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from represent.config import (
    CurrencyConfig, 
    ClassificationConfig, 
    SamplingConfig,
    load_currency_config,
    get_default_currency_config,
    save_currency_config
)
from represent.dataloader import MarketDepthDataset, create_streaming_dataloader
from represent.constants import SAMPLES, ASK_PRICE_COLUMNS, BID_PRICE_COLUMNS, ASK_VOL_COLUMNS, BID_VOL_COLUMNS


def generate_sample_market_data(num_rows: int = SAMPLES + 10000) -> pl.DataFrame:
    """Generate realistic sample market data for testing."""
    print(f"📊 Generating {num_rows:,} rows of sample market data...")
    
    # Base price around 1.2500 for AUD/USD
    base_price = 1.25000
    
    # Generate time series with random walk
    np.random.seed(42)  # For reproducible results
    price_changes = np.random.normal(0, 0.00005, num_rows)  # Small random changes
    mid_prices = base_price + np.cumsum(price_changes)
    
    # Generate bid/ask spreads (typical 1-3 pips)
    spreads = np.random.uniform(0.00001, 0.00003, num_rows)
    
    # Create sample market data
    data = {
        'ts_event': np.arange(num_rows) * 1000000,  # Microsecond timestamps
        'ts_recv': np.arange(num_rows) * 1000000 + 100,
        'rtype': np.ones(num_rows, dtype=np.int32),
        'publisher_id': np.ones(num_rows, dtype=np.int32),
        'symbol': ['AUDUSD'] * num_rows,
    }
    
    # Generate 10 levels of market depth
    for i, col_name in enumerate(ASK_PRICE_COLUMNS):
        # Ask prices increase with level
        level_offset = (i + 1) * 0.00001  # 1 pip per level
        data[col_name] = mid_prices + spreads/2 + level_offset
    
    for i, col_name in enumerate(BID_PRICE_COLUMNS):
        # Bid prices decrease with level  
        level_offset = (i + 1) * 0.00001  # 1 pip per level
        data[col_name] = mid_prices - spreads/2 - level_offset
    
    # Generate volumes (decreasing with level)
    for i, col_name in enumerate(ASK_VOL_COLUMNS):
        base_volume = np.random.uniform(1000000, 5000000, num_rows)  # 1-5M base volume
        level_multiplier = 1.0 / (i + 1)  # Decreasing volume with level
        data[col_name] = base_volume * level_multiplier
    
    for i, col_name in enumerate(BID_VOL_COLUMNS):
        base_volume = np.random.uniform(1000000, 5000000, num_rows)  # 1-5M base volume
        level_multiplier = 1.0 / (i + 1)  # Decreasing volume with level
        data[col_name] = base_volume * level_multiplier
    
    # Add count columns (simple ones for now)
    for col_name in ['ask_ct_00', 'ask_ct_01', 'ask_ct_02', 'ask_ct_03', 'ask_ct_04',
                     'ask_ct_05', 'ask_ct_06', 'ask_ct_07', 'ask_ct_08', 'ask_ct_09',
                     'bid_ct_00', 'bid_ct_01', 'bid_ct_02', 'bid_ct_03', 'bid_ct_04',
                     'bid_ct_05', 'bid_ct_06', 'bid_ct_07', 'bid_ct_08', 'bid_ct_09']:
        data[col_name] = np.ones(num_rows, dtype=np.int32)
    
    return pl.DataFrame(data)


def demo_currency_configs():
    """Demonstrate different currency configurations."""
    print("🌍 Currency Configuration Demo")
    print("=" * 50)
    
    # 1. Show default configurations for different currencies
    currencies = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'NZDUSD']
    
    print("\n1. Default Currency Configurations:")
    print("-" * 40)
    
    for currency in currencies:
        config = get_default_currency_config(currency)
        print(f"\n{currency}:")
        print(f"  - True pip size: {config.classification.true_pip_size}")
        print(f"  - Micro pip size: {config.classification.micro_pip_size}")
        print(f"  - Classification bins: {config.classification.nbins}")
        print(f"  - Lookforward input: {config.classification.lookforward_input}")
        print(f"  - Coverage percentage: {config.sampling.coverage_percentage}")
        print(f"  - Description: {config.description}")
    
    # 2. Create and save custom configuration
    print("\n\n2. Creating Custom Currency Configuration:")
    print("-" * 40)
    
    custom_config = CurrencyConfig(
        currency_pair="CUSTOM",
        classification=ClassificationConfig(
            true_pip_size=0.0001,
            nbins=9,
            lookforward_input=3000,
            ticks_per_bin=50
        ),
        sampling=SamplingConfig(
            sampling_mode='random',
            coverage_percentage=0.5,
            seed=123
        ),
        description="Custom configuration for demo purposes"
    )
    
    print(f"Custom config created for {custom_config.currency_pair}")
    print(f"Classification settings: {custom_config.classification.nbins} bins, {custom_config.classification.lookforward_input} lookforward")
    print(f"Sampling settings: {custom_config.sampling.sampling_mode} mode, {custom_config.sampling.coverage_percentage*100}% coverage")
    
    # Save the configuration
    config_dir = Path("./demo_configs")
    saved_path = save_currency_config(custom_config, config_dir)
    print(f"Saved to: {saved_path}")
    
    return custom_config


def demo_dataloader_with_currency(sample_data: pl.DataFrame):
    """Demonstrate DataLoader usage with currency configuration."""
    print("\n\n3. DataLoader with Currency Configuration:")
    print("-" * 40)
    
    # Test different currency configurations
    test_configs = [
        ('AUDUSD', 'AUD/USD optimized settings'),
        ('EURUSD', 'EUR/USD optimized settings'),
        ('USDJPY', 'USD/JPY with different pip size')
    ]
    
    for currency, description in test_configs:
        print(f"\n🔧 Testing {currency} ({description}):")
        
        # Create dataset with currency configuration
        dataset = MarketDepthDataset(
            data_source=sample_data,
            currency=currency,  # This loads currency-specific config
            features=['volume'],  # Single feature for simplicity
            batch_size=500
        )
        
        print(f"  ✅ Dataset created with currency: {currency}")
        print(f"  - Classification bins: {dataset.classification_config.nbins}")
        print(f"  - True pip size: {dataset.classification_config.true_pip_size}")
        print(f"  - Lookforward input: {dataset.classification_config.lookforward_input}")
        print(f"  - Sampling mode: {dataset.sampling_config.sampling_mode}")
        print(f"  - Coverage: {dataset.sampling_config.coverage_percentage*100}%")
        print(f"  - Output shape: {dataset.output_shape}")
        
        # Create PyTorch DataLoader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        print("  📦 PyTorch DataLoader created")
        print(f"  - Number of batches available: {len(dataset)}")
        
        # Test first batch
        try:
            start_time = time.perf_counter()
            batch_iter = iter(dataloader)
            features, targets = next(batch_iter)
            end_time = time.perf_counter()
            
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            
            print(f"  ⚡ First batch processed in {processing_time:.2f}ms")
            print(f"  - Features shape: {features.shape}")
            print(f"  - Targets shape: {targets.shape}")
            print(f"  - Feature tensor dtype: {features.dtype}")
            print(f"  - Target tensor dtype: {targets.dtype}")
            
            # Show classification distribution
            unique_targets, counts = torch.unique(targets, return_counts=True)
            print(f"  - Target distribution: {dict(zip(unique_targets.tolist(), counts.tolist()))}")
            
        except Exception as e:
            print(f"  ❌ Error processing batch: {e}")


def demo_manual_vs_currency_config(sample_data: pl.DataFrame):
    """Compare manual configuration vs currency-based configuration."""
    print("\n\n4. Manual vs Currency Configuration Comparison:")
    print("-" * 40)
    
    # Manual configuration
    print("\n🔧 Manual Configuration:")
    manual_dataset = MarketDepthDataset(
        data_source=sample_data,
        classification_config={
            'true_pip_size': 0.0001,
            'nbins': 13,
            'lookforward_input': 5000
        },
        sampling_config={
            'sampling_mode': 'consecutive',
            'coverage_percentage': 1.0
        },
        features=['volume']
    )
    
    print(f"  - Classification bins: {manual_dataset.classification_config.nbins}")
    print(f"  - Lookforward input: {manual_dataset.classification_config.lookforward_input}")
    print(f"  - Sampling mode: {manual_dataset.sampling_config.sampling_mode}")
    print(f"  - Available batches: {len(manual_dataset)}")
    
    # Currency-based configuration (AUDUSD)
    print("\n🌍 Currency-based Configuration (AUDUSD):")
    currency_dataset = MarketDepthDataset(
        data_source=sample_data,
        currency='AUDUSD',  # Loads AUDUSD-optimized settings
        features=['volume']
    )
    
    print(f"  - Classification bins: {currency_dataset.classification_config.nbins}")
    print(f"  - Lookforward input: {currency_dataset.classification_config.lookforward_input}")
    print(f"  - Sampling mode: {currency_dataset.sampling_config.sampling_mode}")
    print(f"  - Coverage percentage: {currency_dataset.sampling_config.coverage_percentage}")
    print(f"  - Available batches: {len(currency_dataset)}")
    
    # Performance comparison
    print("\n⚡ Performance Comparison:")
    
    datasets = [
        ('Manual Config', manual_dataset),
        ('AUDUSD Config', currency_dataset)
    ]
    
    for name, dataset in datasets:
        start_time = time.perf_counter()
        
        # Process first batch
        dataloader = DataLoader(dataset, batch_size=1)
        batch_iter = iter(dataloader)
        features, targets = next(batch_iter)
        
        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000
        
        print(f"  {name}: {processing_time:.2f}ms")


def demo_streaming_with_currency():
    """Demonstrate streaming DataLoader with currency configuration."""
    print("\n\n5. Streaming DataLoader with Currency Configuration:")
    print("-" * 40)
    
    # Create streaming dataset with currency config
    streaming_dataset = create_streaming_dataloader(
        buffer_size=SAMPLES,
        features=['volume'],
        classification_config=None  # Will use defaults
    )
    
    # Override with currency config after creation
    streaming_dataset.currency = 'EURUSD'
    currency_config = load_currency_config('EURUSD')
    streaming_dataset.classification_config = currency_config.classification
    streaming_dataset.sampling_config = currency_config.sampling
    
    print("  ✅ Streaming dataset created with EURUSD configuration")
    print(f"  - Ring buffer size: {streaming_dataset.ring_buffer_size}")
    print(f"  - Ready for processing: {streaming_dataset.is_ready_for_processing}")
    print(f"  - Classification bins: {streaming_dataset.classification_config.nbins}")
    
    # Generate some streaming data
    print("\n  📡 Simulating streaming data...")
    sample_streaming_data = generate_sample_market_data(SAMPLES + 100)
    
    # Add data in chunks
    chunk_size = 1000
    for i in range(0, len(sample_streaming_data), chunk_size):
        chunk = sample_streaming_data.slice(i, chunk_size)
        streaming_dataset.add_streaming_data(chunk)
        
        if streaming_dataset.is_ready_for_processing:
            print(f"    📊 Added chunk {i//chunk_size + 1}, buffer ready: {streaming_dataset.ring_buffer_size}/{SAMPLES}")
            break
        else:
            print(f"    📊 Added chunk {i//chunk_size + 1}, buffer size: {streaming_dataset.ring_buffer_size}/{SAMPLES}")
    
    if streaming_dataset.is_ready_for_processing:
        print("\n  ⚡ Testing streaming representation generation...")
        start_time = time.perf_counter()
        representation = streaming_dataset.get_current_representation()
        end_time = time.perf_counter()
        
        processing_time = (end_time - start_time) * 1000
        print(f"    Generated representation in {processing_time:.2f}ms")
        print(f"    Shape: {representation.shape}")
        print(f"    Dtype: {representation.dtype}")


def demo_config_persistence():
    """Demonstrate saving and loading currency configurations."""
    print("\n\n6. Configuration Persistence Demo:")
    print("-" * 40)
    
    config_dir = Path("./demo_configs")
    config_dir.mkdir(exist_ok=True)
    
    # Create multiple currency configurations
    configs = {
        'TESTCUR1': CurrencyConfig(
            currency_pair='TESTCUR1',
            classification=ClassificationConfig(nbins=9, lookforward_input=3000),
            sampling=SamplingConfig(coverage_percentage=0.8, sampling_mode='random'),
            description='Test currency 1 - moderate volatility'
        ),
        'TESTCUR2': CurrencyConfig(
            currency_pair='TESTCUR2', 
            classification=ClassificationConfig(nbins=5, lookforward_input=1000),
            sampling=SamplingConfig(coverage_percentage=0.6, sampling_mode='consecutive'),
            description='Test currency 2 - high volatility, quick decisions'
        )
    }
    
    # Save configurations
    print("  💾 Saving configurations...")
    for currency, config in configs.items():
        saved_path = save_currency_config(config, config_dir)
        print(f"    {currency}: {saved_path}")
    
    # Load configurations back
    print("\n  📂 Loading configurations...")
    for currency in configs.keys():
        try:
            loaded_config = load_currency_config(currency, config_dir)
            print(f"    {currency}: ✅ Loaded successfully")
            print(f"      - Bins: {loaded_config.classification.nbins}")
            print(f"      - Coverage: {loaded_config.sampling.coverage_percentage}")
            print(f"      - Description: {loaded_config.description}")
        except Exception as e:
            print(f"    {currency}: ❌ Failed to load: {e}")
    
    # Test with DataLoader
    print("\n  🧪 Testing loaded config with DataLoader...")
    sample_data = generate_sample_market_data(SAMPLES + 500)
    
    for currency in configs.keys():
        try:
            dataset = MarketDepthDataset(
                data_source=sample_data,
                currency=currency,  # This will load from the saved config
                features=['volume']
            )
            print(f"    {currency}: ✅ DataLoader created with {len(dataset)} batches")
        except Exception as e:
            print(f"    {currency}: ❌ DataLoader failed: {e}")


def main():
    """Run all currency configuration demos."""
    print("🚀 Currency Configuration Demo Script")
    print("=" * 60)
    
    try:
        # Generate sample data
        sample_data = generate_sample_market_data()
        
        # Run demonstrations
        demo_currency_configs()
        demo_dataloader_with_currency(sample_data)
        demo_manual_vs_currency_config(sample_data)
        demo_streaming_with_currency()
        demo_config_persistence()
        
        print("\n" + "=" * 60)
        print("✅ All currency configuration demos completed successfully!")
        print("\nKey takeaways:")
        print("- Currency configurations provide optimized settings for different currency pairs")
        print("- PyTorch DataLoader seamlessly integrates with currency configs")
        print("- Configurations can be saved/loaded for reuse")
        print("- Performance is maintained across different configuration options")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()