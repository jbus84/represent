#!/usr/bin/env python3
"""
Simple demonstration of currency configuration usage in PyTorch DataLoader.

This script shows the essential features:
1. Load currency-specific configurations
2. Create DataLoader with currency configuration
3. Compare different currency settings
"""

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from represent.config import (
    get_default_currency_config
)
from represent.dataloader import MarketDepthDataset
from represent.constants import SAMPLES, ASK_PRICE_COLUMNS, BID_PRICE_COLUMNS, ASK_VOL_COLUMNS, BID_VOL_COLUMNS


def generate_minimal_market_data(num_rows: int = SAMPLES + 6000) -> pl.DataFrame:
    """Generate minimal sample market data for testing."""
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
    
    # Add count columns
    for col_name in ['ask_ct_00', 'ask_ct_01', 'ask_ct_02', 'ask_ct_03', 'ask_ct_04',
                     'ask_ct_05', 'ask_ct_06', 'ask_ct_07', 'ask_ct_08', 'ask_ct_09',
                     'bid_ct_00', 'bid_ct_01', 'bid_ct_02', 'bid_ct_03', 'bid_ct_04',
                     'bid_ct_05', 'bid_ct_06', 'bid_ct_07', 'bid_ct_08', 'bid_ct_09']:
        data[col_name] = np.ones(num_rows, dtype=np.int32)
    
    return pl.DataFrame(data)


def demo_currency_config_basics():
    """Show basic currency configuration loading."""
    print("🌍 Currency Configuration Basics")
    print("=" * 40)
    
    currencies = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDJPY']
    
    for currency in currencies:
        config = get_default_currency_config(currency)
        print(f"\n{currency}:")
        print(f"  - Pip size: {config.classification.true_pip_size}")
        print(f"  - Bins: {config.classification.nbins}")
        print(f"  - Lookforward: {config.classification.lookforward_input}")
        print(f"  - Coverage: {config.sampling.coverage_percentage}")


def demo_dataloader_currency_usage():
    """Demonstrate DataLoader with currency configuration."""
    print("\n\n🔧 DataLoader with Currency Configuration")
    print("=" * 40)
    
    # Generate test data
    sample_data = generate_minimal_market_data()
    
    # Test with different currencies
    currencies = ['AUDUSD', 'EURUSD']
    
    for currency in currencies:
        print(f"\n📊 Testing {currency}:")
        
        try:
            # Create dataset with currency configuration
            dataset = MarketDepthDataset(
                data_source=sample_data,
                currency=currency,
                features=['volume'],
                batch_size=500
            )
            
            print("  ✅ Dataset created successfully")
            print(f"  - Classification bins: {dataset.classification_config.nbins}")
            print(f"  - True pip size: {dataset.classification_config.true_pip_size}")
            print(f"  - Sampling mode: {dataset.sampling_config.sampling_mode}")
            print(f"  - Available batches: {len(dataset)}")
            print(f"  - Output shape: {dataset.output_shape}")
            
            if len(dataset) > 0:
                # Create PyTorch DataLoader and test first batch
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                
                batch_iter = iter(dataloader)
                features, targets = next(batch_iter)
                
                print("  ⚡ First batch processed successfully")
                print(f"  - Features shape: {features.shape}")
                print(f"  - Targets shape: {targets.shape}")
                print(f"  - Features dtype: {features.dtype}")
                print(f"  - Targets dtype: {targets.dtype}")
                
                # Show target statistics
                unique_targets = torch.unique(targets)
                print(f"  - Unique targets: {unique_targets.tolist()}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")


def demo_manual_vs_currency():
    """Compare manual configuration vs currency-based configuration."""
    print("\n\n⚖️  Manual vs Currency Configuration")
    print("=" * 40)
    
    sample_data = generate_minimal_market_data()
    
    # Manual configuration
    print("\n🔧 Manual Configuration:")
    try:
        manual_dataset = MarketDepthDataset(
            data_source=sample_data,
            classification_config={
                'true_pip_size': 0.0001,
                'nbins': 7,  # Simpler for testing
                'lookforward_input': 2000  # Smaller for sample data
            },
            sampling_config={
                'sampling_mode': 'consecutive',
                'coverage_percentage': 0.5  # Process subset
            },
            features=['volume']
        )
        
        print(f"  ✅ Manual config: {len(manual_dataset)} batches available")
        print(f"  - Bins: {manual_dataset.classification_config.nbins}")
        print(f"  - Lookforward: {manual_dataset.classification_config.lookforward_input}")
        
    except Exception as e:
        print(f"  ❌ Manual config failed: {e}")
    
    # Currency-based configuration
    print("\n🌍 Currency Configuration (AUDUSD):")
    try:
        currency_dataset = MarketDepthDataset(
            data_source=sample_data,
            currency='AUDUSD',
            features=['volume']
        )
        
        # Override some settings for our test data
        currency_dataset.classification_config.lookforward_input = 2000
        currency_dataset.sampling_config.coverage_percentage = 0.5
        
        # Re-analyze with new settings
        currency_dataset._analyze_and_select_end_ticks()
        
        print(f"  ✅ Currency config: {len(currency_dataset)} batches available")
        print(f"  - Bins: {currency_dataset.classification_config.nbins}")
        print(f"  - Coverage: {currency_dataset.sampling_config.coverage_percentage}")
        
    except Exception as e:
        print(f"  ❌ Currency config failed: {e}")


def demo_different_features():
    """Demonstrate currency config with different features."""
    print("\n\n🎛️  Currency Configuration with Different Features")
    print("=" * 40)
    
    sample_data = generate_minimal_market_data()
    
    feature_sets = [
        (['volume'], "Single feature (volume only)"),
        (['volume', 'variance'], "Two features (volume + variance)"),
    ]
    
    for features, description in feature_sets:
        print(f"\n📊 {description}:")
        
        try:
            dataset = MarketDepthDataset(
                data_source=sample_data,
                currency='EURUSD',
                features=features,
                batch_size=500
            )
            
            # Adjust for test data
            dataset.classification_config.lookforward_input = 2000
            dataset.sampling_config.coverage_percentage = 0.3
            dataset._analyze_and_select_end_ticks()
            
            print(f"  ✅ Dataset created with {len(features)} features")
            print(f"  - Output shape: {dataset.output_shape}")
            print(f"  - Available batches: {len(dataset)}")
            
            if len(dataset) > 0:
                dataloader = DataLoader(dataset, batch_size=1)
                batch_iter = iter(dataloader)
                features_tensor, targets = next(batch_iter)
                
                print(f"  ⚡ Batch shape: {features_tensor.shape}")
                
        except Exception as e:
            print(f"  ❌ Error with {len(features)} features: {e}")


def main():
    """Run focused currency configuration demo."""
    print("🚀 Simple Currency Configuration Demo")
    print("=" * 50)
    
    try:
        demo_currency_config_basics()
        demo_dataloader_currency_usage()
        demo_manual_vs_currency()
        demo_different_features()
        
        print("\n" + "=" * 50)
        print("✅ Currency configuration demo completed!")
        print("\nKey Features Demonstrated:")
        print("- Currency-specific configurations (AUDUSD, EURUSD, etc.)")
        print("- Automatic optimization per currency pair")
        print("- PyTorch DataLoader integration")
        print("- Different feature combinations")
        print("- Performance comparison manual vs currency config")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()