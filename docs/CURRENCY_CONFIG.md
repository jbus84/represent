# Currency Configuration Guide

This guide demonstrates how to use currency-specific configurations with the PyTorch DataLoader in the represent package.

## Overview

The represent package supports currency-specific configurations that optimize market depth processing for different currency pairs. Each currency has unique characteristics (pip sizes, volatility patterns, liquidity) that require tailored processing parameters.

## Key Features

- **Automatic Optimization**: Pre-configured settings for major currency pairs
- **Pip Size Handling**: Correct pip sizes for different currency types (e.g., JPY pairs)
- **Classification Tuning**: Optimized bin counts and thresholds per currency
- **Sampling Strategies**: Currency-specific coverage percentages and sampling modes
- **PyTorch Integration**: Seamless integration with PyTorch DataLoader

## Quick Start

### Basic Usage with Currency Configuration

```python
from represent.dataloader import MarketDepthDataset
from torch.utils.data import DataLoader

# Create dataset with currency-specific configuration
dataset = MarketDepthDataset(
    data_source="path/to/your/data.dbn",
    currency="AUDUSD",  # Automatically loads AUDUSD optimizations
    features=['volume']
)

# Create PyTorch DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Use in training loop
for features, targets in dataloader:
    # features shape: (8, 402, 500) for single feature
    # targets shape: (8, 1) for classification
    pass
```

### Multi-Feature with Currency Configuration

```python
# Multi-feature dataset with currency optimization
dataset = MarketDepthDataset(
    data_source="path/to/your/data.dbn",
    currency="EURUSD",
    features=['volume', 'variance']  # Multiple features
)

# features shape will be: (batch_size, 2, 402, 500)
```

## Supported Currencies

### Major Pairs (USD-based)
- **AUDUSD**: Australian Dollar / US Dollar
- **EURUSD**: Euro / US Dollar  
- **GBPUSD**: British Pound / US Dollar
- **NZDUSD**: New Zealand Dollar / US Dollar

### JPY Pairs (Special pip sizing)
- **USDJPY**: US Dollar / Japanese Yen
- **EURJPY**: Euro / Japanese Yen
- **GBPJPY**: British Pound / Japanese Yen

## Currency-Specific Optimizations

### AUDUSD Configuration
```python
{
    "true_pip_size": 0.0001,        # Standard pip size
    "nbins": 13,                    # 13-bin classification
    "lookforward_input": 5000,      # 5000 ticks lookforward
    "coverage_percentage": 0.8,     # Process 80% of data
    "sampling_mode": "random"       # Random sampling
}
```

### USDJPY Configuration  
```python
{
    "true_pip_size": 0.01,          # JPY pip size (different!)
    "micro_pip_size": 0.001,        # JPY micro pip
    "nbins": 9,                     # Fewer bins for JPY dynamics
    "lookforward_input": 5000,      # 5000 ticks lookforward
    "coverage_percentage": 0.6,     # Process 60% of data
}
```

### GBPUSD Configuration
```python
{
    "true_pip_size": 0.0001,        # Standard pip size
    "nbins": 13,                    # 13-bin classification
    "lookforward_input": 3000,      # Shorter window (more volatile)
    "coverage_percentage": 0.7,     # Process 70% of data
}
```

## Manual vs Currency Configuration

### Manual Configuration
```python
# Manual configuration - you specify all parameters
dataset = MarketDepthDataset(
    data_source="data.dbn",
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
```

### Currency Configuration
```python
# Currency configuration - optimized settings loaded automatically
dataset = MarketDepthDataset(
    data_source="data.dbn", 
    currency='AUDUSD',  # Loads AUDUSD-optimized settings
    features=['volume']
)
```

## Advanced Usage

### Custom Currency Configuration

```python
from represent.config import CurrencyConfig, ClassificationConfig, SamplingConfig

# Create custom configuration
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
    description="Custom configuration for exotic pair"
)

# Save for reuse
from represent.config import save_currency_config
save_currency_config(custom_config, Path("./configs"))

# Use with dataset
dataset = MarketDepthDataset(
    data_source="data.dbn",
    currency="CUSTOM",  # Loads from saved config
    features=['volume']
)
```

### Override Currency Settings

```python
# Start with currency config but override specific settings
dataset = MarketDepthDataset(
    data_source="data.dbn",
    currency='EURUSD'  # Base configuration
)

# Override specific settings
dataset.classification_config.lookforward_input = 2000  # Shorter lookforward
dataset.sampling_config.coverage_percentage = 0.5      # Process less data

# Re-analyze with new settings
dataset._analyze_and_select_end_ticks()
```

## Output Shapes by Feature Count

The output tensor shape depends on the number of features selected:

- **Single feature**: `(402, 500)` - 2D tensor
- **Multiple features**: `(N, 402, 500)` - 3D tensor where N = number of features

```python
# Single feature
dataset = MarketDepthDataset(currency='AUDUSD', features=['volume'])
# Output shape: (402, 500)

# Two features  
dataset = MarketDepthDataset(currency='AUDUSD', features=['volume', 'variance'])
# Output shape: (2, 402, 500)

# Three features
dataset = MarketDepthDataset(currency='AUDUSD', features=['volume', 'variance', 'trade_counts'])
# Output shape: (3, 402, 500)
```

## Performance Considerations

Currency configurations are optimized for performance:

- **Random Sampling**: Reduces data processing by using coverage percentages
- **Optimized Lookforward**: Balanced between accuracy and speed per currency
- **Memory Efficiency**: Pre-allocated buffers scale with feature count
- **Classification Bins**: Optimal bin counts for each currency's volatility

### Performance Targets (maintained across all currencies)

- **Single Record**: <1ms processing time
- **Array Generation**: <10ms for complete feature arrays
- **Batch Processing**: <50ms for 500-record batches
- **Memory Usage**: Linear scaling with feature count

## Complete Working Example

```python
#!/usr/bin/env python3
"""Complete example using currency configuration with PyTorch DataLoader."""

import torch
from torch.utils.data import DataLoader
from represent.dataloader import MarketDepthDataset

def main():
    # Load data with EURUSD optimizations
    dataset = MarketDepthDataset(
        data_source="eurusd_data.dbn",
        currency="EURUSD",
        features=['volume', 'variance']
    )
    
    print(f"Dataset configured for EURUSD:")
    print(f"- Classification bins: {dataset.classification_config.nbins}")
    print(f"- Pip size: {dataset.classification_config.true_pip_size}")
    print(f"- Features: {dataset.features}")
    print(f"- Output shape: {dataset.output_shape}")
    print(f"- Available batches: {len(dataset)}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    
    # Training loop
    for epoch in range(10):
        for batch_idx, (features, targets) in enumerate(dataloader):
            # features: (16, 2, 402, 500) - batch of 16, 2 features, 402x500 market depth
            # targets: (16, 1) - classification targets
            
            # Your model training here
            print(f"Epoch {epoch}, Batch {batch_idx}: {features.shape} -> {targets.shape}")
            
            if batch_idx >= 2:  # Just show first few batches
                break

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

1. **Insufficient Data**: Ensure your data has enough rows for the lookforward window
   ```
   ⚠️  Insufficient data: need 55500 rows, got 51000
   ```
   Solution: Reduce `lookforward_input` or increase your dataset size

2. **No Batches Available**: Check if your data meets minimum requirements
   ```python
   if len(dataset) == 0:
       # Reduce requirements
       dataset.classification_config.lookforward_input = 2000
       dataset.sampling_config.coverage_percentage = 0.5
       dataset._analyze_and_select_end_ticks()
   ```

3. **Currency Not Found**: Custom currencies need to be saved first
   ```python
   from represent.config import save_currency_config
   save_currency_config(custom_config, config_dir)
   ```

### Debug Information

```python
# Check configuration details
print(f"Currency: {dataset.currency}")
print(f"Classification config: {dataset.classification_config}")
print(f"Sampling config: {dataset.sampling_config}")
print(f"Available end ticks: {len(dataset._end_tick_positions)}")
```

## Summary

Currency configuration provides:

✅ **Automatic Optimization**: Pre-tuned settings for major currency pairs  
✅ **Correct Pip Handling**: Proper pip sizes for different currency types  
✅ **PyTorch Integration**: Seamless DataLoader compatibility  
✅ **Multi-Feature Support**: Works with all feature combinations  
✅ **Performance**: Maintains <10ms latency targets  
✅ **Flexibility**: Override settings as needed  

This makes currency-specific market depth processing both easy and optimized for real-world trading applications.