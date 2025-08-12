# Focused Configuration Guide

This guide demonstrates how to use the new focused configuration models in the represent package.

## Overview

The represent package now uses three focused configuration models instead of a monolithic RepresentConfig:
- **DatasetBuilderConfig**: For creating comprehensive symbol datasets
- **GlobalThresholdConfig**: For calculating consistent classification thresholds  
- **MarketDepthProcessorConfig**: For processing market data into ML tensors

Each configuration model contains only the parameters needed by its respective module.

## Key Features

- **Focused Configurations**: Separate configs eliminate parameter pollution between modules
- **Automatic Optimization**: Currency-specific settings applied automatically
- **Type Safety**: Pydantic validation ensures correct parameter types and values
- **Computed Fields**: Auto-calculated values like `min_required_samples` and `time_bins`
- **Factory Functions**: Convenient functions for creating compatible configurations

## Quick Start

### Individual Configuration Creation

```python
from represent import DatasetBuilderConfig, GlobalThresholdConfig, MarketDepthProcessorConfig

# Dataset Builder - for creating symbol datasets
dataset_config = DatasetBuilderConfig(
    currency="AUDUSD",
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500
)

# Global Threshold - for consistent classification
threshold_config = GlobalThresholdConfig(
    currency="AUDUSD", 
    nbins=13,
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500,
    jump_size=100
)

# Market Depth Processor - for ML tensor processing
processor_config = MarketDepthProcessorConfig(
    features=['volume', 'variance'],
    samples=50000,
    ticks_per_bin=100
)
```

### Compatible Configuration Creation

```python
from represent import create_compatible_configs

# Create all three configs with consistent parameters
dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
    currency="AUDUSD",
    features=['volume', 'variance'],
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500,
    nbins=13,
    samples=50000
)

# All configs will have consistent currency and price movement parameters
print(f"All currencies match: {dataset_cfg.currency == threshold_cfg.currency}")  # True
print(f"Price params consistent: {dataset_cfg.lookback_rows == threshold_cfg.lookback_rows}")  # True
```

## Usage Examples

### Dataset Building

```python
from represent import DatasetBuilder, DatasetBuildConfig

# Use focused config for dataset building
dataset_config = DatasetBuilderConfig(
    currency="AUDUSD",
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500
)

# Dataset build configuration (legacy)
dataset_build_config = DatasetBuildConfig(
    currency="AUDUSD",
    force_uniform=True,
    nbins=13
)

# Create builder with focused config
builder = DatasetBuilder(
    config=dataset_config,
    dataset_config=dataset_build_config
)
```

### Global Threshold Calculation

```python
from represent import GlobalThresholdCalculator

# Use focused config for threshold calculation
threshold_config = GlobalThresholdConfig(
    currency="AUDUSD",
    nbins=13,
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500,
    sample_fraction=0.5
)

calculator = GlobalThresholdCalculator(config=threshold_config)
thresholds = calculator.calculate_global_thresholds("/path/to/dbn/files/")
```

### Market Depth Processing

```python
from represent import MarketDepthProcessor

# Use focused config for processing
processor_config = MarketDepthProcessorConfig(
    features=['volume', 'variance'],
    samples=50000,
    ticks_per_bin=100
)

processor = MarketDepthProcessor(config=processor_config)

# Process market data
import polars as pl
df = pl.read_parquet("symbol_data.parquet")
tensor = processor.process(df)
print(f"Tensor shape: {tensor.shape}")  # (2, 402, 500) for 2 features
```

## Factory Functions

### Individual Factory Functions

```python
from represent import (
    create_dataset_builder_config,
    create_threshold_config,
    create_processor_config
)

# Create specific configs
dataset_cfg = create_dataset_builder_config(currency="EURUSD", lookback_rows=3000)
threshold_cfg = create_threshold_config(currency="EURUSD", nbins=9)
processor_cfg = create_processor_config(features=['volume'], samples=25000)
```

### Currency-Specific Optimizations

The package automatically applies optimizations based on currency:

```python
# GBPUSD gets shorter lookforward for volatility
dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(currency="GBPUSD")
print(f"GBPUSD lookforward: {dataset_cfg.lookforward_input}")  # 3000 instead of 5000

# JPY pairs get different pip sizes and fewer bins
dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(currency="USDJPY")
print(f"USDJPY nbins: {threshold_cfg.nbins}")  # 9 instead of 13
print(f"USDJPY micro_pip_size: {processor_cfg.micro_pip_size}")  # 0.001 instead of 0.00001
```

## Legacy Compatibility

The package maintains backward compatibility through the legacy `create_represent_config` function:

```python
from represent import create_represent_config

# Legacy function returns tuple of three configs
configs = create_represent_config(
    currency="AUDUSD",
    features=['volume'],
    lookback_rows=5000
)

dataset_cfg, threshold_cfg, processor_cfg = configs
```

## Migration Guide

### From Old RepresentConfig

**Old way:**
```python
from represent.config import RepresentConfig

config = RepresentConfig(
    currency="AUDUSD",
    features=['volume'],
    lookback_rows=5000,
    nbins=13,
    samples=50000
)
```

**New way:**
```python
from represent import create_compatible_configs

dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
    currency="AUDUSD",
    features=['volume'],
    lookback_rows=5000,
    nbins=13,
    samples=50000
)
```

### Module Usage Updates

**DatasetBuilder:**
```python
# Old: builder = DatasetBuilder(config)
# New: builder = DatasetBuilder(config=dataset_cfg)
```

**GlobalThresholdCalculator:**
```python  
# Old: calculator = GlobalThresholdCalculator(config)
# New: calculator = GlobalThresholdCalculator(config=threshold_cfg)
```

**MarketDepthProcessor:**
```python
# Old: processor = MarketDepthProcessor(config)  
# New: processor = MarketDepthProcessor(config=processor_cfg)
```

## Configuration Reference

### DatasetBuilderConfig Fields
- `currency`: Currency pair (e.g., "AUDUSD") 
- `lookback_rows`: Historical data rows for price calculation
- `lookforward_input`: Future data rows for price calculation
- `lookforward_offset`: Offset before future window starts
- `min_required_samples`: Auto-computed minimum samples needed

### GlobalThresholdConfig Fields  
- All DatasetBuilderConfig fields plus:
- `nbins`: Number of classification bins
- `max_samples_per_file`: Performance optimization limit
- `sample_fraction`: Fraction of files to use for threshold calculation
- `jump_size`: Sampling step size for performance

### MarketDepthProcessorConfig Fields
- `features`: List of features to extract (['volume', 'variance', 'trade_counts'])
- `samples`: Number of samples to process 
- `ticks_per_bin`: Ticks per time bin
- `micro_pip_size`: Price precision
- `time_bins`: Auto-computed time dimension
- `output_shape`: Auto-computed tensor shape

## Best Practices

1. **Use Compatible Configs**: Always use `create_compatible_configs()` when you need configs for multiple modules
2. **Validate Early**: Config validation happens at creation time - catch errors early  
3. **Currency Optimization**: Let the package apply currency-specific optimizations automatically
4. **Focused Usage**: Only create the configs you actually need for your specific modules
5. **Type Safety**: Leverage Pydantic validation - the configs will tell you if parameters are invalid

## Advanced Usage

### Custom Validation

```python
from represent import DatasetBuilderConfig

try:
    config = DatasetBuilderConfig(
        currency="INVALID",  # Will raise validation error
        lookback_rows=-100   # Will raise validation error  
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### Computed Fields

```python
config = DatasetBuilderConfig(
    lookback_rows=5000,
    lookforward_input=4000,
    lookforward_offset=500
)

print(f"Min required samples: {config.min_required_samples}")  # 9500 (auto-computed)
```