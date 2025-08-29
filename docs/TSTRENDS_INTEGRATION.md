# TStrends Integration Guide

## Overview

The represent package now includes modular target generators based on the [tstrends library](https://github.com/agpenas/tstrends), which implements academic approaches for trend labelling in financial time series.

## Available TStrends Generators

### 1. **BinaryCTLGenerator** - Binary Cumulative Trend Labelling
```python
from represent import BinaryCTLGenerator, ModularDatasetBuilder

generator = BinaryCTLGenerator(
    omega=0.02,  # Threshold parameter for trend detection
    target_name="binary_ctl_label"
)

builder = ModularDatasetBuilder([generator])
dataset = builder.build_dataset(market_data)
# Result: binary_ctl_label column with binary trend labels
```

### 2. **TernaryCTLGenerator** - Ternary Cumulative Trend Labelling
```python
from represent import TernaryCTLGenerator

generator = TernaryCTLGenerator(
    marginal_change_thres=0.02,  # Marginal change threshold
    window_size=10,              # Window size for trend detection
    target_name="ternary_ctl_label"
)
# Result: Three-class labels (up, down, sideways)
```

### 3. **OracleBinaryTrendGenerator** - Oracle Binary Labelling
```python
from represent import OracleBinaryTrendGenerator

generator = OracleBinaryTrendGenerator(
    transaction_cost=0.001,  # Transaction cost for optimization
    target_name="oracle_binary_label"
)
# Result: Optimal binary trend labels using future price knowledge
```

### 4. **OracleTernaryTrendGenerator** - Oracle Ternary Labelling
```python
from represent import OracleTernaryTrendGenerator

generator = OracleTernaryTrendGenerator(
    transaction_cost=0.001,        # Transaction cost
    neutral_reward_factor=0.5,     # Neutral reward factor
    target_name="oracle_ternary_label"
)
# Result: Optimal ternary trend labels
```

### 5. **TunedTrendGenerator** - Parameter-Tuned Labelling
```python
from represent import TunedTrendGenerator

generator = TunedTrendGenerator(
    base_labeller_type="binary_ctl",  # Base labeller to tune
    omega=0.02,                       # Initial parameters
    target_name="tuned_trend_label"
)
# Result: Optimized trend labels with tuned parameters
```

## Installation

The tstrends library is required for these generators:

```bash
uv add git+https://github.com/agpenas/tstrends.git
```

Or if using pip:
```bash
pip install git+https://github.com/agpenas/tstrends.git
```

## Factory Pattern Support

TStrends generators are automatically registered with the factory:

```python
from represent import TargetGeneratorFactory

# Available tstrends generator types
available = TargetGeneratorFactory.list_available()
print(available)
# Output includes: 'binary_ctl', 'ternary_ctl', 'oracle_binary', 'oracle_ternary', 'tuned_trend'

# Create using factory
generator = TargetGeneratorFactory.create("binary_ctl", omega=0.02)
```

## Mixed Target Generation

Combine tstrends generators with existing represent generators:

```python
from represent import (
    ModularDatasetBuilder,
    QuantileClassificationGenerator,
    DirectionalMFEGenerator,
    BinaryCTLGenerator,
    OracleBinaryTrendGenerator
)

generators = [
    # Traditional represent generators
    QuantileClassificationGenerator(nbins=13),
    DirectionalMFEGenerator(lookforward_horizon=3000),
    
    # TStrends academic approaches
    BinaryCTLGenerator(omega=0.02),
    OracleBinaryTrendGenerator(transaction_cost=0.001),
]

builder = ModularDatasetBuilder(generators)
dataset = builder.build_dataset(market_data)

# Result dataset contains:
# - classification_label (quantile-based)
# - mfe_buy_bps, mfe_sell_bps (MFE regression)
# - binary_ctl_label (Binary CTL)
# - oracle_binary_label (Oracle binary)
```

## Configuration-Based Creation

```python
from represent import create_modular_builder

configs = [
    {"type": "quantile_classification", "nbins": 13},
    {"type": "binary_ctl", "omega": 0.02},
    {"type": "oracle_binary", "transaction_cost": 0.001},
    {"type": "ternary_ctl", "marginal_change_thres": 0.02, "window_size": 10}
]

builder = create_modular_builder(configs)
dataset = builder.build_dataset(market_data)
```

## Academic References

The tstrends library implements approaches from academic literature:

1. **Cumulative Trend Labelling (CTL)**: Binary and ternary trend detection based on cumulative price movements
2. **Oracle Labelling**: Optimal labelling using future price knowledge for benchmarking
3. **Parameter Tuning**: Optimization of labelling parameters for improved performance

## Key Benefits

1. **Academic Rigor**: Implements peer-reviewed trend labelling approaches
2. **Optimal Benchmarks**: Oracle labellers provide upper bounds for performance
3. **Modular Integration**: Seamlessly integrates with existing represent generators
4. **Parameter Optimization**: Built-in tuning for optimal labelling parameters
5. **Multiple Approaches**: Binary, ternary, and tuned variants available

## Example Usage

```python
#!/usr/bin/env python3
from represent import (
    ModularDatasetBuilder,
    BinaryCTLGenerator,
    OracleBinaryTrendGenerator
)
import polars as pl
import numpy as np

# Create sample market data
prices = np.random.randn(1000).cumsum() + 100
market_data = pl.DataFrame({
    "timestamp": np.arange(1000),
    "mid_price": prices,
    "volume": np.random.exponential(1000, 1000)
})

# Create generators
generators = [
    BinaryCTLGenerator(omega=0.02, target_name="ctl_binary"),
    OracleBinaryTrendGenerator(transaction_cost=0.001, target_name="oracle_binary")
]

# Build dataset
builder = ModularDatasetBuilder(generators)
dataset = builder.build_dataset(market_data)

print(f"Dataset shape: {dataset.shape}")
print(f"Columns: {dataset.columns}")

# Analyze labels
ctl_labels = dataset["ctl_binary"].to_numpy()
oracle_labels = dataset["oracle_binary"].to_numpy()

print(f"CTL unique labels: {np.unique(ctl_labels)}")
print(f"Oracle unique labels: {np.unique(oracle_labels)}")
```

## Error Handling

If tstrends is not installed, the generators will raise helpful error messages:

```python
try:
    from represent import BinaryCTLGenerator
    generator = BinaryCTLGenerator(omega=0.02)
except ImportError as e:
    print("Install tstrends: uv add git+https://github.com/agpenas/tstrends.git")
```

The modular system gracefully handles missing dependencies, so other generators continue to work even if tstrends is not available.

## Performance Considerations

- **Oracle labellers** are computationally intensive as they use future price knowledge
- **CTL approaches** are more efficient for real-time applications
- **Tuned generators** may require additional computation time for parameter optimization
- All generators support the same memory-efficient processing as other represent generators

This integration brings academic trend labelling approaches into the practical, high-performance represent ecosystem.