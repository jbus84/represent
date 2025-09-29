# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Represent** is a high-performance Python package for quantitative finance ML applications that provides:

1. **Normalized LOB Representations**: Transform raw limit order book data into ML-ready tensor formats
2. **Modular Target Generation**: 15+ sophisticated labeling approaches with Bayesian parameter optimization
3. **Research Integration**: Academic TStrends methods with optimized parameters for real-world performance

**CRITICAL: This system is performance-optimized for ML training. Focus on efficiency and correctness.**

## Core Architecture

### 1. LOB Data Processing Pipeline

The package transforms raw market data into normalized tensor representations:

```python
Raw DBN/Parquet → MarketDepthProcessor → Normalized Tensors (N, 402, 500)
                                       ↑
                                Multi-feature extraction:
                                - Volume depth
                                - Price variance  
                                - Trade counts
```

**Key Components:**
- `MarketDepthProcessor`: Core LOB → tensor transformation
- `MarketDepthProcessorConfig`: Type-safe configuration with auto-computed fields
- **Output Shape**: (N_features, 402_price_levels, 500_time_bins)

### 2. Modular Target Generation Architecture

**IMPORTANT: All target generation uses pluggable `TargetGenerator` interface with factory pattern.**

```python
Raw Data → Multiple TargetGenerators → Combined Dataset
           ↑                          ↑
    - Quantile Classification    Multiple target columns
    - GA Labeling (NEW)         with optimized parameters
    - Academic TStrends
    - Regression targets
```

**Key Components:**
- `TargetGenerator` base interface for all labeling approaches
- `TargetGeneratorFactory` for creating generators by name
- `ModularDatasetBuilder` combines multiple generators
- **15+ generators available** from classification to evolutionary optimization

## Available Target Generators

### Classification Methods

1. **Quantile Classification** (`quantile_classification`)
   - Traditional percentile-based balanced labeling
   - Parameters: `nbins` (number of classes)
   - Use case: Balanced multi-class direction prediction

2. **GA Labeling** (`ga_labeling`) ⭐ **NEW**
   - Genetic algorithm-optimized trading labels with memory-efficient chunked processing
   - **OPTIMIZED**: population_size=50, max_generations=75, lookforward_window=250
   - **CORRECTED**: transaction_cost=0.00007 (proper 0.7 pip conversion)
   - **Memory Efficient**: Uses chunked population processing and int8 chromosomes
   - Use case: Performance-optimized evolutionary trading signals with realistic parameters

3. **Binary CTL** (`binary_ctl`)
   - Academic binary trend labeling from TStrends
   - **OPTIMIZED**: omega=0.0 (240.20% returns)
   - Use case: Research benchmarking, binary trend detection

4. **Ternary CTL** (`ternary_ctl`)
   - Academic ternary trend labeling (Down/Neutral/Up)
   - **CORRECTED**: marginal_change_thres=0.0005, window_size=10
   - **Performance**: Balanced distribution (27.6% Down, 44.0% Neutral, 28.4% Up)
   - Use case: 3-class trend analysis with balanced labels

5. **Oracle Binary/Ternary** (`oracle_binary`, `oracle_ternary`)
   - Theoretical optimal labels for benchmarking
   - **OPTIMIZED**: Various transaction cost and neutral factor parameters
   - Use case: Performance upper bounds, research comparison

### Regression Methods

1. **Log Return Horizons** (`log_return_horizons`) ⭐ **NEW**
   - Multi-horizon log return predictions (1k-5k ticks)
   - Output: 5 continuous targets for different time scales
   - Use case: Multi-scale trading strategies

2. **Directional MFE** (`directional_mfe`)
   - Maximum Favorable Excursion for long/short positions
   - Output: Buy MFE and Sell MFE in basis points
   - Use case: Position sizing optimization

3. **Volatility Scaled Returns** (`volatility_scaled_returns`)
   - Adaptive returns with dynamic volatility-based barriers
   - Use case: Regime-aware trading with risk management

4. **Remaining Value Tuner** (`remaining_value_tuner`)
   - Trend potential prediction (continuous trend magnitude)
   - Use case: Advanced entry/exit timing

5. **Rolling Volatility** (`volatility`)
   - Future volatility forecasting
   - Use case: Risk management, options trading

## Parameter Optimization Results

**All generators include OPTIMIZED parameters from Bayesian optimization:**

| Method | Returns | Key Insight |
|--------|---------|-------------|
| GA Labeling | **71.34%** | Evolutionary optimization dominates |
| Binary CTL | **240.20%** | Zero omega filtering optimal |
| Ternary CTL | Balanced | Corrected parameters for practical classification |
| Oracle Binary | 1.23% | Minimal transaction costs optimal |
| Oracle Ternary | 0.18% | Low neutral factor favors directional signals |

**Optimization Benefits:**
- **Transaction Cost Aware**: All optimized for realistic 0.7 pip fees (0.00007 decimal)
- **Returns-Based Fitness**: Parameters maximize actual trading performance
- **Bayesian Efficiency**: Gaussian Process finds global optima
- **Practical Constraints**: Maintains realistic trading requirements with hundreds of tick lookahead
- **Memory Efficient**: GA uses chunked processing to handle large datasets without memory issues

## Development Workflow

### Primary Workflow: Target Generation

```python
from represent import ModularDatasetBuilder, TargetGeneratorFactory

# Create generators with optimized parameters
generators = [
    # Traditional balanced classification
    TargetGeneratorFactory.create("quantile_classification", nbins=13),
    
    # Evolutionary optimization (OPTIMIZED & MEMORY EFFICIENT)
    TargetGeneratorFactory.create("ga_labeling", 
                                 population_size=50, max_generations=75,
                                 lookforward_window=250, 
                                 transaction_cost=0.00007),
    
    # Multi-horizon regression
    TargetGeneratorFactory.create("log_return_horizons", 
                                 horizons=[1000, 2000, 3000, 4000, 5000]),
    
    # Academic methods (OPTIMIZED)
    TargetGeneratorFactory.create("binary_ctl", omega=0.0),
    TargetGeneratorFactory.create("ternary_ctl", 
                                 marginal_change_thres=0.0446, window_size=501),
]

# Build dataset with all target types
builder = ModularDatasetBuilder(generators)
dataset = builder.build_from_parquet("symbol_data.parquet")

# Result: Multiple optimized target columns
```

### LOB Processing Workflow

```python
from represent import MarketDepthProcessor
from represent.configs import MarketDepthProcessorConfig

# Configure multi-feature processing
config = MarketDepthProcessorConfig(
    features=['volume', 'variance', 'trade_counts'],
    samples=50000,
    ticks_per_bin=100
)

processor = MarketDepthProcessor(config)
tensor_data = processor.process(market_data)  # Shape: (3, 402, 500)
```

## Key Data Structures

### Input Data Requirements
- **DBN files**: `.dbn` or `.dbn.zst` (Databento format)
- **Parquet files**: Polars-compatible with required columns
- **Required columns**: `mid_price`, `ts_event`, optionally `symbol`

### Output Formats
- **LOB Tensors**: (N_features, 402, 500) normalized representations
- **Target DataFrames**: Polars DataFrames with multiple target columns
- **Combined Datasets**: Features + targets ready for ML training

### Multi-Feature Shapes
- **1 feature**: (402, 500) - 2D tensor
- **2+ features**: (N, 402, 500) - 3D tensor with feature dimension first

## Performance Requirements

**Critical Performance Targets:**
- **LOB Processing**: 300+ samples/second
- **Target Generation**: 1500+ samples/second for all methods
- **Memory Usage**: <8GB RAM for large datasets
- **Storage Efficiency**: 90% reduction with target-only architecture

## Development Standards

### Code Organization
- Use modular `TargetGenerator` interface for all labeling logic
- Implement factory pattern for generator creation
- Type-safe Pydantic configurations with auto-computed fields
- Performance-critical: vectorized operations with Polars/NumPy

### Testing Requirements
- **80% code coverage minimum** (mandatory for all PRs)
- Performance regression tests for critical paths
- Integration tests for complete workflows
- Realistic fixtures, avoid excessive mocking

### Configuration System
- **Focused configs**: Separate Pydantic models per module
- **Auto-computed fields**: Derive dependent parameters automatically
- **Type safety**: Full validation with descriptive errors
- **Backwards compatibility**: Legacy configs still supported

## Instructions for Claude

When working on this codebase:

1. **TARGET GENERATION FIRST** - All labeling uses the modular `TargetGenerator` interface
2. **PERFORMANCE CRITICAL** - Every change must consider performance impact
3. **80% COVERAGE MANDATORY** - All code must maintain coverage threshold
4. **OPTIMIZED PARAMETERS** - Use the Bayesian-optimized parameters for all generators
5. **TYPE SAFETY** - Fix all type annotations, use Pydantic configs
6. **VECTORIZED OPERATIONS** - Use Polars/NumPy for all data operations
7. **NO BACKWARDS COMPATIBILITY** - Remove old approaches that don't fit architecture
8. **NO FALLBACKS** - Never use default/fallback parameters that hide bugs or inconsistencies
9. **TEST THOROUGHLY** - Include performance tests with benchmarks
10. **VALIDATE AT STARTUP** - Pre-validate schemas, use lookups over calculations
11. **MODULAR DESIGN** - Each target generator should be independent and composable

## Key Files and Components

### Core Processing
- `represent/pipeline.py` - MarketDepthProcessor (LOB → tensors)
- `represent/configs.py` - Type-safe configuration models
- `represent/modular_dataset_builder.py` - Multi-target dataset builder

### Target Generators
- `represent/target_generators/base.py` - Base TargetGenerator interface
- `represent/target_generators/classification.py` - Classification generators
- `represent/target_generators/regression.py` - Regression generators
- `represent/target_generators/ga_labeling.py` - Genetic algorithm labeling (NEW)
- `represent/target_generators/tstrends_labeling.py` - Academic TStrends methods
- `represent/target_generators/factory.py` - Generator factory pattern

### Optimization
- `represent/parameter_optimization.py` - Bayesian parameter optimization
- Results files: `optimized_*_params.json` - Optimized parameters for all methods

### Examples & Visualization
- `examples/labeling_approaches_visualization.py` - Complete comparison plots
- Generates 4 professional plots comparing all 15+ labeling approaches

This architecture provides a comprehensive, performance-optimized foundation for quantitative finance ML applications with state-of-the-art labeling methods and parameter optimization.