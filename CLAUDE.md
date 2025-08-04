# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **high-performance** Python package called "represent" that creates normalized market depth representations from limit order book (LOB) data using a **parquet-based machine learning pipeline**. The core objective is to produce market depth arrays with pre-computed classification labels for efficient ML training.

**CRITICAL: This system must be extremely performance-optimized for ML training applications. Every millisecond matters.**

### New Architecture (v2.0.0) - DBN→Parquet→ML Pipeline

The package follows a **two-stage pipeline** for maximum performance:

1. **Offline Preprocessing**: Convert DBN files to labeled parquet datasets
2. **Online Training**: Lazy loading from parquet for memory-efficient ML training

## Core Workflow

### Stage 1: DBN to Labeled Parquet Conversion

```python
from represent import convert_dbn_file

# Convert DBN file to labeled parquet dataset  
stats = convert_dbn_file(
    dbn_path="market_data.dbn.zst",
    output_path="labeled_dataset.parquet", 
    currency="AUDUSD",                    # Currency-specific classification config
    features=['volume', 'variance'],      # Multi-feature extraction
    symbol_filter="M6AM4"                 # Optional symbol filtering
)
```

**Key Features:**
- **Pre-computed Classification Labels**: Price movement classification during conversion
- **Multi-feature Extraction**: Volume, variance, and trade count features
- **Currency-specific Configs**: AUDUSD, GBPUSD, EURJPY market configurations
- **Efficient Storage**: Compressed parquet with optimal row groups

### Stage 2: Lazy ML Training

```python  
from represent import create_market_depth_dataloader

# Create lazy dataloader for memory-efficient training
dataloader = create_market_depth_dataloader(
    parquet_path="labeled_dataset.parquet",
    batch_size=32,
    shuffle=True,
    sample_fraction=0.1,                  # Use 10% of dataset for quick iteration
    num_workers=4                         # Parallel loading
)

# Standard PyTorch training loop
for features, labels in dataloader:
    # features: torch.Tensor shape (batch_size, [N_features,] 402, 500)  
    # labels: torch.Tensor shape (batch_size,) with classification targets
    outputs = model(features)
    loss = criterion(outputs, labels)
    # ... training logic
```

**Key Benefits:**
- **Memory Efficient**: Load only required batches, not entire dataset
- **Pre-computed Labels**: No runtime classification overhead
- **Lazy Loading**: Train on datasets larger than available RAM
- **PyTorch Native**: Direct tensor output for ML workflows

## Core Data Structures

### Market Depth Features

1. **Volume Features** (Default):
   - Source: `ask_sz_XX`, `bid_sz_XX` columns from DBN
   - Output: Normalized cumulative volume differences
   - Shape: (402, 500) for single feature

2. **Variance Features**:
   - Source: Price variance data from DBN files  
   - Output: Normalized cumulative variance differences
   - Shape: (402, 500) for single feature

3. **Trade Count Features**:
   - Source: `ask_ct_XX`, `bid_ct_XX` columns from DBN
   - Output: Normalized cumulative trade count differences
   - Shape: (402, 500) for single feature

### Multi-Feature Output Shapes

**Shape Determination Logic:**
- **1 feature**: Output shape `(402, 500)` - 2D tensor
- **2+ features**: Output shape `(N, 402, 500)` - 3D tensor with feature dimension first

```python
# Examples:
features=['volume'] → tensor shape (402, 500)
features=['volume', 'variance'] → tensor shape (2, 402, 500)  
features=['volume', 'variance', 'trade_counts'] → tensor shape (3, 402, 500)
```

### Price and Time Dimensions

```python
PRICE_LEVELS = 402       # Total price bins (200 bid + 200 ask + 2 mid)
TIME_BINS = 500          # Time dimension  
MICRO_PIP_SIZE = 0.00001 # Price precision
TICKS_PER_BIN = 100      # Tick aggregation for time bins
```

## Classification System

### Pre-computed Classification Labels

Labels are computed during DBN→parquet conversion based on price movement:

```python
# Classification logic (applied during conversion)
def classify_price_movement(start_price, end_price, thresholds):
    movement = (end_price - start_price) / MICRO_PIP_SIZE
    if movement > thresholds.up_threshold:
        return 2  # UP
    elif movement < thresholds.down_threshold:  
        return 0  # DOWN
    else:
        return 1  # NEUTRAL
```

### Currency-Specific Configuration

Each currency has optimized classification thresholds:

```python
# AUDUSD Configuration (represent/configs/audusd.json)
{
  "classification": {
    "nbins": 13,
    "up_threshold": 5.0,     # micro-pips
    "down_threshold": -5.0,   # micro-pips
    "lookforward_input": 5000,
    "lookback_rows": 5000,
    "lookforward_offset": 500
  }
}
```

## Development Setup

The project uses Python 3.12, uv for package management, and modern Python packaging standards.

```bash
# Primary development commands  
make install      # Install dependencies and setup environment
make test         # Run all tests with coverage (requires 80%)
make test-fast    # Run tests excluding performance tests
make coverage-html # Generate HTML coverage report
make lint         # Run linting checks
make typecheck    # Run type checking
make format       # Format code
make build        # Build package
make clean        # Clean build artifacts

# Direct uv commands (fallback)
uv sync --all-extras
uv run pytest --cov=represent  
uv run ruff check .
uv run pyright
uv build
```

## Development Standards

### Code Organization
- Use Single Responsibility Principle consistently
- Prefer Pydantic models over standard Python classes
- Keep all related functionality in single modules, not split across files

### Error Handling
- Always provide graceful degradation paths
- Log errors with appropriate context for debugging

### Data Processing (Performance Critical)

#### Parquet-Based Processing Requirements:
- **Lazy Loading**: Load only required data chunks from parquet
- **Vectorized Operations**: Use polars for high-performance data operations
- **Pre-allocated Buffers**: No dynamic memory allocation in hot paths
- **Feature-agnostic Pipeline**: Same processing handles volume, variance, trade counts
- **Memory Pre-allocation**: Allocate buffers for maximum enabled features at initialization
- **Schema Validation**: Validate feature availability at startup, not runtime

#### DBN Processing Requirements:
- **Streaming Decompression**: Handle zstandard compression efficiently
- **Batch Processing**: Process large DBN files in configurable chunks
- **Price Mapping**: Map prices to 402-level grid using lookup tables
- **Multi-feature Extraction**: Extract all enabled features in single pass

### Testing (Performance Focused)
- Organize tests by domain matching source structure
- Use realistic fixtures, avoid excessive mocking in integration tests
- Test error conditions and recovery scenarios
- **MANDATORY: 80% code coverage minimum** - all PRs must maintain this threshold
- **MANDATORY: Performance test critical paths with benchmarks**
- **Benchmark against target latencies: <10ms for array generation, <50ms for batch processing**
- **Memory usage tests** - ensure no memory leaks in long-running processes
- **Coverage reporting** - use `make test-coverage` and `make coverage-html` for detailed reports

## Performance Requirements (NON-NEGOTIABLE)

**CRITICAL LATENCY TARGETS:**
- **DBN Conversion**: <1000 samples/second sustained processing
- **Parquet Loading**: <10ms for single sample loading (with caching)
- **Batch Processing**: <50ms for 32-sample batch generation
- **Feature Processing**: <2ms additional latency per additional feature (linear scaling)
- **Memory Usage**: <4GB RAM for training on multi-GB parquet datasets

**THROUGHPUT REQUIREMENTS:**
- **Training**: 1000+ samples/second during ML training
- **Conversion**: 500+ samples/second during DBN→parquet conversion
- **Parallel Processing**: Must scale linearly with CPU cores

**MEMORY CONSTRAINTS:**
- **Training Memory**: <4GB RAM regardless of parquet dataset size
- **Conversion Memory**: <8GB RAM during DBN processing
- **Cache Efficiency**: >90% cache hit rate for frequently accessed samples
- **Feature Scaling**: Linear memory usage per additional feature

## Key Components

### DBN to Parquet Converter (`represent/converter.py`)

High-performance converter from DBN files to labeled parquet datasets.

```python
class DBNToParquetConverter:
    """
    Convert DBN files to labeled parquet datasets with:
    - Automatic classification labeling
    - Multi-feature extraction (volume, variance, trade_counts)
    - Currency-specific configurations
    - Efficient batch processing
    """
```

### Lazy Parquet DataLoader (`represent/lazy_dataloader.py`)

Memory-efficient PyTorch dataloader for parquet datasets.

```python
class LazyParquetDataLoader:
    """
    Lazy loading dataloader with:
    - Memory usage independent of dataset size
    - LRU caching for performance
    - Configurable sampling strategies
    - Direct tensor deserialization
    """
```

### Market Depth Processor (`represent/pipeline.py`)

Core market depth processing logic for feature extraction.

```python
class MarketDepthProcessor:
    """
    Process market data to generate:
    - Price level mapping (402 levels)
    - Time bin aggregation (500 bins)  
    - Multi-feature extraction
    - Normalized output tensors
    """
```

## API Reference

### Main Entry Points

```python
# DBN Conversion
from represent import convert_dbn_file, batch_convert_dbn_files

# Lazy DataLoader
from represent import create_market_depth_dataloader

# Core Processing  
from represent import MarketDepthProcessor, process_market_data

# Configuration
from represent import load_currency_config, ClassificationConfig
```

### Currency Configurations

Available currency configurations:
- **AUDUSD**: `represent/configs/audusd.json`
- **GBPUSD**: `represent/configs/gbpusd.json` 
- **EURJPY**: `represent/configs/eurjpy.json`

Each config includes optimized classification thresholds and processing parameters.

## Instructions for Claude

When working on this codebase:

1. **PARQUET-FIRST ARCHITECTURE** - All data processing should assume parquet-based lazy loading
2. **PERFORMANCE FIRST** - Every code change must be evaluated for performance impact
3. **80% COVERAGE MANDATORY** - All code must maintain 80% test coverage minimum
4. **NO RING BUFFERS** - The old ring buffer architecture has been completely replaced
5. **TYPE SAFETY** - Fix all type annotations for better IDE support and runtime safety
6. **LAZY LOADING ONLY** - Remove any references to streaming/real-time data ingestion
7. **PRE-COMPUTED LABELS** - Classification happens during conversion, not training
8. **MEMORY EFFICIENCY** - Optimize for training on datasets larger than RAM
9. **VECTORIZED OPERATIONS** - Use polars/numpy for all data operations
10. **VALIDATE AT STARTUP** - Pre-validate schemas, use lookup tables over calculations
11. **TEST THOROUGHLY** - Include performance regression tests with every change
12. **DOCUMENT PERFORMANCE** - Explain performance implications of design decisions

## Migration from v1.x

**CURRENT ARCHITECTURE COMPONENTS:**

✅ **ACTIVE COMPONENTS:**
- `converter.py` - DBN to labeled parquet conversion with classification
- `dataloader.py` - Lazy parquet loading for ML training  
- `config.py` - Currency-specific configurations with YAML support
- `api.py` - High-level convenience API for common workflows
- `constants.py` - Feature definitions and output shape calculations

**WORKFLOW INTEGRATION:**
- Use `convert_dbn_file()` for offline preprocessing
- Use `LazyParquetDataLoader` for memory-efficient training
- Leverage currency configs for market-specific optimizations
- Pre-computed labels eliminate runtime classification overhead

## Example Workflows

### Complete ML Pipeline

```python
from represent import convert_dbn_file, create_market_depth_dataloader
import torch
import torch.nn as nn

# 1. Convert DBN to labeled parquet (run once)
stats = convert_dbn_file(
    dbn_path="data/AUDUSD-20240101.dbn.zst",
    output_path="data/AUDUSD_labeled.parquet",
    currency="AUDUSD",
    features=['volume', 'variance'],
    chunk_size=50000
)

# 2. Create lazy dataloader for training
dataloader = create_market_depth_dataloader(
    parquet_path="data/AUDUSD_labeled.parquet", 
    batch_size=32,
    shuffle=True,
    sample_fraction=0.2,  # Use 20% of data
    num_workers=4
)

# 3. Train PyTorch model
model = nn.Sequential(
    nn.Conv2d(2, 32, 3),  # 2 features: volume + variance
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 3)      # 3-class classification
)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for features, labels in dataloader:
        # features: (32, 2, 402, 500) for volume+variance
        # labels: (32,) with classification targets 0,1,2
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Batch Processing Multiple Files

```python
from represent import batch_convert_dbn_files

# Convert all DBN files in directory
results = batch_convert_dbn_files(
    input_directory="data/dbn_files/",
    output_directory="data/parquet_datasets/", 
    currency="AUDUSD",
    pattern="*.dbn*",
    features=['volume'],
    chunk_size=25000
)

print(f"Converted {len(results)} files successfully")
```

This architecture provides maximum performance for ML training while maintaining flexibility for different market configurations and feature combinations.