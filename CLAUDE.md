# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **high-performance** Python package called "represent" that creates normalized market depth representations from limit order book (LOB) data using a **parquet-based machine learning pipeline**. The core objective is to produce market depth arrays with pre-computed classification labels for efficient ML training.

**CRITICAL: This system must be extremely performance-optimized for ML training applications. Every millisecond matters.**

### New Architecture (v3.0.0) - DBN→Parquet→Classification→ML Pipeline

The package follows a **three-stage pipeline** for maximum flexibility and performance:

1. **Stage 1: Raw Data Processing**: Convert DBN files to unlabeled parquet datasets grouped by symbol
2. **Stage 2: Post-Processing Classification**: Apply symbol-specific classification to parquet data
3. **Stage 3: ML Training**: Lazy loading from classified parquet for memory-efficient ML training

## Core Workflow

### Stage 1: DBN to Unlabeled Parquet Conversion

```python
from represent import convert_dbn_to_parquet

# Convert DBN file to unlabeled parquet dataset with symbol grouping
stats = convert_dbn_to_parquet(
    dbn_path="market_data.dbn.zst",
    output_dir="/data/parquet/",           # Directory for symbol-grouped parquet files
    features=['volume', 'variance'],       # Multi-feature extraction
    group_by_symbol=True                   # Create separate files per symbol
)

# Output: /data/parquet/AUDUSD_M6AM4.parquet, /data/parquet/AUDUSD_M6AM5.parquet, etc.
```

**Key Features:**
- **Symbol-Grouped Storage**: Separate parquet files for each symbol
- **Unlabeled Data**: Raw market depth features without classification
- **Multi-feature Extraction**: Volume, variance, and trade count features  
- **Efficient Storage**: Compressed parquet with optimal row groups per symbol

### Stage 2: Post-Processing Classification

```python
from represent import apply_classification_to_parquet

# Apply uniform classification to common symbols only
classification_stats = apply_classification_to_parquet(
    parquet_dir="/data/parquet/",
    output_dir="/data/classified/",
    currency="AUDUSD",                     # Currency-specific thresholds
    min_samples=10000,                     # Only classify symbols with sufficient data
    target_distribution="uniform"          # Ensure uniform class distribution
)

# Output: /data/classified/AUDUSD_M6AM4_labeled.parquet with classification_label column
```

**Key Benefits:**
- **Symbol-Specific Analysis**: Different symbols can have different characteristics
- **Uniform Distribution**: Apply balanced sampling to achieve equal class representation
- **Flexible Thresholds**: Easy to experiment with different classification strategies
- **Common Symbols Only**: Focus on symbols with sufficient data for reliable classification

### Stage 3: Lazy ML Training

```python  
from represent import create_market_depth_dataloader

# Create lazy dataloader for memory-efficient training
dataloader = create_market_depth_dataloader(
    parquet_dir="/data/classified/",       # Directory with classified parquet files
    batch_size=32,
    shuffle=True,
    sample_fraction=0.1,                   # Use 10% of dataset for quick iteration
    num_workers=4,                         # Parallel loading
    symbols=["M6AM4", "M6AM5"]            # Optional: specific symbols only
)

# Standard PyTorch training loop with guaranteed uniform distribution
for features, labels in dataloader:
    # features: torch.Tensor shape (batch_size, [N_features,] 402, 500)  
    # labels: torch.Tensor shape (batch_size,) with uniform distribution (7.69% each class)
    outputs = model(features)
    loss = criterion(outputs, labels)
    # ... training logic
```

**Key Benefits:**
- **Guaranteed Uniform Distribution**: Each class has equal representation
- **Symbol Flexibility**: Train on specific symbols or all available symbols
- **Memory Efficient**: Load only required batches, not entire dataset
- **Reproducible**: Consistent classification across training runs

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

High-performance converter from DBN files to unlabeled parquet datasets with symbol grouping.

```python
class DBNToParquetConverter:
    """
    Convert DBN files to unlabeled parquet datasets with:
    - Symbol-based file grouping
    - Multi-feature extraction (volume, variance, trade_counts)
    - Efficient batch processing
    - No classification overhead during conversion
    """
```

### Post-Processing Classifier (`represent/classifier.py`)

Symbol-aware classification system for parquet datasets.

```python
class ParquetClassifier:
    """
    Apply classification to parquet datasets with:
    - Symbol-specific threshold calculation
    - Uniform distribution guarantee
    - Common symbols filtering
    - Flexible classification strategies
    """
```

### Lazy Parquet DataLoader (`represent/lazy_dataloader.py`)

Memory-efficient PyTorch dataloader for classified parquet datasets.

```python
class LazyParquetDataLoader:
    """
    Lazy loading dataloader with:
    - Multi-symbol dataset support
    - Guaranteed uniform class distribution
    - Memory usage independent of dataset size
    - Symbol-specific sampling strategies
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
# Stage 1: DBN to Parquet Conversion
from represent import convert_dbn_to_parquet, batch_convert_dbn_files

# Stage 2: Post-Processing Classification
from represent import apply_classification_to_parquet, ParquetClassifier

# Stage 3: ML Training DataLoader
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

### Complete ML Pipeline (3-Stage)

```python
from represent import convert_dbn_to_parquet, apply_classification_to_parquet, create_market_depth_dataloader
import torch
import torch.nn as nn

# Stage 1: Convert DBN to unlabeled parquet with symbol grouping
print("Stage 1: Converting DBN to parquet...")
conversion_stats = convert_dbn_to_parquet(
    dbn_path="data/AUDUSD-20240101.dbn.zst",
    output_dir="/data/parquet/",
    features=['volume', 'variance'],
    group_by_symbol=True
)

# Stage 2: Apply uniform classification to common symbols
print("Stage 2: Applying uniform classification...")
classification_stats = apply_classification_to_parquet(
    parquet_dir="/data/parquet/",
    output_dir="/data/classified/",
    currency="AUDUSD",
    min_samples=10000,                    # Only symbols with sufficient data
    target_distribution="uniform"         # Guarantee uniform class distribution
)

# Stage 3: Create lazy dataloader for training
print("Stage 3: Creating training dataloader...")
dataloader = create_market_depth_dataloader(
    parquet_dir="/data/classified/",
    batch_size=32,
    shuffle=True,
    sample_fraction=0.2,                  # Use 20% of data
    num_workers=4,
    symbols=["M6AM4", "M6AM5"]           # Train on specific symbols
)

# Train PyTorch model with guaranteed uniform distribution
model = nn.Sequential(
    nn.Conv2d(2, 32, 3),                 # 2 features: volume + variance
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 13)                    # 13-class uniform classification
)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for features, labels in dataloader:
        # features: (32, 2, 402, 500) for volume+variance
        # labels: (32,) with uniform distribution (7.69% each class 0-12)
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Symbol-Specific Analysis

```python
from represent import ParquetClassifier
import polars as pl

# Load specific symbol data for analysis
symbol_data = pl.read_parquet("/data/parquet/AUDUSD_M6AM4.parquet")
print(f"Symbol M6AM4 has {len(symbol_data):,} samples")

# Analyze price movement distribution for this symbol
classifier = ParquetClassifier(currency="AUDUSD")
movement_stats = classifier.analyze_symbol_distribution(symbol_data)

print(f"Price movement characteristics for M6AM4:")
print(f"  Mean: {movement_stats['mean']:.6f}")
print(f"  Std: {movement_stats['std']:.6f}")
print(f"  Suitable for classification: {movement_stats['sufficient_data']}")
```

### Batch Processing Multiple Files

```python
from represent import batch_convert_dbn_files

# Convert all DBN files to symbol-grouped parquet
results = batch_convert_dbn_files(
    input_directory="data/dbn_files/",
    output_directory="/data/parquet/", 
    features=['volume', 'variance'],
    group_by_symbol=True,
    pattern="*.dbn*"
)

print(f"Converted {len(results)} files successfully")
print(f"Generated parquet files in /data/parquet/ grouped by symbol")
```

This architecture provides maximum performance for ML training while maintaining flexibility for different market configurations and feature combinations.