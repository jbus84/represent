# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **high-performance** Python package called "represent" that creates normalized market depth representations from limit order book (LOB) data using a **parquet-based machine learning pipeline**. The core objective is to produce market depth arrays with pre-computed classification labels for efficient ML training.

**CRITICAL: This system must be extremely performance-optimized for ML training applications. Every millisecond matters.**

### Streamlined 2-Stage Architecture (v4.0.0) - DBN→Classified-Parquet→ML Pipeline

The package follows a **streamlined two-stage pipeline** for maximum performance and uniform distribution:

1. **Stage 1: Direct DBN-to-Classified-Parquet**: Load DBN file, split by symbol, apply classification, save symbol-specific classified parquet files
2. **Stage 2: ML Training**: Lazy loading from classified parquet for memory-efficient ML training

**Key Performance Improvements:**
- **No Intermediate Files**: Direct conversion eliminates I/O overhead
- **Symbol-Level Processing**: Each symbol processed independently with full history
- **True Uniform Distribution**: Classification applied with full dataset context
- **Faster Loading**: Single parquet files per symbol load much faster

**Streamlined Processing Logic:**
1. **Load DBN File**: Read entire DBN file into polars DataFrame
2. **Split by Symbol**: Group data by symbol column
3. **Filter by Thresholds**: Keep symbols with ≥ (input_rows + lookforward_rows) samples
4. **Apply Classification**: For each symbol, add classification_label column based on price movement
5. **Filter Rows**: Drop rows without sufficient historical (input_rows) or future (lookforward_rows) data
6. **Save Symbol Parquets**: Save each symbol's DataFrame as {currency}_{symbol}_classified.parquet

## Core Workflow

### Stage 1: Direct DBN-to-Classified-Parquet Conversion

```python
from represent import ParquetClassifier

# Process DBN file directly to classified parquet files by symbol
classifier = ParquetClassifier(
    currency="AUDUSD",
    features=['volume', 'variance'],       # Multi-feature extraction
    input_rows=5000,                       # Historical data required for features
    lookforward_rows=500,                  # Future data required for classification
    min_symbol_samples=1000,               # Minimum samples per symbol
    force_uniform=True                     # Guarantee uniform class distribution
)

stats = classifier.process_dbn_to_classified_parquets(
    dbn_path="market_data.dbn.zst",
    output_dir="/data/classified/",        # Directory for symbol-specific classified files
)

# Output: /data/classified/AUDUSD_M6AM4_classified.parquet, /data/classified/AUDUSD_M6AM5_classified.parquet, etc.
```

**Key Features:**
- **Single-Pass Processing**: DBN → Classified Parquet in one step
- **Symbol Isolation**: Each symbol processed independently with full context
- **Uniform Distribution**: True uniform distribution using full dataset
- **Row Filtering**: Automatic filtering of rows without sufficient history/future
- **Preserved Schema**: Original DBN columns + classification_label column
- **Fast Loading**: Optimized parquet files for ML training

### Stage 2: ML Training (External Implementation)

The classified parquet files are ready for ML training. **Dataloader functionality has been moved out of the represent package** to allow for customization in your ML training repository.

**See `DATALOADER_MIGRATION_GUIDE.md` for comprehensive instructions on rebuilding the dataloader with Claude.**

**Expected Workflow:**
```python
# In your ML training repository, implement a custom dataloader
# that reads the classified parquet files from Stage 1

# Standard PyTorch training loop structure:
for features, labels in your_custom_dataloader:
    # features: torch.Tensor shape (batch_size, [N_features,] 402, 500)  
    # labels: torch.Tensor shape (batch_size,) with uniform distribution (7.69% each class)
    outputs = model(features)
    loss = criterion(outputs, labels)
    # ... training logic
```

**Key Benefits:**
- **Guaranteed Uniform Distribution**: Each class has equal representation from Stage 1
- **Symbol Flexibility**: Train on specific symbols or all available symbols
- **Custom Implementation**: Tailor dataloader to your specific ML framework needs
- **Faster Loading**: Single parquet files per symbol load much faster
- **No Intermediate Processing**: Files are ready for ML training immediately

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

### ParquetClassifier (`represent/parquet_classifier.py`)

Primary streamlined DBN-to-classified-parquet processor.

```python
class ParquetClassifier:
    """
    Process DBN files directly to classified parquet files with:
    - Single-pass DBN → Classified Parquet processing
    - Symbol-level processing with full context
    - True uniform distribution using quantile-based classification
    - Row filtering for insufficient history/future data
    - Preserved schema with added classification_label column
    """
```

### Custom DataLoader (External Implementation)

**Dataloader functionality moved to ML training repositories.**

See `DATALOADER_MIGRATION_GUIDE.md` for comprehensive instructions on rebuilding with:
- Multi-symbol dataset support
- Guaranteed uniform class distribution  
- Memory usage independent of dataset size
- Symbol-specific sampling strategies

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
# Primary: Streamlined DBN-to-Classified-Parquet Processing
from represent import ParquetClassifier, process_dbn_to_classified_parquets
from represent import classify_parquet_file, batch_classify_parquet_files

# ML Training DataLoader (implement in your ML repo)
# See DATALOADER_MIGRATION_GUIDE.md for implementation instructions

# Core Processing  
from represent import MarketDepthProcessor, process_market_data

# Configuration
from represent import create_represent_config

# Alternative: Unlabeled conversion approach (if needed)
from represent import convert_dbn_to_parquet, batch_convert_unlabeled
```

### Dynamic Configuration Generation

The represent package now uses **dynamic configuration generation** instead of static files:

```python
from represent import generate_classification_config_from_parquet

# Generate optimized config from actual parquet data
config, metrics = generate_classification_config_from_parquet(
    parquet_files="/path/to/data.parquet",
    currency="AUDUSD"
)

print(f"Quality: {metrics['validation_metrics']['quality']}")
print(f"Average deviation: {metrics['validation_metrics']['avg_deviation']:.1f}%")
```

**Benefits of Dynamic Configuration:**
- ✅ **No Static Files**: Eliminates need for `represent/configs/` directory
- ✅ **Data-Driven**: Optimized thresholds based on actual price movements
- ✅ **Uniform Distribution**: Quantile-based approach ensures balanced classes
- ✅ **Quality Assessment**: Real-time validation and quality metrics

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
- Implement custom dataloader in your ML repository (see DATALOADER_MIGRATION_GUIDE.md)
- Leverage currency configs for market-specific optimizations
- Pre-computed labels eliminate runtime classification overhead

## Example Workflows

### Complete ML Pipeline (3-Stage)

```python
from represent import convert_dbn_to_parquet, classify_parquet_file
# Note: Dataloader functionality moved to your ML training repo

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

# Stage 3: ML Training (implement custom dataloader in your ML repo)
print("Stage 3: Ready for ML training...")
print("Classified parquet files available in /data/classified/")
print("See DATALOADER_MIGRATION_GUIDE.md to implement custom dataloader")

# Example training structure (implement custom_dataloader):
# for features, labels in your_custom_dataloader:
#     # features: (32, 2, 402, 500) for volume+variance
#     # labels: (32,) with uniform distribution (7.69% each class 0-12)
#     outputs = model(features)
#     loss = criterion(outputs, labels)
#     
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
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