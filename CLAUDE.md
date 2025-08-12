# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **high-performance** Python package called "represent" that creates normalized market depth representations from limit order book (LOB) data using a **parquet-based machine learning pipeline**. The core objective is to produce comprehensive symbol datasets from multiple DBN files for efficient ML training.

**CRITICAL: This system must be extremely performance-optimized for ML training applications. Every millisecond matters.**

### Symbol-Split-Merge Architecture (v5.0.0) - Multi-DBN→Comprehensive-Datasets→ML Pipeline

The package now follows a **symbol-split-merge architecture** for creating comprehensive symbol datasets from multiple DBN files:

1. **Phase 1: Symbol Splitting**: For each DBN file, split by symbol into intermediate parquet files
2. **Phase 2: Symbol Merging**: Merge all instances of each symbol across files into comprehensive datasets
3. **Phase 3: ML Training**: Lazy loading from comprehensive symbol datasets for memory-efficient ML training

**Key Architecture Benefits:**
- **Comprehensive Symbol Coverage**: Each symbol's complete history across multiple files
- **Large Dataset Creation**: Merge symbol data from multiple DBN files for robust training
- **Symbol-Specific Datasets**: Each symbol gets its own comprehensive parquet dataset
- **Optimized Storage**: Symbol datasets are much larger and more comprehensive than individual file splits
- **Better ML Training**: Train on symbol's complete history rather than fragmented data

**Symbol-Split-Merge Processing Logic:**
1. **Split Phase**: For each DBN file, split by symbol → `{dbn_name}_{symbol}.parquet`
2. **Registry Phase**: Track all symbol files across all DBN inputs
3. **Merge Phase**: For each symbol, merge all its files → `{currency}_{symbol}_dataset.parquet`
4. **Classification**: Apply uniform classification during merge with full symbol context
5. **Dataset Ready**: Comprehensive symbol datasets ready for ML training

## Core Workflow

### New Primary Workflow: Symbol-Split-Merge Dataset Building

```python
from represent import DatasetBuilder, DatasetBuildConfig, build_datasets_from_dbn_files

# Build comprehensive symbol datasets from multiple DBN files
config = create_represent_config("AUDUSD", features=['volume', 'variance'])
dataset_config = DatasetBuildConfig(
    currency="AUDUSD",
    min_symbol_samples=60500,          # Must be >= samples + lookback + lookforward + offset (50K + 5K + 5K + 500)
    force_uniform=True,                # Ensure uniform class distribution
    keep_intermediate=False            # Clean up intermediate split files
)

# Process multiple DBN files into comprehensive symbol datasets
# Use at least 10 DBN files to ensure sufficient data for minimum sample requirements
results = build_datasets_from_dbn_files(
    config=config,
    dbn_files=[
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240101.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240102.dbn.zst", 
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240103.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240104.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240105.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240106.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240107.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240108.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240109.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240110.dbn.zst"
    ],
    output_dir="/data/symbol_datasets/",
    dataset_config=dataset_config
)

# Output: /data/symbol_datasets/AUDUSD_M6AM4_dataset.parquet
#         /data/symbol_datasets/AUDUSD_M6AM5_dataset.parquet (comprehensive symbol datasets)
```

**Processing Flow:**
```
DBN File 1 → Split by Symbol → M6AM4.parquet, M6AM5.parquet, ...
DBN File 2 → Split by Symbol → M6AM4.parquet, M6AM5.parquet, ...
DBN File 3 → Split by Symbol → M6AM4.parquet, M6AM5.parquet, ...
                                        ↓
Merge Phase: M6AM4_dataset.parquet ← All M6AM4.parquet files
             M6AM5_dataset.parquet ← All M6AM5.parquet files
```

**Key Features:**
- **Two-Phase Processing**: Split all DBN files, then merge by symbol
- **Comprehensive Coverage**: Each symbol dataset contains data from all input files
- **Large Dataset Creation**: Symbol datasets are much larger than individual file processing
- **Symbol Registry**: Automatic tracking of which symbols appear in which files
- **Configurable Storage**: Keep or cleanup intermediate split files
- **Uniform Distribution**: True uniform classification using full symbol context

### Alternative Workflow: Directory-Based Processing

```python
from represent import batch_build_datasets_from_directory

# Process all DBN files in a directory
results = batch_build_datasets_from_directory(
    config=config,
    input_directory="data/dbn_files/",
    output_dir="/data/symbol_datasets/",
    file_pattern="*.dbn*",
    dataset_config=dataset_config
)
```

### ML Training (External Implementation)

The comprehensive symbol datasets are ready for ML training. **Dataloader functionality has been moved out of the represent package** to allow for customization in your ML training repository.

**Expected Workflow:**
```python
# In your ML training repository, implement a custom dataloader
# that reads the comprehensive symbol datasets

# Standard PyTorch training loop structure:
for features, labels in your_custom_dataloader:
    # features: torch.Tensor shape (batch_size, [N_features,] 402, 500)  
    # labels: torch.Tensor shape (batch_size,) with uniform distribution
    outputs = model(features)
    loss = criterion(outputs, labels)
    # ... training logic
```

**Key Benefits:**
- **Comprehensive Training Data**: Each symbol's full history for robust training
- **Large Dataset Support**: Train on multi-GB symbol datasets efficiently
- **Symbol-Specific Training**: Focus on specific symbols or use all available
- **Uniform Distribution**: Guaranteed class balance within each symbol dataset
- **Memory Efficient**: Lazy loading supports datasets larger than RAM

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

### Symbol-Specific Classification with First-Half Training

**NEW APPROACH (v5.0.0+)**: Classification is now performed on individual symbol datasets using the first half of each symbol's data to define classification bins, then applying those bins to the entire symbol dataset.

**Key Benefits:**
- **Symbol-Specific Boundaries**: Each symbol gets classification thresholds tailored to its price movement distribution
- **First-Half Training**: Uses first 50% of symbol data to define bins, preventing data leakage
- **Consistent Boundaries**: Same classification boundaries applied to entire symbol dataset
- **Uniform Distribution**: Quantile-based approach ensures balanced class distribution
- **Full Context**: Classification uses symbol's complete merged dataset from all DBN files

### Classification Process

```python
# New classification logic (applied per symbol during dataset creation)
def apply_symbol_classification(symbol_df, nbins=13, force_uniform=True):
    """
    Apply classification using first half of symbol data to define bins:
    
    1. Calculate price movements using lookback vs lookforward methodology
    2. Use first half of data to determine quantile boundaries
    3. Apply those boundaries to classify the entire symbol dataset
    4. Ensure uniform distribution across all classes
    """
    
    # Step 1: Calculate price movements for entire symbol dataset
    for stop_row in range(lookback_rows, len(data) - (lookforward_input + lookforward_offset)):
        lookback_mean = mean(mid_prices[stop_row - lookback_rows:stop_row])
        lookforward_mean = mean(mid_prices[stop_row + 1 + lookforward_offset:stop_row + lookforward_input])
        price_movements[stop_row] = (lookforward_mean - lookback_mean) / lookback_mean
    
    # Step 2: Use first half of data to define classification bins
    first_half_size = len(valid_movements) // 2
    training_movements = valid_movements[:first_half_size]
    
    # Step 3: Create quantile boundaries from first half
    quantiles = np.linspace(0, 1, nbins + 1)
    quantile_boundaries = np.quantile(training_movements, quantiles)
    
    # Step 4: Apply classification to ALL data using first-half boundaries
    classification_labels = np.digitize(valid_movements, quantile_boundaries[1:-1])
    classification_labels = np.clip(classification_labels, 0, nbins - 1)
    
    return classified_symbol_data
```

**Advantages over Previous Approaches:**
- **No Data Leakage**: First-half training prevents future information bleeding into classification
- **Symbol Adaptation**: Each symbol gets boundaries fitted to its specific movement characteristics  
- **Uniform Distribution**: True uniform distribution achieved using quantile-based binning
- **Full Symbol Context**: Uses complete merged symbol data (not individual DBN file fragments)
- **Polars Optimized**: Vectorized operations for high-performance processing

### Currency-Specific Configuration

Each currency uses focused configuration models for each module:

```python
from represent import (
    DatasetBuilderConfig, GlobalThresholdConfig, MarketDepthProcessorConfig,
    create_compatible_configs
)

# Create compatible configs for all modules
dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
    currency="AUDUSD",
    features=['volume', 'variance'],
    lookback_rows=5000,        # Historical data for price movement calculation
    lookforward_input=5000,    # Future data for price movement calculation  
    lookforward_offset=500,    # Offset before future window starts
    jump_size=100,            # Sampling interval for performance
    nbins=13                  # Number of classification bins
)
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

#### Symbol-Split-Merge Processing Requirements:
- **Streaming DBN Processing**: Handle large DBN files efficiently during split phase
- **Intermediate File Management**: Efficient creation and cleanup of symbol split files
- **Symbol Registry Tracking**: Track symbol files across multiple DBN inputs efficiently
- **Large Dataset Merging**: Efficiently merge multiple symbol files using polars concat
- **Memory Management**: Process large symbol merges without excessive RAM usage

#### Parquet-Based Processing Requirements:
- **Lazy Loading**: Load only required data chunks from comprehensive datasets
- **Vectorized Operations**: Use polars for high-performance data operations
- **Pre-allocated Buffers**: No dynamic memory allocation in hot paths
- **Feature-agnostic Pipeline**: Same processing handles volume, variance, trade counts
- **Schema Validation**: Validate feature availability at startup, not runtime

#### DBN Processing Requirements:
- **Streaming Decompression**: Handle zstandard compression efficiently
- **Multi-file Processing**: Process multiple DBN files in sequence efficiently
- **Symbol Splitting**: Extract symbols from each DBN file in single pass
- **Multi-feature Extraction**: Extract all enabled features during split phase

### Testing (Performance Focused)
- Organize tests by domain matching source structure
- Use realistic fixtures, avoid excessive mocking in integration tests
- Test error conditions and recovery scenarios
- **MANDATORY: 80% code coverage minimum** - all PRs must maintain this threshold
- **MANDATORY: Performance test critical paths with benchmarks**
- **Test Symbol-Split-Merge performance with multiple large DBN files**
- **Memory usage tests** - ensure no memory leaks in long-running processes
- **Coverage reporting** - use `make test-coverage` and `make coverage-html` for detailed reports

## Performance Requirements (NON-NEGOTIABLE)

**CRITICAL LATENCY TARGETS:**
- **DBN Split Phase**: <500 samples/second per DBN file during symbol splitting
- **Symbol Merge Phase**: <2000 samples/second during symbol dataset merging
- **Dataset Loading**: <10ms for single sample loading from comprehensive datasets
- **Batch Processing**: <50ms for 32-sample batch generation from symbol datasets
- **Memory Usage**: <8GB RAM for processing multiple large DBN files

**THROUGHPUT REQUIREMENTS:**
- **Split Processing**: 300+ samples/second per DBN during split phase
- **Merge Processing**: 1500+ samples/second during symbol merging
- **Training**: 1000+ samples/second during ML training from comprehensive datasets
- **Parallel Processing**: Must scale linearly with CPU cores

**MEMORY CONSTRAINTS:**
- **Split Phase Memory**: <4GB RAM per DBN file during splitting
- **Merge Phase Memory**: <8GB RAM during symbol dataset creation
- **Training Memory**: <4GB RAM regardless of comprehensive dataset size
- **Cache Efficiency**: >90% cache hit rate for frequently accessed samples

## Key Components

### DatasetBuilder (`represent/dataset_builder.py`)

Primary symbol-split-merge dataset builder for creating comprehensive symbol datasets.

```python
class DatasetBuilder:
    """
    Symbol-Split-Merge Dataset Builder:
    - Phase 1: Split multiple DBN files by symbol into intermediate files
    - Phase 2: Merge each symbol across all files into comprehensive datasets
    - Symbol registry tracks which symbols appear in which files
    - Configurable intermediate file cleanup
    - Pre-computed classification labels using full symbol context
    """
```

### Custom DataLoader (External Implementation)

**Dataloader functionality moved to ML training repositories.**

See `DATALOADER_MIGRATION_GUIDE.md` for comprehensive instructions on rebuilding with:
- Comprehensive symbol dataset support
- Guaranteed uniform class distribution within each symbol
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
# Primary: Symbol-Split-Merge Dataset Building
from represent import DatasetBuilder, DatasetBuildConfig
from represent import build_datasets_from_dbn_files, batch_build_datasets_from_directory

# Alternative: Streamlined DBN-to-Classified-Parquet Processing  
from represent import ParquetClassifier, process_dbn_to_classified_parquets

# Legacy: Unlabeled conversion approach (if needed)
from represent import convert_dbn_to_parquet, batch_convert_unlabeled

# Core Processing  
from represent import MarketDepthProcessor, process_market_data

# Configuration
from represent import create_represent_config
```

### Dynamic Configuration Generation

The represent package uses **focused configuration models** for each module:

```python
from represent import (
    DatasetBuilderConfig, GlobalThresholdConfig, MarketDepthProcessorConfig,
    create_compatible_configs
)

# Create focused configurations for each module
dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
    currency="AUDUSD",
    features=['volume', 'variance'],
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500
)

print(f"Dataset config currency: {dataset_cfg.currency}")
print(f"Processor config features: {processor_cfg.features}")
print(f"Threshold config lookback rows: {threshold_cfg.lookback_rows}")
```

## Instructions for Claude

When working on this codebase:

1. **SYMBOL-SPLIT-MERGE FIRST** - All new data processing should use the symbol-split-merge architecture
2. **PERFORMANCE FIRST** - Every code change must be evaluated for performance impact
3. **80% COVERAGE MANDATORY** - All code must maintain 80% test coverage minimum
4. **COMPREHENSIVE DATASETS** - Focus on creating large, comprehensive symbol datasets
5. **TYPE SAFETY** - Fix all type annotations for better IDE support and runtime safety
6. **LAZY LOADING ONLY** - Remove any references to streaming/real-time data ingestion
7. **PRE-COMPUTED LABELS** - Classification happens during dataset creation, not training
8. **MEMORY EFFICIENCY** - Optimize for processing multiple large DBN files
9. **VECTORIZED OPERATIONS** - Use polars/numpy for all data operations
10. **VALIDATE AT STARTUP** - Pre-validate schemas, use lookup tables over calculations
11. **TEST THOROUGHLY** - Include performance regression tests with every change
12. **NO BACKWARDS COMPATIBILITY** - Remove old approaches that don't fit new architecture

## Migration from Previous Versions

**NEW v5.0.0 ARCHITECTURE COMPONENTS:**

✅ **PRIMARY COMPONENTS:**
- `dataset_builder.py` - Symbol-split-merge dataset builder for comprehensive datasets
- `configs.py` - Focused configuration models for each module
- `api.py` - High-level convenience API updated for new architecture

**WORKFLOW INTEGRATION:**
- Use `build_datasets_from_dbn_files()` for comprehensive dataset creation
- Use `batch_build_datasets_from_directory()` for directory processing  
- Implement custom dataloader in your ML repository (see DATALOADER_MIGRATION_GUIDE.md)
- Leverage focused configuration models for each module
- Pre-computed labels eliminate runtime classification overhead

## Example Workflows

### Complete Symbol-Split-Merge Pipeline

```python
from represent import create_represent_config, DatasetBuildConfig
from represent import build_datasets_from_dbn_files

# Create configuration
config = create_represent_config(
    currency="AUDUSD",
    features=['volume', 'variance'],
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500
)

# Create dataset building configuration  
dataset_config = DatasetBuildConfig(
    currency="AUDUSD",
    min_symbol_samples=60500,     # Auto-calculated: samples + lookback + lookforward + offset
    force_uniform=True,           # Guarantee uniform class distribution
    keep_intermediate=False       # Clean up intermediate split files
)

# Build comprehensive symbol datasets from multiple DBN files
# Use at least 10 DBN files to ensure sufficient data for each symbol
print("Building comprehensive symbol datasets...")
results = build_datasets_from_dbn_files(
    config=config,
    dbn_files=[
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240101.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240102.dbn.zst", 
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240103.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240104.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240105.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240106.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240107.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240108.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240109.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240110.dbn.zst"
    ],
    output_dir="/data/symbol_datasets/",
    dataset_config=dataset_config,
    verbose=True
)

print(f"Created {results['phase_2_stats']['datasets_created']} comprehensive symbol datasets")
print(f"Total samples: {results['phase_2_stats']['total_samples']:,}")
print("Ready for ML training with comprehensive symbol data!")

# ML Training (implement custom dataloader in your ML repo)
# See DATALOADER_MIGRATION_GUIDE.md for implementation instructions
```

### Directory-Based Dataset Building

```python
from represent import batch_build_datasets_from_directory

# Build datasets from all DBN files in a directory
results = batch_build_datasets_from_directory(
    config=config,
    input_directory="data/audusd_dbn_files/",
    output_dir="/data/symbol_datasets/",
    file_pattern="*.dbn*",
    dataset_config=dataset_config
)

print(f"Processed {len(results['input_files'])} DBN files")
print(f"Generated {results['phase_2_stats']['datasets_created']} symbol datasets")
```

### Symbol-Specific Dataset Analysis

```python
import polars as pl

# Load comprehensive symbol dataset for analysis
symbol_dataset = pl.read_parquet("/data/symbol_datasets/AUDUSD_M6AM4_dataset.parquet")

print(f"Symbol M6AM4 comprehensive dataset:")
print(f"  Total samples: {len(symbol_dataset):,}")
print(f"  Date range: {symbol_dataset['ts_event'].min()} to {symbol_dataset['ts_event'].max()}")

# Analyze class distribution
class_dist = symbol_dataset['classification_label'].value_counts().sort('classification_label')
print(f"  Class distribution: {class_dist}")
print(f"  Uniform distribution achieved: {class_dist['count'].std() < 0.1 * class_dist['count'].mean()}")
```

This architecture provides maximum performance and comprehensive data coverage for ML training while maintaining efficiency through the two-phase symbol-split-merge approach.