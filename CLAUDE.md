# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **high-performance** Python package called "represent" that creates normalized market depth representations from limit order book (LOB) data. The core objective is to produce market depth arrays that can include multiple features:

1. **Base Feature**: `normed_abs_combined` numpy array (shape 402x500) - normalized cumulative market volume differences between ask and bid sides
2. **Extended Features**: Additional dimensions for variance and trade count data
3. **Multi-dimensional Output**: Configurable ND arrays with features in PyTorch-compatible dimensions

**CRITICAL: This system must be extremely performance-optimized for real-time trading applications. Every millisecond matters.**

### Core Functionality

The package processes market depth data to create multi-dimensional representations of limit order book dynamics:

1. **Price Binning**: Converts raw prices to integer micro-pip format and maps them to 402 price levels (200 bid + 200 ask + 2 for mid-price)
2. **Time Aggregation**: Groups tick data into 500 time bins (100 ticks per bin by default)
3. **Feature Extraction**: Extracts multiple features from market data:
   - **Volume**: Order book volumes mapped to 2D grid (price levels × time bins)
   - **Variance**: Price variance calculated from `market_depth_extraction_micro_pips_var` 
   - **Trade Counts**: Count data available in DBN files
4. **Cumulative Processing**: Calculates cumulative market depth for ask and bid sides across all features
5. **Multi-dimensional Output**: Creates configurable output arrays:
   - **Single Feature**: (402, 500) - traditional 2D representation
   - **Multi-feature**: (N, 402, 500) - ND array with features in first dimension (PyTorch compatible)
   - **Feature Selection**: Configurable combination of volume, variance, and trade counts

### Input Data Sources

- **DBN Files**: Databento compressed market data files (.dbn.zst format)
  - Contains volume, variance (`market_depth_extraction_micro_pips_var`), and trade count data
  - Must support extraction of multiple feature types from same source
- **Streaming Data**: Real-time market data in Polars DataFrame format
- **Market Data**: 10-level market by price (MBP-10) limit order book data with extended features

### PyTorch Integration

The package serves as a **ultra-fast** PyTorch-compatible data loading module that:
- Dynamically generates multi-dimensional feature arrays for ML model inputs with **<10ms latency**
- Supports configurable feature selection (volume, variance, trade counts) with **same performance targets**
- Provides **zero-copy** batching and streaming capabilities for training
- Maintains rolling windows of 50K historical samples with **O(1) insertion/removal**
- Processes data in batches of 500 records with **vectorized operations only**
- Uses **memory-mapped files** and **pre-allocated buffers** for maximum throughput
- **Feature dimension management**: Automatically handles PyTorch-compatible tensor shapes (N_features, 402, 500)

## Development Setup

The project uses Python 3.12, uv for package management, and modern Python packaging standards. **All repository interactions should be run via the Makefile where sensible.**

```bash
# Primary development commands (use these)
make install      # Install dependencies and setup environment
make test         # Run all tests with coverage (requires 80%)
make test-fast    # Run tests excluding performance tests with coverage
make test-coverage # Run tests with coverage report
make coverage-html # Generate HTML coverage report
make lint         # Run linting checks
make typecheck    # Run type checking
make format       # Format code
make build        # Build package
make clean        # Clean build artifacts

# Direct uv commands (fallback when Makefile targets don't exist)
uv sync --all-extras
uv run pytest --cov=represent
uv run ruff check .
uv run pyright
uv build
```

### Key Constants (from notebook analysis)

```python
MICRO_PIP_SIZE = 0.00001  # Price precision
TICKS_PER_BIN = 100      # Tick aggregation
SAMPLES = 50000          # Historical context window
PRICE_LEVELS = 402       # Total price bins (200 bid + 200 ask + 2 mid)
TIME_BINS = 500          # Time dimension

# Extended Features
FEATURE_TYPES = ['volume', 'variance', 'trade_counts']  # Available feature types
DEFAULT_FEATURES = ['volume']  # Default feature selection for backward compatibility
MAX_FEATURES = 3         # Maximum number of features that can be selected

# Feature Index Mapping (consistent ordering in multi-feature tensors)
FEATURE_INDEX_MAP = {
    'volume': 0,
    'variance': 1, 
    'trade_counts': 2
}
```

## Extended Features Architecture

### Feature Types and Data Sources

1. **Volume Features** (Default/Existing):
   - Source: `ask_sz_XX`, `bid_sz_XX` columns
   - Processing: Same as current `normed_abs_combined` logic
   - Output: Normalized cumulative volume differences

2. **Variance Features** (New):
   - Source: `market_depth_extraction_micro_pips_var` from DBN files
   - Processing: Extract variance data, map to price levels, apply cumulative processing
   - Output: Normalized cumulative variance differences across price levels

3. **Trade Count Features** (New):
   - Source: Trade count data available in DBN files (ask_ct_XX, bid_ct_XX columns)
   - Processing: Count aggregation, map to price levels, apply cumulative processing  
   - Output: Normalized cumulative trade count differences across price levels

### API Design for Feature Selection

```python
# Single feature (backward compatible) - always 2D output
processor = create_processor(features=['volume'])
result = processor.process(data)  # Shape: (402, 500)

# Two features - always 3D output  
processor = create_processor(features=['volume', 'variance'])
result = processor.process(data)  # Shape: (2, 402, 500)

# Three features - always 3D output
processor = create_processor(features=['volume', 'variance', 'trade_counts'])
result = processor.process(data)  # Shape: (3, 402, 500)

# Individual feature selection
processor = create_processor(features=['variance'])  # Just variance
result = processor.process(data)  # Shape: (402, 500)

processor = create_processor(features=['trade_counts'])  # Just trade counts  
result = processor.process(data)  # Shape: (402, 500)

# PyTorch integration - output shape determined by feature count
dataset = MarketDepthDataset(features=['volume'])  # 2D tensors
batch = dataset.get_current_representation()  # Shape: (402, 500)

dataset = MarketDepthDataset(features=['volume', 'variance'])  # 3D tensors
batch = dataset.get_current_representation()  # Shape: (2, 402, 500)
```

### Output Shape Rules

**Simple Dimensional Logic:**
- **1 feature**: Output shape `(402, 500)` - 2D tensor
- **2 features**: Output shape `(2, 402, 500)` - 3D tensor with feature dimension first
- **3 features**: Output shape `(3, 402, 500)` - 3D tensor with feature dimension first

**Feature Index Mapping:**
```python
FEATURE_INDEX_MAP = {
    'volume': 0,
    'variance': 1, 
    'trade_counts': 2
}

# For multi-feature tensors, features are ordered by index
features=['variance', 'volume'] → shape (2, 402, 500) where:
# result[0] = volume (index 0)
# result[1] = variance (index 1)
```

### Configuration Options

**Simple Configuration Example:**
```python
# Configuration is now simply based on feature selection
processor = create_processor(
    features=['volume', 'variance', 'trade_counts'],  # Shape: (3, 402, 500)
    normalize_features=True,     # Normalize each feature independently  
    cache_features=True,         # Cache processed features for reuse
    validate_features=True,      # Validate feature availability at initialization
)

# DataLoader configuration
dataset = MarketDepthDataset(
    features=['volume', 'variance'],  # Shape: (2, 402, 500)
    buffer_size=SAMPLES,
    validate_features=True,
)
```

**Shape Determination Logic:**
```python
def determine_output_shape(features):
    """Simple logic: shape determined by feature count."""
    if len(features) == 1:
        return (PRICE_LEVELS, TIME_BINS)  # (402, 500)
    else:
        return (len(features), PRICE_LEVELS, TIME_BINS)  # (N, 402, 500)
    
# Examples:
features=['volume'] → shape (402, 500)
features=['volume', 'variance'] → shape (2, 402, 500) 
features=['volume', 'variance', 'trade_counts'] → shape (3, 402, 500)
features=['variance'] → shape (402, 500)
```

### Performance Requirements for Extended Features

**CRITICAL: Extended features must maintain same performance targets:**
- **Feature Processing**: <10ms for any combination of features (1-3 features)
- **Memory Usage**: Linear scaling - max 3x base memory for all 3 features
- **Batch Processing**: Same <50ms target regardless of feature count
- **Cache Efficiency**: Pre-allocate buffers for all enabled features
- **Shape Handling**: Zero overhead for output shape determination based on feature count

## Development Standards

### Code Organization
- Use Single Responsibility Principle consistently
- Prefer Pydantic models over standard Python classes
- Never create modules with suffixes like `_extended`, `_coverage`, `_enhanced`
- Keep all related functionality in single modules, not split across files

### Error Handling
- Always provide graceful degradation paths
- Log errors with appropriate context for debugging

### Data Processing (Performance Critical)
- Maintain exactly 50K historical samples for ML context **using ring buffers**
- Process in batches of exactly 500 records (TICKS_PER_BIN = 100) **with SIMD operations**
- Use Polars for high-performance data operations on streaming data **with lazy evaluation**
- Use Databento for DBN file processing and decompression **with streaming decompression**
- **Pre-validate data schemas at startup** - avoid runtime validation in hot paths
- Handle zstandard compression for DBN files **with parallel decompression**
- Map prices to 402-level grid centered on mid-price **using lookup tables, not calculations**
- **Pre-allocate all arrays** - no dynamic memory allocation in processing loops
- **Use numba/cython** for critical path functions if pure NumPy isn't fast enough

#### Extended Features Processing Requirements:
- **Feature-agnostic processing**: Same pipeline handles volume, variance, and trade counts
- **Memory pre-allocation**: Allocate buffers for maximum enabled features at initialization
- **Vectorized multi-feature operations**: Process all enabled features in single pass
- **Feature dimension management**: Handle (N_features, 402, 500) shapes efficiently
- **Backward compatibility**: Single feature mode must perform identically to original
- **Schema validation**: Validate feature availability in data source at startup
- **Feature extraction**: Extract variance from `market_depth_extraction_micro_pips_var` efficiently
- **Count aggregation**: Handle trade count data with same performance as volume data

### Testing (Performance Focused)
- Organize tests by domain matching source structure
- Use realistic fixtures, avoid excessive mocking in integration tests
- Test error conditions and recovery scenarios
- **MANDATORY: 80% code coverage minimum** - all PRs must maintain this threshold
- **MANDATORY: Performance test critical paths with benchmarks**
- **Every PR must include performance regression tests**
- **Benchmark against target latencies: <10ms for array generation, <1ms for single record processing**
- **Memory usage tests** - ensure no memory leaks in long-running processes
- **Load tests** - verify performance under sustained 10K+ records/second
- **Coverage reporting** - use `make test-coverage` and `make coverage-html` for detailed reports


## Recent Updates

### Classification Logic Integration
- Added classification class generation logic from `notebooks/market_depth_extraction_micro_pips.py` 
- Integrated price movement-based classification with configurable bin thresholds
- Enhanced PyTorch dataloader with classification target generation capabilities
- Updated tests to cover new classification functionality

## Development Workflow

### Environment Setup
- **Always use Makefile targets for common operations** (make install, make test, etc.)
- Use `uv` for all package management and virtual environment handling
- Use `direnv` for environment variable management  
- Bootstrap with `make install` command
- Run tests with `make test` before commits
- Use pre-commit hooks for code quality

### Package Deployment Options

**Local Development Reference (Recommended for now):**
```bash
# In consuming projects, reference locally:
uv add --editable /path/to/represent

# Or in pyproject.toml:
[tool.uv.sources]
represent = { path = "../represent", editable = true }
```

**Private Package Hosting (Future option):**
- **GitHub Packages**: Host privately on GitHub with token-based access
- **GitLab Package Registry**: If using GitLab for source control
- **Private PyPI Server**: For enterprise environments (e.g., devpi, Artifactory)
- **Git Dependencies**: Direct git+ssh references in uv/pip

For current development, local editable installs are most practical until the API stabilizes.

### Adding New Features
- Start with data models and validation
- Implement core business logic with error handling
- Add comprehensive tests (unit, integration, e2e)
- Update monitoring and health checks
- Document API changes and operational impact

### Debugging and Monitoring
- Use structured logging with appropriate levels
- Monitor end-to-end latency and throughput
- Track data quality metrics continuously
- Set up alerts for critical system metrics
- Maintain operational runbooks for common issues



## Key Architecture Patterns

### Market Depth Processing Pipeline

The core processing follows this pattern from the notebook analysis:

```python
# 1. Load and decompress DBN data
with zstd.ZstdDecompressor() as decompressor:
    data = db.DBNStore.from_file("trades.dbn")
    df = data.to_df()

# 2. Convert to micro-pip integers
df[price_columns] = (df[price_columns] / MICRO_PIP_SIZE).round().astype(int)

# 3. Create time bins and aggregate
sub_df = sub_df.with_columns((pl.int_range(0, SAMPLES) // TICKS_PER_BIN).alias("tick_bin"))

# 4. Group by time bins and map to price grid
grouped_data = sub_df.group_by(["tick_bin"]).agg([...])

# 5. Create cumulative volume arrays
ask_market_volume = np.cumsum(mapped_volumes, axis=0)
bid_market_volume = np.cumsum(mapped_volumes, axis=0)

# 6. Generate final normalized output
normed_abs_combined = normalize_market_difference(ask_market_volume, bid_market_volume)
```

### Data Structure Patterns

```python
# Price column definitions (from notebook)
ASK_PRICE_COLUMNS = [f"ask_px_{str(i).zfill(2)}" for i in range(10)]
BID_PRICE_COLUMNS = [f"bid_px_{str(i).zfill(2)}" for i in range(10)]
ASK_VOL_COLUMNS = [f"ask_sz_{str(i).zfill(2)}" for i in range(10)]
BID_VOL_COLUMNS = [f"bid_sz_{str(i).zfill(2)}" for i in range(10)]
```


## Performance Requirements (NON-NEGOTIABLE)

**CRITICAL LATENCY TARGETS:**
- **Single Record Processing**: <1ms per record (any feature combination)
- **Array Generation**: <10ms for complete arrays (single feature: 402x500, multi-feature: Nx402x500)
- **Feature Processing**: <2ms additional latency per additional feature (linear scaling)
- **Batch Processing**: <50ms for 500-record batch end-to-end (any feature combination)
- **Rolling Window Updates**: <5ms for 50K sample window maintenance (any feature combination)

**THROUGHPUT REQUIREMENTS:**
- **Sustained**: 10K+ records/second without performance degradation
- **Peak**: 50K+ records/second burst capacity
- **Parallel Processing**: Must scale linearly with CPU cores

**MEMORY CONSTRAINTS:**
- **Core Processing**: <1GB RAM for single feature, <3GB for all features
- **Total System**: <4GB for single feature, <12GB for complete application with all features
- **Memory Allocation**: Zero dynamic allocation in hot paths
- **Cache Efficiency**: >95% L1 cache hit rate for price lookups
- **Feature Scaling**: Linear memory usage per additional feature (no exponential growth)

**ARCHITECTURAL PERFORMANCE REQUIREMENTS:**
- **Context Size**: Maintain exactly 50K samples in O(1) ring buffer
- **Batch Size**: Process exactly 500 records per batch with vectorized operations
- **Data Structures**: All critical path data structures must be cache-aligned
- **Thread Safety**: Lock-free data structures for concurrent access

## Instructions for Claude

When working on this codebase:

1. **PERFORMANCE FIRST** - Every code change must be evaluated for performance impact. Profile before and after.
2. **80% COVERAGE MANDATORY** - All code must maintain 80% test coverage minimum (use `make test-coverage`)
3. **Use Makefile first** - Always check for and use Makefile targets before running direct commands
4. **Zero tolerance for performance regressions** - Any change that increases latency must be rejected
5. **Zero tolerance for coverage drops** - Any change that drops coverage below 80% must be rejected
6. **Benchmark everything** - Include performance tests with every significant change
7. **Memory efficiency** - Use pre-allocated buffers, avoid dynamic allocation in hot paths
8. **Follow the domain organization** - Don't suggest technical layer organization
9. **Always add error handling** - But ensure error paths are also optimized
10. **Validate data efficiently** - Pre-validate schemas, use lookup tables over calculations
11. **Think about operations** - Include monitoring for performance metrics specifically
12. **Test thoroughly** - Include performance regression tests (use `make test`)
13. **Document performance decisions** - Explain why specific optimizations were chosen
14. **Optimize first, then simplify** - Performance takes precedence over elegance in this system

### Extended Features Implementation Guidelines

When implementing extended features (variance, trade counts):

1. **Backward Compatibility**: Existing API must work unchanged with `features=['volume']` default
2. **Feature Selection**: Support `features` parameter in all processor creation functions  
3. **Simple Dimension Logic**: Output shape determined solely by feature count (1→2D, 2+→3D)
4. **Feature Ordering**: Use consistent FEATURE_INDEX_MAP ordering in multi-feature tensors
5. **Data Source Validation**: Check feature availability in data at initialization
6. **Memory Pre-allocation**: Allocate for maximum requested features upfront
7. **Vectorized Processing**: Process all features in single data pass when possible
8. **PyTorch Integration**: Ensure DataLoader handles both 2D and 3D tensors correctly
9. **Performance Testing**: Benchmark each feature combination against targets
10. **Schema Evolution**: Handle data sources that may not have all features available
11. **Feature Extraction**: Implement efficient extraction from `market_depth_extraction_micro_pips_var`
12. **Configuration Validation**: Validate feature selection and availability at creation time
13. **Output Shape Consistency**: Ensure predictable tensor shapes based on feature count
14. **Individual Features**: Support any single feature as 2D output (volume, variance, or trade_counts)
15. **Caching Strategy**: Implement optional feature caching for repeated processing