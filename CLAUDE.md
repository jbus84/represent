# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **high-performance** Python package called "represent" that creates normalized market depth representations from limit order book (LOB) data. The core objective is to produce the `normed_abs_combined` numpy array (shape 402x500) that represents normalized cumulative market volume differences between ask and bid sides across price levels and time bins.

**CRITICAL: This system must be extremely performance-optimized for real-time trading applications. Every millisecond matters.**

### Core Functionality

The package processes market depth data to create visual representations of limit order book dynamics:

1. **Price Binning**: Converts raw prices to integer micro-pip format and maps them to 402 price levels (200 bid + 200 ask + 2 for mid-price)
2. **Time Aggregation**: Groups tick data into 500 time bins (100 ticks per bin by default)
3. **Volume Mapping**: Maps order book volumes to a 2D grid (price levels × time bins)
4. **Cumulative Volume**: Calculates cumulative market depth for ask and bid sides
5. **Normalization**: Creates the final `normed_abs_combined` array showing normalized volume differences with sign preservation (positive for ask dominance, negative for bid dominance)

### Input Data Sources

- **DBN Files**: Databento compressed market data files (.dbn.zst format)
- **Streaming Data**: Real-time market data in Polars DataFrame format
- **Market Data**: 10-level market by price (MBP-10) limit order book data

### PyTorch Integration

The package serves as a **ultra-fast** PyTorch-compatible data loading module that:
- Dynamically generates `normed_abs_combined` arrays for ML model inputs with **<10ms latency**
- Provides **zero-copy** batching and streaming capabilities for training
- Maintains rolling windows of 50K historical samples with **O(1) insertion/removal**
- Processes data in batches of 500 records with **vectorized operations only**
- Uses **memory-mapped files** and **pre-allocated buffers** for maximum throughput

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
```

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
- **Single Record Processing**: <1ms per record
- **Array Generation**: <10ms for complete `normed_abs_combined` array (402x500)
- **Batch Processing**: <50ms for 500-record batch end-to-end
- **Rolling Window Updates**: <5ms for 50K sample window maintenance

**THROUGHPUT REQUIREMENTS:**
- **Sustained**: 10K+ records/second without performance degradation
- **Peak**: 50K+ records/second burst capacity
- **Parallel Processing**: Must scale linearly with CPU cores

**MEMORY CONSTRAINTS:**
- **Core Processing**: <1GB RAM for processing components
- **Total System**: <4GB for complete application including buffers
- **Memory Allocation**: Zero dynamic allocation in hot paths
- **Cache Efficiency**: >95% L1 cache hit rate for price lookups

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