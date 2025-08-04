# Represent Library Examples (v2.0.0)

This directory contains examples demonstrating the new **parquet-based machine learning pipeline** for market depth processing and PyTorch integration.

## New Architecture Overview

The represent library now uses a **two-stage pipeline**:
1. **Offline Preprocessing**: Convert DBN files to labeled parquet datasets
2. **Online Training**: Lazy loading from parquet for memory-efficient ML training

## Directory Structure

- **`new_architecture/`** - Complete workflow examples for the new DBN→Parquet→ML pipeline
- **`real_data/`** - Examples using the new parquet-based dataloader
- **`visualization/`** - Market depth visualization using core processing pipeline
- **`random_access_evaluation/`** - Performance benchmarks for lazy dataloader random access

## Quick Start

For the new architecture, follow this sequence:
1. `new_architecture/dbn_to_parquet_example.py` - Complete workflow demonstration
2. `real_data/parquet_dataloader_example.py` - Parquet-based training
3. `visualization/generate_visualization.py` - Core processing visualization

## Example Categories

### 1. New Architecture (`new_architecture/`)

Complete workflow examples for the v2.0.0 architecture:
- **`dbn_to_parquet_example.py`** - End-to-end DBN→Parquet→ML pipeline demonstration

**Features demonstrated:**
- DBN file to labeled parquet conversion
- Automatic classification labeling during conversion
- Currency-specific configurations (AUDUSD, GBPUSD, EURJPY)
- Multi-feature extraction (volume, variance, trade counts)
- Lazy loading parquet dataloader for training
- Memory-efficient ML training on large datasets

### 2. Parquet-Based Training (`real_data/`)

Examples using the new lazy loading dataloader:
- **`parquet_dataloader_example.py`** - Memory-efficient training from parquet files

**Features demonstrated:**
- Lazy loading from parquet datasets
- Memory usage independent of dataset size
- High-performance tensor generation
- PyTorch integration with pre-computed labels
- Configurable sampling strategies

### 3. Core Processing (`visualization/`)

Core market depth processing examples:
- **`generate_visualization.py`** - Market depth heatmap generation

**Features demonstrated:**
- Raw market data processing
- 402×500 market depth representation
- Price level mapping and time bin aggregation

### 4. Performance Evaluation (`random_access_evaluation/`)

Comprehensive benchmarks for lazy dataloader performance:
- **`lazy_dataloader_random_access_benchmark.py`** - Full performance benchmark suite
- **`minimal_test.py`** - Quick functionality verification
- **`usage_examples.py`** - Configuration examples for different scenarios

**Features demonstrated:**
- Random access latency measurement (<1ms target)
- Batch loading throughput (>10K samples/sec target)
- 50K subset sampling for ML training
- Cache effectiveness analysis with different sizes
- Memory efficiency monitoring
- Production-ready performance validation

## New Workflow

### Stage 1: Convert DBN to Labeled Parquet

```bash
# Use the conversion script or example
uv run python examples/new_architecture/dbn_to_parquet_example.py

# Or use the CLI tool
uv run python scripts/convert_dbn_to_parquet.py \
  --input data/market_data.dbn.zst \
  --output data/labeled_dataset.parquet \
  --currency AUDUSD
```

### Stage 2: Train with Parquet Dataloader

```bash
# Use the parquet dataloader example
uv run python examples/real_data/parquet_dataloader_example.py
```

## Prerequisites

### Data Setup

For DBN conversion examples:
```bash
# Ensure DBN files are available
ls data/*.dbn.zst

# Or create sample DBN files for testing
```

For parquet training examples:
```bash
# Convert DBN to parquet first (Stage 1)
# Then run parquet training examples (Stage 2)
```

### Environment Setup

```bash
# Install the library with all dependencies
uv sync --all-extras

# Verify installation
uv run python -c "from represent import convert_dbn_file, create_market_depth_dataloader; print('✅ Installation OK')"
```

## Running Examples

### Complete Workflow (Recommended)

```bash
# 1. Run complete workflow example
uv run python examples/new_architecture/dbn_to_parquet_example.py

# 2. Run parquet training example  
uv run python examples/real_data/parquet_dataloader_example.py

# 3. Generate visualization
uv run python examples/visualization/generate_visualization.py

# 4. Evaluate random access performance (optional)
uv run python examples/random_access_evaluation/minimal_test.py
```

### Individual Components

```bash
# DBN conversion workflow
uv run python examples/new_architecture/dbn_to_parquet_example.py

# Parquet-based training
uv run python examples/real_data/parquet_dataloader_example.py

# Core processing visualization
uv run python examples/visualization/generate_visualization.py

# Performance evaluation
uv run python examples/random_access_evaluation/minimal_test.py
uv run python examples/random_access_evaluation/usage_examples.py
```

## Key Features Demonstrated

### Performance Optimizations
- **Pre-computed Labels**: Classification during conversion, not training
- **Lazy Loading**: Memory usage independent of dataset size
- **Columnar Storage**: Efficient parquet compression and querying
- **Batch Processing**: Optimized tensor generation

### ML Integration
- **PyTorch Native**: Direct tensor output for training
- **Pre-computed Classification**: 3-class price movement prediction
- **Multi-feature Support**: Volume, variance, and trade count features
- **Memory Efficient**: Train on datasets larger than RAM

### Production Ready
- **Currency Configurations**: Optimized settings for AUDUSD, GBPUSD, EURJPY
- **Batch Conversion**: Process multiple DBN files efficiently
- **Performance Monitoring**: Built-in benchmarking and statistics
- **Scalable Architecture**: Linear memory scaling with dataset size

## Migration from v1.x

The old streaming/ring buffer architecture has been completely replaced. 

**Old workflow (v1.x):**
```python
# DEPRECATED - Ring buffer streaming
dataset = MarketDepthDataset(buffer_size=50000)
dataset.add_streaming_data(data)
representation = dataset.get_current_representation()
```

**New workflow (v2.0.0):**
```python
# NEW - Parquet-based pipeline
# Stage 1: Convert DBN to parquet (run once)
convert_dbn_file(dbn_path, parquet_path, currency='AUDUSD')

# Stage 2: Lazy loading for training
dataloader = create_market_depth_dataloader(parquet_path, batch_size=32)
for features, labels in dataloader:
    # Train your model
```

This new architecture provides significant performance improvements and enables training on much larger datasets while using less memory.