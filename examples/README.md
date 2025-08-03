# Represent Library Examples

This directory contains organized examples demonstrating various uses of the `represent` library for market depth processing, PyTorch integration, and performance analysis.

## Directory Structure

- **`basic_usage/`** - Simple introductory examples and quickstart guides
- **`dataloader_performance/`** - Performance benchmarking and optimization examples
- **`pytorch_integration/`** - PyTorch dataloader and ML training examples
- **`real_data/`** - Examples using real market data files (.dbn format)
- **`visualization/`** - Market depth visualization and plotting examples
- **`extended_features/`** - Multi-feature processing demonstrations (volume, variance, trade counts)

## Quick Start

For beginners, start with:
1. `basic_usage/pytorch_quickstart.py` - Basic library usage
2. `visualization/generate_visualization.py` - Generate market depth visualizations
3. `dataloader_performance/dataloader_performance_demo.py` - Performance analysis

## Example Categories

### 1. Basic Usage (`basic_usage/`)

Simple examples to get started with the library:
- **`pytorch_quickstart.py`** - Basic PyTorch tensor generation
- **`simple_background_usage.py`** - Background processing introduction  
- **`simple_currency_config_demo.py`** - Currency configuration examples

### 2. Performance Analysis (`dataloader_performance/`)

Performance benchmarking and optimization examples:
- **`dataloader_performance_demo.py`** - Comprehensive performance analysis
- **`production_dataloader_example.py`** - Production-ready dataloader setup

### 3. PyTorch Integration (`pytorch_integration/`)

ML and PyTorch-specific examples:
- **`pytorch_training_example.py`** - Model training with market data
- **`pytorch_inference_example.py`** - Model inference examples
- **`pytorch_streaming_example.py`** - Real-time streaming integration
- **`background_training_demo.py`** - Background training workflows

### 4. Real Data Processing (`real_data/`)

Examples using actual market data files:
- **`dataloader_real_data_example.py`** - Real DBN file processing
- **`currency_config_demo.py`** - Currency-specific configurations

### 5. Visualization (`visualization/`)

Market depth visualization examples:
- **`generate_visualization.py`** - Basic market depth heatmaps

### 6. Extended Features (`extended_features/`)

Multi-feature processing demonstrations:
- **`extended_features_visualization.py`** - Volume, variance, and trade count features

## Running Examples

### Prerequisites

Ensure you have the represent library installed and data files available:

```bash
# Install the library
uv sync --all-extras

# Ensure data files are available
ls data/glbx-mdp3-*.dbn.zst
```

### Running Individual Examples

```bash
# Basic usage
uv run python examples/basic_usage/pytorch_quickstart.py

# Performance analysis  
uv run python examples/dataloader_performance/dataloader_performance_demo.py

# Visualization
uv run python examples/visualization/generate_visualization.py

# Extended features
uv run python examples/extended_features/extended_features_visualization.py

# Real data processing
uv run python examples/real_data/dataloader_real_data_example.py
```

### Key Features Demonstrated

- **High-Performance Processing**: <10ms market depth generation
- **Multi-Feature Support**: Volume, variance, and trade count features
- **PyTorch Integration**: Native tensor output for ML workflows
- **Real-Time Streaming**: Continuous data processing capabilities
- **Production Ready**: Optimized for trading applications
