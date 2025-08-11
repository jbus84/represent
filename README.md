# Represent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-104%20passed-green.svg)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-74%25-green.svg)](#testing)

High-performance Python package for creating normalized market depth representations from limit order book data using a **symbol-split-merge architecture**. Built for machine learning applications requiring comprehensive, uniform datasets from multiple DBN files.

## 🚀 Key Features

- **📊 Symbol-Split-Merge Architecture**: Process multiple DBN files into comprehensive symbol datasets
- **⚡ High Performance**: 1500+ samples/second processing with memory-efficient streaming
- **🎯 Uniform Distribution**: Guaranteed balanced class distributions for optimal ML training
- **🔧 Three Core Modules**: Clean, focused architecture for dataset building, market processing, and thresholds
- **📈 Multi-Feature Support**: Volume, variance, and trade count features
- **🧠 Framework Agnostic**: Compatible with PyTorch, TensorFlow, or custom ML frameworks

## 📦 Installation

```bash
# Using uv (recommended)
uv add represent

# Using pip
pip install represent

# Development installation
git clone <repository-url>
cd represent
uv sync --all-extras
```

## 🏗️ Three Core Modules

### 1. 📊 Dataset Builder (`dataset_builder`)
**Primary module for creating comprehensive symbol datasets from multiple DBN files**

```python
from represent import build_datasets_from_dbn_files, DatasetBuildConfig, create_represent_config

# Configure processing
config = create_represent_config(
    currency="AUDUSD",
    features=['volume', 'variance'],
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500
)

dataset_config = DatasetBuildConfig(
    currency="AUDUSD",
    features=['volume', 'variance'],
    force_uniform=True  # Ensures balanced class distribution
)

# Build comprehensive symbol datasets from multiple DBN files
results = build_datasets_from_dbn_files(
    config=config,
    dbn_files=[
        "data/AUDUSD-20240101.dbn.zst",
        "data/AUDUSD-20240102.dbn.zst", 
        "data/AUDUSD-20240103.dbn.zst"
    ],
    output_dir="symbol_datasets/",
    dataset_config=dataset_config
)

# Output: symbol_datasets/AUDUSD_M6AM4_dataset.parquet (comprehensive symbol data)
print(f"Created {results['phase_2_stats']['datasets_created']} symbol datasets")
print(f"Total samples: {results['phase_2_stats']['total_samples']:,}")
```

**Key Functions:**
- `build_datasets_from_dbn_files()` - Process multiple DBN files
- `batch_build_datasets_from_directory()` - Process entire directories
- `DatasetBuilder` - Advanced processing with custom workflows

### 2. ⚡ Market Depth Processor (`market_depth_processor`)
**High-performance processor for converting market data into normalized tensors**

```python
from represent import MarketDepthProcessor, create_processor, process_market_data
import polars as pl

# Create processor for specific features
config = create_represent_config("AUDUSD", features=['volume', 'variance'])
processor = MarketDepthProcessor(config=config, features=['volume', 'variance'])

# Load market data
market_data = pl.read_parquet("symbol_datasets/AUDUSD_M6AM4_dataset.parquet")

# Process into normalized tensor representation
tensor_data = processor.process(market_data)

# Output shape: (2, 402, 500) for 2 features, 402 price levels, 500 time bins
print(f"Tensor shape: {tensor_data.shape}")
print(f"Data type: {tensor_data.dtype}")

# Convenience function for single-use processing
tensor_data = process_market_data(market_data, config=config, features=['volume'])
```

**Key Functions:**
- `MarketDepthProcessor` - Main processor class
- `process_market_data()` - Single-use convenience function  
- `create_processor()` - Factory function for processor creation

### 3. 📏 Global Threshold Calculator (`global_threshold_calculator`)
**Calculate consistent classification thresholds across multiple files for uniform distributions**

```python
from represent import calculate_global_thresholds, GlobalThresholdCalculator

# Calculate thresholds from sample of DBN files
config = create_represent_config("AUDUSD")
thresholds = calculate_global_thresholds(
    config=config,
    data_directory="data/databento/AUDUSD/",
    sample_fraction=0.5,  # Use 50% of files for threshold calculation
    verbose=True
)

print(f"Generated {thresholds.nbins} classification bins")
print(f"Based on {thresholds.sample_size:,} price movements")

# Use calculated thresholds for consistent classification
dataset_config = DatasetBuildConfig(
    global_thresholds=thresholds,  # Apply same thresholds to all processing
    force_uniform=True
)

# Advanced usage with custom calculator
calculator = GlobalThresholdCalculator(config=config)
thresholds = calculator.calculate_thresholds_from_directory(
    data_directory="data/databento/AUDUSD/",
    sample_fraction=0.3
)
```

**Key Functions:**
- `calculate_global_thresholds()` - Main threshold calculation function
- `GlobalThresholdCalculator` - Advanced threshold calculation class
- `GlobalThresholds` - Result object containing threshold data

## 🚀 Complete Workflow Example

```python
from represent import (
    create_represent_config, 
    DatasetBuildConfig,
    build_datasets_from_dbn_files,
    calculate_global_thresholds,
    MarketDepthProcessor
)

# Step 1: Configure processing parameters
config = create_represent_config(
    currency="AUDUSD",
    features=['volume', 'variance'],
    lookback_rows=5000,
    lookforward_input=5000, 
    lookforward_offset=500
)

# Step 2: Calculate global thresholds for consistent classification
thresholds = calculate_global_thresholds(
    config=config,
    data_directory="data/databento/AUDUSD/",
    sample_fraction=0.5
)

# Step 3: Build comprehensive symbol datasets
dataset_config = DatasetBuildConfig(
    currency="AUDUSD", 
    features=['volume', 'variance'],
    global_thresholds=thresholds,  # Use calculated thresholds
    force_uniform=True
)

results = build_datasets_from_dbn_files(
    config=config,
    dbn_files=[
        "data/AUDUSD-20240101.dbn.zst",
        "data/AUDUSD-20240102.dbn.zst",
        "data/AUDUSD-20240103.dbn.zst"
    ],
    output_dir="symbol_datasets/",
    dataset_config=dataset_config
)

# Step 4: Process datasets for ML training (in your ML repository)
processor = MarketDepthProcessor(config=config, features=['volume', 'variance'])

# Load a comprehensive symbol dataset
import polars as pl
symbol_data = pl.read_parquet("symbol_datasets/AUDUSD_M6AM4_dataset.parquet")

# Convert to tensor for ML training
tensor_data = processor.process(symbol_data)
# Shape: (2, 402, 500) - 2 features, 402 price levels, 500 time bins

print(f"✅ Created {results['phase_2_stats']['datasets_created']} symbol datasets")
print(f"✅ Ready for ML training with {tensor_data.shape} tensor shape")
```

## 🎯 Feature Types and Output Shapes

**Available Features:**
- **Volume**: Market depth from order sizes - `(402, time_bins)`
- **Variance**: Price volatility patterns - `(402, time_bins)`
- **Trade Counts**: Transaction activity levels - `(402, time_bins)`

**Multi-Feature Output Shapes:**
- **1 feature**: `(402, 500)` - 2D tensor
- **2+ features**: `(N, 402, 500)` - 3D tensor with feature dimension first

```python
# Examples of different feature configurations
config = create_represent_config("AUDUSD", features=['volume'])
# Output: (402, 500)

config = create_represent_config("AUDUSD", features=['volume', 'variance']) 
# Output: (2, 402, 500)

config = create_represent_config("AUDUSD", features=['volume', 'variance', 'trade_counts'])
# Output: (3, 402, 500)
```

## 🏗️ Symbol-Split-Merge Architecture

The package uses a two-phase architecture for creating comprehensive symbol datasets:

### **Phase 1: Symbol Splitting**
Each DBN file is split by symbol into intermediate parquet files
- **Input**: Multiple DBN files (e.g., AUDUSD-20240101.dbn.zst, AUDUSD-20240102.dbn.zst)
- **Output**: Intermediate symbol files (e.g., file1_M6AM4.parquet, file2_M6AM4.parquet)
- **Performance**: 300+ samples/second per DBN file

### **Phase 2: Symbol Merging**  
All instances of each symbol are merged into comprehensive datasets
- **Input**: All symbol files across multiple DBN files
- **Output**: Comprehensive symbol datasets (e.g., AUDUSD_M6AM4_dataset.parquet)
- **Performance**: 1500+ samples/second during merging
- **Features**: Pre-computed classification labels with uniform distribution

### **Phase 3: ML Training** (External Implementation)
Comprehensive symbol datasets ready for custom dataloader implementation

```python
# Implement in your ML training repository:
from your_ml_package import create_custom_dataloader
import torch

# Load comprehensive symbol dataset
dataloader = create_custom_dataloader(
    parquet_path="symbol_datasets/AUDUSD_M6AM4_dataset.parquet",
    batch_size=32,
    shuffle=True
)

# Standard PyTorch training loop
for features, labels in dataloader:
    # features: torch.Tensor shape [32, 2, 402, 500] for volume+variance
    # labels: torch.Tensor shape [32] with uniform distribution
    outputs = model(features)
    loss = criterion(outputs, labels)
    # ... training logic
```

## ⚙️ Configuration System

### **RepresentConfig**
Central configuration object for all processing parameters:

```python
from represent import RepresentConfig, create_represent_config

# Manual configuration
config = RepresentConfig(
    currency="AUDUSD",
    features=['volume', 'variance'], 
    samples=50000,
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500,
    nbins=13,
    batch_size=1500
)

# Convenience function with currency-specific optimizations
config = create_represent_config(
    currency="AUDUSD",  # Auto-configures pip size, time bins, etc.
    features=['volume'],
    samples=25000
)

# Access configuration parameters
print(f"Time bins: {config.time_bins}")           # Auto-calculated
print(f"Min samples: {config.min_symbol_samples}") # Auto-calculated
print(f"Pip size: {config.true_pip_size}")        # Currency-specific
```

### **DatasetBuildConfig**
Configuration for dataset building process:

```python
from represent import DatasetBuildConfig

dataset_config = DatasetBuildConfig(
    currency="AUDUSD",
    features=['volume', 'variance'],
    min_symbol_samples=10000,     # Minimum samples per symbol
    force_uniform=True,           # Ensure balanced class distribution
    nbins=13,                     # Number of classification bins
    keep_intermediate=False       # Clean up intermediate files
)
```

## 📁 Data Formats

**Input Requirements:**
- **DBN files**: `.dbn` or `.dbn.zst` (compressed recommended)
- **Required columns**: `ask_px_00-09`, `bid_px_00-09`, `ask_sz_00-09`, `bid_sz_00-09`
- **Optional columns**: `ask_ct_00-09`, `bid_ct_00-09` (for trade count features)

**Output Format:**
- **Comprehensive symbol datasets**: One parquet file per symbol containing merged data
- **Pre-classified**: Uniform distribution labels ready for training
- **Tensor-ready**: Direct loading into ML frameworks with consistent shapes

## ⚡ Performance

- **DBN Processing**: 300+ samples/second during symbol splitting
- **Symbol Merging**: 1500+ samples/second during dataset creation  
- **ML Training**: 1000+ samples/second from comprehensive datasets
- **Memory Usage**: <8GB RAM for processing multiple large DBN files
- **Scalability**: Linear scaling with CPU cores

## 🧪 Development

```bash
# Install dependencies
uv sync --all-extras

# Run tests
make test                 # Full test suite with coverage
make test-fast           # Quick tests (excludes performance tests)

# Code quality
make lint                # Linting and type checking
make format             # Code formatting

# Build package
make build              # Build distribution packages
```

### Testing
- **104 tests passing** with comprehensive coverage
- **74% code coverage** focused on critical paths
- **Performance tests** for latency requirements
- **Integration tests** for complete workflows

## 📊 Examples

Check out the `examples/` directory for complete demonstrations:

```bash
# Symbol-split-merge demonstration
python examples/symbol_split_merge_demo.py

# Quick start examples  
python examples/quick_start_examples.py

# Feature extraction demo
python examples/demonstrate_feature_extraction.py
```

## 📈 Architecture Benefits

**Why Symbol-Split-Merge?**
- **Comprehensive Datasets**: Each symbol contains complete history from multiple files
- **Memory Efficient**: Stream large DBN files without loading into RAM
- **Uniform Distribution**: Balanced class labels for optimal ML training  
- **Production Ready**: Handle 10+ DBN files efficiently with automatic validation

**Clean Three-Module Design:**
- **dataset_builder**: High-level dataset creation from DBN files
- **market_depth_processor**: Low-level market data tensor processing
- **global_threshold_calculator**: Consistent classification across files

## 📄 License

MIT License - see LICENSE file for details.

---

**🏗️ Production-ready symbol-split-merge architecture for comprehensive ML datasets with memory-efficient processing and guaranteed uniform class distribution**