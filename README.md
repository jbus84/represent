# Represent v5.0.0

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-217%20passed-green.svg)](https://github.com/your-repo/represent)
[![Coverage](https://img.shields.io/badge/coverage-76%25-yellow.svg)](https://github.com/your-repo/represent)
[![Performance](https://img.shields.io/badge/latency-<10ms-orange.svg)](https://github.com/your-repo/represent)

High-performance Python package for creating normalized market depth representations from limit order book data using a **symbol-split-merge architecture** for **comprehensive ML datasets** with **memory-efficient processing** and **guaranteed uniform distribution**.

## 🚀 Key Features

- **🏗️ Symbol-Split-Merge Architecture**: Multiple DBN files → Comprehensive Symbol Datasets → ML Training
- **💪 Automatic Sample Requirements**: Auto-calculates minimum samples (60,500+ per symbol by default)
- **🧠 Memory-Efficient Processing**: Streams large DBN files without loading entire datasets into memory
- **📊 Comprehensive Symbol Datasets**: Each symbol contains data merged from multiple DBN files for robust training
- **⚖️ Guaranteed Uniform Distribution**: 7.69% per class (13-bin) for optimal ML training
- **🔄 Two-Phase Processing**: Split DBN files by symbol → Merge symbol data across all files
- **🎯 Multi-Feature Support**: Volume, variance, and trade count features (configurable)
- **🧠 Framework Agnostic**: Output compatible with PyTorch, TensorFlow, or custom ML frameworks
- **⚡ Performance Optimized**: RepresentConfig system with memory-efficient streaming
- **📈 Production Ready**: Handles 10+ DBN files efficiently with automatic requirement validation

## 📦 Installation

```bash
# Using uv (recommended)
uv add represent

# Using pip
pip install represent

# Development installation
git clone <repository-url> && cd represent
uv sync --all-extras
```

## 🔧 v5.0.0 Symbol-Split-Merge Architecture

### **Symbol-Split-Merge Processing**
New architecture processes multiple DBN files to create comprehensive symbol datasets:

**Processing Phases:**
- **Phase 1**: Split each DBN file by symbol into intermediate files
- **Phase 2**: Merge all instances of each symbol across files into comprehensive datasets
- **Memory Efficiency**: Streams DBN data without loading entire files into RAM
- **Automatic Requirements**: Calculates minimum samples needed (samples + lookback + lookforward + offset)

**Architecture Benefits:**
- **Comprehensive Coverage**: Each symbol dataset contains complete history from multiple files
- **Large Dataset Creation**: Symbol datasets are much larger and more comprehensive
- **Memory Efficient**: Handles DBN files >20GB with <8GB RAM usage
- **Automatic Validation**: Ensures minimum sample requirements are met before processing
- **Better ML Training**: Train on symbol's complete merged history rather than fragmented data

### **Symbol-Split-Merge Workflow:**

**Step 1: Configure with Automatic Requirements**
```python
from represent import create_represent_config, DatasetBuildConfig

# Create configuration - minimum samples auto-calculated
config = create_represent_config(
    currency="AUDUSD",
    features=['volume', 'variance'],
    samples=50000,              # Base samples needed
    lookback_rows=5000,         # Historical data required
    lookforward_input=5000,     # Future data required
    lookforward_offset=500      # Offset before future window
)

# DatasetBuilder auto-calculates min_symbol_samples = 60,500
dataset_config = DatasetBuildConfig(
    currency="AUDUSD",
    features=['volume', 'variance'],
    # min_symbol_samples automatically set to 60,500
    force_uniform=True
)
```

**Step 2: Process Multiple DBN Files**
```python
from represent import build_datasets_from_dbn_files

# Process 10+ DBN files into comprehensive symbol datasets
results = build_datasets_from_dbn_files(
    config=config,
    dbn_files=[
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240101.dbn.zst",
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240102.dbn.zst",
        # ... 8 more files for comprehensive coverage
        "/Users/danielfisher/data/databento/AUDUSD-micro/AUDUSD-20240110.dbn.zst"
    ],
    output_dir="/data/symbol_datasets/",
    dataset_config=dataset_config
)

# Output: Comprehensive symbol datasets (e.g., AUDUSD_M6AM4_dataset.parquet)
print(f"Created {results['phase_2_stats']['datasets_created']} comprehensive datasets")
print(f"Each dataset contains merged data from {len(results['input_files'])} DBN files")
```

## 🏗️ Symbol-Split-Merge Architecture

### **Phase 1: Split DBN Files by Symbol**
Each DBN file is split by symbol into intermediate files, with memory-efficient streaming processing.

### **Phase 2: Merge Symbol Data Across Files**
All instances of each symbol are merged into comprehensive datasets with uniform classification.

### **Phase 3: ML Training (External Implementation)**
Comprehensive symbol datasets ready for custom dataloader implementation in your ML training repository.

## 🚀 Quick Start

### Phase 1-2: Multiple DBN Files to Comprehensive Symbol Datasets

**🎯 RECOMMENDED: Use symbol-split-merge for comprehensive ML datasets**

```python
from represent import create_represent_config, DatasetBuildConfig, batch_build_datasets_from_directory

# Step 1: Create configuration with automatic minimum calculation
config = create_represent_config(
    currency="AUDUSD",
    features=['volume', 'variance'],
    samples=50000,              # Base samples needed
    lookback_rows=5000,         # Historical data for classification
    lookforward_input=5000,     # Future data for classification
    lookforward_offset=500      # Offset before future window
)

dataset_config = DatasetBuildConfig(
    currency="AUDUSD",
    features=['volume', 'variance'],
    # min_symbol_samples = 60,500 (auto-calculated)
    force_uniform=True
)

print(f"Minimum samples per symbol: {config.samples + config.lookback_rows + config.lookforward_input + config.lookforward_offset:,}")

# Step 2: Process entire directory of DBN files into comprehensive datasets
results = batch_build_datasets_from_directory(
    config=config,
    input_directory="/Users/danielfisher/data/databento/AUDUSD-micro/",  # Directory with 10+ DBN files
    output_dir="/data/symbol_datasets/",
    dataset_config=dataset_config
)

# Output: /data/symbol_datasets/AUDUSD_M6AM4_dataset.parquet (comprehensive symbol datasets)
print(f"Generated {results['phase_2_stats']['datasets_created']} comprehensive symbol datasets")
print(f"Total samples: {results['phase_2_stats']['total_samples']:,}")
print(f"Each dataset contains merged data from {len(results['input_files'])} DBN files")
print(f"All datasets meet minimum sample requirements for ML training")
```

**⚠️ Why Symbol-Split-Merge Architecture Matters:**
- **❌ Single-file processing**: Limited data per symbol, fragmented training datasets
- **✅ Symbol-split-merge**: Comprehensive symbol datasets with complete history across multiple files
- **Result**: Robust ML training with comprehensive, large-scale symbol-specific datasets

### Phase 3: ML Training (External Implementation)

The comprehensive symbol datasets are ready for ML training. **Dataloader functionality has been moved to your ML training repository** for maximum customization.

**See `DATALOADER_MIGRATION_GUIDE.md` for comprehensive instructions on rebuilding the dataloader with Claude.**

```python
# Expected workflow in your ML training repository:
from your_ml_package import create_custom_dataloader  # Implement using guide
from represent import create_represent_config

# Use same config for consistent parameters
config = create_represent_config("AUDUSD")

# Implement custom dataloader following migration guide
dataloader = create_custom_dataloader(
    parquet_path="/data/symbol_datasets/AUDUSD_M6AM4_dataset.parquet",  # Comprehensive dataset
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Standard PyTorch training loop structure:
for batch_features, batch_labels in dataloader:
    # batch_features: torch.Tensor of shape [32, 2, 402, 500] for 2 features 
    # batch_labels: torch.Tensor of shape [32] with uniform distribution (7.69% each class)
    # Note: Comprehensive datasets provide much larger training data per symbol
    model_output = model(batch_features)
    loss = criterion(model_output, batch_labels)
    # ... training logic
```

## 🔥 Complete Symbol-Split-Merge Pipeline Example

```python
from represent import create_represent_config, DatasetBuildConfig, batch_build_datasets_from_directory
import torch
import torch.nn as nn

# Phase 1: Configure with automatic minimum sample calculation
config = create_represent_config(
    currency="AUDUSD",
    features=['volume', 'variance'],
    samples=50000,              # Base samples needed
    lookback_rows=5000,         # Historical data for classification
    lookforward_input=5000,     # Future data for classification
    lookforward_offset=500      # Offset before future window
)

dataset_config = DatasetBuildConfig(
    currency="AUDUSD",
    features=['volume', 'variance'],
    # min_symbol_samples = 60,500 (auto-calculated)
    force_uniform=True,
    keep_intermediate=False
)

print(f"🎯 Minimum samples per symbol: {config.samples + config.lookback_rows + config.lookforward_input + config.lookforward_offset:,}")

# Phase 2: Process all DBN files into comprehensive symbol datasets
print("🔄 Processing DBN directory into comprehensive symbol datasets...")
results = batch_build_datasets_from_directory(
    config=config,
    input_directory="/Users/danielfisher/data/databento/AUDUSD-micro/",  # 10+ DBN files
    output_dir="/data/symbol_datasets/",
    dataset_config=dataset_config,
    verbose=True
)

# Phase 3: ML training (implement custom dataloader in your ML repo)
print("🔄 Phase 3: Comprehensive symbol datasets ready for training...")
print(f"📁 Symbol datasets available in: /data/symbol_datasets/")
print(f"📊 Created {results['phase_2_stats']['datasets_created']} comprehensive datasets")
print(f"📈 Total samples: {results['phase_2_stats']['total_samples']:,}")
print("📖 See DATALOADER_MIGRATION_GUIDE.md for dataloader implementation")

# Example training structure (implement in your ML training repository):
"""
# In your ML training repo, implement custom dataloader using the migration guide:
dataloader = your_custom_dataloader(
    parquet_path="data/classified/AUDUSD_M6AM4_classified.parquet",
    batch_size=32,
    shuffle=True,
    sample_fraction=0.3
)

# Standard PyTorch training loop:
for epoch in range(5):
    for features, labels in dataloader:
        # features: (32, 2, 402, 250) for volume+variance with dynamic TIME_BINS=250
        # labels: (32,) with uniform distribution (7.69% each class 0-12)
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""

print("✅ Complete streamlined pipeline finished successfully!")
```

## 📋 Step-by-Step: Bin Generation → Classified Parquet Files

Here's a complete workflow showing how to generate global classification bins and apply them to build classified parquet files:

### **Step 1: Generate Global Classification Bins**

```python
from represent import calculate_global_thresholds
import json

# Calculate global thresholds from your AUDUSD dataset
# This analyzes price movements across multiple files to determine optimal bin boundaries
data_directory = "/Users/danielfisher/data/databento/AUDUSD-micro"
config = create_represent_config("AUDUSD")

print("🎯 Step 1: Generating global classification bins...")
global_thresholds = calculate_global_thresholds(
    config=config,
    data_directory=data_directory,
    sample_fraction=0.5,               # Use first 50% of files for bin calculation
    verbose=True
)

print(f"✅ Generated {global_thresholds.nbins} classification bins")
print(f"📊 Based on {global_thresholds.sample_size:,} price movements from {global_thresholds.files_analyzed} files")

# Show the generated bin boundaries
print(f"\n🎯 Generated Classification Bins:")
for i, boundary in enumerate(global_thresholds.quantile_boundaries[:-1]):
    print(f"   Bin {i}: Price movements ≤ {boundary:.2f} micro pips")

# Save bins for reuse across multiple processing runs
bin_data = {
    "currency": "AUDUSD",
    "nbins": global_thresholds.nbins,
    "boundaries": global_thresholds.quantile_boundaries.tolist(),
    "sample_size": global_thresholds.sample_size,
    "files_analyzed": global_thresholds.files_analyzed,
    "price_stats": global_thresholds.price_movement_stats
}

with open("audusd_classification_bins.json", "w") as f:
    json.dump(bin_data, f, indent=2)

print("💾 Saved bins to: audusd_classification_bins.json")
```

### **Step 2: Apply Bins to Generate Classified Parquet Files**

```python
from represent import process_dbn_to_classified_parquets
from pathlib import Path

# Get list of all DBN files to process
data_dir = Path("/Users/danielfisher/data/databento/AUDUSD-micro")
dbn_files = sorted(data_dir.glob("*.dbn*"))

print(f"\n🔄 Step 2: Applying bins to generate classified parquet files...")
print(f"📊 Found {len(dbn_files)} DBN files to process")

# Process each DBN file using the SAME global thresholds
output_dir = "classified_parquet_output"
all_results = []

for i, dbn_file in enumerate(dbn_files[:5]):  # Process first 5 files for demo
    print(f"\n📄 Processing {i+1}/{5}: {dbn_file.name}")
    
    # Apply the global bins to classify each row in this file
    results = process_dbn_to_classified_parquets(
        config=config,
        dbn_path=dbn_file,
        output_dir=output_dir,
        global_thresholds=global_thresholds,    # 🎯 Same bins for ALL files!
        force_uniform=True,                    # Ensure uniform class distribution
        verbose=False
    )
    
    all_results.append(results)
    
    print(f"   ✅ Generated {results['symbols_processed']} symbol files")
    print(f"   📊 Classified {results['total_classified_samples']:,} rows")
    print(f"   🎯 Each row assigned to one of 13 classes using global bins")

# Summary of all generated files
total_symbols = sum(r['symbols_processed'] for r in all_results)
total_rows = sum(r['total_classified_samples'] for r in all_results)

print(f"\n✅ Classified Parquet Generation Complete!")
print(f"📊 Total symbol files generated: {total_symbols}")
print(f"📊 Total rows classified: {total_rows:,}")
print(f"📁 Output directory: {output_dir}/")
```

### **Step 3: Examine Generated Classified Parquet Files**

```python
import polars as pl
from pathlib import Path

# Look at the generated classified parquet files
classified_dir = Path(output_dir)
classified_files = list(classified_dir.glob("*_classified.parquet"))

print(f"\n📊 Step 3: Examining generated classified parquet files...")
print(f"📁 Found {len(classified_files)} classified parquet files:")

for file in classified_files[:3]:  # Show first 3 files
    print(f"   📄 {file.name}")

# Load and examine one file to see the row-level classification
sample_file = classified_files[0]
df = pl.read_parquet(sample_file)

print(f"\n🔍 Sample file structure: {sample_file.name}")
print(f"📊 Total rows: {len(df):,}")
print(f"📊 Columns: {df.columns}")

# Show sample classified rows
sample_rows = df.select([
    'ts_event', 'symbol', 'ask_px_00', 'bid_px_00', 
    'price_movement', 'classification_label'
]).head(10)

print(f"\n📋 Sample rows showing price movement → classification:")
print(sample_rows)

# Show classification distribution
labels = df['classification_label'].to_numpy()
unique_labels, counts = pl.Series(labels).value_counts().sort('classification_label').to_numpy().T

print(f"\n📈 Classification distribution (each row classified individually):")
for label, count in zip(unique_labels, counts):
    percentage = (count / len(df)) * 100
    print(f"   Class {label:2d}: {count:6,} rows ({percentage:5.1f}%)")

print(f"\n🎯 Key Results:")
print(f"✅ Each row has its own classification_label based on price_movement")
print(f"✅ Same global bins used across all files ensure consistency") 
print(f"✅ Same price movement always gets same classification across files")
print(f"✅ Ready for ML training with consistent, comparable labels")
```

### **Step 4: Verify Consistency Across Files**

```python
# Verify that the same bins were applied consistently across all files
print(f"\n🔍 Step 4: Verifying consistency across files...")

consistency_check = {}
for file in classified_files[:3]:  # Check first 3 files
    df = pl.read_parquet(file)
    symbol = file.stem.split('_')[1]
    
    # Check label distribution
    labels = df['classification_label'].to_numpy()
    unique_labels = sorted(list(set(labels)))
    
    consistency_check[symbol] = {
        'file': file.name,
        'rows': len(df),
        'unique_labels': unique_labels,
        'min_label': min(labels),
        'max_label': max(labels)
    }

print("📊 Consistency check across files:")
for symbol, info in consistency_check.items():
    print(f"   {symbol}: {info['rows']:,} rows, labels {info['min_label']}-{info['max_label']}")

print(f"\n🎉 WORKFLOW COMPLETE!")
print(f"✅ Generated global classification bins from sample data")
print(f"✅ Applied same bins consistently to all files") 
print(f"✅ Each row classified individually based on price movement")
print(f"✅ All files use identical bin boundaries for consistency")
print(f"🚀 Ready for ML training with high-quality, consistent labels!")
```

**🎯 Why This Approach Works:**
- **Consistent bins**: Same price movement gets same label across all files
- **Row-level classification**: Each price movement gets individually classified
- **Uniform distribution**: Global bins ensure balanced class representation
- **ML-ready**: Classified parquet files ready for direct PyTorch loading

## 🎯 Feature Types

Configure any combination of features for different analyses:

- **Volume**: Traditional market depth (order sizes) - shape: `(402, time_bins)`
- **Variance**: Market volatility patterns - shape: `(402, time_bins)`  
- **Trade Counts**: Activity levels from transaction counts - shape: `(402, time_bins)`

**Multi-Feature Output Shapes:**
- 1 feature: `(402, time_bins)` - 2D tensor (time_bins=250 for AUDUSD)
- 2+ features: `(N, 402, time_bins)` - 3D tensor with feature dimension first

## 🎲 Dynamic Classification System

### **No Static Config Files** 
All classification thresholds are computed dynamically from your data using quantile analysis.

### **Guaranteed Uniform Distribution**
- **13-bin classification**: Each class gets exactly 7.69% of samples
- **Optimal for ML training**: No class imbalance issues
- **Quality metrics**: Average deviation typically <2%

### **Currency Support**
Works with any currency pair - automatic adaptation to market characteristics:
- **AUDUSD, EURUSD, GBPUSD**: 0.0001 pip size
- **USDJPY, EURJPY**: 0.01 pip size (JPY pairs)
- **Any other pair**: Automatic detection

## ⚙️ Simplified Configuration

### **Flat, User-Friendly Structure**
The new `RepresentConfig` eliminates complex nested structures with a simple, flat interface:

```python
from represent import RepresentConfig, create_represent_config

# Simple configuration with fully configurable parameters
config = RepresentConfig(
    currency="AUDUSD", 
    nbins=13,
    samples=50000,
    features=["volume", "variance"],
    lookback_rows=3000,     # ✅ Fully configurable (no more hardcoded 2000!)
    lookforward_input=4000, # ✅ Fully configurable
    batch_size=1500,        # ✅ Configurable batch processing
)

# Or use the convenience function with currency-specific optimizations
config = create_represent_config(
    currency="GBPUSD",      # Automatically gets lookforward_input=3000 for volatility
    samples=25000,
    features=["volume"]
)

# Direct access to all parameters - no nested structures!
print(f"Lookback: {config.lookback_rows}")        # Direct access
print(f"Lookforward: {config.lookforward_input}") # No config.classification.lookforward_input
print(f"Batch Size: {config.batch_size}")         # No more hardcoded values
print(f"Max Samples: {config.max_samples_per_file}")  # New: Performance parameters
print(f"Target Samples: {config.target_samples}")     # New: From dependency injection
```

### **Currency-Specific Optimizations**
Automatic optimizations based on market characteristics:
- **GBPUSD**: `lookforward_input=3000` (shorter window for high volatility)
- **JPY pairs**: `true_pip_size=0.01`, `nbins=9` (adapted for JPY dynamics)
- **All others**: Standard settings with `lookforward_input=5000`

### **Key Improvements**
- ✅ **No Hardcoded Values**: All timing parameters fully configurable
- ✅ **Flat Structure**: Direct access to all parameters  
- ✅ **Auto-Computed Fields**: `time_bins`, `min_symbol_samples` computed automatically
- ✅ **Validation**: Built-in parameter validation with helpful error messages
- ✅ **No Static Files**: No more complex nested config files to manage

## 🚀 High-Level API

```python
from represent import RepresentAPI

# Use the high-level API for complete workflows
api = RepresentAPI()
config = create_represent_config("AUDUSD", features=['volume', 'variance'])

# Run streamlined processing
results = api.process_dbn_to_classified_parquets(
    config=config,
    dbn_path="data.dbn",
    output_dir="/data/classified/",
    force_uniform=True
)

print(f"✅ Processing complete! {results['symbols_processed']} symbols processed")
print(f"📁 Classified data ready with {results['total_classified_samples']:,} samples")
```

## 🎨 Symbol-Split-Merge Demo

**Run the complete symbol-split-merge demonstration:**

```bash
# Run symbol-split-merge demo with mock data
python examples/symbol_split_merge_demo.py

# Run complete workflow with visualizations (requires real DBN data)
python examples/complete_workflow_demo.py

# Run quick start examples
python examples/quick_start_examples.py
```

The symbol-split-merge demos showcase:

### **🏗️ Symbol-Split-Merge Processing**
- **Phase 1**: Split multiple DBN files by symbol with memory-efficient streaming
- **Phase 2**: Merge symbol data across all files into comprehensive datasets
- **Automatic Requirements**: Calculate and enforce minimum sample requirements (60,500+)
- **Memory Efficiency**: Process DBN files >20GB with <8GB RAM usage

### **📊 Comprehensive Symbol Datasets**  
- **Large-Scale Data**: Each symbol contains merged data from 10+ DBN files
- **Uniform Distribution**: Guaranteed 7.69% per class using symbol-specific classification
- **Complete History**: Full symbol trading history for robust ML training
- **Sample Validation**: Automatic verification that datasets meet minimum requirements

### **🎨 Feature Visualization**
- **Volume Features**: Traditional market depth patterns from comprehensive data
- **Variance Features**: Price volatility across multiple time periods
- **Classification Analysis**: Distribution plots showing uniform class balance
- **Multi-File Coverage**: Visualize data coverage across source DBN files

### **⚡ Production Performance**
- **Memory Streaming**: Handle large DBN files without loading into memory
- **Batch Processing**: Efficient processing of 10+ DBN files simultaneously
- **Auto-Scaling**: Processing rate scales with available CPU cores
- **Sample Tracking**: Real-time progress and sample count validation

**Generated Output:**
```
/tmp/complete_workflow_output/
├── visualizations/
│   ├── AUDUSD_M6AM4_volume_visualization.png    # Volume feature patterns
│   ├── AUDUSD_M6AM4_variance_visualization.png  # Variance feature patterns
│   ├── AUDUSD_M6AM4_classification_analysis.png # Classification distribution
│   └── ...
├── AUDUSD_M6AM4_dataset.parquet                 # Comprehensive symbol dataset
├── AUDUSD_M6AU4_dataset.parquet                 # Another symbol dataset
└── ...
```

## ⚡ Performance

- **Phase 1 (Split)**: 300+ samples/second per DBN file during symbol splitting
- **Phase 2 (Merge)**: 1500+ samples/second during symbol dataset merging  
- **Phase 3 (Training)**: 1000+ samples/second during ML training from comprehensive datasets
- **Memory Usage**: <8GB RAM for processing multiple large DBN files
- **Scalability**: Linear scaling with CPU cores, handles datasets >100GB
- **Sample Requirements**: Automatic validation ensures 60,500+ samples per symbol

## 📊 Data Formats

**Supported Inputs:**
- **DBN files**: `.dbn`, `.dbn.zst` (compressed recommended)
- **Any currency pair**: Automatic pip size detection

**Required Columns:**
- Price levels: `ask_px_00-09`, `bid_px_00-09`
- Volume levels: `ask_sz_00-09`, `bid_sz_00-09`
- Trade counts: `ask_ct_00-09`, `bid_ct_00-09` (for trade count features)

**Output Format:**
- **Comprehensive Symbol Datasets**: Each symbol's complete history from multiple DBN files
- **Parquet files**: Optimized for ML training with memory-efficient loading
- **Pre-classified**: Uniform distribution labels ready for training
- **Tensor-ready**: Direct loading into PyTorch tensors with consistent shapes

## 🧪 Development

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest --cov=represent

# Code quality
uv run ruff format && uv run ruff check && uv run pyright

# Performance tests
uv run pytest -m performance
```

## 🔄 Migration Guide (v3.x → v4.x)

### **1. Update Function Calls**

**Old API:**
```python
# Multiple parameters - hard to maintain
calculator = GlobalThresholdCalculator(currency="AUDUSD", nbins=13, max_samples_per_file=10000)
converter = UnlabeledDBNConverter(currency="AUDUSD", batch_size=100)
```

**New API:**
```python
# Single config parameter - consistent and maintainable
config = create_represent_config("AUDUSD")
calculator = GlobalThresholdCalculator(config=config)
converter = UnlabeledDBNConverter(config=config)
```

### **2. Update Test Code**

**Old Test Setup:**
```python
def test_old_api():
    calc = GlobalThresholdCalculator(currency="AUDUSD", nbins=13)
    # Many individual parameters to mock
```

**New Test Setup:**
```python
def setup_method(self):
    self.config = create_represent_config("AUDUSD")

def test_new_api(self):
    calc = GlobalThresholdCalculator(config=self.config)
    # Single config object - easier to test
```

### **3. Benefits of Migration**

- ✅ **Reduced Complexity**: 10+ parameters → 1 config object
- ✅ **Type Safety**: Pydantic validation catches errors early
- ✅ **Consistency**: All components use same configuration
- ✅ **Testability**: Easy to mock and test with single config
- ✅ **Performance**: Eliminated parameter duplication

## 📈 Architecture Details

- **Price Levels**: 402 levels (200 bid + 200 ask + 2 mid)
- **Time Bins**: Dynamic computation based on samples/ticks_per_bin
- **Classification**: 13-bin uniform distribution by default
- **Memory**: Linear scaling with feature count
- **Thread Safety**: Concurrent access support
- **Configuration**: RepresentConfig with dependency injection

## 🎯 Why Streamlined Architecture?

### **Removed Complexity:**
- ❌ Intermediate unlabeled parquet files - direct processing
- ❌ Multi-stage pipeline overhead - single-pass approach
- ❌ Static config files - replaced with dynamic generation
- ❌ Complex stage management - streamlined workflow

### **Added Benefits:**
- ✅ Faster processing with single-pass DBN-to-classified-parquet
- ✅ Guaranteed uniform distribution for optimal ML training
- ✅ Symbol-specific processing with full context
- ✅ Reduced storage requirements with no intermediate files
- ✅ Simplified API with direct processing

## 🧪 Testing & Code Quality

### **Test Suite Organization**
The test suite has been reorganized to align with the current architecture:

- **✅ 217 Tests Passing**: Comprehensive coverage of all core functionality
- **✅ 76% Code Coverage**: Focus on critical paths and user-facing APIs  
- **✅ Removed Legacy Tests**: Eliminated 4 outdated test modules that no longer matched current architecture
- **✅ Added New Test Coverage**: Enhanced tests for `GlobalThresholdCalculator`, `ParquetClassifier`, and `RepresentConfig`

### **Removed Outdated Components:**
- ❌ `test_legacy_dataloader.py` - Old ring buffer architecture  
- ❌ `test_lazy_dataloader_new.py` - Moved dataloader functionality to ML training repos
- ❌ `test_reference_implementation.py` - Notebook-based reference code
- ❌ `test_benchmarks.py` - Benchmarks against removed reference implementation
- ❌ `reference_implementation.py` - Static reference module
- ❌ `lazy_dataloader.py` - DataLoader functionality moved to external implementation

### **Enhanced Test Coverage:**
- ✅ **Global Threshold Calculator**: Comprehensive tests for threshold calculation logic
- ✅ **API Integration**: Tests for high-level API consistency and parameter handling  
- ✅ **Configuration System**: Tests for RepresentConfig dynamic computation
- ✅ **Error Handling**: Robust tests for edge cases and error conditions

### **Test Execution:**
```bash
# Run full test suite with coverage
make test

# Run quick tests (excluding performance tests)
make test-fast

# Generate HTML coverage report
make coverage-html
```

## 📄 License

MIT License - see LICENSE file for details.

---

**🏗️ Production-ready symbol-split-merge architecture for comprehensive ML datasets with memory-efficient processing and guaranteed uniform class distribution**