# Represent v4.0.0

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-204%20passed-green.svg)](https://github.com/your-repo/represent)
[![Coverage](https://img.shields.io/badge/coverage-86.4%25-brightgreen.svg)](https://github.com/your-repo/represent)
[![Performance](https://img.shields.io/badge/latency-<10ms-orange.svg)](https://github.com/your-repo/represent)

High-performance Python package for creating normalized market depth representations from limit order book data using a **streamlined DBN-to-parquet pipeline** with **dynamic classification** and **guaranteed uniform distribution**.

## 🚀 Key Features

- **🔄 Streamlined Pipeline**: DBN→Classified Parquet→ML Training (single-pass processing)
- **⚡ Dynamic Classification**: Adaptive thresholds computed from actual market data
- **📊 Guaranteed Uniform Distribution**: 7.69% per class (13-bin) for optimal ML training
- **💾 Symbol-Grouped Processing**: Separate parquet files per symbol for targeted analysis
- **🔋 Memory-Efficient Training**: Lazy loading with configurable batch sizes for large datasets
- **🎯 Multi-Feature Support**: Volume, variance, and trade count features (configurable)
- **🧠 PyTorch Integration**: Production-ready DataLoader with direct tensor operations
- **⚡ Performance Optimized**: 86.4% test coverage with <25s test execution
- **📈 Real-World Tested**: Validated on AUDUSD, GBPUSD, and EURJPY market data

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

## 🏗️ Streamlined 2-Stage Architecture

### **Stage 1: DBN → Classified Parquet (Direct Processing)**
Process raw DBN files directly to classified parquet datasets with uniform distribution, grouped by symbol.

### **Stage 2: ML Training**
Memory-efficient lazy loading for PyTorch training with optimal class balance.

## 🚀 Quick Start

### Stage 1: DBN to Classified Parquet (Global Threshold Approach)

**🎯 RECOMMENDED: Use global thresholds for consistent classification across files**

```python
from represent import calculate_global_thresholds, process_dbn_to_classified_parquets

# Step 1: Calculate global thresholds from sample of your data files
global_thresholds = calculate_global_thresholds(
    data_directory="/path/to/your/dbn/files/",
    currency="AUDUSD", 
    sample_fraction=0.5,                    # Use 50% of files for threshold calculation
    nbins=13                                # 13-class uniform distribution
)

print(f"Global thresholds calculated from {global_thresholds.files_analyzed} files")

# Step 2: Process all files using the same global thresholds
stats = process_dbn_to_classified_parquets(
    dbn_path="data/glbx-mdp3-20240403.mbp-10.dbn.zst",
    output_dir="data/classified/",          # Directory for classified symbol files
    currency="AUDUSD",
    features=['volume', 'variance'],        # Multi-feature extraction
    global_thresholds=global_thresholds,    # 🎯 Consistent thresholds across ALL files!
    min_symbol_samples=1000,                # Only symbols with sufficient data
    force_uniform=True                      # Guarantee uniform class distribution
)

# Output: data/classified/AUDUSD_M6AM4_classified.parquet, AUDUSD_M6AU4_classified.parquet, etc.
print(f"Generated {stats['symbols_processed']} classified symbol files")
print(f"Total classified samples: {stats['total_classified_samples']:,}")
print(f"All files use consistent global thresholds - same price movement = same label!")
```

**⚠️ Why Global Thresholds Matter:**
- **❌ Per-file quantiles**: Same price movement gets different labels in different files
- **✅ Global thresholds**: Same price movement gets the same label across ALL files
- **Result**: Consistent, comparable training data for better ML performance

### Stage 2: ML Training with Lazy Loading

```python
from represent.lazy_dataloader import create_parquet_dataloader

# Create memory-efficient dataloader
dataloader = create_parquet_dataloader(
    parquet_path="data/classified/AUDUSD_M6AM4_classified.parquet",
    batch_size=32,
    shuffle=True,
    num_workers=4                          # Parallel loading for performance
)

# Use in PyTorch training loop
for batch_features, batch_labels in dataloader:
    # batch_features: torch.Tensor of shape [32, 2, 402, 500] for 2 features
    # batch_labels: torch.Tensor of shape [32] with uniform distribution (7.69% each class)
    model_output = model(batch_features)
    loss = criterion(model_output, batch_labels)
    # ... training logic
```

## 🔥 Complete Pipeline Example with Global Thresholds

```python
from represent import calculate_global_thresholds, process_dbn_to_classified_parquets, create_parquet_dataloader
import torch
import torch.nn as nn

# Stage 1A: Calculate global thresholds (do this once for your dataset)
print("🎯 Calculating global thresholds from sample data...")
global_thresholds = calculate_global_thresholds(
    data_directory="/path/to/your/audusd/data/",  # e.g., AUDUSD-micro dataset
    currency="AUDUSD",
    sample_fraction=0.5,  # Use 50% of files for threshold calculation
    nbins=13
)

# Stage 1B: Process files with consistent global thresholds
print("🔄 Stage 1: Processing DBN with global thresholds...")
processing_stats = process_dbn_to_classified_parquets(
    dbn_path="data/AUDUSD-20240101.dbn.zst",
    output_dir="data/classified/",
    currency="AUDUSD",
    features=['volume', 'variance'],
    global_thresholds=global_thresholds,  # 🎯 Consistent across all files!
    min_symbol_samples=5000,
    force_uniform=True
)

# Stage 2: ML training with guaranteed uniform distribution
print("🔄 Stage 2: Creating ML training dataloader...")
dataloader = create_parquet_dataloader(
    parquet_path="data/classified/AUDUSD_M6AM4_classified.parquet",
    batch_size=32,
    shuffle=True,
    sample_fraction=0.3
)

# Train PyTorch model with optimal class balance
model = nn.Sequential(
    nn.Conv2d(2, 32, 3),                   # 2 features: volume + variance
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 13)                      # 13-class uniform classification
)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

print("🔄 Training with guaranteed uniform distribution...")
for epoch in range(5):
    for features, labels in dataloader:
        # features: (32, 2, 402, 500) for volume+variance
        # labels: (32,) with uniform distribution (7.69% each class 0-12)
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

print("🎯 Step 1: Generating global classification bins...")
global_thresholds = calculate_global_thresholds(
    data_directory=data_directory,
    currency="AUDUSD",
    nbins=13,                           # Create 13 classification bins (0-12)
    sample_fraction=0.5,               # Use first 50% of files for bin calculation
    lookforward_rows=500,              # Price movement prediction horizon
    max_samples_per_file=10000,        # Sample size per file for efficiency
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
        dbn_path=dbn_file,
        output_dir=output_dir,
        currency="AUDUSD",
        global_thresholds=global_thresholds,    # 🎯 Same bins for ALL files!
        features=["volume"],                    # Market depth features to extract
        min_symbol_samples=1000,               # Minimum rows required per symbol
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

- **Volume**: Traditional market depth (order sizes) - shape: `(402, 500)`
- **Variance**: Market volatility patterns - shape: `(402, 500)`
- **Trade Counts**: Activity levels from transaction counts - shape: `(402, 500)`

**Multi-Feature Output Shapes:**
- 1 feature: `(402, 500)` - 2D tensor
- 2+ features: `(N, 402, 500)` - 3D tensor with feature dimension first

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

# Run streamlined processing
results = api.process_dbn_to_classified_parquets(
    dbn_path="data.dbn",
    output_dir="/data/classified/",
    currency="AUDUSD",
    features=['volume', 'variance'],
    min_symbol_samples=5000,
    force_uniform=True
)

print(f"✅ Processing complete! {results['symbols_processed']} symbols processed")
print(f"📁 Classified data ready with {results['total_classified_samples']:,} samples")
```

## ⚡ Performance

- **Stage 1 (Processing)**: 500+ samples/second direct DBN-to-classified-parquet processing
- **Stage 2 (Training)**: 1000+ samples/second during ML training
- **Memory Usage**: <4GB RAM regardless of dataset size
- **Real-time compatible**: Processes live market data

## 📊 Data Formats

**Supported Inputs:**
- **DBN files**: `.dbn`, `.dbn.zst` (compressed recommended)
- **Any currency pair**: Automatic pip size detection

**Required Columns:**
- Price levels: `ask_px_00-09`, `bid_px_00-09`
- Volume levels: `ask_sz_00-09`, `bid_sz_00-09`
- Trade counts: `ask_ct_00-09`, `bid_ct_00-09` (for trade count features)

**Output Format:**
- **Parquet files**: Optimized for ML training
- **Symbol-grouped**: Separate files per symbol for targeted analysis
- **Tensor-ready**: Direct loading into PyTorch tensors

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

## 📈 Architecture Details

- **Price Levels**: 402 levels (200 bid + 200 ask + 2 mid)
- **Time Bins**: 500 bins (configurable ticks per bin)
- **Classification**: 13-bin uniform distribution by default
- **Memory**: Linear scaling with feature count
- **Thread Safety**: Concurrent access support

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

## 📄 License

MIT License - see LICENSE file for details.

---

**🏎️ Production-ready market data processing with guaranteed uniform class distribution for optimal ML training**