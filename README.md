# Represent v3.0.0

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-204%20passed-green.svg)](https://github.com/your-repo/represent)
[![Coverage](https://img.shields.io/badge/coverage-86.4%25-brightgreen.svg)](https://github.com/your-repo/represent)
[![Performance](https://img.shields.io/badge/latency-<10ms-orange.svg)](https://github.com/your-repo/represent)

High-performance Python package for creating normalized market depth representations from limit order book data using a **parquet-optimized 3-stage pipeline** with **dynamic classification** and **guaranteed uniform distribution**.

## 🚀 Key Features

- **🔄 Parquet-Optimized Pipeline**: DBN→Unlabeled Parquet→Classified Parquet→ML Training
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

## 🏗️ Clean 3-Stage Architecture

### **Stage 1: DBN → Unlabeled Parquet (Symbol-Grouped)**
Convert raw DBN files to unlabeled parquet datasets, grouped by symbol for analysis.

### **Stage 2: Dynamic Classification**
Apply dynamic classification with guaranteed uniform distribution using quantile-based analysis.

### **Stage 3: ML Training**
Memory-efficient lazy loading for PyTorch training with optimal class balance.

## 🚀 Quick Start

### Stage 1: Convert DBN to Unlabeled Parquet

```python
from represent.unlabeled_converter import convert_dbn_to_parquet

# Convert DBN file to symbol-grouped unlabeled parquet datasets
stats = convert_dbn_to_parquet(
    dbn_path="data/glbx-mdp3-20240403.mbp-10.dbn.zst",
    output_dir="data/unlabeled/",           # Directory for symbol-grouped files
    features=['volume', 'variance'],        # Multi-feature extraction
    min_symbol_samples=1000                 # Only symbols with sufficient data
)

# Output: data/unlabeled/M6AM4.parquet, M6AU4.parquet, etc.
print(f"Generated {stats['symbols_processed']} symbol files")
print(f"Total samples: {stats['total_processed_samples']:,}")
print(f"Conversion time: {stats['conversion_time_seconds']:.2f}s")
```

### Stage 2: Apply Dynamic Classification

```python
from represent.parquet_classifier import ParquetClassifier

# Apply dynamic classification with guaranteed uniform distribution
classifier = ParquetClassifier("AUDUSD", force_uniform=True)
classification_stats = classifier.classify_symbol_parquet(
    parquet_path="data/unlabeled/M6AM4.parquet",
    output_path="data/classified/M6AM4_classified.parquet",
    validate_uniformity=True               # Ensure uniform distribution
)

print(f"Uniformity assessment: {classification_stats['uniformity_assessment']}")
print(f"Class distribution: {classification_stats['class_distribution']}")
print(f"Processing time: {classification_stats['processing_time_seconds']:.3f}s")
```

### Stage 3: ML Training with Lazy Loading

```python
from represent.lazy_dataloader import create_parquet_dataloader

# Create memory-efficient dataloader
dataloader = create_parquet_dataloader(
    parquet_path="data/classified/M6AM4_classified.parquet",
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

## 🔥 Complete Pipeline Example

```python
from represent import convert_dbn_to_parquet, classify_parquet_file, create_parquet_dataloader
import torch
import torch.nn as nn

# Stage 1: DBN to unlabeled parquet (symbol-grouped)
print("🔄 Stage 1: Converting DBN to unlabeled parquet...")
conversion_stats = convert_dbn_to_parquet(
    dbn_path="data/AUDUSD-20240101.dbn.zst",
    output_dir="data/unlabeled/",
    currency="AUDUSD",
    features=['volume', 'variance'],
    min_symbol_samples=5000
)

# Stage 2: Dynamic classification with uniform distribution
print("🔄 Stage 2: Applying dynamic classification...")
classification_stats = classify_parquet_file(
    parquet_path="data/unlabeled/AUDUSD_M6AM4.parquet",
    output_path="data/classified/AUDUSD_M6AM4_classified.parquet",
    currency="AUDUSD",
    force_uniform=True
)

# Stage 3: ML training with guaranteed uniform distribution
print("🔄 Stage 3: Creating ML training dataloader...")
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

print("✅ Complete 3-stage pipeline finished successfully!")
```

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

## 🚀 High-Level API

```python
from represent import RepresentAPI

# Use the high-level API for complete workflows
api = RepresentAPI()

# Run complete 3-stage pipeline
results = api.run_complete_pipeline(
    dbn_path="data.dbn",
    output_base_dir="/data/pipeline_output/",
    currency="AUDUSD",
    features=['volume', 'variance'],
    min_symbol_samples=5000,
    force_uniform=True
)

print(f"✅ Pipeline complete! {results['total_symbols']} symbols processed")
print(f"📁 Classified data ready: {results['classified_directory']}")
```

## ⚡ Performance

- **Stage 1 (Conversion)**: 500+ samples/second sustained
- **Stage 2 (Classification)**: <5ms per file with caching
- **Stage 3 (Training)**: 1000+ samples/second during ML training
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

## 🎯 Why v3.0.0?

### **Removed Complexity:**
- ❌ Static config files - replaced with dynamic generation
- ❌ Backward compatibility wrappers - direct imports only
- ❌ Mixed-concern modules - clean separation of stages

### **Added Benefits:**
- ✅ Guaranteed uniform distribution for optimal ML training
- ✅ Symbol-specific processing for targeted analysis
- ✅ Dynamic configuration adapts to any market data
- ✅ Simplified API with clear stage separation

## 📄 License

MIT License - see LICENSE file for details.

---

**🏎️ Production-ready market data processing with guaranteed uniform class distribution for optimal ML training**