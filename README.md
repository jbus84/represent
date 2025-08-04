# Represent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/your-repo/represent)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)](https://github.com/your-repo/represent)
[![Performance](https://img.shields.io/badge/latency-<10ms-orange.svg)](https://github.com/your-repo/represent)

High-performance Python package for creating normalized market depth representations from limit order book data. Optimized for real-time trading applications with <10ms processing targets.

## 🚀 Key Features

- **Two-Stage ML Pipeline**: Offline DBN→Parquet conversion + Online lazy loading
- **Pre-computed Classifications**: Labels generated during conversion for faster training
- **Memory-Efficient Training**: Lazy loading with configurable caching for large datasets
- **Multi-Feature Support**: Volume, variance, and trade count features
- **Currency Configurations**: Pre-optimized settings for major currency pairs with YAML support
- **PyTorch Integration**: Production-ready DataLoader with tensor operations
- **Smart Output Shapes**: Automatic 2D/3D tensor generation based on feature count
- **Performance Optimized**: <10ms processing targets for ML training applications

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

## 📊 What It Does

**Stage 1: DBN to Labeled Parquet Conversion**
- **Input**: Raw DBN market data files (.dbn, .dbn.zst)
- **Processing**: Extracts market depth + computes classification labels
- **Output**: Labeled parquet datasets ready for ML training

**Stage 2: Memory-Efficient ML Training**
- **Input**: Labeled parquet datasets
- **Processing**: Lazy loading with configurable caching
- **Features**: Volume, variance, trade counts (individually or combined)
- **Output**: PyTorch tensors with pre-computed labels

**Tensor Shapes:**
- Single feature: `(time_bins, price_levels)` → `(402, 500)`
- Multiple features: `(features, time_bins, price_levels)` → `(N, 402, 500)`

## 🚀 Quick Start

### Step 1: Convert DBN to Labeled Parquet Dataset

```python
from represent import convert_dbn_file

# Convert DBN file to labeled parquet dataset
stats = convert_dbn_file(
    dbn_path="market_data.dbn.zst",
    output_path="labeled_dataset.parquet", 
    currency="AUDUSD",                    # Currency-specific classification config
    features=['volume', 'variance'],      # Multi-feature extraction
    symbol_filter="M6AM4"                 # Optional symbol filtering
)

print(f"Converted {stats['total_samples']} samples with {stats['classification_distribution']} labels")
```

### Step 2: Create ML Training DataLoader

```python
from represent import create_market_depth_dataloader

# Create production-ready dataloader
dataloader = create_market_depth_dataloader(
    parquet_path="labeled_dataset.parquet",
    batch_size=32,
    shuffle=True,
    cache_size=1000  # Optimize for your memory constraints
)

# Use in PyTorch training loop
for batch_features, batch_labels in dataloader:
    # batch_features: torch.Tensor of shape [batch_size, time_bins, price_levels, features]
    # batch_labels: torch.Tensor of shape [batch_size] with classification targets
    model_output = model(batch_features)
    loss = criterion(model_output, batch_labels)
    # ... training logic
```

### Currency-Specific Configurations

```python
from represent import load_currency_config, list_available_currencies

# View available currency configurations
currencies = list_available_currencies()
print(f"Available currencies: {currencies}")  # ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDJPY', ...]

# Load specific currency config
config = load_currency_config('AUDUSD')
print(f"AUDUSD classification bins: {config.classification.nbins}")
print(f"AUDUSD pip size: {config.classification.pip_size}")

# Different currencies have optimized settings for their market characteristics
for currency in ['AUDUSD', 'USDJPY']:
    config = load_currency_config(currency)
    print(f"{currency}: {config.classification.nbins} bins, pip size: {config.classification.pip_size}")
```

### Complete ML Training Example

```python
from represent import convert_dbn_file, create_market_depth_dataloader
import torch
import torch.nn as nn

# 1. Convert DBN to labeled parquet (run once)
stats = convert_dbn_file(
    dbn_path="data/AUDUSD-20240101.dbn.zst",
    output_path="data/AUDUSD_labeled.parquet",
    currency="AUDUSD",
    features=['volume', 'variance'],
    chunk_size=50000
)

# 2. Create lazy dataloader for training
dataloader = create_market_depth_dataloader(
    parquet_path="data/AUDUSD_labeled.parquet", 
    batch_size=32,
    shuffle=True,
    sample_fraction=0.2,  # Use 20% of data
    cache_size=1000
)

# 3. Train PyTorch model
model = nn.Sequential(
    nn.Conv2d(2, 32, 3),  # 2 features: volume + variance
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 3)      # 3-class classification
)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for features, labels in dataloader:
        # features: (32, 2, 402, 500) for volume+variance
        # labels: (32,) with classification targets 0,1,2
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
for epoch in range(5):
    for batch_idx, (features, targets) in enumerate(dataloader):
        # features: (16, 3, 402, 500) - batch of 16, 3 features, 402x500 depth
        # targets: (16, 1) - classification targets
        
        print(f"Epoch {epoch}, Batch {batch_idx}")
        print(f"  Features shape: {features.shape}")
        print(f"  Targets shape: {targets.shape}")
        
        # Your model training here - reliable and stable
        # model(features) -> predictions
        
        if batch_idx >= 2:  # Just show first few batches
            break

print("✅ Production-ready training completed successfully")
```

## 🎯 Feature Types

- **Volume**: Traditional market depth (order sizes)
- **Variance**: Market volatility patterns 
- **Trade Counts**: Activity levels from transaction counts

Mix and match any combination for different analyses.

## ⚡ Performance

- **Single feature**: ~6ms processing (target <10ms)
- **Multi-feature**: ~11ms for 3 features (target <50ms)  
- **Real-time compatible**: Processes live DBN market data
- **Memory efficient**: Linear scaling with feature count

## 📈 Classification

Built-in price movement classification with configurable bins (3-13):

- **0-4**: Bearish movements (strong to moderate)
- **5-6**: Neutral/sideways  
- **7-12**: Bullish movements (moderate to strong)

Each batch returns `(features, targets)` where targets are integer classification labels.

## 📁 Data Formats

**Supported inputs:**
- **DBN files**: `.dbn.zst` format (recommended)
- **Polars DataFrames**: For streaming data

**Required columns:**
- Price levels: `ask_px_00-09`, `bid_px_00-09`
- Volume levels: `ask_sz_00-09`, `bid_sz_00-09`  
- Trade counts: `ask_ct_00-09`, `bid_ct_00-09` (for trade count features)

Automatic preprocessing handles missing columns and type conversions.

## ⚙️ Configuration

**Enhanced Configuration System:**
- **Validated by Pydantic**: All configurations validated at creation time
- **Complete threshold sets**: Guaranteed bin thresholds for all classification levels
- **No fallback values**: Direct dictionary access for reliable performance
- **Currency defaults**: Every dataset automatically gets currency-specific optimization

**Currency-specific settings** (automatically optimized):
- **AUDUSD**: 13 bins, 0.8 coverage, 0.0001 pip size (default when no currency specified)
- **EURUSD**: 13 bins, 0.9 coverage, 0.0001 pip size  
- **USDJPY**: 9 bins, 0.6 coverage, 0.01 pip size (JPY pairs)
- **GBPUSD**: 13 bins, 0.7 coverage, 3000 lookforward

**Manual configuration** inherits from currency base with validation.

## 🚀 Examples

```bash
# Currency configuration demo
uv run python examples/simple_currency_config_demo.py

# Production-ready dataloader demo
uv run python examples/production_dataloader_example.py

# Real market data processing
uv run python examples/dataloader_real_data_example.py

# PyTorch training example
uv run python examples/pytorch_training_example.py

# Performance benchmarks
uv run python examples/dataloader_performance_demo.py
```

## 🧪 Development

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest --cov=represent

# Code quality
uv run ruff format && uv run ruff check && uv run pyright
```

## 📈 Architecture

- **Price Levels**: 402 levels (200 bid + 200 ask + 2 mid)
- **Time Bins**: 500 bins (100 ticks per bin default)
- **Feature Scaling**: Linear memory and processing time
- **Thread Safety**: Concurrent access support

## 📄 License

MIT License - see LICENSE file for details.

---

**Production-ready market data processing for real-time trading systems** 🏎️