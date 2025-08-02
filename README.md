# Represent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/your-repo/represent)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)](https://github.com/your-repo/represent)
[![Performance](https://img.shields.io/badge/latency-<10ms-orange.svg)](https://github.com/your-repo/represent)

High-performance Python package for creating normalized market depth representations from limit order book data. Optimized for real-time trading applications with <10ms processing targets.

## 🚀 Key Features

- **<10ms Processing**: Ultra-fast market depth array generation
- **PyTorch Integration**: Native DataLoader with tensor operations
- **Multi-Feature Support**: Volume, variance, and trade count features
- **Currency Configurations**: Pre-optimized settings for major currency pairs
- **Validated Configurations**: Pydantic validation ensures reliable threshold access
- **Smart Output Shapes**: Automatic 2D/3D tensor generation based on feature count
- **Real Market Data**: Processes DBN files and streaming market data

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

Transforms limit order book data into normalized ML-ready representations:

- **Input**: Raw market data (DBN files or DataFrames)
- **Processing**: Maps to 402 price levels × 500 time bins
- **Features**: Volume, variance, trade counts (individually or combined)
- **Output**: Normalized tensors ready for PyTorch models

**Output Shapes:**
- Single feature: `(402, 500)` 
- Multiple features: `(N, 402, 500)`

## 🚀 Quick Start

### Basic Processing

```python
from represent import process_market_data
import polars as pl

# Single feature extraction
volume_features = process_market_data(df)  # (402, 500)

# Multiple features  
multi_features = process_market_data(df, features=['volume', 'variance'])  # (2, 402, 500)
```

### PyTorch DataLoader Integration

```python
import torch
from torch.utils.data import DataLoader
from represent.dataloader import MarketDepthDataset

# Create dataset with currency optimization
dataset = MarketDepthDataset(
    data_source="market_data.dbn",
    currency="AUDUSD",  # Auto-loads AUDUSD optimized settings
    features=['volume', 'variance']
)
# Note: If no currency specified, automatically defaults to AUDUSD configuration

# Standard PyTorch DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

for features, targets in dataloader:
    # features: (8, 2, 402, 500) - batch of 8, 2 features, 402x500 depth
    # targets: (8, 1) - classification targets
    pass
```

### Currency-Specific Configurations

```python
# Different currencies have optimized settings
currencies = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDJPY']

for currency in currencies:
    dataset = MarketDepthDataset(
        data_source="data.dbn",
        currency=currency,  # Loads currency-specific optimization
        features=['volume']
    )
    print(f"{currency}: {dataset.classification_config.nbins} bins")
    # AUDUSD: 13 bins, USDJPY: 9 bins (different pip sizes)
```

### AsyncDataLoader Example

```python
import time
from represent.dataloader import MarketDepthDataset, AsyncDataLoader

# Create dataset with multiple features
dataset = MarketDepthDataset(
    data_source="market_data.dbn",
    currency="EURUSD",
    features=['volume', 'variance', 'trade_counts'],  # 3 features
    batch_size=500
)

# Create AsyncDataLoader for background batch generation
async_loader = AsyncDataLoader(
    dataset=dataset,
    background_queue_size=8,  # Keep 8 batches ready
    prefetch_batches=4        # Pre-generate 4 batches on startup
)

print(f"Dataset: {len(dataset)} batches available")
print(f"Output shape per sample: {dataset.output_shape}")  # (3, 402, 500)

# Start background batch production
async_loader.start_background_production()

# Zero-latency training loop
for epoch in range(5):
    for batch_idx in range(min(3, len(dataset))):  # Show first 3 batches
        start_time = time.perf_counter()
        
        # Get batch instantly (pre-generated in background)
        batch = async_loader.get_batch()
        
        batch_time = (time.perf_counter() - start_time) * 1000
        
        print(f"Epoch {epoch}, Batch {batch_idx}")
        print(f"  Batch retrieved in: {batch_time:.2f}ms")
        print(f"  Batch shape: {batch.shape}")
        print(f"  Queue status: {async_loader.queue_status['queue_size']}/{async_loader.queue_status['max_queue_size']} batches ready")
        
        # Your model training here - batches generate in background
        # model(batch) -> predictions
        time.sleep(0.1)  # Simulate training time

# Cleanup
async_loader.stop()
print(f"Average retrieval time: {async_loader.average_retrieval_time_ms:.2f}ms")
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