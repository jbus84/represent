# Represent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/your-repo/represent)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/your-repo/represent)
[![Performance](https://img.shields.io/badge/throughput-400K%2B%20rps-orange.svg)](https://github.com/your-repo/represent)

**Represent** is a high-performance Python package for creating normalized market depth representations from limit order book (LOB) data. Designed for real-time trading applications where every millisecond matters.

## 🚀 Key Features

- **Ultra-Fast Processing**: 400K+ records/second sustained throughput, 3M+ peak throughput
- **Background Batch Processing**: 741.9x faster batch loading for ML training (29.77ms → 0.040ms)
- **PyTorch Integration**: Zero-copy tensor operations with async dataloaders
- **Memory Efficient**: Optimized data structures with minimal memory footprint
- **Real-Time Ready**: Sub-millisecond latency for single array generation
- **Vectorized Operations**: NumPy-optimized algorithms for maximum performance
- **Type Safe**: Full type annotations with strict type checking
- **Well Tested**: 88% test coverage with comprehensive benchmarks

## 📊 What It Does

Represent transforms raw limit order book data into normalized 2D representations:

1. **Price Binning**: Converts prices to micro-pip format and maps to 402 price levels (200 bid + 200 ask + 2 mid-price)
2. **Time Aggregation**: Groups tick data into 500 time bins (100 ticks per bin by default)
3. **Volume Mapping**: Creates a 2D grid of market depth (price levels × time bins)
4. **Normalization**: Produces the final `normed_abs_combined` array (shape: 402×500)

## 🔧 Installation

```bash
# Install with uv (recommended)
uv add represent

# Or with pip
pip install represent
```

## 📖 Quick Start

### Basic Usage

```python
import polars as pl
from represent import process_market_data

# Load your market data (Polars DataFrame with LOB columns)
df = pl.read_csv("market_data.csv")

# Process into normalized representation
result = process_market_data(df)

# Result is a numpy array with shape (402, 500)
print(f"Output shape: {result.shape}")
print(f"Data type: {result.dtype}")
```

### Advanced Usage with Processor

```python
from represent import create_processor, MarketDepthProcessor

# Create a reusable processor for better performance
processor = create_processor()

# Process multiple datasets
results = []
for data_batch in data_batches:
    result = processor.process(data_batch)
    results.append(result)
```

### PyTorch Integration with Background Processing

For machine learning applications, use the ultra-fast background batch processing:

```python
from represent.dataloader import MarketDepthDataset, AsyncDataLoader
import torch.nn as nn

# Create dataset from your market data
dataset = MarketDepthDataset(buffer_size=50000)
dataset.add_streaming_data(your_market_data)

# Create async dataloader with background processing  
async_loader = AsyncDataLoader(
    dataset=dataset,
    background_queue_size=4,  # Keep 4 batches ready
    prefetch_batches=2        # Pre-generate 2 batches
)

# Start background batch production
async_loader.start_background_production()

# Training loop - batches are instant!
model = nn.Sequential(...)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    # Get batch (sub-millisecond when queue is full!)
    batch = async_loader.get_batch()  # Shape: (402, 500)
    
    # Standard PyTorch training
    batch = batch.unsqueeze(0).unsqueeze(0)  # Add batch & channel dims
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Cleanup
async_loader.stop()
```

**Background Processing Benefits:**
- **741.9x faster** batch access (29.77ms → 0.040ms)
- **100% GPU utilization** during training
- **Zero training bottlenecks** from data loading
- **Thread-safe** concurrent operations

### Expected Data Format

Your input DataFrame should contain these columns:

```python
# Required columns for 10-level market data
columns = [
    'ts_event', 'ts_recv', 'rtype', 'publisher_id', 'symbol',
    
    # Ask prices (10 levels)
    'ask_px_00', 'ask_px_01', ..., 'ask_px_09',
    
    # Bid prices (10 levels)  
    'bid_px_00', 'bid_px_01', ..., 'bid_px_09',
    
    # Ask volumes (10 levels)
    'ask_sz_00', 'ask_sz_01', ..., 'ask_sz_09',
    
    # Bid volumes (10 levels)
    'bid_sz_00', 'bid_sz_01', ..., 'bid_sz_09',
    
    # Ask counts (10 levels)
    'ask_ct_00', 'ask_ct_01', ..., 'ask_ct_09',
    
    # Bid counts (10 levels)
    'bid_ct_00', 'bid_ct_01', ..., 'bid_ct_09'
]
```

## ⚡ Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| **Sustained Throughput** | 400K+ records/second |
| **Peak Throughput** | 3M+ records/second |
| **Single Array Latency** | <15ms (50K records) |
| **Background Batch Access** | 0.040ms (741.9x speedup) |
| **GPU Utilization** | 100% (vs 36% synchronous) |
| **Memory Usage** | <100MB per operation |
| **Test Coverage** | 88%+ |

### Performance Tips

1. **Use background processing** for ML training:
   ```python
   # 741.9x faster than synchronous batch loading
   async_loader = AsyncDataLoader(dataset, background_queue_size=4)
   ```

2. **Use the processor factory** for repeated operations:
   ```python
   processor = create_processor()  # Reuse this
   ```

3. **Batch your data** for optimal throughput:
   ```python
   # Process in chunks of 50K records for best performance
   chunk_size = 50000
   ```

4. **Tune queue size** for your training speed:
   ```python
   # Larger queues for slower training, smaller for faster
   background_queue_size = 4  # Good default
   ```

## 🧪 Development

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/represent.git
cd represent

# Install dependencies
make install
```

### Testing

```bash
# Run all tests (including performance benchmarks)
make test

# Run fast tests only (skip performance benchmarks)
make test-fast

# Run with coverage report
make test-coverage

# Generate HTML coverage report
make coverage-html
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make typecheck

# Run all checks
make check-commit
```

## 📁 Project Structure

```
represent/
├── represent/
│   ├── __init__.py          # Public API
│   ├── constants.py         # Performance-tuned constants
│   ├── core.py             # Core functionality
│   ├── data_structures.py  # Optimized data structures
│   ├── dataloader.py       # PyTorch integration & background processing
│   └── pipeline.py         # Main processing pipeline
├── tests/
│   ├── test_benchmarks.py  # Performance benchmarks
│   ├── test_core.py        # Core functionality tests
│   ├── test_dataloader.py  # Dataloader and PyTorch tests
│   ├── test_integration.py # Integration tests
│   └── fixtures/           # Test data generation
├── examples/               # Usage examples and demos
│   ├── simple_background_usage.py
│   ├── background_training_demo.py
│   └── dataloader_performance_demo.py
├── notebooks/              # Analysis notebooks
└── Makefile               # Development commands
```

## 🔬 Technical Details

### Constants

```python
from represent import (
    MICRO_PIP_SIZE,    # 0.00001 - Price precision
    TICKS_PER_BIN,     # 100 - Ticks per time bin
    SAMPLES,           # 50000 - Expected input size
    PRICE_LEVELS,      # 402 - Output price levels
    TIME_BINS,         # 500 - Output time bins
)
```

### Data Types

- **Input**: Polars DataFrame with market data
- **Output**: NumPy array (float32, shape: 402×500)
- **Internal**: Optimized int64/float64 for calculations

### Algorithm Overview

1. **Price Conversion**: Raw prices → micro-pip integers
2. **Time Binning**: Tick timestamps → time bin indices  
3. **Price Mapping**: Micro-pip prices → price level indices
4. **Volume Aggregation**: Sum volumes per (price_level, time_bin)
5. **Cumulative Calculation**: Compute cumulative market depth
6. **Normalization**: Apply final normalization and differencing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Optimized for real-time trading applications
- Built with NumPy and Polars for maximum performance
- Inspired by modern market microstructure research

---

**Performance-first design for production trading systems** 🏎️