# Represent Library Examples (v3.0.0)

This directory contains examples demonstrating the **clean 3-stage pipeline** for market depth processing with **guaranteed uniform distribution** and **dynamic classification**.

## Clean 3-Stage Architecture (v3.0.0)

The represent library uses a **clean separation of concerns**:

1. **Stage 1: DBN → Unlabeled Parquet (Symbol-Grouped)** - Raw data conversion with symbol separation
2. **Stage 2: Dynamic Classification (Uniform Distribution)** - Quantile-based classification with guaranteed class balance
3. **Stage 3: ML Training (Memory-Efficient)** - Lazy loading with optimal class distribution

## Key v3.0.0 Benefits

- ✅ **Guaranteed Uniform Distribution**: 7.69% per class (13-bin) for optimal ML training
- ✅ **Dynamic Configuration**: No static config files - thresholds computed from data
- ✅ **Symbol-Grouped Processing**: Separate files per symbol for targeted analysis
- ✅ **Memory-Efficient**: <4GB RAM regardless of dataset size
- ✅ **Clean API**: Direct imports, no backward compatibility wrappers

## Quick Start Examples

### 🚀 Complete 3-Stage Pipeline
```bash
python examples/api_usage_examples.py
```
Comprehensive demonstration of all v3.0.0 features and workflows.

### 🔄 Stage-by-Stage Processing
```bash
python examples/new_architecture/dbn_to_parquet_example.py
```
Detailed walkthrough of each stage with performance metrics.

### ⚡ Dynamic Classification Demo
```bash
python examples/classification_analysis/dynamic_config_demo.py
```
Shows quantile-based config generation achieving uniform distribution.

## Directory Structure

### **`api_usage_examples.py`** - 🌟 Main Examples File
Complete v3.0.0 API demonstration with 7 examples:
1. Complete 3-stage pipeline
2. Stage-by-stage processing
3. Dynamic classification
4. Batch processing
5. Currency configurations
6. Feature combinations
7. Memory optimization strategies

### **`new_architecture/`** - Core Pipeline Examples
- **`dbn_to_parquet_example.py`** - Complete 3-stage workflow with performance metrics

### **`classification_analysis/`** - Dynamic Classification Examples
- **`dynamic_config_demo.py`** - Quantile-based config generation
- **`final_quantile_validation.py`** - Validation achieving <2% deviation
- **`realistic_market_demo.py`** - Real market data classification

### **`real_data/`** - Production Examples
- **`parquet_dataloader_example.py`** - Memory-efficient training from classified parquet

### **`visualization/`** - Market Depth Visualization
- **`generate_visualization.py`** - Market depth tensor visualization

### **`random_access_evaluation/`** - Performance Benchmarks
- **`lazy_dataloader_random_access_benchmark.py`** - Memory and performance testing

## Example Categories

### 1. Complete Pipeline Examples

**Stage 1: DBN → Unlabeled Parquet**
```python
from represent import convert_dbn_to_parquet

stats = convert_dbn_to_parquet(
    dbn_path="data.dbn.zst",
    output_dir="/data/unlabeled/",
    features=['volume', 'variance'],
    min_symbol_samples=1000
)
```

**Stage 2: Dynamic Classification**
```python
from represent import classify_parquet_file

stats = classify_parquet_file(
    parquet_path="/data/unlabeled/AUDUSD_M6AM4.parquet",
    output_path="/data/classified/AUDUSD_M6AM4_classified.parquet",
    currency="AUDUSD",
    force_uniform=True  # Guaranteed uniform distribution
)
```

**Stage 3: ML Training**
```python
from represent import create_parquet_dataloader

dataloader = create_parquet_dataloader(
    parquet_path="/data/classified/AUDUSD_M6AM4_classified.parquet",
    batch_size=32,
    shuffle=True,
    sample_fraction=0.2
)

for features, labels in dataloader:
    # features: (32, [N_features,] 402, 500)
    # labels: (32,) with uniform distribution (7.69% each class)
    pass
```

### 2. High-Level API Examples

**Complete Pipeline in One Call**
```python
from represent import RepresentAPI

api = RepresentAPI()

results = api.run_complete_pipeline(
    dbn_path="data.dbn",
    output_base_dir="/data/pipeline/",
    currency="AUDUSD",
    features=['volume', 'variance'],
    force_uniform=True
)
```

### 3. Dynamic Classification Examples

**Generate Optimal Configuration**
```python
from represent import RepresentAPI

api = RepresentAPI()

config_result = api.generate_classification_config(
    parquet_files="/data/unlabeled/AUDUSD_M6AM4.parquet",
    currency="AUDUSD",
    nbins=13
)

print(f"Quality: {config_result['metrics']['validation_metrics']['quality']:.1%}")
```

### 4. Memory Optimization Examples

**Different Memory Strategies**
```python
# Quick iteration (10% data, 500 cache)
dataloader = create_parquet_dataloader(
    parquet_path="data.parquet",
    sample_fraction=0.1,
    cache_size=500
)

# Full training (100% data, 2000 cache)
dataloader = create_parquet_dataloader(
    parquet_path="data.parquet",
    sample_fraction=1.0,
    cache_size=2000
)
```

## Running Examples

### Prerequisites
```bash
# Install represent with development dependencies
uv sync --all-extras

# Ensure you have DBN data files in data/ directory
```

### Run Individual Examples
```bash
# Complete API demonstration
python examples/api_usage_examples.py

# 3-stage pipeline walkthrough
python examples/new_architecture/dbn_to_parquet_example.py

# Dynamic classification demo
python examples/classification_analysis/dynamic_config_demo.py

# Production memory-efficient training
python examples/real_data/parquet_dataloader_example.py
```

### Batch Processing Examples
```bash
# Process multiple DBN files
python -c "
from represent import batch_convert_unlabeled, batch_classify_parquet_files

# Stage 1: Batch convert
unlabeled_results = batch_convert_unlabeled('data/dbn/', 'data/unlabeled/')

# Stage 2: Batch classify
classified_results = batch_classify_parquet_files('data/unlabeled/', 'data/classified/')

print(f'Processed {len(classified_results)} files with uniform distribution')
"
```

## Performance Expectations

### Stage 1: DBN Conversion
- **Throughput**: 500+ samples/second sustained
- **Memory**: <8GB during conversion
- **Output**: Symbol-grouped parquet files

### Stage 2: Classification
- **Speed**: <5ms per file with caching
- **Quality**: <2% deviation from uniform distribution
- **Guarantee**: Exactly 7.69% per class (13-bin)

### Stage 3: ML Training
- **Throughput**: 1000+ samples/second during training
- **Memory**: <4GB RAM regardless of dataset size
- **Distribution**: Guaranteed uniform class balance

## Migration from v2.0.0

### Removed (No Longer Needed)
- ❌ Static config files (replaced with dynamic generation)
- ❌ `create_market_depth_dataloader()` (use `create_parquet_dataloader()`)
- ❌ `convert_dbn_file()` (use 3-stage pipeline instead)

### New in v3.0.0
- ✅ `convert_dbn_to_parquet()` - Stage 1: Symbol-grouped unlabeled conversion
- ✅ `classify_parquet_file()` - Stage 2: Dynamic uniform classification
- ✅ `create_parquet_dataloader()` - Stage 3: Memory-efficient ML training
- ✅ Guaranteed uniform distribution for optimal ML training

## Tips for Production Use

1. **Memory Optimization**: Use `sample_fraction` and `cache_size` to control memory usage
2. **Symbol Analysis**: Process symbols individually for targeted strategies
3. **Dynamic Classification**: Let the system compute optimal thresholds from your data
4. **Batch Processing**: Use batch functions for multiple files
5. **High-Level API**: Use `RepresentAPI.run_complete_pipeline()` for simple workflows

---

**🎯 All examples demonstrate production-ready workflows with guaranteed uniform class distribution for optimal ML training performance.**