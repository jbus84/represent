# Dataloader Migration Guide

This guide provides comprehensive instructions for rebuilding the represent dataloader logic in your ML training repository using Claude Code.

## Overview

The represent package previously included dataloader functionality that has been removed to keep the package focused on data conversion and feature extraction. This guide will help you rebuild a custom dataloader tailored to your specific ML training needs.

## Architecture Summary

The dataloader you'll build should follow this architecture:

```
Classified Parquet Files → Lazy DataLoader → PyTorch Training Loop
```

**Input**: Symbol-specific classified parquet files from represent (e.g., `AUDUSD_M6AM4_classified.parquet`)
**Output**: PyTorch-compatible batches with features and labels

## Core Requirements

### 1. Parquet File Structure

Your represent-generated parquet files contain:
- **Original DBN columns**: `ts_event`, `price`, `size_delta`, `side`, `action`, etc.
- **Feature columns**: Computed market depth features (402 price levels × 500 time bins)
- **Classification column**: `classification_label` (0-12 for 13-class uniform distribution)
- **Metadata columns**: Symbol-specific information

### 2. Expected Output Shapes

- **Single feature**: `(batch_size, 402, 500)`
- **Multi-feature**: `(batch_size, N_features, 402, 500)`
- **Labels**: `(batch_size,)` with uniform distribution

## Step-by-Step Implementation Guide

### Step 1: Create the Core DataLoader Class

Ask Claude to create a `RepresentDataLoader` class with these specifications:

```python
"""
Create a PyTorch DataLoader class for represent-generated parquet files with:

1. Lazy loading - only load batches as needed, not entire dataset
2. Multi-symbol support - load from multiple parquet files in a directory
3. Feature flexibility - handle volume, variance, trade_counts features
4. Uniform sampling - maintain uniform class distribution across batches
5. Memory efficiency - work with datasets larger than available RAM
6. Symbol filtering - optional filtering to specific symbols
7. Sample fraction - ability to train on subset of data for quick iteration

Key methods needed:
- __init__(parquet_dir, batch_size, features, shuffle=True, sample_fraction=1.0, symbols=None)
- __len__() - return number of batches
- __iter__() - iterate over batches
- _load_parquet_metadata() - scan parquet files and build index
- _sample_batch() - sample batch maintaining uniform distribution
- _extract_features() - extract market depth features from parquet data
"""
```

### Step 2: Implement Lazy Loading Strategy

Ask Claude to implement lazy loading with these requirements:

```python
"""
Implement lazy loading strategy that:

1. Scans all parquet files on initialization to build metadata index
2. Tracks samples per class per symbol for uniform sampling
3. Uses polars for efficient parquet reading with column selection
4. Implements LRU cache for recently accessed parquet chunks
5. Pre-allocates tensors for batch assembly to avoid memory fragmentation
6. Handles variable-length symbols (some symbols may have more data than others)

Memory constraints:
- Keep metadata index in memory (lightweight)
- Load only current batch data into memory
- Cache last N parquet chunks for efficiency
- Target <4GB RAM usage regardless of dataset size
"""
```

### Step 3: Add Feature Extraction Logic

Ask Claude to implement feature extraction:

```python
"""
Create feature extraction that converts parquet data to PyTorch tensors:

1. Volume features: Extract from market depth columns and normalize
2. Variance features: Extract price variance data and normalize  
3. Trade count features: Extract from trade count columns and normalize
4. Multi-feature stacking: Combine multiple features into 3D tensor
5. Normalization: Apply consistent normalization across features
6. GPU compatibility: Return tensors ready for GPU transfer

Feature shapes to support:
- ['volume'] → (batch_size, 402, 500)
- ['volume', 'variance'] → (batch_size, 2, 402, 500)
- ['volume', 'variance', 'trade_counts'] → (batch_size, 3, 402, 500)
"""
```

### Step 4: Implement Uniform Sampling

Ask Claude to create uniform sampling logic:

```python
"""
Implement uniform class distribution sampling:

1. Track class distribution across all symbols in dataset
2. Sample equal numbers from each class (0-12) per batch
3. Handle class imbalance by oversampling minority classes
4. Implement stratified sampling within symbols
5. Maintain randomness while ensuring uniformity
6. Support sample_fraction parameter for quick iteration

Requirements:
- Each batch should have ~equal representation of all 13 classes
- Randomization should be reproducible with seed
- Handle edge cases where some classes have fewer samples
- Efficient sampling without loading entire dataset
"""
```

### Step 5: Add Performance Optimizations

Ask Claude to optimize for performance:

```python
"""
Add performance optimizations:

1. Multi-threading: Use ThreadPoolExecutor for parallel parquet loading
2. Prefetching: Load next batch while current batch is being processed  
3. Caching: LRU cache for parquet chunks to avoid repeated disk I/O
4. Memory pooling: Pre-allocate tensor buffers and reuse them
5. Vectorized operations: Use polars/numpy for all data operations
6. Lazy evaluation: Defer expensive operations until actually needed

Performance targets:
- <50ms per batch generation
- >90% cache hit rate for frequently accessed data
- Linear scaling with number of CPU cores
- <4GB memory usage regardless of dataset size
"""
```

### Step 6: Add Configuration and Validation

Ask Claude to add configuration support:

```python
"""
Add configuration and validation:

1. Schema validation: Verify parquet files have expected columns
2. Feature validation: Check that requested features are available
3. Class distribution reporting: Log actual vs target class distribution
4. Symbol filtering: Validate symbol names and availability
5. Configuration file support: Load settings from YAML/JSON
6. Error handling: Graceful handling of corrupted files or missing data

Configuration options:
- Feature normalization methods
- Cache size limits  
- Threading parameters
- Sampling strategies
- Validation strictness levels
"""
```

## Integration Examples

### Basic Training Loop

```python
# After implementing the dataloader
from your_ml_package import RepresentDataLoader

dataloader = RepresentDataLoader(
    parquet_dir="/data/classified/",
    batch_size=32,
    features=['volume', 'variance'],
    shuffle=True,
    sample_fraction=0.1,
    symbols=["M6AM4", "M6AM5"]
)

for epoch in range(10):
    for features, labels in dataloader:
        # features: (32, 2, 402, 500)
        # labels: (32,) with uniform distribution
        outputs = model(features)
        loss = criterion(outputs, labels)
        # ... training logic
```

### Validation Split

```python
# Create train/validation splits
train_loader = RepresentDataLoader(
    parquet_dir="/data/classified/",
    batch_size=32,
    sample_fraction=0.8,  # 80% for training
    shuffle=True
)

val_loader = RepresentDataLoader(
    parquet_dir="/data/classified/", 
    batch_size=32,
    sample_fraction=0.2,  # 20% for validation
    shuffle=False
)
```

## Testing Requirements

Ask Claude to create comprehensive tests:

```python
"""
Create test suite covering:

1. Unit tests for each major component
2. Integration tests with real parquet files
3. Performance benchmarks with timing assertions
4. Memory usage tests to verify <4GB constraint
5. Edge case tests (empty files, single class, etc.)
6. Deterministic tests with fixed random seeds

Test data requirements:
- Generate small test parquet files with known distributions
- Test with various feature combinations
- Test with different symbol counts
- Verify uniform distribution in output batches
"""
```

## Migration Checklist

- [ ] Implement core RepresentDataLoader class
- [ ] Add lazy loading with polars backend
- [ ] Implement feature extraction for volume/variance/trade_counts
- [ ] Add uniform sampling with stratification
- [ ] Optimize for performance with caching and threading
- [ ] Add configuration and validation
- [ ] Create comprehensive test suite
- [ ] Benchmark against performance targets
- [ ] Document API and usage examples
- [ ] Integrate with existing ML training pipeline

## Performance Targets

Your rebuilt dataloader should meet these targets:

- **Batch Generation**: <50ms per 32-sample batch
- **Memory Usage**: <4GB RAM regardless of dataset size  
- **Cache Efficiency**: >90% hit rate for frequently accessed data
- **Throughput**: >1000 samples/second during training
- **CPU Scaling**: Linear performance improvement with additional cores

## Support

When working with Claude on this implementation:

1. **Provide Context**: Share this guide and your specific ML framework requirements
2. **Iterative Development**: Build and test one component at a time
3. **Performance Focus**: Ask Claude to benchmark each optimization
4. **Error Handling**: Ensure robust handling of edge cases and corrupted data
5. **Documentation**: Have Claude document the API and provide usage examples

This approach will give you a custom dataloader optimized for your specific use case while maintaining the performance and uniform distribution guarantees of the original represent dataloader.