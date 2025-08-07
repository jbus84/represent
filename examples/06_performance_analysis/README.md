# ⚡ Performance Analysis Examples

Performance testing and benchmarking examples to optimize processing speeds, memory usage, and training throughput.

## 📋 Files in this directory

### **dataloader_performance_benchmark.py**
Comprehensive dataloader performance analysis:
- Memory access pattern benchmarks
- Loading speed measurements across different batch sizes
- Multi-worker performance comparison
- Memory usage profiling during training

### **random_access_benchmark.py**
Memory access pattern analysis:
- Sequential vs random access performance
- Cache efficiency measurements
- Optimal access pattern identification
- Memory bandwidth utilization

### **throughput_analysis.py**
Processing throughput measurements:
- DBN to parquet conversion speeds
- Classification processing rates
- End-to-end pipeline benchmarks
- Bottleneck identification

### **memory_efficiency_tests.py**
Memory usage optimization analysis:
- Memory usage across different dataset sizes
- Lazy loading efficiency validation
- Memory leak detection
- Optimal configuration recommendations

## 🎯 Performance Targets

### **Critical Latency Targets (NON-NEGOTIABLE)**
- **DBN Conversion**: <1000 samples/second sustained processing
- **Parquet Loading**: <10ms for single sample loading (with caching)
- **Batch Processing**: <50ms for 32-sample batch generation
- **Feature Processing**: <2ms additional latency per additional feature
- **Memory Usage**: <4GB RAM for training on multi-GB parquet datasets

### **Throughput Requirements**
- **Training**: 1000+ samples/second during ML training
- **Conversion**: 500+ samples/second during DBN→parquet conversion
- **Parallel Processing**: Must scale linearly with CPU cores

## ⚡ Benchmarking Framework

### **Performance Testing Setup**
```python
import time
import psutil
import torch
from represent.lazy_dataloader import create_parquet_dataloader

def benchmark_dataloader(parquet_path, batch_size, num_batches=100):
    """Benchmark dataloader performance."""
    
    # Create dataloader
    dataloader = create_parquet_dataloader(
        parquet_path=parquet_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Memory baseline
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Timing benchmark
    start_time = time.perf_counter()
    samples_processed = 0
    
    for i, (features, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        samples_processed += features.shape[0]
    
    end_time = time.perf_counter()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    processing_time = end_time - start_time
    samples_per_second = samples_processed / processing_time
    memory_used = end_memory - start_memory
    
    return {
        'samples_per_second': samples_per_second,
        'memory_used_mb': memory_used,
        'processing_time': processing_time,
        'samples_processed': samples_processed
    }
```

### **Memory Usage Analysis**
```python
import tracemalloc

def profile_memory_usage(parquet_path):
    """Profile memory usage during loading."""
    
    tracemalloc.start()
    
    # Load data
    dataloader = create_parquet_dataloader(parquet_path, batch_size=32)
    
    # Process several batches
    for i, (features, labels) in enumerate(dataloader):
        if i >= 10:
            break
    
    # Get memory snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    return top_stats
```

## 📊 Benchmark Results Examples

### **Dataloader Performance (Target: 1000+ samples/sec)**
```
Batch Size  | Samples/sec | Memory (MB) | Latency/batch
------------|-------------|-------------|---------------
16          | 1,250       | 245         | 12.8ms
32          | 1,450       | 315         | 22.1ms ✅
64          | 1,380       | 445         | 46.4ms ✅  
128         | 1,150       | 725         | 111.3ms ❌
```

### **Feature Processing Performance**
```
Features    | Processing Time | Memory Usage | Samples/sec
------------|-----------------|--------------|------------
volume      | 8.2ms           | 180 MB       | 1,420
vol+var     | 10.1ms (+1.9ms) | 245 MB       | 1,180 ✅
vol+var+tc  | 12.8ms (+2.7ms) | 320 MB       | 980
```

### **Memory Scaling Analysis**
```
Dataset Size | Memory Usage | Loading Time | Efficiency
-------------|--------------|--------------|------------
100MB        | 185 MB       | 2.1s         | 95%
500MB        | 210 MB       | 3.8s         | 92% ✅
2GB          | 285 MB       | 8.2s         | 88% ✅
10GB         | 445 MB       | 15.1s        | 82% ✅
```

## 🔧 Running Performance Tests

### **Complete Performance Suite**
```bash
# Run all performance benchmarks
python 06_performance_analysis/dataloader_performance_benchmark.py

# Memory access patterns
python 06_performance_analysis/random_access_benchmark.py  

# Processing throughput
python 06_performance_analysis/throughput_analysis.py

# Memory efficiency validation
python 06_performance_analysis/memory_efficiency_tests.py
```

### **Custom Benchmarking**
```python
# Test your specific configuration
from performance_analysis.dataloader_performance_benchmark import benchmark_suite

results = benchmark_suite(
    parquet_path="your_classified_data.parquet",
    batch_sizes=[16, 32, 64],
    num_workers_list=[2, 4, 8],
    feature_combinations=[
        ['volume'],
        ['volume', 'variance'],
        ['volume', 'variance', 'trade_counts']
    ]
)
```

## 🎯 Optimization Strategies

### **Memory Optimization**
```python
# Optimal configuration for 8GB RAM system
dataloader = create_parquet_dataloader(
    parquet_path="large_dataset.parquet",
    batch_size=32,           # Sweet spot for most hardware
    num_workers=4,           # Match CPU cores
    sample_fraction=0.1,     # Start with subset
    pin_memory=True,         # For GPU training
    persistent_workers=True  # Reduce worker startup overhead
)
```

### **Throughput Optimization**  
```python
# High-throughput configuration
dataloader = create_parquet_dataloader(
    parquet_path="dataset.parquet", 
    batch_size=64,           # Larger batches for throughput
    num_workers=8,           # Maximum parallelism
    prefetch_factor=4,       # Aggressive prefetching
    drop_last=True          # Consistent batch sizes
)
```

### **Memory-Constrained Optimization**
```python
# Low-memory configuration
dataloader = create_parquet_dataloader(
    parquet_path="dataset.parquet",
    batch_size=16,           # Smaller batches
    num_workers=2,           # Fewer workers
    sample_fraction=0.05,    # Small dataset subset
    persistent_workers=False # Reduce memory overhead
)
```

## 📈 Performance Monitoring

### **Real-time Monitoring**
```python
import time
import psutil
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.process = psutil.Process()
    
    def record_batch(self, start_time, batch_size):
        batch_time = time.perf_counter() - start_time
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        self.times.append(batch_time)
        self.memory_usage.append(memory_mb)
        
        if len(self.times) >= 10:
            avg_time = sum(self.times) / len(self.times)
            samples_per_sec = batch_size / avg_time
            avg_memory = sum(self.memory_usage) / len(self.memory_usage)
            
            print(f"Avg: {samples_per_sec:.0f} samples/sec, {avg_memory:.0f} MB")

# Usage in training loop
monitor = PerformanceMonitor()
for features, labels in dataloader:
    start_time = time.perf_counter()
    
    # ... training logic ...
    
    monitor.record_batch(start_time, features.shape[0])
```

## 🚨 Performance Regression Detection

### **Automated Performance Tests**
```python
def performance_regression_test():
    """Detect performance regressions."""
    
    # Baseline performance (update these based on your hardware)
    BASELINE_SAMPLES_PER_SEC = 1000
    BASELINE_MEMORY_MB = 300
    TOLERANCE = 0.1  # 10% tolerance
    
    results = benchmark_dataloader("test_data.parquet", batch_size=32)
    
    # Check performance regression
    if results['samples_per_second'] < BASELINE_SAMPLES_PER_SEC * (1 - TOLERANCE):
        raise AssertionError(f"Performance regression: {results['samples_per_second']} < {BASELINE_SAMPLES_PER_SEC}")
    
    if results['memory_used_mb'] > BASELINE_MEMORY_MB * (1 + TOLERANCE):
        raise AssertionError(f"Memory regression: {results['memory_used_mb']} > {BASELINE_MEMORY_MB}")
    
    print("✅ Performance tests passed!")
```

## 📊 Expected Output Files

Performance tests generate:
```
06_performance_analysis/outputs/
├── benchmark_results.json           # Detailed benchmark data
├── memory_profile.json             # Memory usage analysis  
├── performance_charts.png          # Visualization of results
├── optimization_recommendations.txt # Specific optimization advice
└── regression_test_results.log     # Pass/fail status
```

## ➡️ Next Steps

After performance analysis:
- Use optimization recommendations to tune your configuration
- `04_ml_training/` - Apply optimizations to your training pipeline
- `07_advanced_features/` - Explore advanced performance techniques
- Monitor production performance using the monitoring tools