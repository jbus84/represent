# 🧠 ML Training Examples

Machine learning integration and training examples showing how to use classified parquet data with PyTorch models.

## 📋 Files in this directory

### **streamlined_dataloader_simple_demo.py**
Simple demonstration of the lazy parquet dataloader:
- Memory-efficient loading from classified parquet files
- Basic PyTorch tensor integration
- Perfect for getting started with ML training

### **pytorch_training_example.py**
Complete PyTorch training example:
- Full CNN model for market depth prediction
- Training loop with validation
- Model evaluation and metrics
- Production-ready training pipeline

### **performance_benchmark.py**
Training performance analysis:
- Memory usage during training
- Loading speed benchmarks
- Throughput measurements
- Optimization recommendations

### **real_data_ml_example.py**
Real market data ML example:
- Uses actual AUDUSD classified data
- Multi-symbol training strategies
- Cross-validation approaches
- Realistic training scenarios

## 🎯 Key Features

### **Lazy Loading**
- **Memory Efficient**: Load only required batches, not entire dataset
- **Large Dataset Support**: Train on datasets larger than RAM
- **Configurable Batch Sizes**: Optimize for your hardware
- **Parallel Loading**: Multi-worker support for performance

### **Guaranteed Uniform Distribution**
- **Balanced Classes**: Each class gets exactly 7.69% of samples (13-class)
- **No Class Imbalance**: Optimal for ML training
- **Consistent Across Files**: Global thresholds ensure uniform distribution

### **Multi-Feature Support**
```python
# Single feature: volume only
batch_features.shape = (batch_size, 402, 500)

# Multi-feature: volume + variance  
batch_features.shape = (batch_size, 2, 402, 500)

# Three features: volume + variance + trade_counts
batch_features.shape = (batch_size, 3, 402, 500)
```

## 🚀 Quick Start Training

### **Basic DataLoader Setup**
```python
from represent.lazy_dataloader import create_parquet_dataloader

# Create memory-efficient dataloader
dataloader = create_parquet_dataloader(
    parquet_path="classified/AUDUSD_M6AM4_classified.parquet",
    batch_size=32,
    shuffle=True,
    sample_fraction=0.2,  # Use 20% of data for quick iteration
    num_workers=4
)

# Use in training loop
for features, labels in dataloader:
    # features: torch.Tensor [32, 402, 500] or [32, N, 402, 500]
    # labels: torch.Tensor [32] with uniform distribution
    model_output = model(features)
    loss = criterion(model_output, labels)
```

### **Multi-Symbol Training**
```python
# Train on multiple symbols simultaneously
dataloader = create_parquet_dataloader(
    parquet_path="classified/",  # Directory with multiple symbol files
    batch_size=32,
    symbols=["M6AM4", "M6AU4", "M6BM4"],  # Specific symbols
    sample_fraction=0.1
)
```

## 🏗️ Model Architectures

### **2D CNN for Single Feature**
```python
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3),       # Input: (batch, 1, 402, 500)
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 13)                      # 13-class output
)
```

### **3D CNN for Multi-Feature**
```python
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3),       # Input: (batch, 3, 402, 500)  
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(128, 13)                     # 13-class output
)
```

## ⚡ Performance Optimization

### **Memory Management**
```python
# Optimize for your hardware
dataloader = create_parquet_dataloader(
    parquet_path="classified/large_dataset.parquet",
    batch_size=16,        # Reduce if memory limited
    num_workers=8,        # Match your CPU cores
    sample_fraction=0.05  # Start small for testing
)
```

### **Training Speed**
```python
# Pre-allocate tensors for consistent performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for features, labels in dataloader:
    features = features.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    # ... training logic
```

## 📊 Expected Performance

### **Loading Performance**
- **Batch Loading**: <10ms per 32-sample batch
- **Memory Usage**: <4GB RAM regardless of parquet size
- **Throughput**: 1000+ samples/second during training

### **Model Training**
- **Classification Accuracy**: Typically 60-85% on AUDUSD (depends on model/features)
- **Training Speed**: 100-500 batches/second (depends on model complexity)
- **Memory Scaling**: Linear with batch size and feature count

## 🔧 Running Examples

```bash
# Start with simple dataloader demo
python 04_ml_training/streamlined_dataloader_simple_demo.py

# Try full PyTorch training
python 04_ml_training/pytorch_training_example.py

# Benchmark performance
python 04_ml_training/performance_benchmark.py

# Real data example (requires processed AUDUSD data)
python 04_ml_training/real_data_ml_example.py
```

## 💡 Training Tips

### **Feature Selection**
```python
# Volume only: Good for basic price prediction
features=['volume']

# Volume + Variance: Captures volatility patterns  
features=['volume', 'variance']

# All features: Maximum information but more complex
features=['volume', 'variance', 'trade_counts']
```

### **Hyperparameter Tuning**
```python
# Start with these parameters
batch_size = 32           # Good balance of speed/stability
learning_rate = 0.001     # Standard Adam learning rate
sample_fraction = 0.2     # Use subset for faster iteration

# Gradually increase as you optimize
sample_fraction = 1.0     # Use full dataset for final training
```

### **Model Validation**
```python
# Use temporal splits for market data
train_symbols = ["M6AM4", "M6AU4"]  # Earlier time periods
val_symbols = ["M6BM4", "M6CU4"]    # Later time periods

# Avoid data leakage across time
```

## ➡️ Next Steps

After training your models:
- `06_performance_analysis/` - Optimize training performance
- `05_visualization/` - Visualize model predictions and market data  
- `07_advanced_features/` - Explore advanced training techniques