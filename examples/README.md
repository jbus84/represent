# 🚀 Represent Package Examples

This directory contains a comprehensive demonstration of all core functionality of the represent package.

## 📋 Available Examples

### **🌟 comprehensive_demo.py - Complete Demonstration**

**This is the main example that demonstrates all functionality:**

- **🎨 Multi-Feature Extraction**: Volume, variance, and trade count features with RGB visualization
- **📈 Classification Distributions**: With and without force_uniform comparison  
- **⚡ DataLoader Performance**: Benchmarking different configurations
- **🧠 ML Sample Generation**: Creating training samples aligned with multi-feature extraction

**Run the comprehensive demo:**
```bash
python examples/comprehensive_demo.py
```

**Features:**
- Uses consistent synthetic dataset across all demonstrations
- Professional visualization style matching reference images
- Generates HTML and Markdown reports with all results
- Creates visualization files for each functionality area
- Includes performance metrics and statistical analysis

## 📊 Expected Outputs

Running the comprehensive demo generates:

### **Generated Files:**
```
comprehensive_demo_output/
├── synthetic_market_data.parquet          # Demo dataset
├── feature_extraction_demo.png            # Feature visualization
├── classification_distribution_demo.png   # Classification analysis
├── dataloader_performance_demo.png        # Performance benchmarks
├── ml_sample_generation_demo.png          # ML sample visualization
├── comprehensive_demo_report.html         # Interactive HTML report
├── comprehensive_demo_report.md           # Markdown documentation
└── demo_results.json                      # Raw results data
```

### **HTML Report Features:**
- 📊 Interactive navigation between sections
- 📈 Professional visualizations embedded
- 📋 Comprehensive statistics and metrics
- 🎨 Responsive design with modern styling
- 💻 Code examples for each functionality

### **Functionality Demonstrated:**

#### **🎨 Multi-Feature Extraction**
- **Volume Features**: Traditional market depth from order sizes
- **Variance Features**: Price volatility patterns across levels
- **Trade Count Features**: Activity patterns from transaction counts
- **RGB Combination**: Multi-feature visualization with proper normalization
- **Shape Handling**: Single feature (402, 500) vs multi-feature (N, 402, 500)

#### **📈 Classification Analysis**
- **With Force Uniform**: Guaranteed 7.69% per class distribution
- **Without Force Uniform**: Natural (biased) price movement distribution
- **Quality Metrics**: Standard deviation analysis for uniformity
- **Visual Comparison**: Side-by-side distribution analysis

#### **⚡ Performance Benchmarking**
- **Multiple Configurations**: Different batch sizes and worker counts
- **Throughput Analysis**: Samples per second measurements
- **Memory Usage**: RAM consumption tracking
- **Efficiency Metrics**: Performance per MB analysis

#### **🧠 ML Sample Generation**
- **Multi-Feature Tensors**: Proper tensor shapes for PyTorch
- **Label Integration**: Classified labels aligned with features
- **Memory Efficiency**: Optimized tensor formats
- **Code Examples**: Direct PyTorch integration patterns

## 🚀 Quick Start

### **1. Run Complete Demo**
```bash
# Run comprehensive demonstration
python examples/comprehensive_demo.py

# View HTML report
open comprehensive_demo_output/comprehensive_demo_report.html
```

### **2. Integration Examples**

#### **Feature Extraction**
```python
from represent import MarketDepthProcessor

processor = MarketDepthProcessor()
features = processor.extract_features(
    data=market_data,
    features=["volume", "variance", "trade_counts"]
)

# Output shapes:
# Single feature: (402, 500)
# Multi-feature: (3, 402, 500)
```

#### **Classification with Force Uniform**
```python
from represent import process_dbn_to_classified_parquets

results = process_dbn_to_classified_parquets(
    dbn_path="data.dbn",
    output_dir="classified/",
    features=["volume", "variance", "trade_counts"],
    force_uniform=True  # Guarantee uniform distribution
)
```

#### **ML Training Integration**
```python
from represent import create_parquet_dataloader
import torch.nn as nn

# Create dataloader
dataloader = create_parquet_dataloader(
    parquet_path="classified/data.parquet",
    batch_size=32,
    features=["volume", "variance", "trade_counts"]
)

# Model for multi-feature input
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3),  # 3 features
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 13)  # 13-class classification
)

# Training loop
for features, labels in dataloader:
    # features: (32, 3, 402, 500)
    # labels: (32,) with uniform distribution
    outputs = model(features)
    loss = criterion(outputs, labels)
    # ... training logic
```

## 🎯 Key Benefits Demonstrated

### **✅ Multi-Feature Support**
- Seamless extraction of different feature types
- Proper normalization and RGB combination
- Flexible single or multi-feature tensor formats

### **✅ Uniform Classification**
- Guaranteed balanced class distribution (7.69% per class)
- Eliminates class imbalance issues in ML training
- Consistent results across different datasets

### **✅ Performance Optimization**
- >1000 samples/second throughput capability
- <4GB RAM usage for large datasets
- Scalable parallel processing

### **✅ Production Ready**
- Professional visualization and reporting
- Comprehensive error handling
- Memory-efficient tensor formats
- Direct PyTorch integration

## 🔧 Customization

### **Modify Features**
```python
# In comprehensive_demo.py
self.features = ["volume"]  # Single feature
# or
self.features = ["volume", "variance"]  # Two features
```

### **Change Classification Bins**
```python
# In comprehensive_demo.py  
self.nbins = 9   # 9-class classification
# or
self.nbins = 15  # 15-class classification
```

### **Adjust Dataset Size**
```python
# In create_synthetic_dataset() method
n_samples = 50000  # Larger dataset
n_symbols = 5      # More symbols
```

## 📈 Performance Targets

The demonstration validates these performance targets:

- **Throughput**: >1000 samples/second for ML training
- **Memory**: <4GB RAM regardless of dataset size  
- **Latency**: <50ms for 32-sample batch generation
- **Accuracy**: <2% deviation from uniform distribution
- **Efficiency**: Linear scaling with features and batch size

## 🎉 Next Steps

After running the comprehensive demo:

1. **📊 Review the HTML report** for detailed analysis
2. **🔍 Examine generated visualizations** to understand each feature
3. **💻 Adapt code examples** for your own datasets
4. **⚡ Optimize configurations** based on performance results
5. **🚀 Integrate into your ML pipeline** using demonstrated patterns

---

**The comprehensive demo provides everything needed to understand and integrate the represent package into production ML workflows.**