# Represent Examples

This directory contains comprehensive examples demonstrating all aspects of the represent package, organized by functionality and complexity level.

## 📂 Directory Structure

### 🚀 **01_getting_started/**
Basic examples for new users to get familiar with the package
- **API usage examples** - High-level API demonstrations
- **Configuration demos** - Different ways to configure the system
- **Simple currency examples** - Basic currency-specific processing

### 🎯 **02_global_thresholds/**
**RECOMMENDED APPROACH** - Examples showing global threshold calculation for consistent classification
- **Global threshold workflow** - Complete bin generation → classification pipeline
- **Simple threshold demo** - Quick verification of the approach
- **Row classification verification** - Confirms each row gets classified individually
- **Comprehensive threshold analysis** - Full analysis with visualizations

### 📊 **03_data_processing/**
Examples focused on converting and processing market data
- **DBN to parquet conversion** - Raw data processing examples
- **Streamlined classification** - Direct DBN → classified parquet processing
- **Optimized processing demos** - Performance-optimized workflows

### 🧠 **04_ml_training/**
Machine learning integration and training examples
- **PyTorch integration** - How to use with PyTorch models
- **Dataloader examples** - Memory-efficient data loading for training
- **Real data ML examples** - Training with actual market data
- **Performance benchmarking** - ML training performance analysis

### 📈 **05_visualization/**
Data visualization and analysis examples
- **Market depth visualization** - Visual analysis of market depth data
- **Multi-feature visualization** - Visualizing different feature types
- **Extended features examples** - Advanced visualization techniques

### ⚡ **06_performance_analysis/**
Performance testing and benchmarking examples
- **Random access benchmarks** - Memory access pattern analysis
- **Dataloader performance** - Loading speed optimization
- **Throughput analysis** - Processing rate measurements
- **Memory efficiency tests** - RAM usage optimization

### 🔬 **07_advanced_features/**
Advanced usage patterns and specialized examples
- **Extended features** - Volume, variance, trade count features
- **Real market data processing** - Production-scale examples
- **Sample outputs** - Pre-generated results for reference
- **Advanced configurations** - Complex setup examples

## 🎯 **Recommended Learning Path**

### **For New Users:**
1. Start with `01_getting_started/api_usage_examples.py`
2. Try `02_global_thresholds/simple_global_threshold_demo.py`
3. Run `04_ml_training/streamlined_dataloader_simple_demo.py`

### **For Production Use:**
1. Use `02_global_thresholds/global_threshold_classification_demo.py` for your workflow
2. Integrate `04_ml_training/pytorch_training_example.py` for ML training
3. Check `06_performance_analysis/` for optimization

### **For Research/Analysis:**
1. Explore `05_visualization/` for data analysis
2. Use `07_advanced_features/` for specialized processing
3. Reference `07_advanced_features/sample_outputs/` for expected results

## 🌐 **Global Threshold Approach (RECOMMENDED)**

The examples in `02_global_thresholds/` demonstrate the **correct approach** for consistent classification across multiple files:

**❌ Problem:** Per-file quantile calculation creates inconsistent classifications
**✅ Solution:** Global thresholds ensure same price movement = same label across ALL files

**Key Benefits:**
- Consistent classification across your entire dataset
- Same price movement always gets the same label
- Better ML model performance
- Comparable results between files

## 📊 **Data Sources**

Many examples reference the AUDUSD-micro dataset at:
```
/Users/danielfisher/data/databento/AUDUSD-micro
```

This path contains real AUDUSD market data files used for demonstration. Update paths in examples to point to your own data.

## 🚀 **Quick Start**

```bash
# Navigate to examples directory
cd examples/

# Run the simple global threshold demo
python 02_global_thresholds/simple_global_threshold_demo.py

# Try ML training integration
python 04_ml_training/streamlined_dataloader_simple_demo.py

# Generate visualizations
python 05_visualization/generate_visualization.py
```

## 📝 **Output Files**

Examples generate outputs in their respective directories:
- `02_global_thresholds/` → Threshold analysis and classified parquet files
- `04_ml_training/` → Training performance metrics
- `05_visualization/` → Charts and analysis plots
- `06_performance_analysis/` → Benchmark results
- `07_advanced_features/sample_outputs/` → Reference outputs

## 🔧 **Dependencies**

Examples may require additional packages:
```bash
pip install matplotlib scipy polars torch databento
```

## 📚 **Documentation**

Each directory contains its own README with specific details about the examples in that category. Start with the directory most relevant to your use case!