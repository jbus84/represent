# 🚀 Represent Package Examples

This directory contains a clean, focused example demonstrating the complete workflow of the `represent` package using all three core modules.

## 📋 Available Examples

### **🌟 `complete_workflow_demo.py` - Complete Three-Module Demo**

This is the primary comprehensive example that showcases the complete represent workflow using all three core modules in sequence. It demonstrates how the modules work together to create a complete ML-ready dataset pipeline.

**What it demonstrates:**

1. **🎯 Global Threshold Calculator**
   - Calculate consistent classification thresholds from sample data
   - Generate quantile-based boundaries for uniform distribution
   - Visualize threshold distribution and boundary values

2. **🏗️ Dataset Builder** 
   - Create comprehensive symbol datasets using calculated thresholds
   - Demonstrate symbol-split-merge architecture
   - Generate uniform class distributions across datasets

3. **⚡ Market Depth Processor**
   - Process market data into normalized tensors
   - Generate multiple features (volume, variance, trade_counts)
   - Create comprehensive visualizations including RGB feature combinations

**Key Features:**
- **Unified HTML Report** - Complete results in a single comprehensive report
- **Step-by-Step Progress** - Clear logging of each processing stage
- **Comprehensive Visualizations** - Multiple charts and plots for analysis
- **Mock Data Support** - Works even without DBN files for demonstration
- **Performance Metrics** - Processing times, memory usage, and data quality metrics

**Run the complete demo:**
```bash
python examples/complete_workflow_demo.py
```

**Output:**
- Creates `examples/complete_workflow_output/` directory
- Generates individual feature visualizations (volume, variance, trade_counts)
- Creates RGB feature combination plots
- Produces comprehensive HTML report with all results

## 🎯 Core Workflow Illustrated

### Three-Module Architecture

The example demonstrates the clean three-module architecture of represent:

1. **Global Threshold Calculator** (`global_threshold_calculator`)
   - Analyzes sample data to determine optimal classification boundaries
   - Ensures consistent thresholds across multiple files
   - Provides quantile-based uniform distribution

2. **Dataset Builder** (`dataset_builder`)
   - Uses calculated thresholds to create symbol datasets
   - Implements symbol-split-merge for comprehensive coverage
   - Generates ML-ready datasets with uniform class distribution

3. **Market Depth Processor** (`market_depth_processor`)  
   - Converts market data into normalized tensors
   - Supports multiple features simultaneously
   - Provides high-performance processing for ML training

### Feature Generation Pipeline

The demo shows how to generate and visualize multiple features:

- **Volume Features**: Market depth from order sizes
- **Variance Features**: Price volatility patterns  
- **Trade Count Features**: Transaction activity levels

All features are generated on-demand with consistent normalization and shape:
- **1 feature**: `(402, 500)` - 2D tensor
- **2+ features**: `(N, 402, 500)` - 3D tensor with feature dimension first

### Visualization Capabilities

The example creates comprehensive visualizations:

- **Individual Feature Plots**: Heatmaps for each feature type
- **RGB Combinations**: Multi-channel visualization using feature combinations
- **Statistical Analysis**: Distribution plots and summary statistics
- **Quality Metrics**: Data completeness and processing performance

### Report Generation

All results are compiled into a unified HTML report containing:

- **Step-by-Step Results**: Progress and outcomes from each module
- **Performance Metrics**: Processing times, memory usage, sample counts
- **Visual Analysis**: All generated plots and charts
- **Data Quality**: Statistics on created datasets and features

## 🚀 Getting Started

1. **Run the demo**:
   ```bash
   python examples/complete_workflow_demo.py
   ```

2. **View results**: 
   - Open `examples/complete_workflow_output/complete_workflow_report.html`
   - Review individual visualization files in the output directory

3. **Customize for your data**:
   - Modify the DBN file paths in the script
   - Adjust feature configurations as needed
   - Update currency and processing parameters

## 💡 Key Benefits Demonstrated

- **Clean Architecture**: Three focused modules with clear responsibilities
- **End-to-End Pipeline**: Complete workflow from raw data to ML-ready tensors
- **Performance Optimized**: High-speed processing with memory efficiency
- **Comprehensive Reporting**: Detailed analysis and visualization of results
- **Flexible Configuration**: Easy customization for different currencies and features
- **Production Ready**: Robust error handling and fallback mechanisms

This example serves as both a demonstration and a template for implementing represent in your own ML projects.