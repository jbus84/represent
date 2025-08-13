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

### Three-Module Architecture with Focused Configurations

The example demonstrates the clean three-module architecture of represent with the new focused configuration system:

1. **Global Threshold Calculator** (`global_threshold_calculator`)
   - **Configuration**: `GlobalThresholdConfig` - only threshold calculation parameters
   - Analyzes sample data to determine optimal classification boundaries
   - Ensures consistent thresholds across multiple files
   - Provides quantile-based uniform distribution

2. **Dataset Builder** (`dataset_builder`)
   - **Configuration**: `DatasetBuilderConfig` - only dataset building parameters  
   - Uses calculated thresholds to create symbol datasets
   - Implements symbol-split-merge for comprehensive coverage
   - Generates ML-ready datasets with uniform class distribution

3. **Market Depth Processor** (`market_depth_processor`)  
   - **Configuration**: `MarketDepthProcessorConfig` - only feature processing parameters
   - Converts market data into normalized tensors
   - Supports multiple features simultaneously
   - Provides high-performance processing for ML training

### 🆕 New Focused Configuration Architecture

The demo showcases the new configuration approach:

- **Separate Configs**: Each module has its own focused Pydantic configuration
- **Clear Separation**: No confusion between unrelated parameters
- **Better Validation**: Type safety and field validation for each module
- **Compatibility Helper**: `create_compatible_configs()` for workflows using all modules

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

## 🔧 Configuration Examples

The demo shows two different ways to create configurations:

### Individual Focused Configs
```python
from represent.configs import GlobalThresholdConfig, DatasetBuilderConfig, MarketDepthProcessorConfig

# Each module gets its own focused configuration
threshold_config = GlobalThresholdConfig(
    currency="AUDUSD", nbins=13, lookback_rows=1000
)
dataset_config = DatasetBuilderConfig(
    currency="AUDUSD", lookback_rows=1000, lookforward_input=1000
)
processor_config = MarketDepthProcessorConfig(
    features=["volume", "variance"], samples=25000
)
```

### Compatible Configs (Recommended for Multi-Module Workflows)
```python
from represent.configs import create_compatible_configs

# One function creates all three compatible configurations
dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
    currency="AUDUSD",
    features=["volume", "variance"],
    lookback_rows=1000,
    nbins=13
)
```


## 💡 Key Benefits Demonstrated

- **🆕 Focused Configurations**: Each module has its own Pydantic config with only relevant parameters
- **Clean Architecture**: Three focused modules with clear responsibilities  
- **End-to-End Pipeline**: Complete workflow from raw data to ML-ready tensors
- **Performance Optimized**: High-speed processing with memory efficiency
- **Comprehensive Reporting**: Detailed analysis and visualization of results
- **Flexible Configuration**: Easy customization for different currencies and features
- **Type Safety**: Pydantic validation ensures correct parameter usage
- **Better Separation**: No confusion between module-specific parameters
- **Compatibility Helper**: `create_compatible_configs()` for complex workflows
- **Production Ready**: Robust error handling and fallback mechanisms

This example serves as both a demonstration and a template for implementing the new focused configuration architecture in your own ML projects.