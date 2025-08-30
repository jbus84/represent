# Represent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-104%20passed-green.svg)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-74%25-green.svg)](#testing)

High-performance Python package for creating normalized market depth representations from limit order book data using a **symbol-split-merge architecture**. Built for machine learning applications requiring comprehensive, uniform datasets from multiple DBN files.

**🆕 v5.0.0+**: Now features **focused Pydantic configuration models** for each core module, replacing the monolithic configuration approach.

## 🚀 Key Features

- **📊 Symbol-Split-Merge Architecture**: Process multiple DBN files into comprehensive symbol datasets
- **⚡ High Performance**: 1500+ samples/second processing with memory-efficient streaming
- **🎯 Uniform Distribution**: Guaranteed balanced class distributions for optimal ML training
- **🔧 Three Core Modules**: Clean, focused architecture with separate Pydantic configs for each module
- **🆕 Focused Configuration**: Type-safe Pydantic models with auto-computed fields and validation
- **📈 Multi-Feature Support**: Volume, variance, and trade count features
- **🧠 Framework Agnostic**: Compatible with PyTorch, TensorFlow, or custom ML frameworks

## 📦 Installation

```bash
# Using uv (recommended)
uv add represent

# Using pip
pip install represent

# Development installation
git clone <repository-url>
cd represent
uv sync --all-extras
```

## 🏗️ Three Core Modules

### 1. 📊 Dataset Builder (`dataset_builder`)
**Primary module for creating comprehensive symbol datasets from multiple DBN files**

```python
from represent import (
    build_datasets_from_dbn_files, DatasetBuildConfig, 
    DatasetBuilderConfig, create_compatible_configs
)

# Configure processing with NEW focused configs approach
from represent.configs import create_compatible_configs

dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
    currency="AUDUSD",
    features=['volume', 'variance'],
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500
)

dataset_config = DatasetBuildConfig(
    currency="AUDUSD",
    force_uniform=True  # Ensures balanced class distribution
)

# Build comprehensive symbol datasets from multiple DBN files
results = build_datasets_from_dbn_files(
    config=dataset_cfg,
    dbn_files=[
        "data/AUDUSD-20240101.dbn.zst",
        "data/AUDUSD-20240102.dbn.zst", 
        "data/AUDUSD-20240103.dbn.zst"
    ],
    output_dir="symbol_datasets/",
    dataset_config=dataset_config
)

# Output: symbol_datasets/AUDUSD_M6AM4_dataset.parquet (comprehensive symbol data)
print(f"Created {results['phase_2_stats']['datasets_created']} symbol datasets")
print(f"Total samples: {results['phase_2_stats']['total_samples']:,}")
```

**Key Functions:**
- `build_datasets_from_dbn_files()` - Process multiple DBN files
- `batch_build_datasets_from_directory()` - Process entire directories
- `DatasetBuilder` - Advanced processing with custom workflows

### 2. ⚡ Market Depth Processor (`market_depth_processor`)
**High-performance processor for converting market data into normalized tensors**

```python
from represent import MarketDepthProcessor, create_processor, process_market_data
import polars as pl

# Create processor with NEW focused config approach
from represent.configs import MarketDepthProcessorConfig

processor_config = MarketDepthProcessorConfig(
    features=['volume', 'variance'],
    samples=50000,
    ticks_per_bin=100
)
processor = MarketDepthProcessor(config=processor_config)

# Load market data
market_data = pl.read_parquet("symbol_datasets/AUDUSD_M6AM4_dataset.parquet")

# Process into normalized tensor representation
tensor_data = processor.process(market_data)

# Output shape: (2, 402, 500) for 2 features, 402 price levels, 500 time bins
print(f"Tensor shape: {tensor_data.shape}")
print(f"Data type: {tensor_data.dtype}")

# Convenience function for single-use processing
tensor_data = process_market_data(market_data, config=processor_config)
```

**Key Functions:**
- `MarketDepthProcessor` - Main processor class
- `process_market_data()` - Single-use convenience function  
- `create_processor()` - Factory function for processor creation

### 3. 📏 Global Threshold Calculator (`global_threshold_calculator`)
**Calculate consistent classification thresholds across multiple files for uniform distributions**

```python
from represent import calculate_global_thresholds, GlobalThresholdCalculator

# Calculate thresholds from sample of DBN files with NEW focused config
from represent.configs import GlobalThresholdConfig

threshold_config = GlobalThresholdConfig(
    currency="AUDUSD",
    nbins=13,
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500,
    sample_fraction=0.5
)
thresholds = calculate_global_thresholds(
    config=threshold_config,
    data_directory="data/databento/AUDUSD/",
    verbose=True
)

print(f"Generated {thresholds.nbins} classification bins")
print(f"Based on {thresholds.sample_size:,} price movements")

# Use calculated thresholds for consistent classification
dataset_config = DatasetBuildConfig(
    global_thresholds=thresholds,  # Apply same thresholds to all processing
    force_uniform=True
)

# Advanced usage with custom calculator
calculator = GlobalThresholdCalculator(config=threshold_config)
thresholds = calculator.calculate_thresholds_from_directory(
    data_directory="data/databento/AUDUSD/",
    sample_fraction=0.3
)
```

**Key Functions:**
- `calculate_global_thresholds()` - Main threshold calculation function
- `GlobalThresholdCalculator` - Advanced threshold calculation class
- `GlobalThresholds` - Result object containing threshold data

## 🚀 Complete Workflow Example

```python
from represent import (
    DatasetBuildConfig,
    build_datasets_from_dbn_files,
    calculate_global_thresholds,
    MarketDepthProcessor
)
from represent.configs import (
    create_compatible_configs,
    GlobalThresholdConfig,
    DatasetBuilderConfig,
    MarketDepthProcessorConfig
)

# Step 1: Configure processing with NEW focused configs
dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
    currency="AUDUSD",
    features=['volume', 'variance'],
    lookback_rows=5000,
    lookforward_input=5000, 
    lookforward_offset=500
)

# Step 2: Calculate global thresholds for consistent classification
thresholds = calculate_global_thresholds(
    config=threshold_cfg,
    data_directory="data/databento/AUDUSD/",
    sample_fraction=0.5
)

# Step 3: Build comprehensive symbol datasets
dataset_config = DatasetBuildConfig(
    currency="AUDUSD",
    global_thresholds=thresholds,  # Use calculated thresholds
    force_uniform=True
)

results = build_datasets_from_dbn_files(
    config=dataset_cfg,
    dbn_files=[
        "data/AUDUSD-20240101.dbn.zst",
        "data/AUDUSD-20240102.dbn.zst",
        "data/AUDUSD-20240103.dbn.zst"
    ],
    output_dir="symbol_datasets/",
    dataset_config=dataset_config
)

# Step 4: Process datasets for ML training (in your ML repository)
processor = MarketDepthProcessor(config=processor_cfg)

# Load a comprehensive symbol dataset
import polars as pl
symbol_data = pl.read_parquet("symbol_datasets/AUDUSD_M6AM4_dataset.parquet")

# Convert to tensor for ML training
tensor_data = processor.process(symbol_data)
# Shape: (2, 402, 500) - 2 features, 402 price levels, 500 time bins

print(f"✅ Created {results['phase_2_stats']['datasets_created']} symbol datasets")
print(f"✅ Ready for ML training with {tensor_data.shape} tensor shape")
```

## 🎯 Feature Types and Output Shapes

**Available Features:**
- **Volume**: Market depth from order sizes - `(402, time_bins)`
- **Variance**: Price volatility patterns - `(402, time_bins)`
- **Trade Counts**: Transaction activity levels - `(402, time_bins)`

**Multi-Feature Output Shapes:**
- **1 feature**: `(402, 500)` - 2D tensor
- **2+ features**: `(N, 402, 500)` - 3D tensor with feature dimension first

```python
# Examples of different feature configurations with NEW focused configs
from represent.configs import MarketDepthProcessorConfig

# Single feature configuration
processor_cfg = MarketDepthProcessorConfig(features=['volume'])
print(f"Output shape: {processor_cfg.output_shape}")  # (402, 500)

# Multi-feature configuration
processor_cfg = MarketDepthProcessorConfig(features=['volume', 'variance'])
print(f"Output shape: {processor_cfg.output_shape}")  # (2, 402, 500)

# Three features configuration
processor_cfg = MarketDepthProcessorConfig(features=['volume', 'variance', 'trade_counts'])
print(f"Output shape: {processor_cfg.output_shape}")  # (3, 402, 500)
```

## 🏗️ Symbol-Split-Merge Architecture

The package uses a two-phase architecture for creating comprehensive symbol datasets:

### **Phase 1: Symbol Splitting**
Each DBN file is split by symbol into intermediate parquet files
- **Input**: Multiple DBN files (e.g., AUDUSD-20240101.dbn.zst, AUDUSD-20240102.dbn.zst)
- **Output**: Intermediate symbol files (e.g., file1_M6AM4.parquet, file2_M6AM4.parquet)
- **Performance**: 300+ samples/second per DBN file

### **Phase 2: Symbol Merging**  
All instances of each symbol are merged into comprehensive datasets
- **Input**: All symbol files across multiple DBN files
- **Output**: Comprehensive symbol datasets (e.g., AUDUSD_M6AM4_dataset.parquet)
- **Performance**: 1500+ samples/second during merging
- **Features**: Pre-computed classification labels with uniform distribution

### **Phase 3: ML Training** (External Implementation)
Comprehensive symbol datasets ready for custom dataloader implementation

```python
# Implement in your ML training repository:
from your_ml_package import create_custom_dataloader
import torch

# Load comprehensive symbol dataset
dataloader = create_custom_dataloader(
    parquet_path="symbol_datasets/AUDUSD_M6AM4_dataset.parquet",
    batch_size=32,
    shuffle=True
)

# Standard PyTorch training loop
for features, labels in dataloader:
    # features: torch.Tensor shape [32, 2, 402, 500] for volume+variance
    # labels: torch.Tensor shape [32] with uniform distribution
    outputs = model(features)
    loss = criterion(outputs, labels)
    # ... training logic
```

## ⚙️ NEW Configuration System

### **🆕 Focused Configuration Models (v5.0.0+)**
Replace the old monolithic `RepresentConfig` with separate Pydantic models for each module:

```python
from represent.configs import (
    DatasetBuilderConfig, GlobalThresholdConfig, MarketDepthProcessorConfig,
    create_compatible_configs
)

# Create focused configurations
dataset_cfg = DatasetBuilderConfig(
    currency="AUDUSD",
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500
)

threshold_cfg = GlobalThresholdConfig(
    currency="AUDUSD",
    nbins=13,
    lookback_rows=5000,
    lookforward_input=5000,
    lookforward_offset=500
)

processor_cfg = MarketDepthProcessorConfig(
    features=['volume', 'variance'],
    samples=50000
)

# Or use convenience function for compatible configs (RECOMMENDED)
dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
    currency="AUDUSD",    # Auto-configures currency-specific optimizations
    features=['volume'],  # Shared across compatible configs
    samples=25000
)

# Access configuration parameters (with Pydantic validation)
print(f"Dataset currency: {dataset_cfg.currency}")
print(f"Min required samples: {dataset_cfg.min_required_samples}")  # Computed field
print(f"Processor time bins: {processor_cfg.time_bins}")           # Computed field 
print(f"Processor output shape: {processor_cfg.output_shape}")     # Computed field
print(f"Threshold nbins: {threshold_cfg.nbins}")                   # Currency-specific
```

### **🆕 Key Benefits of New Configuration Architecture**

- **✅ Focused Validation**: Each module validates only relevant parameters
- **✅ Type Safety**: Full Pydantic validation with descriptive error messages
- **✅ Auto-Computed Fields**: Properties like `min_required_samples`, `time_bins`, `output_shape`
- **✅ Clear Separation**: No confusion between module-specific parameters
- **✅ Better IDE Support**: Full autocomplete and type hints
- **✅ Currency Optimizations**: Automatic adjustments for different currency pairs
- **✅ Backwards Compatibility**: Legacy `create_represent_config()` still works

### **📝 Migration Guide: Old → New Configuration**

```python
# ❌ OLD APPROACH (still works but deprecated)
from represent import create_represent_config

config = create_represent_config(
    currency="AUDUSD",
    features=['volume', 'variance'],
    lookback_rows=5000,
    nbins=13
)
# Returns tuple of three configs - confusing!

# ✅ NEW APPROACH (recommended)
from represent.configs import create_compatible_configs

dataset_cfg, threshold_cfg, processor_cfg = create_compatible_configs(
    currency="AUDUSD",
    features=['volume', 'variance'],
    lookback_rows=5000,
    nbins=13
)
# Clear separation of concerns, focused validation!

# ✅ OR individual focused configs for specific modules
from represent.configs import MarketDepthProcessorConfig

processor_cfg = MarketDepthProcessorConfig(
    features=['volume', 'variance'],
    samples=50000,
    ticks_per_bin=100
)
print(f"Auto-computed time bins: {processor_cfg.time_bins}")        # 500
print(f"Auto-computed output shape: {processor_cfg.output_shape}")  # (2, 402, 500)
```

### **DatasetBuildConfig**
Configuration for dataset building process:

```python
from represent import DatasetBuildConfig

dataset_config = DatasetBuildConfig(
    currency="AUDUSD",
    min_symbol_samples=10000,     # Minimum samples per symbol
    force_uniform=True,           # Ensure balanced class distribution
    nbins=13,                     # Number of classification bins
    keep_intermediate=False       # Clean up intermediate files
)
```

## 📁 Data Formats

**Input Requirements:**
- **DBN files**: `.dbn` or `.dbn.zst` (compressed recommended)
- **Required columns**: `ask_px_00-09`, `bid_px_00-09`, `ask_sz_00-09`, `bid_sz_00-09`
- **Optional columns**: `ask_ct_00-09`, `bid_ct_00-09` (for trade count features)

**Output Format:**
- **Comprehensive symbol datasets**: One parquet file per symbol containing merged data
- **Pre-classified**: Uniform distribution labels ready for training
- **Tensor-ready**: Direct loading into ML frameworks with consistent shapes

## ⚡ Performance

- **DBN Processing**: 300+ samples/second during symbol splitting
- **Symbol Merging**: 1500+ samples/second during dataset creation  
- **ML Training**: 1000+ samples/second from comprehensive datasets
- **Memory Usage**: <8GB RAM for processing multiple large DBN files
- **Scalability**: Linear scaling with CPU cores

## 🎓 Academic TStrends Integration

The represent package includes optimized implementations of academic trend labeling approaches from the **TStrends research library**. These provide research-backed labeling methods for comparison with traditional approaches.

### 📚 Available TStrends Generators

#### Binary Trend Labeling
```python
from represent.target_generators.tstrends_labeling import BinaryCTLGenerator, OracleBinaryTrendGenerator

# Binary Cumulative Trend Labelling (CTL)
binary_ctl = BinaryCTLGenerator(
    omega=0.0008,  # Ultra-aggressive for responsive trend detection
    target_name="binary_ctl_responsive"
)

# Oracle Binary (optimal binary labels)
oracle_binary = OracleBinaryTrendGenerator(
    transaction_cost=0.0003,  # Optimized transaction cost
    target_name="oracle_binary_optimal"
)
```

#### Ternary Trend Labeling (3-Class)
```python
from represent.target_generators.tstrends_labeling import TernaryCTLGenerator, OracleTernaryTrendGenerator

# Ternary Cumulative Trend Labelling (3-class: Down/Neutral/Up)
ternary_ctl = TernaryCTLGenerator(
    marginal_change_thres=0.0008,  # Ultra-aggressive threshold
    window_size=3,  # Very small window for responsiveness
    target_name="ternary_ctl_responsive"
)

# Oracle Ternary (optimal 3-class labels)
oracle_ternary = OracleTernaryTrendGenerator(
    transaction_cost=0.0001,  # Very low cost for high responsiveness
    neutral_reward_factor=0.3,  # Favor directional signals over neutral
    target_name="oracle_ternary_optimal"
)
```

### 🔧 Parameter Optimization

**⚡ Ultra-Aggressive Parameters for Maximum Responsiveness:**

The TStrends generators have been optimized through systematic parameter search to provide:

- **Responsive Trend Detection**: Ultra-low thresholds (0.0005-0.0012) for quick regime changes
- **Small Windows**: Minimal windows (2-5) for fast adaptation to market movements  
- **Low Transaction Costs**: Optimized costs (0.0001-0.0003) for active trading strategies
- **Directional Bias**: Low neutral factors (0.3) to favor actionable Up/Down signals

### 📊 Label Remapping

**Automatic conversion from TStrends format to standard format:**

```python
# TStrends Original Format → Represent Standard Format
# Binary: {-1, 1} → {0: Down/Sell, 1: Up/Buy}
# Ternary: {-1, 0, 1} → {0: Down/Sell, 1: Neutral/Hold, 2: Up/Buy}

# Usage with modular target generation
from represent import ModularDatasetBuilder, TargetGeneratorFactory

generators = [
    TargetGeneratorFactory.create("quantile_classification", nbins=13),  # Traditional
    TargetGeneratorFactory.create("binary_ctl", omega=0.0008),  # Academic binary
    TargetGeneratorFactory.create("ternary_ctl", marginal_change_thres=0.0008, window_size=3)  # Academic ternary
]

builder = ModularDatasetBuilder(generators)
dataset = builder.build_from_parquet("symbol_data.parquet")
```

### 🎨 Visualization Features

**Enhanced visualization with neutral class hiding:**

- **Binary plots**: Clean Down/Sell (blue) and Up/Buy (red) visualization
- **Ternary plots**: Neutral signals automatically hidden for cleaner charts focusing on actionable signals
- **Academic vs Traditional**: Side-by-side comparison plots showing different approaches on same data

```python
# Generate comprehensive labeling visualization
from represent.examples import labeling_approaches_visualization

# Creates 4 plots:
# 1. classification_approaches_comparison.png - All classification methods
# 2. regression_approaches_comparison.png - All regression methods  
# 3. academic_vs_traditional_comparison.png - TStrends vs traditional side-by-side
# 4. complete_labeling_overview.png - Complete overview of all methods
```

### ✅ Installation Requirements

```bash
# Install TStrends library (optional - required only for academic approaches)
uv add "git+https://github.com/agpenas/tstrends.git"

# TStrends generators will automatically detect availability
# If not installed, generators will raise helpful import errors with installation instructions
```

### 🔬 Research Integration

**Benefits of TStrends Integration:**

- **Research-Backed**: Based on peer-reviewed academic approaches
- **Optimal Benchmarks**: Oracle labeling provides theoretical performance limits
- **Parameter Optimization**: Systematic search for market-specific parameter tuning
- **Academic Comparison**: Compare traditional quantile methods with academic approaches
- **Visualization Ready**: Pre-optimized for clean, publication-ready plots

**Use Cases:**
- **Academic Research**: Benchmark traditional approaches against academic methods
- **Strategy Development**: Use Oracle labels as performance upper bounds
- **Market Analysis**: Compare trend detection across different methodologies
- **Parameter Studies**: Test sensitivity across various market regimes

## 🎯 Comprehensive Labeling Approaches

The represent package provides a complete suite of both **classification** and **regression** target generators through its modular architecture. Each approach is designed for specific ML applications and market analysis scenarios.

### 📊 Classification Approaches

#### 1. **Quantile Classification** (`quantile_classification`)
**Traditional percentile-based discrete labeling for balanced class distributions**

```python
from represent import TargetGeneratorFactory

# Multi-class quantile classification
generator = TargetGeneratorFactory.create(
    "quantile_classification",
    nbins=13,  # Number of classes (e.g., 13 classes for detailed classification)
    target_name="price_direction_13class"
)

# Binary classification
binary_generator = TargetGeneratorFactory.create(
    "quantile_classification", 
    nbins=2,
    target_name="price_direction_binary"
)
```

**Use Cases:**
- **Balanced ML Training**: Guaranteed uniform class distribution
- **Multi-class Prediction**: 5, 13, or 21 classes for granular price direction
- **Baseline Models**: Standard approach for comparison benchmarks

**Output**: Discrete labels `{0, 1, 2, ..., nbins-1}` with uniform distribution

---

#### 2. **Global Threshold Classification** (`global_threshold_classification`)
**Consistent classification boundaries computed across multiple datasets**

```python
# Use pre-calculated global thresholds for consistent labeling
generator = TargetGeneratorFactory.create(
    "global_threshold_classification",
    global_thresholds=thresholds,  # From calculate_global_thresholds()
    target_name="consistent_labels"
)
```

**Use Cases:**
- **Cross-Dataset Consistency**: Same classification boundaries across all datasets
- **Production Models**: Consistent labeling in live trading systems
- **Backtesting**: Historical consistency for strategy validation

**Output**: Discrete labels with globally consistent boundaries

---

### 📈 Regression Approaches

#### 1. **Directional MFE** (`directional_mfe`)
**Maximum Favorable Excursion for both long and short positions**

```python
# Directional MFE for risk-aware position sizing
mfe_generator = TargetGeneratorFactory.create(
    "directional_mfe",
    lookforward_horizon=3000,  # Future window (ticks)
    lookback_window=200,       # Smoothing window (ticks)
    expected_fee_pips=0.7,     # Trading costs
    target_names=("mfe_buy_bps", "mfe_sell_bps")
)
```

**Key Features:**
- **Buy-side MFE**: Maximum profit potential for long positions
- **Sell-side MFE**: Maximum profit potential for short positions  
- **Fee-Adjusted**: Accounts for realistic trading costs
- **Risk Management**: Optimizes position sizing based on profit potential

**Use Cases:**
- **Position Sizing**: Determine optimal trade sizes based on profit potential
- **Risk-Adjusted Trading**: Account for maximum drawdown vs maximum profit
- **Strategy Optimization**: Optimize entry/exit timing for maximum favorable excursion

**Output**: Two continuous targets in basis points (buy MFE, sell MFE)

---

#### 2. **Price Movement** (`price_movement`)
**Simple percentage price change over lookforward window**

```python
# Basic price movement prediction
movement_generator = TargetGeneratorFactory.create(
    "price_movement",
    lookforward_window=5000,  # Future window
    lookback_window=5000,     # Baseline window
    target_name="price_change_bps"
)
```

**Use Cases:**
- **Baseline Regression**: Simple price prediction benchmark
- **Linear Models**: Direct input for linear regression models
- **Feature Engineering**: Component for more complex targets

**Output**: Continuous values in basis points representing price movement

---

#### 3. **Rolling Volatility** (`volatility`)
**Rolling volatility estimation over configurable windows**

```python
# Volatility prediction for risk management
vol_generator = TargetGeneratorFactory.create(
    "volatility",
    window_size=1000,  # Rolling window size
    target_name="rolling_volatility_bps"
)
```

**Use Cases:**
- **Risk Management**: Predict future volatility for position sizing
- **Options Trading**: Volatility forecasting for options strategies
- **Market Regime Detection**: Identify high/low volatility periods

**Output**: Continuous volatility values in basis points

---

#### 4. **Cumulative Returns** (`cumulative_returns`)
**Accumulation of returns over specified lookforward period**

```python
# Cumulative return prediction
cumret_generator = TargetGeneratorFactory.create(
    "cumulative_returns",
    lookforward_samples=3000,  # Number of future samples to accumulate
    target_name="cumret_3000_samples"
)
```

**Use Cases:**
- **Return Forecasting**: Predict total return over holding period
- **Buy-and-Hold Strategies**: Optimal holding period determination
- **Performance Attribution**: Understand return accumulation patterns

**Output**: Continuous values representing cumulative log returns in basis points

---

#### 5. **Volatility-Scaled Returns** (`volatility_scaled_returns`)
**Adaptive returns with dynamic stop-loss/take-profit barriers**

```python
# Advanced volatility-adaptive risk management
vol_scaled_generator = TargetGeneratorFactory.create(
    "volatility_scaled_returns",
    volatility_window=500,      # Window for volatility estimation
    vol_multiplier=2.5,         # Barrier multiplier (2.5x volatility)
    horizon_ticks=1500,         # Evaluation horizon
    min_barrier_bps=3.0,        # Minimum barrier size to avoid noise
    target_name="vol_scaled_adaptive"
)
```

**Key Features:**
- **Adaptive Barriers**: Stop-loss/take-profit levels adjust to market volatility
- **Regime-Aware**: Tight barriers in low-vol, wide barriers in high-vol periods
- **Realistic PnL**: Returns actual breach prices, not theoretical barriers
- **Noise Filtering**: Minimum barrier size prevents meaningless small movements

**Use Cases:**
- **Adaptive Trading**: Risk management that adjusts to market conditions
- **FX Trading**: Common approach in currency trading for dynamic risk control
- **Volatility Strategies**: Trading strategies that adapt to volatility regimes

**Output**: Continuous PnL values in basis points with volatility-adjusted risk management

---

#### 6. **Remaining Value Tuner** (`remaining_value_tuner`) ⭐ *NEW*
**Advanced trend potential labeling inspired by TStrends research**

```python
# Trend potential prediction - NEW advanced approach
remaining_value_generator = TargetGeneratorFactory.create(
    "remaining_value_tuner",
    lookback_rows=5000,           # Historical context window
    lookforward_input=3000,       # Trend evaluation window  
    lookforward_offset=500,       # Offset before evaluation
    trend_threshold_bps=20.0,     # Minimum trend magnitude
    neutral_factor=0.5,           # Neutral zone sizing
    enforce_monotonicity=True,    # Smooth trend transitions
    target_name="remaining_trend_potential"
)
```

**Revolutionary Features:**
- **Trend Potential**: Instead of discrete labels (-1, 0, 1), provides continuous values representing remaining movement potential
- **Future-Aware**: Calculates how much upside/downside remains from current point to future peak/trough
- **Smart Classification**: Automatically distinguishes uptrends, downtrends, and neutral periods
- **Monotonicity Enforcement**: Smooths trend values to prevent unrealistic reversals

**Key Outputs:**
- **Uptrends**: Positive values indicating remaining upside potential (e.g., +185 bps remaining)
- **Downtrends**: Negative values indicating remaining downside potential (e.g., -127 bps remaining)  
- **Neutral Trends**: Small values near zero for sideways markets

**Advanced Use Cases:**
- **Optimal Entry Timing**: Enter trends when remaining potential is highest
- **Position Sizing**: Size positions based on remaining trend magnitude
- **Exit Strategy**: Exit when remaining potential diminishes
- **ML Training**: More informative targets than binary/ternary classification
- **Research**: Academic-quality labeling for advanced strategy development

**Output**: Continuous values in basis points representing remaining trend potential

---

### 🎨 Modular Target Generation

**Combine Multiple Approaches:**

```python
from represent import ModularDatasetBuilder, TargetGeneratorFactory

# Create diverse target generators for comprehensive ML training
generators = [
    # Classification approaches
    TargetGeneratorFactory.create("quantile_classification", nbins=13),
    TargetGeneratorFactory.create("global_threshold_classification", 
                                 global_thresholds=thresholds),
    
    # Regression approaches
    TargetGeneratorFactory.create("directional_mfe", lookforward_horizon=3000),
    TargetGeneratorFactory.create("volatility_scaled_returns", vol_multiplier=2.5),
    TargetGeneratorFactory.create("remaining_value_tuner", trend_threshold_bps=20.0),
    
    # Academic approaches (requires tstrends)
    TargetGeneratorFactory.create("binary_ctl", omega=0.0008),
    TargetGeneratorFactory.create("oracle_ternary", transaction_cost=0.0001),
]

# Build comprehensive dataset with all target types
builder = ModularDatasetBuilder(generators, verbose=True)
dataset = builder.build_from_parquet("symbol_data.parquet")

# Result: Dataset with 8+ different target columns for diverse ML training
print(f"Generated dataset with {len(dataset.columns)} columns")
print(f"Target columns: {[col for col in dataset.columns if col not in ['mid_price', 'ts_event', 'symbol']]}")
```

### 📊 Complete Labeling Visualization

**Generate comprehensive visualization of all approaches:**

```python
# Run the complete labeling demonstration
python examples/labeling_approaches_visualization.py

# Generates 4 detailed comparison plots:
# 1. classification_approaches_comparison.png - All classification methods
# 2. regression_approaches_comparison.png - All regression methods
# 3. academic_vs_traditional_comparison.png - TStrends vs traditional
# 4. complete_labeling_overview.png - Complete overview of all 15+ approaches
```

**Example Output:**
- **18 different target types** across classification and regression
- **Side-by-side comparisons** of traditional vs academic approaches  
- **Parameter information** clearly labeled on each plot
- **Statistical summaries** (mean, std, range) for each target type

### 🚀 Choosing the Right Approach

**For Different ML Applications:**

| **ML Goal** | **Recommended Approach** | **Why** |
|-------------|-------------------------|---------|
| **Multi-class Direction** | `quantile_classification` (nbins=13) | Balanced classes, interpretable |
| **Binary Direction** | `binary_ctl` or `quantile_classification` (nbins=2) | Clean directional signals |
| **Position Sizing** | `directional_mfe` or `remaining_value_tuner` | Risk-aware, magnitude-informed |
| **Risk Management** | `volatility_scaled_returns` | Adaptive to market conditions |
| **Return Forecasting** | `cumulative_returns` | Direct return prediction |
| **Volatility Prediction** | `volatility` | Specialized for volatility forecasting |
| **Research/Benchmarking** | `oracle_ternary` + `quantile_classification` | Theoretical optimum vs practical |
| **Advanced Trading** | `remaining_value_tuner` | Future trend potential |

### ⚙️ Advanced Configuration

**All target generators support extensive customization:**

```python
# Example: Highly customized remaining value tuner for intraday trading
intraday_generator = TargetGeneratorFactory.create(
    "remaining_value_tuner",
    lookback_rows=1000,           # Shorter context for faster markets
    lookforward_input=500,        # Quick trend identification  
    lookforward_offset=50,        # Minimal delay
    trend_threshold_bps=5.0,      # Sensitive to small movements
    neutral_factor=0.3,           # Tight neutral zone
    enforce_monotonicity=False,   # Allow rapid trend changes
    target_name="intraday_trend_potential"
)

# Example: Conservative volatility-scaled for position trading
conservative_vol_scaled = TargetGeneratorFactory.create(
    "volatility_scaled_returns",
    volatility_window=2000,       # Longer volatility estimation
    vol_multiplier=3.0,           # Wide barriers
    horizon_ticks=5000,           # Long evaluation period
    min_barrier_bps=10.0,         # Higher noise threshold
    target_name="position_trading_pnl"
)
```

This comprehensive suite provides everything needed for modern ML applications in quantitative finance, from traditional classification to cutting-edge trend potential prediction.

## 🧪 Development

```bash
# Install dependencies
uv sync --all-extras

# Run tests
make test                 # Full test suite with coverage
make test-fast           # Quick tests (excludes performance tests)

# Code quality
make lint                # Linting and type checking
make format             # Code formatting

# Build package
make build              # Build distribution packages
```

### Testing
- **104 tests passing** with comprehensive coverage
- **74% code coverage** focused on critical paths
- **Performance tests** for latency requirements
- **Integration tests** for complete workflows

## 📊 Examples

Check out the `examples/` directory for complete demonstrations:

```bash
# Symbol-split-merge demonstration
python examples/symbol_split_merge_demo.py

# Quick start examples  
python examples/quick_start_examples.py

# Feature extraction demo
python examples/demonstrate_feature_extraction.py
```

## 📈 Architecture Benefits

**Why Symbol-Split-Merge?**
- **Comprehensive Datasets**: Each symbol contains complete history from multiple files
- **Memory Efficient**: Stream large DBN files without loading into RAM
- **Uniform Distribution**: Balanced class labels for optimal ML training  
- **Production Ready**: Handle 10+ DBN files efficiently with automatic validation

**Clean Three-Module Design with Focused Configs (v5.0.0+):**
- **dataset_builder**: High-level dataset creation (`DatasetBuilderConfig`)
- **market_depth_processor**: Low-level tensor processing (`MarketDepthProcessorConfig`)
- **global_threshold_calculator**: Consistent classification (`GlobalThresholdConfig`)
- **🆕 Focused Architecture**: Each module has its own type-safe Pydantic configuration model
- **🆕 Auto-Computed Fields**: Properties automatically calculated from base parameters
- **🆕 Better Validation**: Module-specific validation with descriptive error messages

## 📄 License

MIT License - see LICENSE file for details.

---

**🏗️ Production-ready symbol-split-merge architecture for comprehensive ML datasets with memory-efficient processing and guaranteed uniform class distribution**