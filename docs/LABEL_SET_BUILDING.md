# 🎯 Label Set Building System

A comprehensive, flexible system for building custom label sets optimized for specific ML training and research objectives.

## 🚀 Quick Start

### List Available Presets
```bash
make list-presets
```

### Build Predefined Label Sets
```bash
# Trading-focused labels
make build-trading-labels

# Academic research labels  
make build-research-labels

# MFE analysis labels
make build-mfe-labels

# Trend analysis labels
make build-trend-labels

# Volatility analysis labels
make build-vol-labels
```

### Interactive Builder
```bash
make build-labels  # Shows options and guidance
```

## 📊 Available Preset Configurations

### 1. **MFE Analysis** (`mfe_analysis`)
- **Purpose**: Maximum Favorable Excursion analysis for both long and short positions
- **Use Cases**: Entry/exit timing, position management, directional strategy development
- **Targets**: 
  - Multi-horizon MFE buy/sell signals (1000 and 2000 tick horizons)
  - Optimized winsorization (0.1%, 99.9%) for realistic extremes

### 2. **Trend Analysis** (`trend_analysis`) 
- **Purpose**: Comprehensive trend detection and remaining value analysis
- **Use Cases**: Trend continuation strategies, momentum trading, regime detection
- **Targets**:
  - Remaining value tuning (2k and 4k horizons)
  - Quantile classification for trend regimes

### 3. **Volatility Analysis** (`volatility_analysis`)
- **Purpose**: Risk management and adaptive trading strategies
- **Use Cases**: Position sizing, volatility breakout strategies, risk modeling
- **Targets**:
  - Volatility-scaled returns (2x and 3x multipliers)
  - Rolling volatility measures

### 4. **Returns Analysis** (`returns_analysis`)
- **Purpose**: Momentum and mean reversion strategy development
- **Use Cases**: Portfolio optimization, momentum strategies, return prediction
- **Targets**:
  - Multi-horizon cumulative returns (500, 1500, 3000 samples)
  - Price movement analysis

### 5. **Comprehensive** (`comprehensive`)
- **Purpose**: Complete suite of all available target generators
- **Use Cases**: Research, comparative analysis, full-spectrum modeling
- **Targets**: All 7 target types with optimized parameters

## 🛠️ Custom Configuration System

### YAML Configuration Format
```yaml
name: "My Custom Label Set"
description: "Description of this configuration"

generators:
  - type: "directional_mfe"
    lookforward_horizon: 1500
    expected_fee_pips: 0.7
    target_names: ["mfe_buy", "mfe_sell"]
  
  - type: "quantile_classification"
    nbins: 13
    lookforward_window: 2500
    target_name: "regime_labels"

visualization: true

output:
  base_name: "my_labels"
  include_timestamp: true
```

### Command Line Usage
```bash
# Custom configuration
python scripts/build_label_set.py --config my_config.yaml

# Preset configuration
python scripts/build_label_set.py --preset mfe_analysis

# Symbol filtering
python scripts/build_label_set.py --preset trend_analysis --symbol EURUSD

# Custom data path
python scripts/build_label_set.py --preset comprehensive --data /path/to/data.parquet
```

## 📁 File Organization

```
represent/
├── configs/label_sets/          # Configuration templates
│   ├── trading_strategy.yaml    # Trading-focused config
│   └── research_academic.yaml   # Research-focused config
├── scripts/
│   ├── build_label_set.py      # Main builder script
│   └── create_sample_data.py   # Sample data generation
└── output/label_sets/          # Generated label sets
    ├── *.parquet              # Dataset files
    └── *_visualization.png    # Visualization plots
```

## 🎯 Available Target Generators

### Classification Generators
- **`quantile_classification`**: Quantile-based classification with configurable bins
- Parameters: `nbins`, `lookforward_window`, `lookback_window`

### Regression Generators  
- **`directional_mfe`**: Maximum Favorable Excursion for buy/sell positions
- **`remaining_value_tuner`**: Trend potential analysis from TStrends research
- **`volatility_scaled_returns`**: Adaptive returns with volatility-based barriers
- **`cumulative_returns`**: Accumulated returns over lookforward windows
- **`price_movement`**: Raw price change analysis
- **`volatility`**: Rolling volatility measures

## 🔧 Advanced Features

### Symbol Filtering
```bash
python scripts/build_label_set.py --preset mfe_analysis --symbol EURUSD
```

### Custom Output Paths
```bash
python scripts/build_label_set.py --preset trend_analysis --output /custom/path/
```

### Template Generation
```bash
python scripts/build_label_set.py --template my_template.yaml
```

## 📊 Output Format

### Parquet Dataset Structure
```
Columns:
- ts_event: Timestamps (microsecond precision)  
- mid_price: Market price series
- symbol: Symbol identifier (if present)
- [target_1]: First target column
- [target_2]: Second target column
- ...
```

### Automatic Visualization
Each label set includes:
- **Time Series Plots**: Target evolution over time
- **Distribution Histograms**: Value frequency distributions
- **Statistical Summary**: Mean, std deviation, valid sample counts

## 🎉 Integration with ML Workflows

### Typical Workflow
1. **Choose Configuration**: Select preset or create custom YAML
2. **Build Label Set**: Run make target or script directly
3. **Validate Output**: Review visualization and statistics
4. **ML Training**: Use generated parquet files in your ML pipeline

### Example ML Integration
```python
import polars as pl
from sklearn.model_selection import train_test_split

# Load label set
df = pl.read_parquet('output/label_sets/mfe_analysis_labels.parquet')

# Extract features and targets
features = df.select(['mid_price', 'ts_event']).to_numpy()
targets = df.select(['mfe_buy_1k', 'mfe_sell_1k']).to_numpy()

# Standard ML workflow
X_train, X_test, y_train, y_test = train_test_split(features, targets)
```

## 🌟 Best Practices

### For Trading Strategy Development
- Start with `build-trading-labels` for systematic strategies
- Use `build-mfe-labels` for entry/exit signal development
- Apply `build-vol-labels` for risk management optimization

### For Academic Research
- Use `build-research-labels` for comprehensive analysis
- Apply `build-comprehensive` for comparative studies
- Create custom configs for specific research questions

### For Production Systems
- Use symbol filtering for individual currency pairs
- Set `include_timestamp: false` for consistent filenames
- Validate label distributions before ML training

## 🚀 Performance Optimizations

- **Vectorized Operations**: All target generators use optimized polars/numpy operations
- **Memory Efficient**: Lazy evaluation and streaming processing where possible
- **Parallel Ready**: Multiple label sets can be built concurrently
- **Scalable**: Handles datasets from thousands to millions of samples

---

The label set building system provides maximum flexibility while maintaining ease of use, enabling both quick prototyping with presets and sophisticated custom configurations for advanced research and production systems.