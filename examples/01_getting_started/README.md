# 🚀 Getting Started Examples

Basic examples for new users to get familiar with the represent package. Start here if you're new to represent!

## 📋 Files in this directory

### **api_usage_examples.py**
Demonstrates the high-level API for common workflows:
- Simple DBN to classified parquet conversion
- Basic configuration options
- Quick processing pipeline setup

### **configuration_examples.py**
Shows different ways to configure the system:
- Currency-specific configurations
- Feature selection (volume, variance, trade_counts)
- Custom processing parameters

### **simple_currency_examples.py**
Basic currency-specific processing examples:
- AUDUSD processing setup
- GBPUSD with volatility optimizations
- JPY pair handling with appropriate pip sizes

## 🎯 Recommended Learning Path

1. **Start here**: `api_usage_examples.py`
   - Learn the basic workflow: DBN → Classified Parquet → ML Training
   - See the streamlined processing approach

2. **Next**: `configuration_examples.py`
   - Understand how to customize processing parameters
   - Learn about different feature types

3. **Then**: `simple_currency_examples.py`
   - See currency-specific optimizations
   - Understand market-specific configurations

## 🏃‍♂️ Quick Start

```bash
# Run the main API example
python 01_getting_started/api_usage_examples.py

# Try different configurations
python 01_getting_started/configuration_examples.py

# Explore currency-specific examples
python 01_getting_started/simple_currency_examples.py
```

## 📊 Expected Output

These examples will generate:
- Classified parquet files in `outputs/` subdirectories
- Processing statistics and timing information
- Sample data ready for ML training

## ➡️ Next Steps

After completing these examples, move to:
- `02_global_thresholds/` - Learn the **RECOMMENDED** global threshold approach
- `04_ml_training/` - See how to use classified data in PyTorch models
- `03_data_processing/` - Explore advanced data processing techniques