# Classification Analysis Examples

This directory contains examples and analysis of the represent package's classification system optimization.

## 🎯 Overview

The represent package uses quantile-based classification to achieve uniform distribution across 13 classes for AUDUSD market data. This analysis demonstrates the optimization process and provides production-ready configurations.

## 📁 Files

### Core Demos
- **`complete_pipeline_demo.py`** - Complete 3-stage pipeline demonstration (DBN→Parquet→Classification→ML)
- **`final_demo.py`** - Final optimized pipeline with quantile-based classification
- **`realistic_market_demo.py`** - Realistic market data processing example
- **`simple_classification_demo.py`** - Simple classification example for beginners

### Optimization & Validation
- **`final_quantile_validation.py`** - **[KEY FILE]** Final quantile-based optimization and validation
- **`final_validation/`** - **[KEY DIRECTORY]** Contains optimized results:
  - `optimized_audusd_config.json` - Production-ready AUDUSD configuration
  - `final_validation_results.json` - Complete validation results
  - `final_quantile_validation.png` - Analysis visualization

### Documentation
- **`ARCHITECTURE_UPDATE_SUMMARY.md`** - Technical summary of v3.0.0 architecture changes
- **`README_FINAL.md`** - Final implementation documentation

## 🚀 Quick Start

### 1. Run Final Validation
```bash
# Run the comprehensive quantile-based validation
python examples/classification_analysis/final_quantile_validation.py
```

### 2. Use Optimized Configuration
The optimized configuration is ready for production use:

```python
from represent import load_config_from_file

# Load optimized AUDUSD configuration
config = load_config_from_file("examples/classification_analysis/final_validation/optimized_audusd_config.json")
```

### 3. Complete Pipeline Demo
```bash
# See the full 3-stage pipeline in action
python examples/classification_analysis/complete_pipeline_demo.py
```

## 📊 Key Results

### Optimization Success
- **Before Optimization**: Only 2-3 classes had data, with 61% concentrated in one class
- **After Optimization**: All 13 classes have data with uniform distribution
- **Average Deviation**: Reduced from 13-45% to **1.6%**
- **Max Deviation**: Reduced from 53% to **3.6%**

### Final Configuration
- **Method**: Quantile-based thresholds for exact uniform distribution
- **Validation**: 1,050+ samples across multiple parquet files
- **Quality**: Significant improvement, ready for production use

## 🔧 Technical Details

### Quantile-Based Approach
The final optimization uses direct quantile calculation to achieve uniform distribution:

1. **Training Data**: Calculate exact quantile positions for 13 classes
2. **Threshold Generation**: Use quantile values as classification boundaries
3. **Validation**: Test on independent dataset to ensure robustness

### Configuration Structure
```json
{
  "classification": {
    "nbins": 13,
    "method": "quantile_optimized",
    "thresholds": [/* 12 precisely calculated thresholds */],
    "validation_quality": "SIGNIFICANT_IMPROVEMENT",
    "max_deviation": 3.6,
    "avg_deviation": 1.6
  }
}
```

## 📈 Performance Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Classes with Data | 2-3 | 13 | 100% coverage |
| Average Deviation | 13-45% | 1.6% | 87-96% reduction |
| Max Deviation | 53% | 3.6% | 93% reduction |
| Distribution Quality | Poor | Good | Production ready |

## 🎉 Next Steps

1. **Production Deployment**: Use `optimized_audusd_config.json` in production
2. **Other Currencies**: Apply similar quantile-based optimization to GBPUSD, EURJPY, etc.
3. **Monitoring**: Track classification quality in production using the validation metrics

The quantile-based optimization has successfully resolved the original classification distribution issues and provides a robust foundation for ML training pipelines.