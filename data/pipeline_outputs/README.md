# Pipeline Outputs - 3-Stage Architecture v2.0.0

This directory contains the outputs from the new **3-Stage Architecture** implementation:

## 🎯 Architecture Overview

**DBN → Unlabeled Parquet → Classification → ML Training**

1. **Stage 1**: Convert DBN files to unlabeled symbol-grouped parquet datasets
2. **Stage 2**: Apply post-processing classification with uniform distribution 
3. **Stage 3**: Lazy ML training with memory-efficient dataloaders

## 📁 Directory Structure

```
pipeline_outputs/
├── unlabeled_parquet/           # Stage 1 outputs
│   ├── AUDUSD_M6AM4.parquet     # Symbol-specific unlabeled data (5.0GB)
│   ├── AUDUSD_M6AU4.parquet     # Symbol-specific unlabeled data (1.1GB)
│   └── AUDUSD_M6AM4_test.parquet # Test data (3.7GB)
├── classified_parquet/          # Stage 2 outputs  
│   └── AUDUSD_M6AM4_test_classified.parquet # Classified with uniform distribution (3.7GB)
├── unlabeled_parquet_small/     # Alternative smaller files
│   ├── AUDUSD_M6AM4.parquet     # (7.1GB)
│   └── AUDUSD_M6AU4.parquet     # (1.3GB)
├── classified_parquet_small/    # Corresponding classified files
│   └── AUDUSD_M6AM4_classified.parquet # (4.8GB)
└── analysis/                    # Analysis outputs
    └── (analysis plots and reports)
```

## ✅ Generated Files Summary

### Unlabeled Parquet Files (Stage 1)
- **AUDUSD_M6AM4.parquet**: 5.0GB - Major symbol with high sample count
- **AUDUSD_M6AU4.parquet**: 1.1GB - Secondary symbol data
- **AUDUSD_M6AM4_test.parquet**: 3.7GB - Test dataset for validation

### Classified Parquet Files (Stage 2)
- **AUDUSD_M6AM4_test_classified.parquet**: 3.7GB - Ready for ML training
  - ✅ Uniform distribution classification applied
  - ✅ 13-class system (0-12) with balanced representation
  - ✅ Data-driven thresholds for optimal uniformity

## 🚀 Usage Examples

### Stage 2: Apply Classification
```python
from represent import classify_parquet_file

stats = classify_parquet_file(
    parquet_path='unlabeled_parquet/AUDUSD_M6AM4.parquet',
    output_path='classified_parquet/AUDUSD_M6AM4_classified.parquet',
    currency='AUDUSD'
)
```

### Stage 3: ML Training
```python
from represent import create_market_depth_dataloader

dataloader = create_market_depth_dataloader(
    parquet_path='classified_parquet/AUDUSD_M6AM4_test_classified.parquet',
    batch_size=32,
    shuffle=True
)

for features, labels in dataloader:
    # features: (32, 402, 500) market depth tensors
    # labels: (32,) classification targets 0-12
    pass
```

## 📊 Data Characteristics

### Market Depth Features
- **Shape**: (402, 500) - 402 price levels × 500 time bins
- **Features**: Volume, variance, trade counts (configurable)
- **Currency**: AUDUSD with optimized thresholds
- **Symbols**: M6AM4, M6AU4 (major AUDUSD futures)

### Classification Labels
- **Classes**: 13 (0-12) representing price movement ranges
- **Distribution**: Uniform target (7.69% per class)
- **Thresholds**: Data-driven from real market analysis
- **Quality**: Validated for balanced ML training

## 🎉 Success Metrics

✅ **Files Generated**: 7 parquet files totaling ~25GB  
✅ **Symbol Coverage**: Multiple AUDUSD symbols processed  
✅ **Classification**: Uniform distribution applied successfully  
✅ **ML Ready**: All files tested with PyTorch dataloaders  
✅ **Performance**: Memory-efficient lazy loading verified  

## 🔧 Technical Details

### Data Source
- **Original DBN Files**: 
  - `glbx-mdp3-20240403.mbp-10.dbn.zst`
  - `glbx-mdp3-20240405.mbp-10.dbn.zst`
- **Processing Date**: August 5, 2025
- **Architecture Version**: v2.0.0

### Configuration
- **Currency**: AUDUSD
- **Features**: Volume (primary), variance, trade counts
- **Batch Size**: 1000-2000 samples per processing batch
- **Min Symbol Samples**: 100-1000 threshold for inclusion

---

**Status**: ✅ All outputs generated successfully  
**Ready for**: Production ML training workflows  
**Architecture**: v2.0.0 3-Stage Pipeline  