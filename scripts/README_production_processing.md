# Production Dataset Processing

This document explains the production dataset processing script and workflow for creating ML-ready datasets from AUDUSD-micro data.

## Overview

The `process_production_datasets.py` script implements a **first-half training approach** to prevent data leakage when creating classified datasets for ML training:

1. **Phase 1**: Uses the first half of DBN files to calculate global classification thresholds
2. **Phase 2**: Processes ALL files using those calculated thresholds to create symbol-specific datasets

## Key Benefits

### 🎯 **No Data Leakage**
- Classification boundaries are determined from first half of data only
- These boundaries are then applied to the entire dataset
- Future information never influences past classifications

### ⚖️ **Uniform Distribution**
- Quantile-based thresholds ensure balanced class distribution
- Each symbol gets consistent classification across all time periods
- Optimal for ML training with balanced classes

### 🏭 **Production Ready**
- Processes 260 DBN files spanning April 2024 to April 2025
- Creates comprehensive symbol-specific parquet datasets  
- Handles large-scale data efficiently with streaming processing

## Configuration

The script uses production-optimized parameters:

```python
config = create_represent_config(
    currency="AUDUSD",
    features=['volume', 'variance'],     # Two complementary features
    lookback_rows=5000,                  # 5K rows historical context
    lookforward_input=5000,              # 5K rows future prediction window
    lookforward_offset=500,              # 500 row offset before future window
    samples=50000,                       # 50K samples per processing batch
    nbins=21                            # 21 classification bins for high resolution
)

dataset_config = DatasetBuildConfig(
    global_thresholds=thresholds,        # Use first-half calculated thresholds
    force_uniform=True,                  # Enforce balanced distribution
    min_symbol_samples=50000,            # Minimum 50K samples per symbol dataset
    keep_intermediate=False              # Clean up intermediate files
)
```

## Usage

### Command Line
```bash
# Run the production processing
make process-production

# Or run directly
python scripts/process_production_datasets.py
```

### Expected Output
The script will create classified datasets in `/Users/danielfisher/data/databento/AUDUSD_classified_datasets/`:

```
AUDUSD_M6AM4_dataset.parquet    # Major symbol dataset (largest)
AUDUSD_M6AU4_dataset.parquet    # Another symbol dataset
AUDUSD_[SYMBOL]_dataset.parquet # Additional symbols as available
```

## Processing Flow

### Phase 1: Global Threshold Calculation
1. **Input**: First 130 DBN files (April 2024 - October 2024)
2. **Sampling**: 30% of first-half files for efficiency
3. **Output**: Global quantile boundaries for 21 classification bins

### Phase 2: Dataset Creation  
1. **Input**: ALL 260 DBN files (full dataset)
2. **Processing**: Symbol-split-merge architecture
3. **Classification**: Apply first-half thresholds to all data
4. **Output**: Symbol-specific parquet files with uniform class distribution

### Phase 3: Validation
1. **Statistics**: Processing time, sample counts, file sizes
2. **Quality**: Verify uniform distribution and no data leakage
3. **Summary**: Ready-to-use datasets for external ML training

## Output Format

Each symbol dataset contains:

- **Market Data**: All required price and volume columns
- **Classification Labels**: 21-bin uniform distribution (0-20)
- **Timestamps**: Full temporal information
- **Symbol**: Specific to one financial instrument

### Example Dataset Structure
```python
import polars as pl
df = pl.read_parquet("AUDUSD_M6AM4_dataset.parquet")

# Columns:
# - ts_event: timestamp
# - bid_px_00-09: bid prices (10 levels)
# - ask_px_00-09: ask prices (10 levels)  
# - bid_sz_00-09: bid sizes (10 levels)
# - ask_sz_00-09: ask sizes (10 levels)
# - classification_label: 0-20 (uniform distribution)
```

## Performance Expectations

Based on the current dataset:
- **Total Files**: 260 DBN files
- **Expected Processing Time**: 2-3 hours for full dataset
- **Output Size**: 5-15 GB total (varies by symbol activity)
- **Processing Rate**: 50,000+ samples/second
- **Memory Usage**: <8GB RAM during processing

## Integration with ML Training

The output datasets are designed for use in external ML training repositories:

```python
# In your ML training repository
import polars as pl
from torch.utils.data import Dataset, DataLoader

class MarketDataset(Dataset):
    def __init__(self, parquet_path):
        self.df = pl.read_parquet(parquet_path)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df[idx]
        # Extract features and label
        features = self.process_market_data(row)  # Your processing
        label = row['classification_label'].item()
        return features, label

# Load comprehensive symbol dataset
dataset = MarketDataset("AUDUSD_M6AM4_dataset.parquet") 
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Standard training loop
for features, labels in dataloader:
    # features: [batch_size, feature_dims]
    # labels: [batch_size] with uniform distribution 0-20
    pass
```

## Quality Guarantees

### ✅ **Data Integrity**
- No future information in past classifications
- Consistent thresholds across all time periods
- Complete symbol coverage with comprehensive datasets

### ✅ **Distribution Quality** 
- Uniform class distribution (each class ~4.76% for 21 bins)
- No class imbalance issues
- Optimal for balanced ML training

### ✅ **Production Scale**
- Handles multi-GB datasets efficiently
- Streaming processing prevents memory issues
- Symbol-specific datasets for targeted training

## Troubleshooting

### Issue: No datasets created
**Cause**: min_symbol_samples threshold too high
**Solution**: Lower the threshold in dataset_config or increase data availability

### Issue: Processing too slow
**Cause**: Large dataset size or limited resources  
**Solution**: Reduce sample_fraction or process in smaller batches

### Issue: Unbalanced distribution
**Cause**: force_uniform=False or insufficient data
**Solution**: Ensure force_uniform=True and adequate sample sizes

## Next Steps

After running production processing:

1. **Validate Output**: Check dataset files and distributions
2. **Transfer Data**: Move datasets to ML training environment  
3. **Implement Dataloader**: Create custom dataloader in ML repository
4. **Begin Training**: Use datasets for model training with balanced classes

The production datasets are now ready for high-quality ML training with no data leakage concerns!