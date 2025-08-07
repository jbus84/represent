# 📊 Data Processing Examples

Examples focused on converting and processing market data from DBN files to classified parquet datasets.

## 📋 Files in this directory

### **fixed_optimized_demo.py**
Updated demonstration of the streamlined processing approach:
- DBN → Classified Parquet in single pass
- Optimized for performance and memory efficiency
- Uses modern represent API

### **Other processing examples**
Additional data processing demonstrations showing:
- Multi-feature extraction (volume, variance, trade_counts)
- Symbol-specific processing optimization
- Batch processing workflows
- Memory-efficient large file handling

## 🔄 Processing Pipeline

### **Input: DBN Files**
```
AUDUSD-20240101.dbn.zst  (Raw market data)
├── Multiple symbols (M6AM4, M6AU4, etc.)
├── Price levels (bid/ask 00-09)  
├── Volume levels (bid_sz/ask_sz 00-09)
└── Trade counts (bid_ct/ask_ct 00-09)
```

### **Output: Classified Parquet Files**
```
classified/
├── AUDUSD_M6AM4_classified.parquet
├── AUDUSD_M6AU4_classified.parquet
└── ... (one file per symbol)
```

Each classified parquet contains:
- **Original market data columns**
- **`price_movement`**: Future price change in micro-pips
- **`classification_label`**: Class 0-12 based on price movement
- **Ready for ML training**: Direct loading into PyTorch DataLoaders

## ⚡ Key Features

### **Streamlined Processing**
- **Single-pass conversion**: DBN → Classified Parquet directly
- **No intermediate files**: Eliminates storage overhead
- **Symbol-grouped output**: Separate files for targeted analysis

### **Multi-Feature Support**
```python
features = ['volume', 'variance', 'trade_counts']
# Generates market depth arrays for each feature type
```

### **Memory Efficiency**
- Processes large DBN files without loading entire dataset
- Configurable batch sizes for optimal memory usage
- Streaming decompression for .zst files

## 🚀 Usage Examples

### **Basic Processing**
```python
from represent import process_dbn_to_classified_parquets

stats = process_dbn_to_classified_parquets(
    dbn_path="data/AUDUSD-20240101.dbn.zst",
    output_dir="classified/",
    currency="AUDUSD",
    features=['volume'],
    min_symbol_samples=1000
)
```

### **Multi-Feature Processing**
```python
stats = process_dbn_to_classified_parquets(
    dbn_path="data/market_data.dbn.zst", 
    output_dir="classified/",
    currency="AUDUSD",
    features=['volume', 'variance', 'trade_counts'],  # 3 features
    global_thresholds=global_thresholds  # For consistency
)
```

### **Batch Processing**
```python
from represent import ParquetClassifier

classifier = ParquetClassifier(currency="AUDUSD")
results = classifier.batch_classify_parquets(
    input_directory="data/dbn_files/",
    output_directory="classified/",
    pattern="*.dbn*"
)
```

## 📊 Performance Expectations

- **Processing Speed**: 500+ samples/second
- **Memory Usage**: <4GB RAM regardless of file size
- **Output Size**: ~60-80% of original DBN size (compressed parquet)
- **Symbol Filtering**: Only symbols with sufficient data included

## 🎯 Configuration Options

### **Processing Parameters**
```python
input_rows=5000          # Historical data required
lookforward_rows=500     # Future data for classification
min_symbol_samples=1000  # Minimum samples per symbol
force_uniform=True       # Guarantee uniform distribution
```

### **Feature Configuration**
```python
features=['volume']                    # Single feature: (402, 500)
features=['volume', 'variance']        # Two features: (2, 402, 500)  
features=['volume', 'variance', 'trade_counts']  # Three features: (3, 402, 500)
```

## 🔧 Running Examples

```bash
# Run the main processing demo
python 03_data_processing/fixed_optimized_demo.py

# Process with different feature combinations
python -c "
from represent import process_dbn_to_classified_parquets
stats = process_dbn_to_classified_parquets(
    'your_data.dbn.zst',
    'output/',
    features=['volume', 'variance']
)
"
```

## 📁 Expected Output Structure

```
output_directory/
├── AUDUSD_M6AM4_classified.parquet    # Symbol-specific files
├── AUDUSD_M6AU4_classified.parquet
├── AUDUSD_M6BM4_classified.parquet
└── ... (one file per symbol with sufficient data)
```

Each parquet file contains:
- **All original DBN columns**
- **Computed columns**: `mid_price`, `future_mid_price`, `price_movement`
- **Classification**: `classification_label` (0-12)
- **ML-ready format**: Direct loading into training pipelines

## ➡️ Next Steps

After processing your data:
- `04_ml_training/` - Use classified parquet files for ML training
- `02_global_thresholds/` - Ensure consistent classification across files
- `05_visualization/` - Visualize your processed market data