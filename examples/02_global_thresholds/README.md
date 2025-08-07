# 🎯 Global Thresholds Examples (RECOMMENDED APPROACH)

**⭐ This is the RECOMMENDED approach for production use!**

Examples demonstrating global threshold calculation for consistent classification across multiple files. This solves the fundamental problem of per-file quantile inconsistency.

## 🚨 Why Global Thresholds Matter

### **❌ Problem: Per-File Quantiles**
```
File 1: Price movement +3.2 μpips → Class 8
File 2: Price movement +3.2 μpips → Class 5  ❌ Different label!
```

### **✅ Solution: Global Thresholds**
```
File 1: Price movement +3.2 μpips → Class 8
File 2: Price movement +3.2 μpips → Class 8  ✅ Consistent!
```

## 📋 Files in this directory

### **🌟 global_threshold_classification_demo.py**
**Complete production workflow:**
- Calculate global thresholds from sample of your dataset
- Process multiple files using the same thresholds
- Generate consistent classifications across all files
- **Use this for your actual projects!**

### **simple_global_threshold_demo.py**
**Quick verification:**
- Simplified version of the global threshold approach
- Perfect for understanding the concept
- Fast execution for testing

### **verify_row_classification.py**
**Verification script:**
- Confirms that each row gets classified individually
- Shows price movement → classification mapping
- Demonstrates row-level processing (not file-level)

### **comprehensive_threshold_analysis.py**
**Deep analysis:**
- Detailed threshold calculation with visualizations
- Quality metrics and validation
- Statistical analysis of price movements

## 🎯 Recommended Usage Flow

### **Step 1: Quick Understanding**
```bash
# Start with simple demo to understand the concept
python 02_global_thresholds/simple_global_threshold_demo.py
```

### **Step 2: Verify Row-Level Processing**
```bash
# Confirm each row gets its own classification
python 02_global_thresholds/verify_row_classification.py
```

### **Step 3: Production Workflow**
```bash
# Use this for your actual data processing
python 02_global_thresholds/global_threshold_classification_demo.py
```

### **Step 4: Deep Analysis (Optional)**
```bash
# For research and optimization
python 02_global_thresholds/comprehensive_threshold_analysis.py
```

## 🔥 Key Benefits

✅ **Consistent Classification**: Same price movement gets same label across ALL files  
✅ **Better ML Performance**: Comparable training data improves model accuracy  
✅ **Production Ready**: Handles multiple files with guaranteed consistency  
✅ **Row-Level Processing**: Each price movement classified individually  
✅ **Uniform Distribution**: Balanced class representation for optimal ML training  

## 📊 Expected Output

These examples generate:
- `audusd_classification_bins.json` - Global threshold configuration
- `classified_output/` - Classified parquet files by symbol
- Processing statistics and timing information
- Validation metrics showing consistency

## 🎯 Production Usage Pattern

```python
# 1. Calculate global thresholds ONCE for your dataset
global_thresholds = calculate_global_thresholds(
    data_directory="/path/to/your/audusd/files/",
    currency="AUDUSD", 
    sample_fraction=0.5,  # Use 50% of files
    nbins=13
)

# 2. Apply to ALL files using same thresholds
for dbn_file in your_files:
    process_dbn_to_classified_parquets(
        dbn_path=dbn_file,
        output_dir="classified/",
        global_thresholds=global_thresholds  # 🎯 Key!
    )
```

## 💡 Data Requirements

These examples expect AUDUSD data at:
```
/Users/danielfisher/data/databento/AUDUSD-micro
```

**Update the path in the examples to point to your own data directory.**

## ➡️ Next Steps

After mastering global thresholds:
- `04_ml_training/` - Use your classified data for ML training
- `06_performance_analysis/` - Optimize processing performance
- `03_data_processing/` - Explore advanced processing features