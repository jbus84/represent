# 🚀 Represent Package Examples

This directory contains comprehensive demonstrations of the **Symbol-Split-Merge Architecture (v5.0.0)** for creating comprehensive symbol datasets from multiple DBN files.

## 📋 Available Examples

### **🌟 symbol_split_merge_demo.py - MAIN DEMONSTRATION**

**The primary example showcasing the symbol-split-merge architecture:**

- **🔄 Multi-File Processing**: Process multiple DBN files to create comprehensive symbol datasets
- **📈 Symbol Coverage**: Merge symbol data across files for complete market history
- **⚡ Performance Optimization**: Two-phase processing with intermediate cleanup
- **🧠 ML-Ready Datasets**: Comprehensive symbol datasets ready for training

**Run the main demo:**
```bash
python examples/symbol_split_merge_demo.py
```

### **🚀 quick_start_examples.py - Getting Started**

**Simple, focused examples for quick learning:**

- Basic configuration setup
- Single DBN file processing  
- Directory batch processing
- Custom dataset builder usage

**Run the quick start examples:**
```bash
python examples/quick_start_examples.py
```

## 📊 Expected Outputs

### **Symbol-Split-Merge Demo Outputs:**
```
symbol_split_merge_demo_output/
├── comprehensive_datasets/
│   ├── AUDUSD_M6AM4_dataset.parquet       # Comprehensive M6AM4 data from all files
│   ├── AUDUSD_M6AM5_dataset.parquet       # Comprehensive M6AM5 data from all files  
│   └── AUDUSD_M6AN4_dataset.parquet       # Comprehensive M6AN4 data from all files
├── processing_report.html                  # Detailed processing analysis
├── symbol_coverage_analysis.png           # Symbol distribution across files
├── comprehensive_dataset_stats.png        # Dataset size and quality metrics
└── demo_results.json                      # Complete processing statistics
```

## 🎯 Symbol-Split-Merge Workflow with New Classification

### **New Classification Approach (v5.0.0+):**

The represent package now uses **per-symbol classification with first-half training**:

- **🔍 Symbol-Specific**: Each symbol gets classification boundaries tailored to its price movement distribution
- **📊 First-Half Training**: Uses first 50% of symbol data to define bins, preventing data leakage  
- **⚖️ Uniform Distribution**: Quantile-based approach ensures balanced class distribution across all symbols
- **🚀 No Data Leakage**: Training boundaries defined before applying to full dataset
- **💾 Polars Optimized**: High-performance vectorized operations for classification

### **Primary Workflow Example:**
```python
from represent import create_represent_config, DatasetBuildConfig
from represent import build_datasets_from_dbn_files

# Create configuration
config = create_represent_config(
    currency="AUDUSD", 
    features=['volume', 'variance']
)

dataset_config = DatasetBuildConfig(
    currency="AUDUSD",
    features=['volume', 'variance'],
    min_symbol_samples=10000,     # Higher threshold for comprehensive datasets
    force_uniform=True,           # Ensure uniform class distribution
    keep_intermediate=False       # Clean up intermediate split files
)

# Build comprehensive symbol datasets from multiple DBN files
results = build_datasets_from_dbn_files(
    config=config,
    dbn_files=[
        "data/AUDUSD-20240101.dbn.zst",
        "data/AUDUSD-20240102.dbn.zst", 
        "data/AUDUSD-20240103.dbn.zst"
    ],
    output_dir="/data/symbol_datasets/",
    dataset_config=dataset_config
)

# Result: Comprehensive symbol datasets ready for ML training
# /data/symbol_datasets/AUDUSD_M6AM4_dataset.parquet (contains ALL M6AM4 data)
# /data/symbol_datasets/AUDUSD_M6AM5_dataset.parquet (contains ALL M6AM5 data)
```

### **Directory Processing Example:**
```python
from represent import batch_build_datasets_from_directory

# Process all DBN files in a directory
results = batch_build_datasets_from_directory(
    config=config,
    input_directory="data/audusd_dbn_files/",
    output_dir="/data/symbol_datasets/",
    file_pattern="*.dbn*",
    dataset_config=dataset_config
)

print(f"Processed {len(results['input_files'])} DBN files")
print(f"Created {results['phase_2_stats']['datasets_created']} comprehensive symbol datasets")
```

## 🚀 Quick Start

### **1. Run Main Demo**
```bash
# Demonstrate symbol-split-merge architecture
python examples/symbol_split_merge_demo.py

# View comprehensive processing report
open symbol_split_merge_demo_output/processing_report.html
```

### **2. Quick Learning Examples**
```bash
# Simple focused examples
python examples/quick_start_examples.py
```

## 🎯 New Classification Benefits

### **📊 First-Half Training Advantages**
- **No Data Leakage**: Classification boundaries defined using only first 50% of symbol data
- **Symbol Adaptation**: Each symbol gets thresholds fitted to its specific price movement patterns
- **Consistent Application**: Same boundaries used for entire symbol dataset after training
- **Quality Validation**: Distribution stability metrics show first-half vs second-half consistency

### **⚖️ Uniform Distribution Guarantee** 
- **Quantile-Based**: Uses first-half data quantiles to ensure balanced classes
- **Symbol-Specific**: Each symbol achieves its own uniform distribution
- **ML-Optimized**: Prevents class imbalance issues during training
- **Statistical Rigor**: Coefficient of variation < 0.1 for good uniformity

### **🚀 Performance Optimization**
- **Polars Vectorized**: High-performance classification using polars operations
- **Memory Efficient**: Processes large symbol datasets without memory issues
- **Fast Binning**: Efficient quantile calculation and np.digitize for classification
- **Batch Processing**: Optimized for processing multiple symbols in parallel

## 📈 Symbol-Split-Merge Benefits

### **🔄 Two-Phase Processing**
- **Phase 1**: Split each DBN file by symbol into intermediate files
- **Phase 2**: Merge all instances of each symbol across files into comprehensive datasets
- **Result**: Each symbol gets its complete history from all input files

### **🎯 Comprehensive Coverage**
- **Traditional**: Process each DBN file independently → fragmented symbol data
- **Symbol-Split-Merge**: Merge symbol data across ALL files → comprehensive symbol datasets
- **ML Training**: Train on complete symbol history rather than fragments

### **⚡ Performance Optimized**
```
Processing Flow:
DBN File 1 → Split by Symbol → M6AM4.parquet, M6AM5.parquet, ...
DBN File 2 → Split by Symbol → M6AM4.parquet, M6AM5.parquet, ...  
DBN File 3 → Split by Symbol → M6AM4.parquet, M6AM5.parquet, ...
                                        ↓
Merge Phase: M6AM4_dataset.parquet ← All M6AM4.parquet files
             M6AM5_dataset.parquet ← All M6AM5.parquet files
```

### **📊 Dataset Quality**
- **Large Datasets**: Symbol datasets much larger than individual file processing
- **Uniform Distribution**: True uniform classification using full symbol context
- **Symbol Registry**: Automatic tracking of symbol presence across files
- **Configurable Cleanup**: Keep or remove intermediate split files

## 🎯 Use Cases

### **✅ Perfect for Symbol-Split-Merge:**
- **Multiple DBN files** covering same market/currency
- **Comprehensive symbol datasets** for robust ML training
- **Large-scale ML pipelines** requiring complete market history
- **Production environments** with regular data ingestion

### **📈 Key Advantages:**
- **Complete Symbol History**: Each dataset contains symbol's full timeline
- **Larger Training Sets**: Much more data per symbol than single-file processing
- **Better Model Performance**: Train on comprehensive market patterns
- **Production Scalability**: Designed for multi-file production workflows

## 🔧 Customization Examples

### **Modify Symbol Coverage Threshold**
```python
dataset_config = DatasetBuildConfig(
    min_symbol_samples=5000,   # Lower threshold = more symbols included
    # or
    min_symbol_samples=50000,  # Higher threshold = only highly active symbols
)
```

### **Configure Intermediate File Handling**
```python
dataset_config = DatasetBuildConfig(
    keep_intermediate=True,        # Keep split files for inspection
    intermediate_dir="/tmp/splits" # Custom intermediate directory
)
```

### **Adjust Processing Parameters**
```python
config = create_represent_config(
    currency="AUDUSD",
    features=['volume', 'variance'],
    lookback_rows=10000,      # Longer historical context
    lookforward_input=1000,   # Shorter prediction window
    jump_size=50             # Sparser sampling for performance
)
```

## 📈 Performance Targets

The symbol-split-merge architecture meets these performance requirements:

### **Processing Performance**
- **Split Phase**: >300 samples/second per DBN file
- **Merge Phase**: >1500 samples/second during symbol dataset creation
- **Total Throughput**: >500 samples/second end-to-end
- **Memory Usage**: <8GB RAM for processing multiple large DBN files

### **Dataset Quality**
- **Comprehensive Coverage**: Each symbol's complete history across all files
- **Uniform Distribution**: <5% deviation from perfect uniformity
- **Symbol Registry**: 100% accurate tracking across files
- **File Efficiency**: Optimized parquet compression and organization

## 🎉 Next Steps

After exploring the examples:

1. **🔄 Start with Main Demo** - Run the symbol-split-merge demonstration
2. **⚡ Optimize Configurations** - Tune parameters for your use case
3. **🚀 Integrate into Production** - Adapt examples to your data pipeline
4. **🧠 Build ML Training** - Use comprehensive datasets for robust models
5. **📊 Scale Up** - Process your full DBN file collections

---

**The Symbol-Split-Merge architecture provides the most comprehensive and efficient way to create ML-ready datasets from multiple DBN files, ensuring complete market coverage and optimal training data quality.**