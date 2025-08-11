# 🚀 Represent Package Examples

This directory contains examples demonstrating the core functionality of the `represent` package, focusing on the **Symbol-Split-Merge Architecture**.

## 📋 Available Examples

### **🌟 `symbol_split_merge_demo.py` - Comprehensive Demo**

This is the primary, comprehensive example that showcases the main `build_datasets_from_dbn_files` workflow. It's the best place to start to understand the library's intended use.

**What it demonstrates:**
-   Processing multiple DBN files at once.
-   How data for the same symbol is merged from different files.
-   Configuration of features and dataset parameters.
-   The final output of comprehensive, symbol-specific Parquet files.

**Run the demo:**
```bash
python examples/symbol_split_merge_demo.py
```

### **🚀 `quick_start_examples.py` - Focused Examples**

This script contains a series of simple, focused functions, each designed to illustrate one of the three core objectives of the library.

**What it demonstrates:**
1.  **DBN to Symbol-Specific Parquet**: The basic workflow of converting DBN files into symbol-specific Parquet files.
2.  **Uniform Classification**: Shows how the library automatically creates uniformly distributed classification labels, which is ideal for ML.
3.  **Multi-Feature Generation**: How to configure the pipeline to generate features for volume, variance, and trade counts, and how to verify them in the output.

**Run the quick start examples:**
```bash
python examples/quick_start_examples.py
```

## 🎯 Core Concepts Illustrated

### Symbol-Split-Merge Workflow
The examples show how the library takes multiple DBN files, splits them by financial symbol, and then merges the data for each symbol into a single, comprehensive Parquet file. This is the foundation of the library's design.

### Feature Generation
You can easily configure the pipeline to generate features for:
-   `volume`
-   `variance`
-   `trade_counts`

The examples show how to specify these features and verify their presence in the output files.

### Uniform Classification
For machine learning, having a balanced (or uniform) distribution of class labels is crucial. The examples demonstrate that the `represent` package automatically handles this by creating classification bins with a nearly equal number of samples in each.
