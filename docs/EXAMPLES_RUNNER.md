# Examples Runner - Comprehensive Testing & Documentation

The `run-all-examples` Makefile target provides automated execution of all examples with comprehensive HTML reporting. This is perfect for:

- **Continuous Integration**: Verify all examples work after code changes
- **Documentation**: Generate visual showcase of package capabilities  
- **Testing**: Validate examples across different environments
- **Onboarding**: Provide new users with comprehensive example outputs

## 🚀 Usage

### **Quick Start**
```bash
# Run all examples and generate HTML report
make run-all-examples

# View the generated report
open examples_report/examples_report.html
```

### **Manual Execution**
```bash
# Run the script directly
python scripts/run_all_examples.py

# View results
ls examples_report/
```

## 📊 Generated Outputs

### **HTML Report (`examples_report/examples_report.html`)**
Comprehensive interactive report featuring:

- **📈 Summary Dashboard**: Success rate, runtime statistics, overview metrics
- **📂 Directory Organization**: Examples grouped by category (01_getting_started → 07_advanced_features)
- **✅ Success/Failure Status**: Clear indicators for each example
- **📄 Output Capture**: Full stdout from each example execution
- **❌ Error Details**: Complete error messages for failed examples  
- **📁 Generated Files**: List of all outputs created by each example
- **⏱️ Performance Metrics**: Execution time for each example
- **🔍 Interactive UI**: Expandable sections, auto-expand failures

### **JSON Results (`examples_report/examples_results.json`)**
Programmatic access to results:
```json
{
  "timestamp": "2024-01-15T10:30:45",
  "summary": {
    "total_examples": 25,
    "successful": 23,
    "failed": 2,
    "total_duration": 127.5
  },
  "results": [
    {
      "name": "simple_api_demo.py",
      "path": "01_getting_started/simple_api_demo.py",
      "success": true,
      "duration": 1.2,
      "output_length": 850,
      "error": null,
      "generated_files_count": 3
    }
  ]
}
```

## 🎯 Example Categories

The runner executes examples in logical order:

### **1. 🚀 Getting Started** (`01_getting_started/`)
- Basic API usage examples
- Configuration demonstrations
- Simple currency examples
- **Purpose**: New user onboarding

### **2. 🎯 Global Thresholds** (`02_global_thresholds/`) 
- **RECOMMENDED APPROACH** examples
- Global threshold calculation workflows
- Consistency verification demos
- **Purpose**: Production-ready classification

### **3. 📊 Data Processing** (`03_data_processing/`)
- DBN to parquet conversion
- Streamlined processing workflows
- Multi-feature extraction
- **Purpose**: Core data processing capabilities

### **4. 🧠 ML Training** (`04_ml_training/`)
- PyTorch integration examples
- Lazy dataloader usage
- Real data training examples
- **Purpose**: Machine learning integration

### **5. 📈 Visualization** (`05_visualization/`)
- Market depth visualizations
- Multi-feature comparisons
- Statistical analysis plots
- **Purpose**: Data analysis and insights

### **6. ⚡ Performance Analysis** (`06_performance_analysis/`)
- Benchmark tests
- Memory usage analysis
- Throughput measurements
- **Purpose**: Performance optimization

### **7. 🔬 Advanced Features** (`07_advanced_features/`)
- Extended feature examples
- Production-scale processing
- Research-level capabilities
- **Purpose**: Expert usage patterns

## ⚙️ Configuration

### **Timeout Settings**
- **Default timeout**: 5 minutes per example
- **Rationale**: Allows for data processing examples without hanging
- **Customization**: Modify `timeout=300` in `run_all_examples.py`

### **Output Detection**
The runner automatically detects generated files:
```python
patterns = [
    "*.png", "*.jpg", "*.jpeg", "*.svg",  # Images
    "*.parquet", "*.csv", "*.json",       # Data files
    "*.html", "*.txt", "*.log",           # Reports
    "outputs/**/*", "classified/**/*"     # Output directories
]
```

### **Error Handling**
- **Graceful degradation**: Failed examples don't stop execution
- **Detailed error capture**: Full stack traces preserved
- **Auto-expand failures**: Failed examples automatically expanded in HTML report

## 🔧 Integration with CI/CD

### **GitHub Actions Integration**
```yaml
name: Examples Test
on: [push, pull_request]

jobs:
  test-examples:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: make install
      - name: Run all examples
        run: make run-all-examples
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: examples-report
          path: examples_report/
```

### **Quality Gates**
```bash
# Fail CI if success rate < 90%
python -c "
import json
with open('examples_report/examples_results.json') as f:
    data = json.load(f)
success_rate = data['summary']['successful'] / data['summary']['total_examples']
assert success_rate >= 0.9, f'Success rate {success_rate:.1%} below 90%'
"
```

## 📈 Performance Expectations

### **Typical Runtime**
- **Getting Started**: ~10-20 seconds
- **Global Thresholds**: ~30-60 seconds (data dependent)
- **Data Processing**: ~60-120 seconds (data dependent)
- **ML Training**: ~20-40 seconds
- **Visualization**: ~15-30 seconds  
- **Performance Analysis**: ~30-60 seconds
- **Advanced Features**: ~60-180 seconds (data dependent)

**Total Expected Runtime**: 3-8 minutes (depends on data availability)

### **Resource Usage**
- **Memory**: <4GB RAM (lazy loading prevents memory bloat)
- **Disk Space**: <500MB for all generated outputs
- **CPU**: Scales with available cores for parallel examples

## 🚨 Common Issues & Solutions

### **Data Path Issues**
Many examples reference:
```
/Users/danielfisher/data/databento/AUDUSD-micro
```

**Solutions:**
1. **Skip data-dependent examples**: They'll show "File not found" errors but continue
2. **Update paths**: Modify examples to point to your data directory  
3. **Use sample data**: Create minimal DBN files for testing

### **Missing Dependencies**
```bash
# Install visualization dependencies
pip install matplotlib seaborn plotly

# Install ML dependencies  
pip install torch torchvision

# Install all extras
uv sync --all-extras
```

### **Permission Issues**
```bash
# Make script executable
chmod +x scripts/run_all_examples.py

# Fix output directory permissions
chmod -R 755 examples_report/
```

## 🎉 Benefits

### **For Developers**
- **Automated Regression Testing**: Catch breaking changes immediately
- **Documentation Generation**: Always up-to-date example showcase
- **Debugging Aid**: Clear output capture and error reporting
- **Performance Monitoring**: Track example execution times

### **For Users**
- **Complete Package Showcase**: See all capabilities in one report
- **Learning Path**: Examples executed in logical progression
- **Troubleshooting**: Compare your outputs with expected results
- **Feature Discovery**: Find examples relevant to your use case

### **For CI/CD**
- **Quality Gates**: Ensure examples always work
- **Artifact Generation**: Preserve example outputs for review
- **Performance Regression**: Track execution time changes
- **Documentation Currency**: Generate fresh docs on each release

## 🔄 Maintenance

### **Adding New Examples**
1. Create example file in appropriate directory (`01_getting_started/` through `07_advanced_features/`)
2. Follow naming convention: `descriptive_name.py`
3. Include proper error handling and output generation
4. Test with: `python examples/XX_category/your_example.py`
5. Verify with runner: `make run-all-examples`

### **Updating Categories**
Modify `ordered_dirs` in `run_all_examples.py`:
```python
ordered_dirs = [
    "01_getting_started",
    "02_global_thresholds", 
    "03_data_processing",
    "04_ml_training",
    "05_visualization", 
    "06_performance_analysis",
    "07_advanced_features",
    "08_new_category"  # Add new category
]
```

---

**🎯 The examples runner provides comprehensive automated testing and documentation generation, ensuring all examples work correctly and showcasing the full capabilities of the represent package.**