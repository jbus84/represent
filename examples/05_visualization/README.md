# 📈 Visualization Examples

Data visualization and analysis examples for exploring market depth data, classification results, and model predictions.

## 📋 Files in this directory

### **generate_visualization.py**
Comprehensive visualization generator:
- Market depth heatmaps showing bid/ask structure
- Price movement distribution histograms  
- Classification label distribution charts
- Time series plots of market activity

### **market_depth_visualization.py**  
Detailed market depth analysis:
- 3D surface plots of market depth evolution
- Volume distribution across price levels
- Depth imbalance visualization
- Real-time depth structure analysis

### **multi_feature_visualization.py**
Multi-feature visualization comparisons:
- Side-by-side volume vs variance vs trade_counts
- Feature correlation analysis
- Combined feature impact visualization
- Feature importance heatmaps

### **extended_features_examples.py**
Advanced visualization for extended features:
- Time-domain feature evolution
- Frequency analysis of market patterns
- Cross-symbol feature comparison
- Statistical distribution analysis

## 🎨 Visualization Types

### **Market Depth Heatmaps**
```python
# Visualize the 402 × 500 market depth array
plt.imshow(market_depth_array, aspect='auto', cmap='RdYlBu')
plt.xlabel('Time Bins (500)')
plt.ylabel('Price Levels (402)')  
plt.title('Market Depth - Volume Distribution')
```

### **Classification Analysis**
```python
# Show price movement → classification mapping  
plt.hist(price_movements, bins=50, alpha=0.7)
plt.axvline(threshold_boundaries, color='red', linestyle='--')
plt.xlabel('Price Movement (micro-pips)')
plt.ylabel('Frequency')
plt.title('Price Movement Distribution with Classification Boundaries')
```

### **Multi-Feature Comparison**
```python
# Compare different feature types
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(volume_features, title='Volume Features')
axes[1].imshow(variance_features, title='Variance Features') 
axes[2].imshow(trade_count_features, title='Trade Count Features')
```

## 🔧 Dependencies

Install visualization dependencies:
```bash
pip install matplotlib seaborn plotly scipy pandas
```

Or with uv:
```bash  
uv add matplotlib seaborn plotly scipy pandas
```

## 📊 Generated Visualizations

### **Market Structure Analysis**
- **Depth Evolution**: How market depth changes over time
- **Price Level Activity**: Which price levels see most activity
- **Volume Imbalances**: Bid vs ask volume differences
- **Time Patterns**: Intraday market depth patterns

### **Classification Quality**
- **Label Distribution**: Verify uniform distribution (7.69% per class)
- **Boundary Analysis**: Visualize classification thresholds
- **Movement Patterns**: Price movement statistical analysis
- **Cross-File Consistency**: Verify global threshold effectiveness

### **Feature Analysis**
- **Feature Correlation**: How different features relate
- **Information Content**: Which features capture most signal
- **Temporal Patterns**: How features evolve over time
- **Symbol Differences**: Feature characteristics across symbols

## 🚀 Usage Examples

### **Basic Market Depth Visualization**
```python
from represent.lazy_dataloader import create_parquet_dataloader
import matplotlib.pyplot as plt

# Load classified data
dataloader = create_parquet_dataloader(
    parquet_path="classified/AUDUSD_M6AM4_classified.parquet",
    batch_size=1
)

# Get one sample
features, labels = next(iter(dataloader))
market_depth = features[0]  # Shape: (402, 500) or (N, 402, 500)

# Visualize
plt.figure(figsize=(12, 8))
if market_depth.dim() == 3:
    # Multi-feature: show first feature
    plt.imshow(market_depth[0], aspect='auto', cmap='RdYlBu')
else:
    # Single feature
    plt.imshow(market_depth, aspect='auto', cmap='RdYlBu')
    
plt.xlabel('Time Bins')
plt.ylabel('Price Levels')  
plt.title('Market Depth Visualization')
plt.colorbar(label='Normalized Volume')
plt.show()
```

### **Classification Distribution Analysis**
```python
import polars as pl
import matplotlib.pyplot as plt

# Load classified parquet
df = pl.read_parquet("classified/AUDUSD_M6AM4_classified.parquet")

# Plot classification distribution
labels = df['classification_label'].to_numpy()
unique_labels, counts = np.unique(labels, return_counts=True)

plt.figure(figsize=(10, 6))
plt.bar(unique_labels, counts)
plt.xlabel('Classification Label')  
plt.ylabel('Count')
plt.title('Classification Distribution (Should be Uniform)')

# Add uniform line
expected_count = len(labels) / 13
plt.axhline(y=expected_count, color='red', linestyle='--', 
           label=f'Expected Uniform: {expected_count:.0f}')
plt.legend()
plt.show()
```

### **Multi-Symbol Comparison**
```python
import matplotlib.pyplot as plt
from pathlib import Path

# Load multiple symbol files
classified_dir = Path("classified/")
symbol_files = list(classified_dir.glob("*_classified.parquet"))

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, file in enumerate(symbol_files[:4]):
    df = pl.read_parquet(file)
    symbol = file.stem.split('_')[1]
    
    # Plot price movement distribution
    movements = df['price_movement'].to_numpy()
    axes[i].hist(movements, bins=50, alpha=0.7)
    axes[i].set_title(f'Price Movements - {symbol}')
    axes[i].set_xlabel('Movement (micro-pips)')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

## 🔧 Running Visualization Examples

```bash
# Generate comprehensive visualizations
python 05_visualization/generate_visualization.py

# Create market depth analysis
python 05_visualization/market_depth_visualization.py

# Compare multiple features
python 05_visualization/multi_feature_visualization.py

# Advanced feature analysis
python 05_visualization/extended_features_examples.py
```

## 📁 Output Files

Visualizations are saved as:
```
05_visualization/outputs/
├── market_depth_heatmap.png
├── classification_distribution.png  
├── price_movement_analysis.png
├── feature_comparison.png
├── multi_symbol_analysis.png
└── ... (various analysis charts)
```

## 💡 Visualization Tips

### **Color Schemes**
```python
# Good color maps for market data
cmap='RdYlBu'      # Red-Yellow-Blue for depth (red=high volume)
cmap='viridis'     # Perceptually uniform
cmap='plasma'      # High contrast for patterns
```

### **Interactive Plots**
```python
# Use Plotly for interactive exploration
import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(z=market_depth_array))
fig.show()  # Interactive zoom/pan
```

### **Statistical Analysis**
```python
# Add statistical annotations
plt.text(0.02, 0.95, f'Mean: {np.mean(data):.2f}', 
         transform=plt.gca().transAxes)
plt.text(0.02, 0.90, f'Std: {np.std(data):.2f}', 
         transform=plt.gca().transAxes)
```

## ➡️ Next Steps

After visualizing your data:
- `06_performance_analysis/` - Analyze processing and training performance
- `04_ml_training/` - Use insights for model improvement
- `07_advanced_features/` - Explore advanced visualization techniques