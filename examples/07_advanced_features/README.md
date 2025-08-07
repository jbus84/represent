# 🔬 Advanced Features Examples

Advanced usage patterns, specialized examples, and cutting-edge features for expert users and researchers.

## 📋 Files in this directory

### **extended_features_examples.py**
Advanced feature extraction examples:
- Multi-dimensional feature combinations
- Custom feature engineering pipelines  
- Time-domain and frequency-domain features
- Advanced statistical feature computation

### **real_market_data_processing.py**
Production-scale examples with real market data:
- Large-scale batch processing workflows
- Multi-currency processing pipelines
- High-frequency data handling
- Production deployment patterns

### **sample_outputs/**
Reference outputs and benchmarks:
- Pre-generated sample outputs for validation
- Expected result formats
- Performance benchmarks from production systems
- Quality assurance reference data

### **advanced_configuration_examples.py**
Complex configuration scenarios:
- Dynamic configuration generation
- Multi-market optimization strategies
- Custom classification schemes
- Advanced parameter tuning

## 🚀 Advanced Capabilities

### **Multi-Currency Processing**
```python
from represent import ParquetClassifier

# Process multiple currencies with optimized settings
currencies = {
    'AUDUSD': {'nbins': 13, 'lookforward_rows': 500},
    'GBPUSD': {'nbins': 11, 'lookforward_rows': 300},  # Higher volatility
    'USDJPY': {'nbins': 9, 'lookforward_rows': 400}    # Different pip structure
}

for currency, config in currencies.items():
    classifier = ParquetClassifier(currency=currency, **config)
    results = classifier.process_dbn_to_classified_parquets(
        dbn_path=f"data/{currency}_data.dbn.zst",
        output_dir=f"classified/{currency.lower()}/"
    )
```

### **Custom Feature Engineering**
```python
from represent import MarketDepthProcessor

# Define custom feature extraction pipeline
class CustomFeatureProcessor(MarketDepthProcessor):
    def extract_momentum_features(self, df):
        """Extract price momentum features."""
        df = df.with_columns([
            (pl.col('ask_px_00').diff().alias('ask_momentum')),
            (pl.col('bid_px_00').diff().alias('bid_momentum')),
            (pl.col('ask_sz_00').rolling_std(5).alias('ask_volatility'))
        ])
        return df
    
    def extract_imbalance_features(self, df):
        """Extract order book imbalance features."""
        df = df.with_columns([
            ((pl.col('ask_sz_00') - pl.col('bid_sz_00')) / 
             (pl.col('ask_sz_00') + pl.col('bid_sz_00'))).alias('imbalance_L1'),
            (pl.sum_horizontal([f'ask_sz_0{i}' for i in range(10)]).alias('total_ask_volume')),
            (pl.sum_horizontal([f'bid_sz_0{i}' for i in range(10)]).alias('total_bid_volume'))
        ])
        return df
```

### **Dynamic Configuration Optimization**
```python
from represent import generate_classification_config_from_parquet

# Auto-optimize configuration based on actual data characteristics
def optimize_for_dataset(data_directory):
    """Generate optimal configuration for specific dataset."""
    
    # Analyze data characteristics
    sample_files = list(Path(data_directory).glob("*.dbn*"))[:10]
    
    configurations = []
    for nbins in [9, 11, 13, 15]:
        for lookforward in [300, 500, 700]:
            config, metrics = generate_classification_config_from_parquet(
                parquet_files=sample_files,
                currency="AUDUSD", 
                nbins=nbins,
                lookforward_rows=lookforward
            )
            
            configurations.append({
                'config': config,
                'quality': metrics['validation_metrics']['quality'],
                'deviation': metrics['validation_metrics']['avg_deviation']
            })
    
    # Select best configuration
    best_config = max(configurations, key=lambda x: x['quality'])
    return best_config['config']
```

## 🔬 Research-Level Features

### **Advanced Statistical Analysis**
```python
import numpy as np
from scipy import stats

def analyze_market_microstructure(classified_df):
    """Advanced microstructure analysis."""
    
    # Price impact analysis
    price_movements = classified_df['price_movement'].to_numpy()
    volume_imbalances = (classified_df['ask_sz_00'] - classified_df['bid_sz_00']).to_numpy()
    
    # Calculate correlations
    correlation = stats.pearsonr(price_movements, volume_imbalances)
    
    # Distribution fitting
    dist_params = stats.norm.fit(price_movements)
    
    # High-frequency patterns
    autocorr = np.correlate(price_movements, price_movements, mode='full')
    
    return {
        'price_volume_correlation': correlation,
        'movement_distribution': dist_params,
        'autocorrelation': autocorr,
        'kurtosis': stats.kurtosis(price_movements),
        'skewness': stats.skew(price_movements)
    }
```

### **Multi-Timeframe Analysis**
```python
def multi_timeframe_classification(df, timeframes=[100, 500, 1000]):
    """Classify at multiple time horizons."""
    
    classifications = {}
    
    for tf in timeframes:
        # Calculate price movements at different horizons
        future_price = pl.col('mid_price').shift(-tf).alias(f'future_price_{tf}')
        movement = ((future_price - pl.col('mid_price')) / MICRO_PIP_SIZE).alias(f'movement_{tf}')
        
        df_tf = df.with_columns([future_price, movement])
        
        # Apply quantile classification for this timeframe
        movements = df_tf[f'movement_{tf}'].to_numpy()
        quantiles = np.linspace(0, 1, 14)  # 13 bins
        boundaries = np.quantile(movements[~np.isnan(movements)], quantiles)
        
        classification = np.digitize(movements, boundaries[1:-1])
        classification = np.clip(classification, 0, 12)
        
        classifications[tf] = classification
    
    return classifications
```

### **Cross-Asset Analysis**
```python
def cross_asset_correlation_analysis(symbol_files):
    """Analyze correlations between different symbols/assets."""
    
    symbol_data = {}
    
    # Load all symbol data
    for file in symbol_files:
        symbol = file.stem.split('_')[1]
        df = pl.read_parquet(file)
        symbol_data[symbol] = df['price_movement'].to_numpy()
    
    # Calculate cross-correlations
    symbols = list(symbol_data.keys())
    n_symbols = len(symbols)
    correlation_matrix = np.zeros((n_symbols, n_symbols))
    
    for i, sym1 in enumerate(symbols):
        for j, sym2 in enumerate(symbols):
            # Align data by timestamp if needed
            corr = np.corrcoef(
                symbol_data[sym1][:1000], 
                symbol_data[sym2][:1000]
            )[0,1]
            correlation_matrix[i,j] = corr
    
    return correlation_matrix, symbols
```

## ⚡ Production-Scale Processing

### **High-Throughput Pipeline**
```python
import concurrent.futures
from pathlib import Path

def parallel_processing_pipeline(data_directory, output_directory, max_workers=8):
    """Process multiple files in parallel for maximum throughput."""
    
    dbn_files = list(Path(data_directory).glob("*.dbn*"))
    
    def process_file(dbn_file):
        """Process single file."""
        try:
            results = process_dbn_to_classified_parquets(
                dbn_path=dbn_file,
                output_dir=output_directory,
                currency="AUDUSD",
                features=['volume', 'variance'],
                min_symbol_samples=1000,
                verbose=False
            )
            return results
        except Exception as e:
            return {'error': str(e), 'file': str(dbn_file)}
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, dbn_file): dbn_file 
            for dbn_file in dbn_files
        }
        
        results = []
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            results.append(result)
    
    return results
```

### **Memory-Mapped Large Dataset Processing**
```python
import mmap

def memory_mapped_processing(large_parquet_file):
    """Process datasets larger than RAM using memory mapping."""
    
    # Use memory-mapped files for very large datasets
    with open(large_parquet_file, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Process in chunks without loading entire file
            chunk_size = 10_000_000  # 10M rows at a time
            
            for offset in range(0, len(mm), chunk_size):
                chunk_data = mm[offset:offset + chunk_size]
                # Process chunk
                yield process_chunk(chunk_data)
```

## 🧪 Experimental Features

### **Adaptive Classification**
```python
class AdaptiveClassifier:
    """Classifier that adapts to changing market conditions."""
    
    def __init__(self, adaptation_window=10000):
        self.adaptation_window = adaptation_window
        self.current_boundaries = None
        
    def adaptive_classify(self, price_movements, timestamps):
        """Classify with adaptive boundaries."""
        
        classifications = []
        
        for i in range(len(price_movements)):
            # Use sliding window for boundary calculation
            start_idx = max(0, i - self.adaptation_window)
            window_movements = price_movements[start_idx:i+1]
            
            if len(window_movements) >= 1000:  # Minimum for reliable quantiles
                quantiles = np.linspace(0, 1, 14)
                boundaries = np.quantile(window_movements, quantiles)
                self.current_boundaries = boundaries
            
            # Classify current movement
            if self.current_boundaries is not None:
                label = np.digitize(price_movements[i], self.current_boundaries[1:-1])
                label = np.clip(label, 0, 12)
            else:
                label = 6  # Neutral class fallback
                
            classifications.append(label)
        
        return np.array(classifications)
```

## 🔧 Running Advanced Examples

```bash
# Extended feature extraction
python 07_advanced_features/extended_features_examples.py

# Production-scale processing
python 07_advanced_features/real_market_data_processing.py

# Advanced configuration optimization
python 07_advanced_features/advanced_configuration_examples.py

# Validate against reference outputs
python 07_advanced_features/sample_outputs/validate_outputs.py
```

## 📊 Research Applications

### **Market Regime Detection**
```python
from sklearn.cluster import KMeans

def detect_market_regimes(price_movements):
    """Detect different market regimes using clustering."""
    
    # Feature engineering for regime detection
    features = np.column_stack([
        np.abs(price_movements),           # Volatility proxy
        np.sign(price_movements),          # Direction
        np.cumsum(price_movements),        # Trend
        np.std(rolling_window(price_movements, 100))  # Rolling volatility
    ])
    
    # Cluster analysis
    kmeans = KMeans(n_clusters=3)  # Bull, Bear, Sideways
    regimes = kmeans.fit_predict(features)
    
    return regimes
```

### **Liquidity Analysis**
```python
def analyze_liquidity_patterns(classified_df):
    """Analyze liquidity patterns across different market conditions."""
    
    # Group by classification labels
    liquidity_by_class = {}
    
    for label in range(13):
        class_data = classified_df.filter(pl.col('classification_label') == label)
        
        # Calculate liquidity metrics
        total_volume = class_data.select([
            pl.sum_horizontal([f'ask_sz_0{i}' for i in range(10)]).alias('ask_vol'),
            pl.sum_horizontal([f'bid_sz_0{i}' for i in range(10)]).alias('bid_vol')
        ])
        
        spread = (class_data['ask_px_00'] - class_data['bid_px_00']).mean()
        
        liquidity_by_class[label] = {
            'avg_volume': total_volume.mean(),
            'avg_spread': spread,
            'sample_count': len(class_data)
        }
    
    return liquidity_by_class
```

## 📁 Sample Outputs Directory

The `sample_outputs/` directory contains:
```
07_advanced_features/sample_outputs/
├── reference_classification_results.parquet  # Expected classification output
├── benchmark_performance_metrics.json        # Performance benchmarks
├── statistical_analysis_results.json         # Market analysis results
├── multi_currency_comparison.csv            # Cross-currency analysis
└── validate_outputs.py                      # Validation script
```

## ➡️ Research and Production

These advanced examples are designed for:
- **Research**: Academic studies, market microstructure analysis
- **Production**: High-scale financial data processing  
- **Experimentation**: Testing new feature engineering approaches
- **Optimization**: Fine-tuning for specific market conditions

Use these patterns as starting points for your own advanced applications!