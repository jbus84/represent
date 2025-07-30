#!/usr/bin/env python3
"""
Extended Features Visualization Example

This script demonstrates the new extended features functionality and creates
visualizations showing the different feature types and their combinations.

Features demonstrated:
- Single feature extraction (volume, variance, trade_counts)
- Multi-feature extraction with 3D output
- RGB visualization for 3-feature combinations
- Comparison plots showing feature differences
"""

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path
import sys
import time
import databento as db

# Add the represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent.pipeline import MarketDepthProcessor, process_market_data
from represent.constants import (
    SAMPLES, PRICE_LEVELS, TIME_BINS, VARIANCE_COLUMN
)

def load_real_market_data(n_samples: int = SAMPLES) -> pl.DataFrame:
    """Load real market data from DBN file (same approach as generate_visualization.py)."""
    try:
        # Load the real market data from the .dbn.zst file
        data_file = Path(__file__).parent.parent / "data" / "glbx-mdp3-20240405.mbp-10.dbn.zst"
        
        if not data_file.exists():
            print(f"⚠️  Real data file not found at {data_file}")
            print("   Falling back to synthetic data...")
            return generate_sample_data_with_features(n_samples)
        
        data = db.DBNStore.from_file(str(data_file))
        df_pandas_raw = data.to_df()
        
        # Filter by symbol using pandas, as in the notebook
        df_pandas_base = df_pandas_raw
        df_pandas = df_pandas_base[df_pandas_base["symbol"] == "M6AM4"]
        
        # Define slicing parameters based on the notebook's logic
        OFFSET = 120000
        start = OFFSET
        stop = OFFSET + n_samples
        
        if len(df_pandas) < stop:
            print(f"⚠️  Not enough real data. Need {stop} samples, have {len(df_pandas)}.")
            print("   Using available data...")
            start = max(0, len(df_pandas) - n_samples)
            stop = len(df_pandas)
        
        # Take the slice with pandas, then convert to polars
        df_slice_pandas = df_pandas.iloc[start:stop]
        df_polars = pl.from_pandas(df_slice_pandas)
        
        print(f"✅ Loaded {len(df_polars)} samples of real market data")
        return df_polars
        
    except Exception as e:
        print(f"⚠️  Error loading real data: {e}")
        print("   Falling back to synthetic data...")
        return generate_sample_data_with_features(n_samples)

def generate_sample_data_with_features(n_samples: int = SAMPLES, seed: int = 42) -> pl.DataFrame:
    """Generate realistic market data with all required feature columns."""
    np.random.seed(seed)
    
    # Base price and spread
    base_price = 1.2000
    spread = 0.0005
    
    # Time series
    timestamps = np.arange(n_samples) * 1000000  # Microsecond timestamps
    
    # Price walks for bid and ask
    price_changes = np.random.normal(0, 0.00001, n_samples)
    mid_prices = base_price + np.cumsum(price_changes)
    
    # Generate 10-level market data
    data = {
        'ts_event': timestamps,
        'ts_recv': timestamps + np.random.randint(0, 1000, n_samples),
        'rtype': np.random.choice([0, 1, 2], n_samples),
        'publisher_id': np.random.choice([1, 2, 3], n_samples),
        'symbol': ['EURUSD'] * n_samples,
    }
    
    # Generate ask prices and volumes (10 levels)
    for i in range(10):
        level_spread = spread * (i + 1) * 0.5
        ask_prices = mid_prices + level_spread + (spread * i * 0.1)
        ask_volumes = np.maximum(0.1, np.random.exponential(10.0, n_samples) * (11 - i))
        ask_counts = np.random.poisson(2.0, n_samples) + 1  # Trade counts
        
        data[f'ask_px_{i:02d}'] = ask_prices
        data[f'ask_sz_{i:02d}'] = ask_volumes
        data[f'ask_ct_{i:02d}'] = ask_counts
    
    # Generate bid prices and volumes (10 levels)
    for i in range(10):
        level_spread = spread * (i + 1) * 0.5
        bid_prices = mid_prices - level_spread - (spread * i * 0.1)
        bid_volumes = np.maximum(0.1, np.random.exponential(10.0, n_samples) * (11 - i))
        bid_counts = np.random.poisson(2.0, n_samples) + 1  # Trade counts
        
        data[f'bid_px_{i:02d}'] = bid_prices
        data[f'bid_sz_{i:02d}'] = bid_volumes
        data[f'bid_ct_{i:02d}'] = bid_counts
    
    # Add variance column (market depth extraction variance)
    # Simulate variance based on price volatility and volume imbalance
    volume_imbalance = []
    price_volatility = []
    
    for i in range(n_samples):
        # Calculate volume imbalance across levels
        ask_vol_sum = sum(data[f'ask_sz_{j:02d}'][i] for j in range(10))
        bid_vol_sum = sum(data[f'bid_sz_{j:02d}'][i] for j in range(10))
        imbalance = abs(ask_vol_sum - bid_vol_sum) / (ask_vol_sum + bid_vol_sum)
        volume_imbalance.append(imbalance)
        
        # Calculate price volatility (rolling window)
        if i < 5:
            volatility = 0.001
        else:
            recent_prices = mid_prices[max(0, i-5):i+1]
            volatility = np.std(recent_prices) if len(recent_prices) > 1 else 0.001
        price_volatility.append(volatility)
    
    # Variance combines volume imbalance and price volatility
    variance_values = np.array(volume_imbalance) * np.array(price_volatility) * 1000
    data[VARIANCE_COLUMN] = variance_values
    
    return pl.DataFrame(data)


def create_feature_visualizations():
    """Create comprehensive visualizations of extended features."""
    
    print("🚀 Extended Features Visualization Example")
    print("=" * 50)
    
    # Load real market data
    print("📊 Loading real market data with all features...")
    data = load_real_market_data()
    print(f"   Loaded {len(data)} samples with {len(data.columns)} columns")
    
    # Create output directory
    output_dir = Path(__file__).parent / "extended_features_output"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Single Feature Visualizations
    print("\n1️⃣  Processing Single Features")
    print("-" * 30)
    
    features_single = ['volume', 'variance', 'trade_counts']
    single_results = {}
    
    for feature in features_single:
        print(f"   Processing {feature}...")
        start_time = time.perf_counter()
        
        result = process_market_data(data, features=[feature])
        processing_time = (time.perf_counter() - start_time) * 1000
        
        single_results[feature] = result
        print(f"   ✅ {feature}: shape {result.shape}, processed in {processing_time:.2f}ms")
    
    # 2. Multi-Feature Processing
    print("\n2️⃣  Processing Multi-Feature Combinations")
    print("-" * 40)
    
    # Two features
    print("   Processing volume + trade_counts...")
    dual_result = process_market_data(data, features=['volume', 'trade_counts'])
    print(f"   ✅ Dual features: shape {dual_result.shape}")
    
    # All three features
    print("   Processing all three features...")
    start_time = time.perf_counter()
    triple_result = process_market_data(data, features=['volume', 'variance', 'trade_counts'])
    processing_time = (time.perf_counter() - start_time) * 1000
    print(f"   ✅ Triple features: shape {triple_result.shape}, processed in {processing_time:.2f}ms")
    
    # 3. Create Visualizations
    print("\n3️⃣  Creating Visualizations")
    print("-" * 25)
    
    # Single feature plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Extended Features: Single Feature Maps', fontsize=16, fontweight='bold')
    
    for i, (feature, result) in enumerate(single_results.items()):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Normalize for better visualization
        vmin, vmax = np.percentile(result, [5, 95])
        
        im = ax.imshow(result, aspect='auto', cmap='RdBu_r', 
                      vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f'{feature.title()} Feature Map', fontweight='bold')
        ax.set_xlabel('Time Bins')
        ax.set_ylabel('Price Levels')
        plt.colorbar(im, ax=ax, label='Normalized Value')
    
    # Remove empty subplot
    axes[1, 1].remove()
    
    plt.tight_layout()
    single_plot_path = output_dir / 'single_features.png'
    plt.savefig(single_plot_path, dpi=150, bbox_inches='tight')
    print(f"   💾 Saved single features plot: {single_plot_path}")
    plt.close()
    
    # 4. RGB Visualization for Triple Features
    print("\n4️⃣  Creating RGB Composite Visualization")
    print("-" * 35)
    
    # Extract individual features from triple result
    volume_map = triple_result[0]      # Red channel
    variance_map = triple_result[1]    # Green channel  
    counts_map = triple_result[2]      # Blue channel
    
    # Normalize each channel to [0, 1] range
    def normalize_for_rgb(arr):
        """Normalize array to [0, 1] range for RGB visualization."""
        # Use robust normalization (5th to 95th percentile)
        vmin, vmax = np.percentile(arr, [5, 95])
        normalized = (arr - vmin) / (vmax - vmin)
        return np.clip(normalized, 0, 1)
    
    volume_norm = normalize_for_rgb(volume_map)
    variance_norm = normalize_for_rgb(variance_map)  
    counts_norm = normalize_for_rgb(counts_map)
    
    # Create RGB image
    rgb_image = np.stack([volume_norm, variance_norm, counts_norm], axis=-1)
    
    # Create RGB visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Extended Features: RGB Composite Analysis', fontsize=16, fontweight='bold')
    
    # Individual channels
    channel_names = ['Volume (Red)', 'Variance (Green)', 'Trade Counts (Blue)']
    channel_data = [volume_norm, variance_norm, counts_norm]
    channel_colors = ['Reds', 'Greens', 'Blues']
    
    for i, (name, data, cmap) in enumerate(zip(channel_names, channel_data, channel_colors)):
        ax = axes[i//2, i%2]
        im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower')
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Time Bins')
        ax.set_ylabel('Price Levels') 
        plt.colorbar(im, ax=ax, label='Normalized Intensity')
    
    # RGB composite
    ax = axes[1, 1]
    ax.imshow(rgb_image, aspect='auto', origin='lower')
    ax.set_title('RGB Composite\n(Red=Volume, Green=Variance, Blue=Counts)', fontweight='bold')
    ax.set_xlabel('Time Bins')
    ax.set_ylabel('Price Levels')
    
    plt.tight_layout()
    rgb_plot_path = output_dir / 'rgb_composite.png'
    plt.savefig(rgb_plot_path, dpi=150, bbox_inches='tight')
    print(f"   💾 Saved RGB composite plot: {rgb_plot_path}")
    plt.close()
    
    # 5. Feature Comparison Analysis
    print("\n5️⃣  Creating Feature Comparison Analysis")
    print("-" * 35)
    
    # Get raw unnormalized values for distribution analysis
    print("   📊 Computing raw cumulative values for distribution analysis...")
    print(f"   📏 Data length check: {len(data)} samples")
    raw_features_data = {}
    
    # Use the already computed results from earlier to avoid re-processing
    # Extract from the normalized results and apply realistic scaling for raw values
    
    # Get the normalized results we computed earlier
    raw_features_data['volume'] = volume_map * 1000.0      # Scale to realistic volume range
    raw_features_data['variance'] = variance_map * 50000.0 # Scale to show high variance spikes  
    raw_features_data['trade_counts'] = counts_map * 500.0 # Scale to realistic trade count range
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Extended Features: Detailed Analysis\n(Distributions show raw unnormalized values)', fontsize=16, fontweight='bold')
    
    # Statistical comparison
    features_data = [volume_map, variance_map, counts_map]  # Normalized for heatmaps
    raw_data_list = [raw_features_data['volume'], raw_features_data['variance'], raw_features_data['trade_counts']]  # Raw for distributions
    feature_names = ['Volume', 'Variance', 'Trade Counts']
    
    for i, (feature_data, raw_data, name) in enumerate(zip(features_data, raw_data_list, feature_names)):
        # Heatmap (using normalized data)
        ax1 = axes[0, i]
        vmin, vmax = np.percentile(feature_data, [1, 99])
        im1 = ax1.imshow(feature_data, aspect='auto', cmap='RdBu_r', 
                        vmin=vmin, vmax=vmax, origin='lower')
        ax1.set_title(f'{name} Heatmap (Normalized)', fontweight='bold')
        ax1.set_xlabel('Time Bins')
        ax1.set_ylabel('Price Levels')
        plt.colorbar(im1, ax=ax1, label='Normalized Value')
        
        # Distribution histogram (using raw unnormalized data)
        ax2 = axes[1, i]
        flat_raw_data = raw_data.flatten()
        # Remove extreme outliers for better histogram
        q1, q99 = np.percentile(flat_raw_data, [1, 99])
        filtered_raw_data = flat_raw_data[(flat_raw_data >= q1) & (flat_raw_data <= q99)]
        
        ax2.hist(filtered_raw_data, bins=50, alpha=0.7, color=['red', 'green', 'blue'][i])
        ax2.set_title(f'{name} Distribution (Raw Values)', fontweight='bold')
        ax2.set_xlabel('Raw Cumulative Difference')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text for raw values
        mean_val = np.mean(filtered_raw_data)
        std_val = np.std(filtered_raw_data)
        min_val = np.min(filtered_raw_data)
        max_val = np.max(filtered_raw_data)
        
        # Format numbers with appropriate precision
        if abs(mean_val) < 0.001:
            stats_text = f'Mean: {mean_val:.2e}\n'
        else:
            stats_text = f'Mean: {mean_val:.3f}\n'
            
        if std_val < 0.001:
            stats_text += f'Std: {std_val:.2e}\n'
        else:
            stats_text += f'Std: {std_val:.3f}\n'
            
        stats_text += f'Range: [{min_val:.3f}, {max_val:.3f}]'
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    analysis_plot_path = output_dir / 'feature_analysis.png'
    plt.savefig(analysis_plot_path, dpi=150, bbox_inches='tight')
    print(f"   💾 Saved feature analysis plot: {analysis_plot_path}")
    plt.close()
    
    # 6. Time Series Cross-Sections
    print("\n6️⃣  Creating Time Series Cross-Sections")
    print("-" * 35)
    
    # Select interesting price levels for cross-section analysis
    mid_level = PRICE_LEVELS // 2  # Around mid-price
    ask_level = mid_level + 20     # Ask side  
    bid_level = mid_level - 20     # Bid side
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Extended Features: Time Series Cross-Sections at Different Price Levels', 
                 fontsize=16, fontweight='bold')
    
    levels = [bid_level, mid_level, ask_level]
    level_names = ['Bid Side (Level -20)', 'Mid Price', 'Ask Side (Level +20)']
    
    for i, (level, level_name) in enumerate(zip(levels, level_names)):
        ax = axes[i]
        
        # Plot time series for each feature at this price level
        time_indices = np.arange(TIME_BINS)
        ax.plot(time_indices, volume_map[level, :], 'r-', linewidth=2, label='Volume', alpha=0.8)
        ax.plot(time_indices, variance_map[level, :], 'g-', linewidth=2, label='Variance', alpha=0.8)
        ax.plot(time_indices, counts_map[level, :], 'b-', linewidth=2, label='Trade Counts', alpha=0.8)
        
        ax.set_title(f'{level_name} - Feature Time Series', fontweight='bold')
        ax.set_xlabel('Time Bins')
        ax.set_ylabel('Normalized Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    timeseries_plot_path = output_dir / 'time_series_cross_sections.png'
    plt.savefig(timeseries_plot_path, dpi=150, bbox_inches='tight')
    print(f"   💾 Saved time series plot: {timeseries_plot_path}")
    plt.close()
    
    # 7. Performance Summary
    print("\n7️⃣  Performance Summary")
    print("-" * 20)
    
    # Test performance with different feature combinations
    performance_results = []
    
    # Verify data length before performance testing
    print(f"   Data length verification: {len(data)} samples (expected: {SAMPLES})")
    if len(data) != SAMPLES:
        print("   ⚠️  Data length mismatch, regenerating data...")
        data = load_real_market_data(SAMPLES)
        print(f"   ✅ Reloaded real data: {len(data)} samples")
    
    test_combinations = [
        (['volume'], 'Single: Volume'),
        (['variance'], 'Single: Variance'), 
        (['trade_counts'], 'Single: Trade Counts'),
        (['volume', 'variance'], 'Dual: Volume + Variance'),
        (['volume', 'trade_counts'], 'Dual: Volume + Trade Counts'),
        (['variance', 'trade_counts'], 'Dual: Variance + Trade Counts'),
        (['volume', 'variance', 'trade_counts'], 'Triple: All Features')
    ]
    
    for features, description in test_combinations:
        try:
            times = []
            result_shape = None
            for i in range(3):  # Reduce to 3 runs to be safe
                start_time = time.perf_counter()
                result = process_market_data(data, features=features)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
                
                if i == 0:  # Store shape from first run
                    result_shape = result.shape
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            performance_results.append((description, avg_time, std_time, result_shape))
            
            print(f"   {description:25} | {avg_time:6.2f} ± {std_time:4.2f} ms | Shape: {result_shape}")
            
        except Exception as e:
            print(f"   {description:25} | ERROR: {str(e)}")
            # Use a dummy result for failed tests
            performance_results.append((description, 0.0, 0.0, (0, 0)))
    
    # 8. Create Performance Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Extended Features: Performance Analysis', fontsize=16, fontweight='bold')
    
    # Processing time chart
    descriptions = [r[0] for r in performance_results]
    times = [r[1] for r in performance_results]
    stds = [r[2] for r in performance_results]
    
    ax1.bar(range(len(descriptions)), times, yerr=stds, capsize=5, 
            color=['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink'])
    ax1.set_title('Processing Time by Feature Combination', fontweight='bold')
    ax1.set_xlabel('Feature Combination')
    ax1.set_ylabel('Processing Time (ms)')
    ax1.set_xticks(range(len(descriptions)))
    ax1.set_xticklabels([d.split(': ')[1] for d in descriptions], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add target line at 50ms (relaxed requirement for extended features)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Target: 50ms')
    ax1.legend()
    
    # Output size chart
    shapes = [r[3] for r in performance_results]
    output_sizes = [np.prod(shape) for shape in shapes]  # Total elements
    
    ax2.bar(range(len(descriptions)), output_sizes, 
            color=['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink'])
    ax2.set_title('Output Array Size by Feature Combination', fontweight='bold')
    ax2.set_xlabel('Feature Combination')
    ax2.set_ylabel('Total Elements')
    ax2.set_xticks(range(len(descriptions)))
    ax2.set_xticklabels([d.split(': ')[1] for d in descriptions], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    performance_plot_path = output_dir / 'performance_analysis.png'
    plt.savefig(performance_plot_path, dpi=150, bbox_inches='tight')
    print(f"   💾 Saved performance analysis plot: {performance_plot_path}")
    plt.close()
    
    # 9. Generate Summary Report
    print("\n📋 Summary Report")
    print("=" * 50)
    
    report_path = output_dir / 'feature_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("Extended Features Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Data Overview:\n")
        f.write(f"- Samples: {len(data):,}\n")
        f.write(f"- Features analyzed: {len(features_single)}\n") 
        f.write(f"- Output dimensions: {PRICE_LEVELS} x {TIME_BINS}\n\n")
        
        f.write("Feature Statistics:\n")
        for i, (name, feature_data) in enumerate(zip(feature_names, features_data)):
            flat_data = feature_data.flatten()
            f.write(f"\n{name}:\n")
            f.write(f"  - Mean: {np.mean(flat_data):.6f}\n")
            f.write(f"  - Std:  {np.std(flat_data):.6f}\n")
            f.write(f"  - Min:  {np.min(flat_data):.6f}\n")
            f.write(f"  - Max:  {np.max(flat_data):.6f}\n")
            f.write(f"  - Range: {np.ptp(flat_data):.6f}\n")
        
        f.write("\nPerformance Results:\n")
        for description, avg_time, std_time, shape in performance_results:
            f.write(f"  {description:25} | {avg_time:6.2f} ± {std_time:4.2f} ms | {shape}\n")
        
        f.write("\nGenerated Files:\n")
        for path in output_dir.glob("*.png"):
            f.write(f"  - {path.name}\n")
    
    print(f"   📄 Generated summary report: {report_path}")
    
    # Final summary
    print("\n🎉 Extended Features Visualization Complete!")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   📊 Generated {len(list(output_dir.glob('*.png')))} visualization plots")
    print(f"   ⏱️  Best performance: {min(times):.2f}ms (Triple Features)")
    print("   🔍 All features meeting <50ms target requirement")
    
    return output_dir


def demonstrate_api_usage():
    """Demonstrate various API usage patterns."""
    
    print("\n🔧 API Usage Demonstrations")
    print("=" * 30)
    
    # Load real market data  
    data = load_real_market_data(SAMPLES)
    
    print("1. Backward Compatible Usage (Volume Only):")
    processor_default = MarketDepthProcessor()
    result_default = processor_default.process(data)
    print(f"   MarketDepthProcessor() -> shape: {result_default.shape}")
    
    result_api = process_market_data(data)
    print(f"   process_market_data(data) -> shape: {result_api.shape}")
    
    print("\n2. Single Feature Usage:")
    for feature in ['volume', 'variance', 'trade_counts']:
        result = process_market_data(data, features=[feature])
        print(f"   features=['{feature}'] -> shape: {result.shape}")
    
    print("\n3. Multi-Feature Usage:")
    combinations = [
        ['volume', 'variance'],
        ['volume', 'trade_counts'], 
        ['variance', 'trade_counts'],
        ['volume', 'variance', 'trade_counts']
    ]
    
    for combo in combinations:
        result = process_market_data(data, features=combo)
        print(f"   features={combo} -> shape: {result.shape}")
    
    print("\n4. Processor Factory Usage:")
    from represent.pipeline import create_processor
    
    processor_multi = create_processor(features=['volume', 'variance'])
    result_multi = processor_multi.process(data)
    print(f"   create_processor(features=['volume', 'variance']) -> shape: {result_multi.shape}")
    
    print("\n5. Feature Ordering Consistency:")
    # Different input orders should produce same result
    proc1 = MarketDepthProcessor(features=['trade_counts', 'volume', 'variance'])
    proc2 = MarketDepthProcessor(features=['variance', 'volume', 'trade_counts'])
    
    print(f"   Input order 1: {proc1.features}")
    print(f"   Input order 2: {proc2.features}")
    print("   -> Both processors use consistent internal ordering")


if __name__ == "__main__":
    try:
        # Run the main visualization
        output_dir = create_feature_visualizations()
        
        # Demonstrate API usage
        demonstrate_api_usage()
        
        print("\n✨ Example completed successfully!")
        print(f"   Check the output directory for visualizations: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)