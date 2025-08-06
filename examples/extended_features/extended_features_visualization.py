"""
Extended features visualization demonstration.

Shows comprehensive analysis of all three feature types (volume, variance, trade_counts)
with detailed visualizations and analysis using the new RepresentConfig system.
"""

import databento as db
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pathlib import Path
from represent import RepresentConfig
from represent.pipeline import process_market_data


def create_rgb_composite(volume_data, variance_data, trade_counts_data):
    """Create RGB composite image from three features."""
    
    # Normalize each feature to [0, 1] for RGB
    def normalize_for_rgb(data):
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        return np.zeros_like(data)
    
    red = normalize_for_rgb(volume_data)
    green = normalize_for_rgb(variance_data)
    blue = normalize_for_rgb(trade_counts_data)
    
    return np.stack([red, green, blue], axis=-1)


def analyze_extended_features():
    """Comprehensive analysis of extended features."""
    
    print("🔬 Extended Features Analysis")
    print("="*35)
    
    # Create configuration with all features
    config = RepresentConfig(
        currency="AUDUSD",
        lookback_rows=5000,      # Configurable
        lookforward_input=3000,  # Configurable
        features=["volume", "variance", "trade_counts"]  # All features
    )
    
    print("📊 Configuration:")
    print(f"   Currency: {config.currency}")
    print(f"   Features: {config.features}")
    print(f"   Lookback: {config.lookback_rows}")
    print(f"   Lookforward: {config.lookforward_input}")
    
    try:
        # Load and process data
        print("\n📊 Loading market data...")
        data = db.DBNStore.from_file("data/glbx-mdp3-20240405.mbp-10.dbn.zst")
        df_pandas = data.to_df()
        df_symbol = df_pandas[df_pandas["symbol"] == "M6AM4"].iloc[120000:170000]
        df_polars = pl.from_pandas(df_symbol)
        
        print(f"✅ Loaded {len(df_polars):,} samples")
        
        # Process individual features
        print("\n⚡ Processing individual features...")
        volume_data = process_market_data(df_polars, features=["volume"])
        variance_data = process_market_data(df_polars, features=["variance"])
        trade_counts_data = process_market_data(df_polars, features=["trade_counts"])
        
        # Process multi-feature tensor
        multi_feature_tensor = process_market_data(df_polars, features=config.features)
        
        print("✅ Individual feature shapes:")
        print(f"   Volume: {volume_data.shape}")
        print(f"   Variance: {variance_data.shape}")
        print(f"   Trade counts: {trade_counts_data.shape}")
        print(f"   Multi-feature tensor: {multi_feature_tensor.shape}")
        
        # Create comprehensive visualization
        create_extended_features_visualization(
            volume_data, variance_data, trade_counts_data, multi_feature_tensor
        )
        
        # Generate analysis report
        generate_feature_analysis_report(
            volume_data, variance_data, trade_counts_data, multi_feature_tensor, config
        )
        
        print("\n✅ Extended features analysis complete!")
        
    except FileNotFoundError:
        print("❌ DBN file not found. Please ensure data files are available.")
        print("💡 Expected: data/glbx-mdp3-20240405.mbp-10.dbn.zst")


def create_extended_features_visualization(volume_data, variance_data, trade_counts_data, multi_tensor):
    """Create comprehensive extended features visualization."""
    
    output_dir = Path("examples/extended_features/extended_features_output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Individual features comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Extended Features Analysis", fontsize=16)
    
    features = [
        (volume_data, "Volume Features", "Volume differences"),
        (variance_data, "Variance Features", "Price variance patterns"),
        (trade_counts_data, "Trade Count Features", "Trade frequency patterns")
    ]
    
    # Top row: Individual features
    for i, (data, title, subtitle) in enumerate(features):
        im = axes[0, i].imshow(data, cmap="RdBu", aspect="auto")
        axes[0, i].set_title(f"{title}\n{subtitle}")
        axes[0, i].set_xlabel("Time Bins")
        axes[0, i].set_ylabel("Price Levels")
        plt.colorbar(im, ax=axes[0, i])
    
    # Bottom row: Multi-feature tensor channels
    for i in range(3):
        im = axes[1, i].imshow(multi_tensor[i], cmap="RdBu", aspect="auto")
        axes[1, i].set_title(f"Multi-Feature Tensor\nChannel {i} ({features[i][0]})")
        axes[1, i].set_xlabel("Time Bins")
        axes[1, i].set_ylabel("Price Levels")
        plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. RGB composite visualization
    rgb_composite = create_rgb_composite(volume_data, variance_data, trade_counts_data)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_composite, aspect="auto")
    plt.title("RGB Composite: Red=Volume, Green=Variance, Blue=Trade Counts")
    plt.xlabel("Time Bins")
    plt.ylabel("Price Levels")
    
    # Add RGB legend
    plt.figtext(0.02, 0.98, 
               "RGB Channels:\n🔴 Red = Volume\n🟢 Green = Variance\n🔵 Blue = Trade Counts",
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "rgb_composite.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature cross-sections
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Time Series Cross-Sections (Middle Price Level)", fontsize=14)
    
    middle_price = volume_data.shape[0] // 2
    time_axis = np.arange(volume_data.shape[1])
    
    axes[0].plot(time_axis, volume_data[middle_price, :], color='red', label='Volume')
    axes[0].set_title("Volume Feature Time Series")
    axes[0].set_ylabel("Normalized Volume")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_axis, variance_data[middle_price, :], color='green', label='Variance')
    axes[1].set_title("Variance Feature Time Series")
    axes[1].set_ylabel("Normalized Variance")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time_axis, trade_counts_data[middle_price, :], color='blue', label='Trade Counts')
    axes[2].set_title("Trade Counts Feature Time Series")
    axes[2].set_xlabel("Time Bins")
    axes[2].set_ylabel("Normalized Trade Counts")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "time_series_cross_sections.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved in: {output_dir}")


def generate_feature_analysis_report(volume_data, variance_data, trade_counts_data, multi_tensor, config):
    """Generate detailed feature analysis report."""
    
    output_dir = Path("examples/extended_features/extended_features_output")
    
    # Calculate statistics
    features_stats = []
    for name, data in [("Volume", volume_data), ("Variance", variance_data), ("Trade Counts", trade_counts_data)]:
        stats = {
            'name': name,
            'shape': data.shape,
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'non_zero_count': int(np.count_nonzero(data)),
            'density': float(np.count_nonzero(data) / data.size * 100)
        }
        features_stats.append(stats)
    
    # Generate report
    report = []
    report.append("Extended Features Analysis Report")
    report.append("="*40)
    report.append("\nConfiguration:")
    report.append(f"  Currency: {config.currency}")
    report.append(f"  Features: {', '.join(config.features)}")
    report.append(f"  Lookback rows: {config.lookback_rows}")
    report.append(f"  Lookforward input: {config.lookforward_input}")
    report.append(f"  Batch size: {config.batch_size}")
    
    report.append("\nMulti-Feature Tensor:")
    report.append(f"  Shape: {multi_tensor.shape}")
    report.append(f"  Features dimension: {multi_tensor.shape[0]}")
    report.append(f"  Spatial dimensions: {multi_tensor.shape[1]}×{multi_tensor.shape[2]}")
    report.append(f"  Total parameters: {multi_tensor.size:,}")
    
    report.append("\nIndividual Feature Analysis:")
    for stats in features_stats:
        report.append(f"\n  {stats['name']} Feature:")
        report.append(f"    Shape: {stats['shape']}")
        report.append(f"    Value range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        report.append(f"    Mean: {stats['mean']:.6f}")
        report.append(f"    Standard deviation: {stats['std']:.6f}")
        report.append(f"    Non-zero pixels: {stats['non_zero_count']:,} ({stats['density']:.2f}%)")
    
    report.append("\nFeature Comparison:")
    report.append(f"  Most active feature: {max(features_stats, key=lambda x: x['density'])['name']}")
    report.append(f"  Highest variance: {max(features_stats, key=lambda x: x['std'])['name']}")
    report.append(f"  Largest range: {max(features_stats, key=lambda x: x['max'] - x['min'])['name']}")
    
    # Save report
    with open(output_dir / "feature_analysis_report.txt", "w") as f:
        f.write("\n".join(report))
    
    print(f"📄 Analysis report saved: {output_dir}/feature_analysis_report.txt")


if __name__ == "__main__":
    analyze_extended_features()