#!/usr/bin/env python3
"""
Comprehensive Report Generator
=============================

Creates comprehensive HTML reports with plots for each processing approach.
Moves all outputs to examples/ folder and removes unnecessary columns.
"""

import base64
import sys
import time
from io import BytesIO
from pathlib import Path

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def analyze_and_clean_dataset(dataset_path: Path, output_dir: Path) -> dict:
    """
    Analyze dataset and create cleaned version without volume_representation.

    Returns analysis results for report generation.
    """
    print(f"📊 Analyzing {dataset_path.name}...")

    # Load dataset
    df = pl.read_parquet(str(dataset_path))

    # Basic analysis
    analysis = {
        'file_name': dataset_path.name,
        'total_samples': len(df),
        'total_columns': len(df.columns),
        'file_size_mb': dataset_path.stat().st_size / 1024 / 1024,
        'symbol': df['symbol'][0] if 'symbol' in df.columns else 'Unknown',
        'has_volume_rep': 'volume_representation' in df.columns,
        'classification_distribution': {},
        'uniformity_achieved': False,
        'feature_columns': [col for col in df.columns if col.endswith('_representation')]
    }

    # Classification analysis
    if 'classification_label' in df.columns:
        class_dist = df['classification_label'].value_counts().sort('classification_label')
        total_samples = len(df)

        for row in class_dist.iter_rows():
            label, count = row
            percentage = (count / total_samples) * 100
            analysis['classification_distribution'][int(label)] = {
                'count': int(count),
                'percentage': percentage
            }

        # Check uniformity
        counts = [row[1] for row in class_dist.iter_rows()]
        std_dev = np.std(counts)
        mean_count = np.mean(counts)
        uniformity_ratio = std_dev / mean_count if mean_count > 0 else float('inf')
        analysis['uniformity_achieved'] = uniformity_ratio < 0.15
        analysis['uniformity_ratio'] = uniformity_ratio

    # Remove volume_representation if it exists (it's just for plotting, not needed in final dataset)
    if analysis['has_volume_rep']:
        print("   ⚠️  Removing volume_representation column (unnecessary for ML training)")
        columns_to_keep = [col for col in df.columns if not col.endswith('_representation')]
        df_cleaned = df.select(columns_to_keep)

        # Save cleaned version
        cleaned_path = output_dir / f"{dataset_path.stem}_cleaned.parquet"
        df_cleaned.write_parquet(str(cleaned_path))

        analysis['cleaned_file'] = cleaned_path
        analysis['cleaned_size_mb'] = cleaned_path.stat().st_size / 1024 / 1024
        analysis['size_reduction'] = (analysis['file_size_mb'] - analysis['cleaned_size_mb']) / analysis['file_size_mb'] * 100

        print(f"   ✅ Cleaned dataset: {analysis['cleaned_size_mb']:.1f} MB ({analysis['size_reduction']:.1f}% reduction)")

    return analysis


def create_visualization_plot(analysis: dict, output_dir: Path) -> str:
    """Create visualization plot for the analysis."""
    if not analysis['classification_distribution']:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{analysis['symbol']} - Dataset Analysis", fontsize=14, fontweight='bold')

    # Classification distribution
    labels = list(analysis['classification_distribution'].keys())
    counts = [analysis['classification_distribution'][label]['count'] for label in labels]
    percentages = [analysis['classification_distribution'][label]['percentage'] for label in labels]

    # Bar chart
    bars = ax1.bar(labels, counts, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Classification Label')
    ax1.set_ylabel('Sample Count')
    ax1.set_title('Classification Distribution')
    ax1.grid(True, alpha=0.3)

    # Add count labels
    for bar, count in zip(bars, counts, strict=False):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9)

    # Pie chart
    ax2.pie(percentages, labels=[f'Class {label}' for label in labels],
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Proportions')

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"{analysis['symbol']}_analysis.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')

    # Convert to base64 for HTML
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_b64 = base64.b64encode(img_buffer.read()).decode()
    plt.close(fig)

    return img_b64


def generate_html_report(analyses: list, approach_name: str, output_dir: Path) -> Path:
    """Generate comprehensive HTML report."""

    # Create visualizations
    plot_htmls = []
    for analysis in analyses:
        img_b64 = create_visualization_plot(analysis, output_dir)
        if img_b64:
            plot_html = f"""
            <div class="dataset-analysis">
                <h3>{analysis['symbol']} Dataset</h3>
                <div class="metrics">
                    <div class="metric">
                        <span class="metric-label">Samples:</span>
                        <span class="metric-value">{analysis['total_samples']:,}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Original Size:</span>
                        <span class="metric-value">{analysis['file_size_mb']:.1f} MB</span>
                    </div>
                    {f'''<div class="metric">
                        <span class="metric-label">Cleaned Size:</span>
                        <span class="metric-value">{analysis['cleaned_size_mb']:.1f} MB</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Size Reduction:</span>
                        <span class="metric-value">{analysis['size_reduction']:.1f}%</span>
                    </div>''' if analysis.get('cleaned_file') else ''}
                    <div class="metric">
                        <span class="metric-label">Uniform Distribution:</span>
                        <span class="metric-value">{'✅ Yes' if analysis['uniformity_achieved'] else '❌ No'}</span>
                    </div>
                </div>
                <div class="visualization">
                    <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto;">
                </div>
            </div>
            """
            plot_htmls.append(plot_html)

    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{approach_name} - Processing Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0; padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px; margin: 0 auto; background: white;
                border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white; padding: 30px; text-align: center;
            }}
            .header h1 {{ margin: 0; font-size: 2.5em; font-weight: bold; }}
            .header p {{ margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }}
            .content {{ padding: 30px; }}
            .dataset-analysis {{
                margin-bottom: 40px; padding: 20px; background: #f8f9fa;
                border-radius: 10px; border-left: 5px solid #3498db;
            }}
            .dataset-analysis h3 {{ color: #2c3e50; margin-top: 0; }}
            .metrics {{
                display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px; margin-bottom: 20px;
            }}
            .metric {{
                display: flex; justify-content: space-between;
                padding: 10px; background: white; border-radius: 5px;
            }}
            .metric-label {{ font-weight: 600; color: #2c3e50; }}
            .metric-value {{ font-weight: bold; color: #3498db; }}
            .visualization {{ text-align: center; margin: 20px 0; }}
            .summary {{
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white; padding: 30px; text-align: center;
            }}
            .summary h2 {{ color: white; margin-bottom: 20px; }}
            .timestamp {{
                text-align: center; padding: 20px; color: #666;
                background: #f8f9fa;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 {approach_name}</h1>
                <p>Processing Report with Analysis & Visualizations</p>
            </div>

            <div class="content">
                {''.join(plot_htmls)}
            </div>

            <div class="summary">
                <h2>📈 Summary</h2>
                <p>Generated {len(analyses)} dataset analyses with visualizations</p>
                <p>All datasets cleaned and optimized for ML training</p>
                <p>Unnecessary representation columns removed</p>
            </div>

            <div class="timestamp">
                Generated on {time.strftime('%Y-%m-%d at %H:%M:%S')} using the represent package
            </div>
        </div>
    </body>
    </html>
    """

    report_path = output_dir / f"{approach_name.lower().replace(' ', '_')}_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return report_path


def main():
    """Process all existing outputs and create comprehensive reports."""
    print("🔧 Creating Comprehensive Reports for All Approaches")
    print("=" * 60)

    # Define output directories to process
    output_dirs = {
        "Fast Demo": "examples/fast_demo_output",
        "Demo Output": "examples/demo_output",
        "Fast Output": "examples/fast_output",
        # Add more as they're created
    }

    for approach_name, output_path in output_dirs.items():
        output_dir = Path(output_path)
        if not output_dir.exists():
            print(f"⚠️  {approach_name}: {output_path} does not exist, skipping...")
            continue

        print(f"\n📊 Processing {approach_name}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find dataset files
        dataset_files = list(output_dir.glob("*_dataset.parquet"))
        if not dataset_files:
            print(f"   ❌ No dataset files found in {output_dir}")
            continue

        # Analyze each dataset
        analyses = []
        for dataset_file in dataset_files:
            analysis = analyze_and_clean_dataset(dataset_file, output_dir)
            analyses.append(analysis)

        # Generate HTML report
        if analyses:
            report_path = generate_html_report(analyses, approach_name, output_dir)
            print(f"   ✅ Generated report: {report_path}")
            print(f"   📊 Analyzed {len(analyses)} datasets")

            # Show summary
            total_original_size = sum(a['file_size_mb'] for a in analyses)
            total_cleaned_size = sum(a.get('cleaned_size_mb', a['file_size_mb']) for a in analyses)
            total_reduction = (total_original_size - total_cleaned_size) / total_original_size * 100 if total_original_size > 0 else 0

            print(f"   💾 Size reduction: {total_original_size:.1f} MB → {total_cleaned_size:.1f} MB ({total_reduction:.1f}% saved)")

    print("\n🎉 COMPREHENSIVE REPORTS COMPLETE!")
    print("📁 All reports and cleaned datasets are in examples/ subdirectories")
    print("📊 Each approach now has:")
    print("   • Cleaned parquet files (no unnecessary columns)")
    print("   • Visualization plots (PNG files)")
    print("   • Comprehensive HTML reports")


if __name__ == "__main__":
    main()
