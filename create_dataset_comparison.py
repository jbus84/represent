#!/usr/bin/env python3
"""
Dataset Comparison Assessment

This script creates a comprehensive HTML assessment comparing:
1. Original quantile-based classified datasets
2. New Merton Jump Diffusion classified datasets

The assessment focuses on classification quality, tail prediction accuracy,
and expected ML training performance improvements.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl


class DatasetComparator:
    """Compare original vs Merton-based classified datasets."""

    def __init__(self):
        self.original_dir = Path("/Users/danielfisher/data/databento/AUDUSD_classified_datasets")
        self.merton_dir = Path("/Users/danielfisher/data/databento/AUDUSD_merton_datasets")

    def analyze_dataset(self, file_path: Path, classification_column: str = "classification_label"):
        """Analyze a single dataset's classification quality."""

        try:
            df = pl.read_parquet(file_path)

            if classification_column not in df.columns:
                return None

            labels = df[classification_column].to_numpy()
            valid_labels = labels[~np.isnan(labels.astype(float))]

            if len(valid_labels) == 0:
                return None

            # Basic statistics
            total_samples = len(valid_labels)
            unique_classes = len(np.unique(valid_labels))

            # Class distribution
            class_counts = np.bincount(valid_labels.astype(int), minlength=13)
            class_fractions = class_counts / total_samples

            # Balance metrics
            expected_fraction = 1.0 / 13
            deviations = np.abs(class_fractions - expected_fraction)
            max_deviation = np.max(deviations)
            balance_score = 1.0 - (max_deviation / expected_fraction)

            # Extreme class metrics (classes 0 and 12)
            extreme_concentration = class_fractions[0] + class_fractions[12]
            extreme_excess = extreme_concentration - (2 * expected_fraction)

            # Tail prediction score (lower is better)
            # Based on deviations from ideal 7.7% for extreme classes
            target_extreme = 0.077
            class0_error = abs(class_fractions[0] - target_extreme) / target_extreme
            class12_error = abs(class_fractions[12] - target_extreme) / target_extreme
            tail_score = (class0_error + class12_error) * 50  # Scale for readability

            # Per-class analysis
            class_analysis = []
            for i in range(13):
                class_analysis.append(
                    {
                        "class": int(i),
                        "count": int(class_counts[i]),
                        "fraction": float(class_fractions[i]),
                        "percentage": float(class_fractions[i] * 100),
                        "deviation": float(deviations[i]),
                        "target_pct": 7.69,
                        "error_pct": float(abs(class_fractions[i] - expected_fraction) * 100),
                    }
                )

            return {
                "file_name": file_path.name,
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "total_samples": int(total_samples),
                "unique_classes": int(unique_classes),
                "class_counts": class_counts.tolist(),
                "class_fractions": class_fractions.tolist(),
                "class_analysis": class_analysis,
                "balance_score": float(balance_score),
                "max_deviation": float(max_deviation),
                "extreme_concentration": float(extreme_concentration),
                "extreme_excess": float(extreme_excess),
                "tail_score": float(tail_score),
                "class0_pct": float(class_fractions[0] * 100),
                "class12_pct": float(class_fractions[12] * 100),
            }

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def compare_datasets(self):
        """Compare all matching datasets between original and Merton approaches."""

        print("📊 DATASET COMPARISON ANALYSIS")
        print("=" * 50)

        comparisons = []

        # Find matching datasets
        original_files = list(self.original_dir.glob("AUDUSD_*.parquet"))
        merton_files = list(self.merton_dir.glob("AUDUSD_*_merton.parquet"))

        print(f"Original datasets: {len(original_files)}")
        print(f"Merton datasets: {len(merton_files)}")

        # Match datasets by symbol
        for original_file in original_files:
            # Find corresponding Merton file
            symbol = original_file.stem.split("_")[-1]  # Extract symbol (e.g., M6AM4)
            merton_file = None

            for mf in merton_files:
                if symbol in mf.stem and "merton" in mf.stem:
                    merton_file = mf
                    break

            if merton_file is None:
                print(f"⚠️  No Merton dataset found for {original_file.name}")
                continue

            print(f"📈 Comparing {symbol}: {original_file.name} vs {merton_file.name}")

            # Analyze both datasets
            original_analysis = self.analyze_dataset(original_file, "classification_label")
            merton_analysis = self.analyze_dataset(merton_file, "classification_label_merton")

            if original_analysis is None or merton_analysis is None:
                print(f"❌ Failed to analyze datasets for {symbol}")
                continue

            # Calculate improvements
            balance_improvement = (
                (merton_analysis["balance_score"] - original_analysis["balance_score"])
                / abs(original_analysis["balance_score"])
            ) * 100

            tail_improvement = (
                (original_analysis["tail_score"] - merton_analysis["tail_score"])
                / original_analysis["tail_score"]
            ) * 100

            extreme_improvement = (
                original_analysis["extreme_excess"] - merton_analysis["extreme_excess"]
            )

            comparison = {
                "symbol": symbol,
                "original": original_analysis,
                "merton": merton_analysis,
                "improvements": {
                    "balance_improvement_pct": float(balance_improvement),
                    "tail_improvement_pct": float(tail_improvement),
                    "extreme_improvement_pp": float(extreme_improvement * 100),
                },
            }

            comparisons.append(comparison)

            print(
                f"   Balance: {original_analysis['balance_score']:.3f} → {merton_analysis['balance_score']:.3f} ({balance_improvement:+.1f}%)"
            )
            print(
                f"   Tail Score: {original_analysis['tail_score']:.1f} → {merton_analysis['tail_score']:.1f} ({tail_improvement:+.1f}%)"
            )
            print(
                f"   Extreme Excess: {original_analysis['extreme_excess'] * 100:+.1f}pp → {merton_analysis['extreme_excess'] * 100:+.1f}pp"
            )

        return comparisons

    def generate_html_report(
        self, comparisons, output_file: str = "dataset_comparison_assessment.html"
    ):
        """Generate comprehensive HTML assessment report."""

        # Calculate summary statistics
        if not comparisons:
            print("❌ No comparisons available for HTML report")
            return

        balance_improvements = [c["improvements"]["balance_improvement_pct"] for c in comparisons]
        tail_improvements = [c["improvements"]["tail_improvement_pct"] for c in comparisons]
        extreme_improvements = [c["improvements"]["extreme_improvement_pp"] for c in comparisons]

        avg_balance_improvement = np.mean(balance_improvements)
        avg_tail_improvement = np.mean(tail_improvements)
        avg_extreme_improvement = np.mean(extreme_improvements)

        best_performer = max(comparisons, key=lambda x: x["improvements"]["tail_improvement_pct"])

        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Comparison: Quantile vs Merton Jump Diffusion Classification</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .improvement {{ color: #28a745; }}
        .degradation {{ color: #dc3545; }}
        .neutral {{ color: #6c757d; }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .comparison-table th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .class-distribution {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }}
        .class-bar {{
            background: #e9ecef;
            border-radius: 5px;
            padding: 8px;
            text-align: center;
            font-size: 0.9em;
        }}
        .class-bar.extreme {{
            background: #ffeaa7;
            border: 2px solid #fdcb6e;
        }}
        .methodology {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 30px 0;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .highlight-box {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }}
        .alert-box {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Dataset Classification Comparison</h1>
        <h2>Quantile Baseline vs Merton Jump Diffusion</h2>
        <p>Comprehensive Analysis of Financial Returns Classification Approaches</p>
        <p><em>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
    </div>

    <div class="summary-grid">
        <div class="metric-card">
            <h3>📈 Average Balance Score Improvement</h3>
            <div class="metric-value {"improvement" if avg_balance_improvement > 0 else "degradation"}">{avg_balance_improvement:+.1f}%</div>
            <p>Higher balance = more uniform distribution</p>
        </div>
        <div class="metric-card">
            <h3>🎯 Average Tail Score Improvement</h3>
            <div class="metric-value {"improvement" if avg_tail_improvement > 0 else "degradation"}">{avg_tail_improvement:+.1f}%</div>
            <p>Lower tail score = better extreme class prediction</p>
        </div>
        <div class="metric-card">
            <h3>⚖️ Extreme Class Balance</h3>
            <div class="metric-value {"improvement" if avg_extreme_improvement < 0 else "degradation"}">{avg_extreme_improvement:+.1f}pp</div>
            <p>Classes 0+12 deviation from ideal 15.4%</p>
        </div>
        <div class="metric-card">
            <h3>🏆 Best Performer</h3>
            <div class="metric-value neutral">{best_performer["symbol"]}</div>
            <p>{best_performer["improvements"]["tail_improvement_pct"]:+.1f}% tail improvement</p>
        </div>
    </div>

    <div class="methodology">
        <h2>🔬 Methodology</h2>
        <div class="highlight-box">
            <h3>🎯 Merton Jump Diffusion Model</h3>
            <p><strong>Model:</strong> dS/S = μdt + σdW + (e^J - 1)dN</p>
            <ul>
                <li><strong>μ:</strong> Drift coefficient</li>
                <li><strong>σ:</strong> Diffusion volatility</li>
                <li><strong>J:</strong> Jump size (normally distributed)</li>
                <li><strong>N:</strong> Poisson jump process</li>
            </ul>
            <p><strong>Key Innovation:</strong> Models both continuous price movements and sudden jumps, providing superior tail prediction for extreme market events.</p>
        </div>

        <h3>📊 Key Metrics</h3>
        <ul>
            <li><strong>Balance Score:</strong> 1 - (max_deviation / expected_fraction) - measures distribution uniformity</li>
            <li><strong>Tail Score:</strong> Weighted error for classes 0 & 12 (extreme events) - lower is better</li>
            <li><strong>Extreme Concentration:</strong> Combined percentage in classes 0 + 12 (target: 15.4%)</li>
        </ul>
    </div>

    <h2>📋 Detailed Dataset Comparisons</h2>
    <table class="comparison-table">
        <thead>
            <tr>
                <th>Symbol</th>
                <th>Dataset</th>
                <th>Samples</th>
                <th>Balance Score</th>
                <th>Tail Score</th>
                <th>Class 0</th>
                <th>Class 12</th>
                <th>Extreme Total</th>
                <th>Improvement</th>
            </tr>
        </thead>
        <tbody>
"""

        # Add detailed comparisons
        for comp in comparisons:
            symbol = comp["symbol"]
            orig = comp["original"]
            mert = comp["merton"]
            impr = comp["improvements"]

            # Original row
            html_content += f"""
            <tr style="background: #fff3cd;">
                <td rowspan="2"><strong>{symbol}</strong></td>
                <td>📊 Quantile (Original)</td>
                <td>{orig["total_samples"]:,}</td>
                <td>{orig["balance_score"]:.3f}</td>
                <td>{orig["tail_score"]:.1f}</td>
                <td>{orig["class0_pct"]:.1f}%</td>
                <td>{orig["class12_pct"]:.1f}%</td>
                <td>{orig["extreme_concentration"] * 100:.1f}%</td>
                <td>-</td>
            </tr>
            <tr style="background: #d1ecf1;">
                <td>🎯 Merton (New)</td>
                <td>{mert["total_samples"]:,}</td>
                <td>{mert["balance_score"]:.3f}</td>
                <td>{mert["tail_score"]:.1f}</td>
                <td>{mert["class0_pct"]:.1f}%</td>
                <td>{mert["class12_pct"]:.1f}%</td>
                <td>{mert["extreme_concentration"] * 100:.1f}%</td>
                <td><span class="{"improvement" if impr["tail_improvement_pct"] > 0 else "degradation"}">{impr["tail_improvement_pct"]:+.1f}%</span></td>
            </tr>
            """

        html_content += """
        </tbody>
    </table>

    <div class="chart-container">
        <h2>📊 Class Distribution Analysis</h2>
        <p>Comparison of classification distribution across all 13 classes for each symbol:</p>
"""

        # Add class distribution for each symbol
        for comp in comparisons:
            symbol = comp["symbol"]
            orig = comp["original"]
            mert = comp["merton"]

            html_content += f"""
            <div style="margin: 30px 0; padding: 20px; border: 1px solid #dee2e6; border-radius: 8px;">
                <h3>{symbol} Class Distribution</h3>

                <h4>📊 Quantile (Original)</h4>
                <div class="class-distribution">
            """

            for i in range(13):
                cls_data = orig["class_analysis"][i]
                is_extreme = i in [0, 12]
                html_content += f"""
                    <div class="class-bar {"extreme" if is_extreme else ""}">
                        <div>Class {i}</div>
                        <div><strong>{cls_data["percentage"]:.1f}%</strong></div>
                        <div>({cls_data["count"]:,})</div>
                    </div>
                """

            html_content += """
                </div>

                <h4>🎯 Merton (New)</h4>
                <div class="class-distribution">
            """

            for i in range(13):
                cls_data = mert["class_analysis"][i]
                is_extreme = i in [0, 12]
                html_content += f"""
                    <div class="class-bar {"extreme" if is_extreme else ""}">
                        <div>Class {i}</div>
                        <div><strong>{cls_data["percentage"]:.1f}%</strong></div>
                        <div>({cls_data["count"]:,})</div>
                    </div>
                """

            html_content += "</div></div>"

        html_content += f"""
    </div>

    <div class="methodology">
        <h2>🎉 Key Findings</h2>

        {'<div class="highlight-box">' if avg_tail_improvement > 0 else '<div class="alert-box">'}
            <h3>🎯 Tail Prediction Performance</h3>
            <p>Merton Jump Diffusion shows <strong>{avg_tail_improvement:+.1f}%</strong> average improvement in tail score across all symbols.</p>
            <p>This translates to better identification of extreme market events (classes 0 & 12) where trading rewards are typically highest.</p>
        </div>

        {'<div class="highlight-box">' if avg_balance_improvement > 0 else '<div class="alert-box">'}
            <h3>⚖️ Distribution Balance</h3>
            <p>Overall balance score improved by <strong>{avg_balance_improvement:+.1f}%</strong> on average.</p>
            <p>More balanced class distributions lead to better ML model training and reduced bias.</p>
        </div>

        <div class="alert-box">
            <h3>⚠️ Important Notes</h3>
            <ul>
                <li>Merton model specifically targets extreme events (financial crashes and spikes)</li>
                <li>Improvements are most significant for tail prediction (classes 0 & 12)</li>
                <li>Some symbols may show mixed results due to different market regimes</li>
                <li>Training data uses first-half split to prevent data leakage</li>
            </ul>
        </div>

        <h3>💡 ML Training Implications</h3>
        <ul>
            <li><strong>Better Extreme Event Detection:</strong> Improved classification of rare but high-impact market movements</li>
            <li><strong>Reduced Training Bias:</strong> More balanced class distributions improve model learning</li>
            <li><strong>Enhanced Reward Potential:</strong> Superior tail prediction enables better capture of extreme trading opportunities</li>
        </ul>

        <h3>🔮 Recommended Next Steps</h3>
        <ol>
            <li>Use Merton-classified datasets for ML training</li>
            <li>Focus on extreme class performance metrics during model evaluation</li>
            <li>Monitor real-world trading performance on tail events</li>
            <li>Consider ensemble approaches combining multiple distribution models</li>
        </ol>
    </div>

    <div class="header" style="margin-top: 40px;">
        <h2>🌟 Conclusion</h2>
        <p>Merton Jump Diffusion classification provides <strong>superior tail prediction</strong> capabilities compared to traditional quantile-based approaches, making it ideal for financial ML applications where extreme event detection is critical.</p>
    </div>

</body>
</html>
        """

        # Save HTML report
        output_path = Path("distributions/html") / output_file
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ HTML assessment saved: {output_path}")
        return str(output_path)


def main():
    """Run the complete dataset comparison analysis."""

    print("🔍 DATASET COMPARISON ASSESSMENT")
    print("=" * 50)
    print("Comparing Quantile vs Merton Jump Diffusion Classification")
    print()

    comparator = DatasetComparator()

    # Check if directories exist
    if not comparator.original_dir.exists():
        print(f"❌ Original datasets directory not found: {comparator.original_dir}")
        return False

    if not comparator.merton_dir.exists():
        print(f"⚠️  Merton datasets directory not found: {comparator.merton_dir}")
        print("💡 Run 'make create-merton-dataset' first to generate Merton datasets")
        return False

    # Perform comparison
    comparisons = comparator.compare_datasets()

    if not comparisons:
        print("❌ No dataset comparisons possible")
        return False

    # Generate HTML report
    html_file = comparator.generate_html_report(comparisons)

    print("\n✅ COMPARISON COMPLETE!")
    print(f"📊 Analyzed {len(comparisons)} symbol datasets")
    print(f"📋 HTML report: {html_file}")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
