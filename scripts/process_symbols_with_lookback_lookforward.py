#!/usr/bin/env python3
"""
Process All Symbols with Lookback-Lookforward Regression Targets

This script uses ModularDatasetBuilder to process symbol datasets with the
lookback-lookforward returns generator for symmetric window analysis:
  - Compares lookback and lookforward window means
  - Multiple time scales (1000, 2500, 5000, 10000 ticks)
  - 6 metrics per window: lookback_mean, lookforward_mean, log_return,
    current_vs_lookback_mean, current_vs_lookforward_mean, current_price

Use case: Multi-scale trend analysis with symmetric windows for predicting
whether the future trend will match or diverge from recent history.

Output: Target-only files with keys (row_idx, timestamp, symbol) + 24 regression target columns
"""

import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent.modular_dataset_builder import ModularDatasetBuilder
from represent.target_generators.factory import TargetGeneratorFactory


def get_lookback_lookforward_generators(symbol_name: str) -> list:
    """
    Create lookback-lookforward regression target generator.

    Uses symmetric windows to compare recent history with future trends.
    Each window produces 6 metrics showing relationship between past and future.

    Args:
        symbol_name: Symbol being processed (for logging)

    Returns:
        List containing configured lookback-lookforward generator
    """
    generators = [
        # Lookback-Lookforward Returns Generator
        # Generates 24 targets: 4 windows × 6 metrics each
        TargetGeneratorFactory.create(
            "lookback_lookforward_returns",
            window_lengths=[1000, 2500, 5000, 10000],  # Multi-scale analysis
            target_prefix="lf",
            # Output columns per window (e.g., for 1000-tick window):
            # - lf_lookback_mean_1000t: Mean price over lookback window (includes current)
            # - lf_lookforward_mean_1000t: Mean price over lookforward window (excludes current)
            # - lf_log_return_1000t: Log return between window means (basis points)
            # - lf_current_vs_lookback_mean_1000t: current - lookback mean (basis points)
            # - lf_current_vs_lookforward_mean_1000t: current - lookforward mean (basis points)
            # - lf_current_price_1000t: Price at end of lookback window
        ),
    ]

    print(f"   📊 Configured {len(generators)} lookback-lookforward generator:")
    for gen in generators:
        info = gen.get_target_info()
        print(f"      • {info['description']}")
        print(f"        Targets: {len(info['target_names'])} total")
        print(f"        Windows: [1000, 2500, 5000, 10000] ticks")
        print(f"        Metrics per window: 6 (means, returns, differences)")

    return generators


def process_symbol_with_lookback_lookforward(input_file: Path, output_dir: Path) -> dict:
    """
    Process a single symbol file with lookback-lookforward targets.

    Args:
        input_file: Path to input parquet file
        output_dir: Directory to save target files

    Returns:
        Dictionary with processing results
    """
    # Extract symbol name
    symbol_name = input_file.stem.split('_')[0]

    print(f"\n🔄 Processing symbol: {symbol_name}")
    print(f"   📁 Input: {input_file}")

    try:
        # Create lookback-lookforward generator
        generators = get_lookback_lookforward_generators(symbol_name)

        # Use ModularDatasetBuilder with chunked processing
        print(f"   🔨 Building targets with lookback-lookforward generator...")
        builder = ModularDatasetBuilder(generators, verbose=True)

        # Process using chunked infrastructure for memory efficiency
        targets_df = builder.build_targets_from_parquet_chunked(
            input_file,
            symbol=symbol_name,
            chunk_size=500_000  # Larger chunks OK for regression
        )

        # Get all target names
        all_target_names = []
        for gen in generators:
            info = gen.get_target_info()
            all_target_names.extend(info['target_names'])

        # Save targets file
        output_file = output_dir / f"{symbol_name}_lookback_lookforward_targets.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        targets_df.write_parquet(output_file)

        print(f"   ✅ Processing complete - {len(targets_df)} samples generated")
        print(f"   💾 Saved to: {output_file}")
        print(f"   🏷️  Total targets: {len(all_target_names)} (4 windows × 6 metrics)")
        print(f"   📋 Output columns: {targets_df.columns}")

        # Return detailed results
        return {
            "success": True,
            "output_file": str(output_file),
            "target_names": all_target_names,
            "target_count": len(targets_df),
            "sample_count": len(targets_df),
            "generators_used": [g.get_target_info()['description'] for g in generators]
        }

    except Exception as e:
        print(f"   ❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """Main execution function."""
    print("🚀 PROCESSING ALL SYMBOLS WITH LOOKBACK-LOOKFORWARD TARGETS")
    print("=" * 80)
    print("Symmetric Window Analysis for Multi-Scale Trend Prediction")
    print()
    print("WINDOW STRUCTURE:")
    print("  For each position i and window length N:")
    print("    • Lookback:    [i - N + 1, i]     (N points, includes current)")
    print("    • Current:     mid_prices[i]      (end of lookback)")
    print("    • Lookforward: [i + 1, i + N]     (N points, excludes current)")
    print()
    print("METRICS GENERATED (6 per window):")
    print("  1. lookback_mean          - Mean price over lookback window")
    print("  2. lookforward_mean       - Mean price over lookforward window")
    print("  3. log_return             - Log return between window means (bps)")
    print("  4. current_vs_lookback    - Current - lookback mean (bps)")
    print("  5. current_vs_lookforward - Current - lookforward mean (bps)")
    print("  6. current_price          - Price at window center")
    print()
    print("WINDOWS: [1000, 2500, 5000, 10000] ticks")
    print("TOTAL TARGETS: 24 (4 windows × 6 metrics)")
    print()
    print("USE CASE:")
    print("  • Multi-scale trend analysis")
    print("  • Symmetric window comparisons")
    print("  • Predict if future matches recent history")
    print("  • Identify mean reversion vs trend continuation")
    print("=" * 80)
    print()

    # Define paths
    input_dir = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs")
    output_dir = Path("/Users/danielfisher/data/databento/symbol_datasets/lookback_lookforward_targets")

    # Find all symbol files
    symbol_files = list(input_dir.glob("*.parquet"))

    if not symbol_files:
        print(f"❌ No parquet files found in {input_dir}")
        return

    print(f"📊 Found {len(symbol_files)} symbol files to process")
    print()

    # Process each symbol with progress bar
    all_results = {}
    successful_symbols = 0
    total_samples_generated = 0

    # Create file-level progress bar
    file_progress = tqdm(symbol_files, desc="Processing symbols", unit="file")

    for symbol_file in file_progress:
        symbol_name = symbol_file.stem.split('_')[0]
        file_progress.set_postfix(symbol=symbol_name)

        print(f"\n{'='*80}")
        print(f"📈 PROCESSING SYMBOL: {symbol_name}")
        print(f"{'='*80}")

        try:
            results = process_symbol_with_lookback_lookforward(symbol_file, output_dir)

            if results.get('success', False):
                successful_symbols += 1
                total_samples_generated += results.get('sample_count', 0)
                print(f"   🎉 Symbol complete: {results['sample_count']:,} samples with 24 targets each")
            else:
                print(f"   ⚠️  Symbol processing failed: {results.get('error', 'Unknown error')}")

            all_results[symbol_file.stem] = results

        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            all_results[symbol_file.stem] = {"success": False, "error": f"Unexpected error: {e}"}

    file_progress.close()

    # Generate summary
    print("\n🎉 PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"   Successfully processed: {successful_symbols}/{len(symbol_files)} symbols")
    print(f"   Total samples generated: {total_samples_generated:,}")
    print("   Total targets per sample: 24")
    print("      • 4 window sizes: [1000, 2500, 5000, 10000] ticks")
    print("      • 6 metrics per window: means, returns, differences")
    print(f"   Output directory: {output_dir}")

    # List output files
    output_files = list(output_dir.glob("*_lookback_lookforward_targets.parquet"))
    if output_files:
        print(f"\n📁 Generated {len(output_files)} target files:")
        for f in sorted(output_files):
            # Get file size
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   • {f.name} ({size_mb:.1f} MB)")

    # Save processing summary
    summary_file = output_dir / "processing_summary.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        f.write("Lookback-Lookforward Targets Processing Summary\n")
        f.write(f"Generated on: {datetime.now()}\n\n")

        f.write("=" * 60 + "\n")
        f.write("CONFIGURATION\n")
        f.write("=" * 60 + "\n")
        f.write("Generator: lookback_lookforward_returns\n")
        f.write("  - Window lengths: [1000, 2500, 5000, 10000] ticks\n")
        f.write("  - Metrics per window: 6\n")
        f.write("  - Total targets: 24\n")
        f.write("  - Target prefix: 'lf'\n\n")

        f.write("=" * 60 + "\n")
        f.write("WINDOW STRUCTURE\n")
        f.write("=" * 60 + "\n")
        f.write("For position i and window length N:\n")
        f.write("  Lookback:    [i - N + 1, i]     (N points, includes current)\n")
        f.write("  Current:     mid_prices[i]      (end of lookback)\n")
        f.write("  Lookforward: [i + 1, i + N]     (N points, excludes current)\n\n")

        f.write("=" * 60 + "\n")
        f.write("METRICS (6 per window)\n")
        f.write("=" * 60 + "\n")
        f.write("1. lookback_mean          - Mean price over lookback window\n")
        f.write("2. lookforward_mean       - Mean price over lookforward window\n")
        f.write("3. log_return             - Log return between means (bps)\n")
        f.write("4. current_vs_lookback    - Current - lookback mean (bps)\n")
        f.write("5. current_vs_lookforward - Current - lookforward mean (bps)\n")
        f.write("6. current_price          - Price at window center\n\n")

        f.write("=" * 60 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Symbols processed: {len(symbol_files)}\n")
        f.write(f"Symbols successful: {successful_symbols}\n")
        f.write(f"Success rate: {100*successful_symbols/len(symbol_files):.1f}%\n")
        f.write(f"Total samples: {total_samples_generated:,}\n")
        f.write("Targets per sample: 24\n\n")

        f.write("=" * 60 + "\n")
        f.write("TARGET DESCRIPTIONS\n")
        f.write("=" * 60 + "\n")
        f.write("Example columns for 1000-tick window:\n")
        f.write("  lf_lookback_mean_1000t\n")
        f.write("  lf_lookforward_mean_1000t\n")
        f.write("  lf_log_return_1000t\n")
        f.write("  lf_current_vs_lookback_mean_1000t\n")
        f.write("  lf_current_vs_lookforward_mean_1000t\n")
        f.write("  lf_current_price_1000t\n\n")
        f.write("Repeated for windows: 2500, 5000, 10000 ticks\n\n")

        f.write("=" * 60 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("=" * 60 + "\n")
        for output_file in sorted(output_files):
            size_mb = output_file.stat().st_size / (1024 * 1024)
            f.write(f"{output_file.name} ({size_mb:.1f} MB)\n")

        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("USE CASES\n")
        f.write("=" * 60 + "\n")
        f.write("1. Multi-scale trend analysis across different timeframes\n")
        f.write("2. Mean reversion detection (lookback vs lookforward divergence)\n")
        f.write("3. Trend continuation prediction (lookback matches lookforward)\n")
        f.write("4. Position timing using window mean comparisons\n")
        f.write("5. Regime identification through multi-scale patterns\n\n")

        f.write("=" * 60 + "\n")
        f.write("NEXT STEPS\n")
        f.write("=" * 60 + "\n")
        f.write("1. Train regression models on 24 target dimensions\n")
        f.write("2. Predict log_return targets for directional signals\n")
        f.write("3. Use current_vs_* targets for entry timing\n")
        f.write("4. Combine multiple window predictions for confidence\n")
        f.write("5. Compare lookback vs lookforward for mean reversion signals\n")
        f.write("6. Use longer windows (5k/10k) for trend confirmation\n")

    print(f"\n📋 Processing summary saved to: {summary_file}")
    print()
    print("=" * 80)
    print("NEXT STEPS FOR ML TRAINING:")
    print("=" * 80)
    print("1. Load lookback-lookforward targets from output directory")
    print("2. Train regression models (LightGBM/XGBoost recommended)")
    print("3. Focus on log_return targets for directional predictions")
    print("4. Use current_vs_* targets for entry/exit timing")
    print("5. Combine multiple window scales for robust signals")
    print()
    print("Example trading signals:")
    print("  # Mean reversion signal")
    print("  lookback_high = pred_lookback_mean > pred_lookforward_mean")
    print("  current_below = pred_current_vs_lookback < 0")
    print("  buy_signal = lookback_high & current_below  # Expect reversion up")
    print()
    print("  # Trend continuation signal")
    print("  positive_return = pred_log_return > 2  # > 2 bps")
    print("  multiple_windows = all([pred_1k, pred_2.5k, pred_5k] > 0)")
    print("  long_signal = positive_return & multiple_windows  # Strong trend")
    print("=" * 80)


if __name__ == "__main__":
    main()
