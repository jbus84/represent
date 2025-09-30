#!/usr/bin/env python3
"""
Process All Symbols with Recommended Regression Targets

This script uses ModularDatasetBuilder to process symbol datasets with the
RECOMMENDED regression targets for FX microstructure prediction:
  1. log_return_horizons (with mean returns)
  2. directional_mfe

These methods are superior to triple barrier for FX because:
  - Continuous targets (not balanced classes)
  - Shorter horizons (50-500 ticks, not 600+)
  - Preserve all price movement information
  - No 32% timeout class wasting data

Output: Target-only files with keys (row_idx, timestamp, symbol) + regression target columns
"""

import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent.modular_dataset_builder import ModularDatasetBuilder
from represent.target_generators.factory import TargetGeneratorFactory


def get_regression_generators(symbol_name: str) -> list:
    """
    Create regression target generators optimized for FX microstructure.

    These use SHORTER horizons than triple barrier (50-500 ticks vs 600+)
    which are within the FX predictability range.

    Args:
        symbol_name: Symbol being processed (for logging)

    Returns:
        List of configured target generators
    """
    generators = [
        # 1. Multi-horizon log returns (PRIMARY RECOMMENDATION)
        # Generates 10 targets: 5 endpoint + 5 mean returns
        TargetGeneratorFactory.create(
            "log_return_horizons",
            horizons=[50, 100, 250, 500, 1000],  # Shorter horizons = more predictable
            lookback_window=1000,
            include_mean_returns=True,  # NEW: Also get mean returns (50-100x smoother!)
            # Output columns:
            # - log_return_50t, log_return_100t, ..., log_return_1000t (endpoint returns)
            # - log_return_mean_50t, log_return_mean_100t, ..., log_return_mean_1000t (mean returns)
        ),

        # 2. Directional MFE (profit potential)
        # Generates 2 targets: buy_mfe, sell_mfe
        TargetGeneratorFactory.create(
            "directional_mfe",
            lookforward_horizon=500,  # Shorter than triple barrier
            lookback_window=200,
            expected_fee_pips=0.7,  # FX transaction cost
            # Output columns:
            # - mfe_buy_bps (best possible long profit)
            # - mfe_sell_bps (best possible short profit)
        ),
    ]

    print(f"   📊 Configured {len(generators)} regression generators:")
    for gen in generators:
        info = gen.get_target_info()
        print(f"      • {info['description']}")
        print(f"        Targets: {len(info['target_names'])} - {info['target_names'][:3]}...")

    return generators


def process_symbol_with_regression_targets(input_file: Path, output_dir: Path) -> dict:
    """
    Process a single symbol file with regression targets.

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
        # Create regression target generators
        generators = get_regression_generators(symbol_name)

        # Use ModularDatasetBuilder with chunked processing
        print(f"   🔨 Building targets with {len(generators)} generators...")
        builder = ModularDatasetBuilder(generators, verbose=True)

        # Process using chunked infrastructure for memory efficiency
        targets_df = builder.build_targets_from_parquet_chunked(
            input_file,
            symbol=symbol_name,
            chunk_size=500_000  # Larger chunks OK for regression (no two-pass)
        )

        # Get all target names
        all_target_names = []
        for gen in generators:
            info = gen.get_target_info()
            all_target_names.extend(info['target_names'])

        # Save targets file
        output_file = output_dir / f"{symbol_name}_regression_targets.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        targets_df.write_parquet(output_file)

        print(f"   ✅ Processing complete - {len(targets_df)} samples generated")
        print(f"   💾 Saved to: {output_file}")
        print(f"   🏷️  Total targets: {len(all_target_names)}")
        print("      - log_return_horizons: 10 targets (5 endpoint + 5 mean)")
        print("      - directional_mfe: 2 targets (buy_mfe, sell_mfe)")
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
    print("🚀 PROCESSING ALL SYMBOLS WITH REGRESSION TARGETS")
    print("=" * 80)
    print("Using RECOMMENDED methods for FX microstructure prediction:")
    print("  1. log_return_horizons (50-1000 ticks) with mean returns")
    print("  2. directional_mfe (profit potential)")
    print()
    print("WHY THESE ARE BETTER THAN TRIPLE BARRIER:")
    print("  ✓ Continuous targets (not balanced classes)")
    print("  ✓ Shorter horizons (50-500 ticks, not 600+)")
    print("  ✓ No 32% timeout class wasting data")
    print("  ✓ Mean returns 50-100x smoother than endpoints")
    print("  ✓ Expected R² = 0.01-0.05 (vs 34% random accuracy)")
    print()
    print("Output: Target-only files with 12 total regression targets per sample")
    print("=" * 80)
    print()

    # Define paths
    input_dir = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs")
    output_dir = Path("/Users/danielfisher/data/databento/symbol_datasets/regression_targets")

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
            results = process_symbol_with_regression_targets(symbol_file, output_dir)

            if results.get('success', False):
                successful_symbols += 1
                total_samples_generated += results.get('sample_count', 0)
                print(f"   🎉 Symbol complete: {results['sample_count']:,} samples with 12 targets each")
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
    print("   Total targets per sample: 12")
    print("      • 10 log return targets (5 endpoint + 5 mean)")
    print("      • 2 MFE targets (buy + sell)")
    print(f"   Output directory: {output_dir}")

    # List output files
    output_files = list(output_dir.glob("*_regression_targets.parquet"))
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
        f.write("Regression Targets Processing Summary\n")
        f.write(f"Generated on: {datetime.now()}\n\n")

        f.write("=" * 60 + "\n")
        f.write("CONFIGURATION\n")
        f.write("=" * 60 + "\n")
        f.write("Method 1: log_return_horizons\n")
        f.write("  - Horizons: [50, 100, 250, 500, 1000] ticks\n")
        f.write("  - Include mean returns: True\n")
        f.write("  - Targets: 10 (5 endpoint + 5 mean)\n")
        f.write("  - Why: Shorter horizons within FX predictability range\n\n")

        f.write("Method 2: directional_mfe\n")
        f.write("  - Lookforward: 500 ticks\n")
        f.write("  - Transaction cost: 0.7 pips\n")
        f.write("  - Targets: 2 (buy_mfe, sell_mfe)\n")
        f.write("  - Why: Captures profit potential for both directions\n\n")

        f.write("=" * 60 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Symbols processed: {len(symbol_files)}\n")
        f.write(f"Symbols successful: {successful_symbols}\n")
        f.write(f"Success rate: {100*successful_symbols/len(symbol_files):.1f}%\n")
        f.write(f"Total samples: {total_samples_generated:,}\n")
        f.write("Targets per sample: 12\n\n")

        f.write("=" * 60 + "\n")
        f.write("ADVANTAGES OVER TRIPLE BARRIER\n")
        f.write("=" * 60 + "\n")
        f.write("✓ Continuous targets (not 33/32/34 balanced classes)\n")
        f.write("✓ Shorter horizons (50-500 ticks vs 600+ ticks)\n")
        f.write("✓ No timeout class (0% waste vs 32% waste)\n")
        f.write("✓ Mean returns 50-100x smoother than endpoints\n")
        f.write("✓ Expected R² = 0.01-0.05 (vs 34% random accuracy)\n")
        f.write("✓ Preserves all price movement information\n\n")

        f.write("=" * 60 + "\n")
        f.write("TARGET DESCRIPTIONS\n")
        f.write("=" * 60 + "\n")
        f.write("Endpoint Returns (5 targets):\n")
        f.write("  log_return_50t, log_return_100t, log_return_250t,\n")
        f.write("  log_return_500t, log_return_1000t\n")
        f.write("  → Entry to exit price return (final P&L)\n\n")

        f.write("Mean Returns (5 targets):\n")
        f.write("  log_return_mean_50t, log_return_mean_100t, log_return_mean_250t,\n")
        f.write("  log_return_mean_500t, log_return_mean_1000t\n")
        f.write("  → Average tick-to-tick return (trend/drift, 50-100x smoother)\n\n")

        f.write("MFE Targets (2 targets):\n")
        f.write("  mfe_buy_bps, mfe_sell_bps\n")
        f.write("  → Best possible profit in each direction (for position sizing)\n\n")

        f.write("=" * 60 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("=" * 60 + "\n")
        for output_file in sorted(output_files):
            size_mb = output_file.stat().st_size / (1024 * 1024)
            f.write(f"{output_file.name} ({size_mb:.1f} MB)\n")

        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("NEXT STEPS\n")
        f.write("=" * 60 + "\n")
        f.write("1. Train regression models (LightGBM, XGBoost, Neural Net)\n")
        f.write("2. Evaluate R² on validation set (expect 0.01-0.05)\n")
        f.write("3. Use endpoint returns for P&L prediction\n")
        f.write("4. Use mean returns for trend/direction confirmation\n")
        f.write("5. Combine both signals for trading decisions\n")
        f.write("6. Apply confidence thresholds before trading\n")

    print(f"\n📋 Processing summary saved to: {summary_file}")
    print()
    print("=" * 80)
    print("NEXT STEPS FOR ML TRAINING:")
    print("=" * 80)
    print("1. Load regression targets from output directory")
    print("2. Train models (LightGBM recommended for tabular data)")
    print("3. Expected performance: R² = 0.01-0.05 (GOOD for FX!)")
    print("4. Combine endpoint + mean predictions for stronger signals")
    print("5. Apply thresholds: only trade when prediction > 2-3 bps")
    print()
    print("Example:")
    print("  endpoint_pred = model_endpoint.predict(features)  # Final P&L")
    print("  mean_pred = model_mean.predict(features)          # Trend")
    print("  long_signal = (endpoint_pred > 2) & (mean_pred > 0)  # Both positive")
    print("=" * 80)


if __name__ == "__main__":
    main()
