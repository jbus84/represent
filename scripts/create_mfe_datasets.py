#!/usr/bin/env python3
"""
Batch create MFE-enhanced datasets for all AUDUSD symbols.
Focus on efficiency and dataset creation only.
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from represent.directional_mfe_calculator import DirectionalMFECalculator, DirectionalMFEConfig


def add_mfe_to_dataset_streaming(input_file: Path, output_dir: Path, config: DirectionalMFEConfig) -> dict:
    """
    Add MFE columns using memory-efficient approach with proper streaming.
    """
    symbol = input_file.stem.split("_")[1] 
    print(f"📊 Processing {symbol}...")
    start_time = time.time()

    # Get dataset info
    total_rows = pl.scan_parquet(input_file).select(pl.len()).collect().item()
    print(f"   📊 Dataset size: {total_rows:,} rows")
    
    output_file = output_dir / f"AUDUSD_{symbol}_MFE_enhanced_dataset.parquet"
    
    # Use a simpler, more reliable approach: create MFE arrays first, then add to dataset
    print(f"   🧮 Calculating MFE arrays...")
    
    # Calculate MFE using the existing optimized calculator but with larger batches
    from represent.directional_mfe_calculator import DirectionalMFECalculator
    calculator = DirectionalMFECalculator(config, verbose=False)
    
    # Use larger batch size for better efficiency on big datasets
    batch_size = min(200000, total_rows // 10)  # 200K rows or 10% of dataset, whichever is smaller
    mfe_result = calculator.calculate_from_parquet(input_file, batch_size=batch_size)
    
    if mfe_result.sample_count == 0:
        print("   ❌ No valid MFE values calculated")
        return None
        
    # Verify arrays match
    if len(mfe_result.mfe_buy_bps) != total_rows:
        raise ValueError(f"MFE array length mismatch")
    
    print(f"   💾 Adding MFE columns to dataset using streaming...")
    
    # Use polars streaming to add columns efficiently
    try:
        # Create the enhanced dataset using lazy operations and stream to disk
        (
            pl.scan_parquet(input_file)
            .with_columns([
                pl.Series("mfe_buy_bps", mfe_result.mfe_buy_bps).alias("mfe_buy_bps"),
                pl.Series("mfe_sell_bps", mfe_result.mfe_sell_bps).alias("mfe_sell_bps")
            ])
            .sink_parquet(output_file, compression="zstd")
        )
        
        print(f"   ✅ Enhanced dataset written: {output_file.stat().st_size / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"   ❌ Error writing enhanced dataset: {e}")
        return None
    
    elapsed = time.time() - start_time
    
    # Calculate statistics from the MFE result
    valid_buy_mask = ~np.isnan(mfe_result.mfe_buy_bps)
    valid_sell_mask = ~np.isnan(mfe_result.mfe_sell_bps)
    valid_both_mask = valid_buy_mask & valid_sell_mask
    
    valid_buy_values = mfe_result.mfe_buy_bps[valid_buy_mask]
    valid_sell_values = mfe_result.mfe_sell_bps[valid_sell_mask]
    valid_both_buy = mfe_result.mfe_buy_bps[valid_both_mask]
    valid_both_sell = mfe_result.mfe_sell_bps[valid_both_mask]
    
    if len(valid_buy_values) > 0:
        buy_stats = {
            "mean": np.mean(valid_buy_values),
            "std": np.std(valid_buy_values),
            "min": np.min(valid_buy_values),
            "max": np.max(valid_buy_values),
        }
        sell_stats = {
            "mean": np.mean(valid_sell_values),
            "std": np.std(valid_sell_values),
            "min": np.min(valid_sell_values),
            "max": np.max(valid_sell_values),
        }
        correlation = np.corrcoef(valid_both_buy, valid_both_sell)[0, 1] if len(valid_both_buy) > 1 else 0.0
        
        valid_count = len(valid_buy_values)
        coverage = valid_count / total_rows * 100
        
        print(f"   ✅ {symbol}: {valid_count:,} valid MFE values / {total_rows:,} total rows ({coverage:.1f}% coverage)")
        print(f"      Processing time: {elapsed:.1f}s")
        print(f"      Buy: {buy_stats['mean']:.2f}±{buy_stats['std']:.2f} BPS, Fee-adj: {buy_stats['mean'] - config.expected_fee_pips:.2f}")
        print(f"      Sell: {sell_stats['mean']:.2f}±{sell_stats['std']:.2f} BPS, Fee-adj: {sell_stats['mean'] - config.expected_fee_pips:.2f}")
        print(f"      Correlation: {correlation:.3f}")
        
        return {
            "symbol": symbol,
            "file_path": output_file,
            "total_rows": total_rows,
            "valid_count": valid_count,
            "coverage": coverage,
            "buy_stats": buy_stats,
            "sell_stats": sell_stats,
            "correlation": correlation,
            "processing_time": elapsed,
        }
    else:
        return None

def add_mfe_to_dataset(input_file: Path, output_dir: Path, config: DirectionalMFEConfig) -> dict:
    """
    Add MFE buy and sell columns to existing production dataset.
    Uses optimized streaming approach for large datasets.
    """
    # For large datasets (>1M rows), use streaming approach
    total_rows = pl.scan_parquet(input_file).select(pl.len()).collect().item()
    
    if total_rows > 1_000_000:
        return add_mfe_to_dataset_streaming(input_file, output_dir, config)
    
    # For smaller datasets, use the original approach (it's more accurate)
    symbol = input_file.stem.split("_")[1]
    print(f"📊 Processing {symbol}...")
    start_time = time.time()

    print(f"   📊 Dataset size: {total_rows:,} rows")

    # Calculate MFE using standard approach for smaller datasets
    calculator = DirectionalMFECalculator(config, verbose=True)
    mfe_result = calculator.calculate_from_parquet(input_file, batch_size=50000)

    if mfe_result.sample_count == 0:
        print("   ❌ No valid MFE values calculated - dataset too small")
        return None

    if len(mfe_result.mfe_buy_bps) != total_rows or len(mfe_result.mfe_sell_bps) != total_rows:
        raise ValueError(f"MFE arrays length mismatch: expected {total_rows}, got {len(mfe_result.mfe_buy_bps)}")

    output_file = output_dir / f"AUDUSD_{symbol}_MFE_enhanced_dataset.parquet"
    print(f"   💾 Creating enhanced dataset: {output_file.name}")
    
    # Use polars lazy operations
    enhanced_lazy = (
        pl.scan_parquet(input_file)
        .with_row_index()
        .with_columns([
            pl.lit(mfe_result.mfe_buy_bps).alias("mfe_buy_bps"),
            pl.lit(mfe_result.mfe_sell_bps).alias("mfe_sell_bps")
        ])
        .drop("index")
    )
    
    enhanced_lazy.sink_parquet(output_file, compression="zstd")
    print(f"   ✅ Enhanced dataset written: {output_file.stat().st_size / (1024**2):.1f} MB")

    elapsed = time.time() - start_time

    # Calculate statistics on valid (non-NaN) MFE values
    valid_buy_mask = ~np.isnan(mfe_result.mfe_buy_bps)
    valid_sell_mask = ~np.isnan(mfe_result.mfe_sell_bps)
    valid_both_mask = valid_buy_mask & valid_sell_mask
    
    valid_buy_values = mfe_result.mfe_buy_bps[valid_buy_mask]
    valid_sell_values = mfe_result.mfe_sell_bps[valid_sell_mask]
    valid_both_buy = mfe_result.mfe_buy_bps[valid_both_mask]
    valid_both_sell = mfe_result.mfe_sell_bps[valid_both_mask]

    buy_stats = {
        "mean": np.mean(valid_buy_values),
        "std": np.std(valid_buy_values),
        "min": np.min(valid_buy_values),
        "max": np.max(valid_buy_values),
    }
    sell_stats = {
        "mean": np.mean(valid_sell_values),
        "std": np.std(valid_sell_values),
        "min": np.min(valid_sell_values),
        "max": np.max(valid_sell_values),
    }

    # Calculate correlation only on rows where both values are valid
    correlation = np.corrcoef(valid_both_buy, valid_both_sell)[0, 1] if len(valid_both_buy) > 1 else 0.0

    valid_count = len(valid_buy_values)
    coverage = valid_count / total_rows * 100

    print(f"   ✅ {symbol}: {valid_count:,} valid MFE values / {total_rows:,} total rows ({coverage:.1f}% coverage)")
    print(f"      Processing time: {elapsed:.1f}s")
    print(f"      Buy: {buy_stats['mean']:.2f}±{buy_stats['std']:.2f} BPS, Fee-adj: {buy_stats['mean'] - config.expected_fee_pips:.2f}")
    print(f"      Sell: {sell_stats['mean']:.2f}±{sell_stats['std']:.2f} BPS, Fee-adj: {sell_stats['mean'] - config.expected_fee_pips:.2f}")
    print(f"      Correlation: {correlation:.3f}")

    return {
        "symbol": symbol,
        "file_path": output_file,
        "total_rows": total_rows,
        "valid_count": valid_count,
        "coverage": coverage,
        "buy_stats": buy_stats,
        "sell_stats": sell_stats,
        "correlation": correlation,
        "processing_time": elapsed,
    }


def main():
    print("🎯 BATCH MFE DATASET ENHANCEMENT")
    print("=====================================")
    print("Adding MFE columns to existing production datasets")

    # Configuration  
    config = DirectionalMFEConfig(
        currency="AUDUSD",
        lookback_window=200,
        lookforward_horizon=3000,
        expected_fee_pips=0.7,
        jump_size=1,  # Process EVERY row (no sampling)
        winsorize_percentile=0.01,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )

    print("⚙️ Configuration:")
    print(f"   Lookback/Lookforward: {config.lookback_window}/{config.lookforward_horizon}")
    print(f"   Processing: EVERY row (no sampling)")
    print(f"   Expected fee: {config.expected_fee_pips} BPS")

    # Setup directories
    input_dir = Path("/Users/danielfisher/data/databento/AUDUSD_classified_datasets")
    output_dir = Path("/Users/danielfisher/data/databento/AUDUSD_MFE_enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all symbol datasets
    symbol_datasets = sorted(input_dir.glob("AUDUSD_*_dataset.parquet"))

    if not symbol_datasets:
        print(f"❌ No symbol datasets found in {input_dir}")
        return

    print(f"\n📊 Found {len(symbol_datasets)} production datasets to enhance")

    # Process each dataset
    results = []
    total_start_time = time.time()

    for input_file in symbol_datasets:
        try:
            result = add_mfe_to_dataset(input_file, output_dir, config)
            if result:
                results.append(result)
            print()  # Empty line between symbols

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue

    total_elapsed = time.time() - total_start_time

    # Summary
    print("🎉 BATCH PROCESSING COMPLETE:")
    print("==============================")
    print(f"✅ Successful: {len(results)} / {len(symbol_datasets)} datasets enhanced")
    print(f"✅ Total time: {total_elapsed:.1f} seconds ({total_elapsed / 60:.1f} minutes)")
    print(f"✅ Output directory: {output_dir}")

    if not results:
        print("❌ No datasets were successfully enhanced")
        return

    # Detailed results
    print("\n📊 SYMBOL ENHANCEMENT SUMMARY:")
    print(
        f"{'Symbol':<8} | {'Total Rows':<10} | {'Valid MFE':<10} | {'Coverage':<8} | {'Buy Ret':<8} | {'Sell Ret':<8} | {'Status'}"
    )
    print("-" * 90)

    total_rows = 0
    total_valid = 0
    for result in sorted(results, key=lambda x: x["total_rows"], reverse=True):
        symbol = result["symbol"]
        rows = result["total_rows"]
        valid = result["valid_count"]
        coverage = result["coverage"]
        buy_return = result["buy_stats"]["mean"] - config.expected_fee_pips
        sell_return = result["sell_stats"]["mean"] - config.expected_fee_pips

        status = "🚀" if buy_return > 0 and sell_return > 0 else "⚠️"

        print(
            f"{symbol:<8} | {rows:<10,} | {valid:<10,} | {coverage:<8.1f}% | {buy_return:<8.2f} | {sell_return:<8.2f} | {status}"
        )
        total_rows += rows
        total_valid += valid

    overall_coverage = total_valid / total_rows * 100

    # Overall statistics
    best_buy = max(results, key=lambda x: x["buy_stats"]["mean"] - config.expected_fee_pips)
    best_sell = max(results, key=lambda x: x["sell_stats"]["mean"] - config.expected_fee_pips)
    most_data = max(results, key=lambda x: x["total_rows"])

    print("\n🏆 HIGHLIGHTS:")
    print(f"   🎯 Total rows processed: {total_rows:,}")
    print(f"   ✅ Valid MFE calculations: {total_valid:,}")
    print(f"   📊 Overall coverage: {overall_coverage:.1f}%")
    print(
        f"   📈 Best buy strategy: {best_buy['symbol']} ({best_buy['buy_stats']['mean'] - config.expected_fee_pips:.2f} BPS)"
    )
    print(
        f"   📉 Best sell strategy: {best_sell['symbol']} ({best_sell['sell_stats']['mean'] - config.expected_fee_pips:.2f} BPS)"
    )
    print(f"   📊 Largest dataset: {most_data['symbol']} ({most_data['total_rows']:,} rows)")

    # Usage instructions
    print("\n💡 USAGE:")
    print(f"   Load enhanced datasets: pl.read_parquet('{output_dir}/AUDUSD_M6AM4_MFE_enhanced_dataset.parquet')")
    print("   New columns added: ['mfe_buy_bps', 'mfe_sell_bps']")
    print("   All original columns preserved + MFE columns")
    print("   Filter valid MFE: df.filter(pl.col('mfe_buy_bps').is_not_null())")

    print("\n🚀 Production datasets enhanced with MFE targets!")

    return results


if __name__ == "__main__":
    main()
