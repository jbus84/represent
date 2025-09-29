#!/usr/bin/env python3
"""
Process All Symbols with Adaptive Triple Barrier Method

This script uses the existing ModularDatasetBuilder infrastructure to process
symbol datasets with ONLY the triple_barrier_adaptive method, using
actual optimized parameters from JSON files.

Output: Target-only files with keys (row_idx, timestamp) + adaptive triple barrier target columns
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import polars as pl
from tqdm import tqdm

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent.target_generators.factory import TargetGeneratorFactory
from represent.modular_dataset_builder import ModularDatasetBuilder

def load_optimized_parameters(symbol_name: str) -> dict[str, dict]:
    """
    Load optimized parameters for a symbol from JSON files.

    Args:
        symbol_name: Name of the symbol (e.g., 'M6AU4')

    Returns:
        Dictionary with optimized parameters for each method
    """
    params_dir = Path("outputs/optimization_results/optimized_parameters")

    # Find the parameter directory for this symbol
    symbol_dirs = list(params_dir.glob(f"{symbol_name}*"))
    if not symbol_dirs:
        raise ValueError(f"No optimization results found for symbol {symbol_name}")

    symbol_dir = symbol_dirs[0]  # Use the first match

    optimized_params = {}

    # Load ONLY triple_barrier_adaptive parameters (as requested)
    adaptive_file = symbol_dir / "triple_barrier_adaptive_params.json"
    if adaptive_file.exists():
        with open(adaptive_file) as f:
            data = json.load(f)
            optimized_params['triple_barrier_adaptive'] = data['optimal_params']
    else:
        raise ValueError(f"No triple_barrier_adaptive parameters found for {symbol_name}")

    return optimized_params

def process_symbol_with_triple_methods_chunked(input_file: Path, output_dir: Path) -> dict:
    """
    Process a single symbol file using ModularDatasetBuilder with chunked processing.

    Args:
        input_file: Path to input parquet file
        output_dir: Directory to save target files

    Returns:
        Dictionary with processing results
    """
    # Extract symbol name correctly
    symbol_name = input_file.stem.split('_')[0]  # Gets 'M6AU4' from filename

    print(f"\n🔄 Processing symbol: {symbol_name}")
    print(f"   📁 Input: {input_file}")

    # Load optimized parameters for this symbol
    try:
        optimized_params = load_optimized_parameters(symbol_name)
        print(f"   📋 Loaded optimized parameters for {symbol_name}")
    except Exception as e:
        print(f"   ❌ Failed to load optimized parameters: {e}")
        return {"error": f"Failed to load optimized parameters: {e}"}

    results = {}

    # Process ONLY with adaptive triple barrier method (as requested)
    triple_methods = ['triple_barrier_adaptive']

    for method in triple_methods:
        try:
            print(f"   🎯 Processing with {method}...")

            # Get optimized parameters for this specific symbol
            if method not in optimized_params:
                print(f"   ⚠️  No optimized parameters found for {method}")
                results[method] = {
                    "success": False,
                    "error": f"No optimized parameters found for {method}",
                    "parameters": None
                }
                continue

            params = optimized_params[method].copy()
            print(f"   ⚙️  Optimized parameters: {params}")

            # Create generator with optimized parameters
            generator = TargetGeneratorFactory.create(method, **params)

            # Use ModularDatasetBuilder with chunked processing
            builder = ModularDatasetBuilder([generator], verbose=True)

            # Process using the builder's new chunked infrastructure
            # Use smaller chunk size for two-pass adaptive processing to avoid memory issues
            targets_df = builder.build_targets_from_parquet_chunked(
                input_file,
                symbol=symbol_name,
                chunk_size=200_000  # Reduced chunk size for memory-efficient two-pass processing
            )

            # Get target info for metadata
            target_info = generator.get_target_info()
            target_names = target_info['target_names']
 
            # Save targets file
            output_file = output_dir / f"{symbol_name}_{method}_targets.parquet"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            targets_df.write_parquet(output_file)

            print(f"   ✅ {method} complete - {len(targets_df)} targets generated")
            print(f"   💾 Saved to: {output_file}")
            print(f"   🏷️  Target columns: {target_names}")
            print(f"   📋 Output columns: {targets_df.columns}")

            # Store results
            results[method] = {
                "success": True,
                "output_file": str(output_file),
                "target_names": target_names,
                "target_count": len(targets_df),
                "parameters": params
            }

        except Exception as e:
            print(f"   ❌ {method} failed: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {
                "success": False,
                "error": str(e),
                "parameters": params if 'params' in locals() else None
            }

    return results

def main():
    """Main execution function."""
    print("🚀 PROCESSING ALL SYMBOLS WITH ADAPTIVE TRIPLE BARRIER METHOD")
    print("=" * 80)
    print("Using ACTUAL optimized parameters and ModularDatasetBuilder infrastructure")
    print("Output: Target-only files with keys + adaptive triple barrier target columns")
    print()

    # Define paths
    input_dir = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs")
    output_dir = Path("/Users/danielfisher/data/databento/symbol_datasets/triple_methods")

    # Find all symbol files
    symbol_files = list(input_dir.glob("*.parquet"))

    if not symbol_files:
        print(f"❌ No parquet files found in {input_dir}")
        return

    print(f"📊 Found {len(symbol_files)} symbol files to process")

    # Process each symbol with progress bar
    all_results = {}
    successful_symbols = 0

    # Create file-level progress bar
    file_progress = tqdm(symbol_files, desc="Processing symbols", unit="file")

    for symbol_file in file_progress:
        symbol_name = symbol_file.stem.split('_')[0]
        file_progress.set_postfix(symbol=symbol_name)

        print(f"\n{'='*60}")
        print(f"📈 PROCESSING SYMBOL: {symbol_name}")
        print(f"{'='*60}")

        try:
            results = process_symbol_with_triple_methods_chunked(symbol_file, output_dir)

            if "error" in results:
                print(f"   ⚠️  Symbol processing failed: {results['error']}")
                all_results[symbol_file.stem] = results
            else:
                # Count successful methods (only adaptive triple barrier now)
                successful_methods = sum(1 for r in results.values() if r.get('success', False))
                print(f"   🎉 Symbol complete: {successful_methods}/1 method successful")

                all_results[symbol_file.stem] = results
                if successful_methods > 0:
                    successful_symbols += 1

        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            all_results[symbol_file.stem] = {"error": f"Unexpected error: {e}"}

    file_progress.close()

    # Generate summary
    print(f"\n🎉 PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"   Successfully processed: {successful_symbols}/{len(symbol_files)} symbols")

    total_methods_successful = 0
    total_methods_attempted = 0

    for symbol_results in all_results.values():
        if "error" not in symbol_results:
            for method_result in symbol_results.values():
                total_methods_attempted += 1
                if method_result.get('success', False):
                    total_methods_successful += 1

    print(f"   Method success rate: {total_methods_successful}/{total_methods_attempted} ({100*total_methods_successful/max(1,total_methods_attempted):.1f}%)")
    print(f"   Output directory: {output_dir}")

    # List output files
    output_files = list(output_dir.glob("*_targets.parquet"))
    if output_files:
        print(f"\n📁 Generated {len(output_files)} target files:")
        for f in sorted(output_files):
            print(f"   • {f.name}")

    # Save processing summary
    summary_file = output_dir / "processing_summary.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        f.write("Triple Methods Processing Summary\n")
        f.write(f"Generated on: {datetime.now()}\n\n")
        f.write(f"Symbols processed: {len(symbol_files)}\n")
        f.write(f"Symbols successful: {successful_symbols}\n")
        f.write(f"Methods attempted: {total_methods_attempted}\n")
        f.write(f"Methods successful: {total_methods_successful}\n")
        f.write(f"Success rate: {100*total_methods_successful/max(1,total_methods_attempted):.1f}%\n\n")

        f.write("Parameters used:\n")
        f.write("Individual optimized parameters loaded from JSON files for each symbol\n\n")

        f.write("Output files (target-only with keys):\n")
        for output_file in sorted(output_files):
            f.write(f"  {output_file.name}\n")

    print(f"\n📋 Processing summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
