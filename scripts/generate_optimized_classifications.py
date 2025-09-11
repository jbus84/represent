#!/usr/bin/env python3
"""
Generate Optimized Classification Outputs

This script creates labeled datasets for each symbol using the optimized parameters
from the parameter optimization process. It generates outputs for all classification
methods: GA Labeling, Binary CTL, Ternary CTL, and Quantile Classification.

Usage:
    python scripts/generate_optimized_classifications.py --input-dir /path/to/inputs --output-dir /path/to/outputs
    python scripts/generate_optimized_classifications.py --symbol M6AH5 --method ga_labeling
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

from represent.modular_dataset_builder import ModularDatasetBuilder
from represent.parameter_storage import ParameterStorage
from represent.target_generators.factory import TargetGeneratorFactory


def load_optimized_parameters(storage_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load all optimized parameters from storage."""
    try:
        storage = ParameterStorage(str(storage_dir))
        all_params = {}
        
        # Discover all parameter files
        for symbol_dir in storage_dir.glob("*"):
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                all_params[symbol] = {}
                
                for param_file in symbol_dir.glob("*_params.json"):
                    method = param_file.stem.replace("_params", "")
                    try:
                        params = storage.load_symbol_parameters(symbol, method)
                        all_params[symbol][method] = params
                        print(f"✅ Loaded {method} parameters for {symbol}")
                    except Exception as e:
                        print(f"⚠️  Failed to load {method} parameters for {symbol}: {e}")
        
        return all_params
    except Exception as e:
        print(f"❌ Failed to load parameters from {storage_dir}: {e}")
        return {}


def create_classification_dataset(
    input_path: Path, 
    symbol: str, 
    method: str, 
    params: Dict[str, Any], 
    output_dir: Path,
    chunk_size: int = 100_000
) -> Path:
    """Create a classification dataset using optimized parameters with memory-efficient chunking."""
    
    print(f"\n🎯 Generating {method.upper()} classification for {symbol}")
    print(f"   📂 Input: {input_path.name}")
    
    # Extract optimal parameters
    optimal_params = params.get("optimal_params", {})
    if not optimal_params:
        raise ValueError(f"No optimal parameters found for {symbol} {method}")
    
    print(f"   🔧 Parameters: {optimal_params}")
    
    # Load input data and check size
    print(f"   📊 Loading input dataset...")
    input_df = pl.read_parquet(input_path)
    n_samples = len(input_df)
    print(f"   📊 Loaded {n_samples:,} samples")
    
    # Determine if we need chunking (GA Labeling is memory intensive)
    use_chunking = method == "ga_labeling" and n_samples > chunk_size
    
    if use_chunking:
        print(f"   🔄 Using chunked processing (chunk_size={chunk_size:,}) to manage memory")
        return _create_classification_chunked(input_path, symbol, method, optimal_params, output_dir, chunk_size)
    else:
        print(f"   🔄 Processing full dataset in memory")
        return _create_classification_full(input_path, symbol, method, optimal_params, output_dir)


def _create_classification_full(
    input_path: Path,
    symbol: str, 
    method: str,
    optimal_params: Dict[str, Any],
    output_dir: Path
) -> Path:
    """Create classification dataset without chunking."""
    # Create target generator with optimized parameters
    try:
        generator = TargetGeneratorFactory.create(method, **optimal_params)
    except Exception as e:
        print(f"   ❌ Failed to create generator: {e}")
        raise
    
    # Create dataset builder
    builder = ModularDatasetBuilder([generator], verbose=True)
    
    # Load and process data
    input_df = pl.read_parquet(input_path)
    
    # Generate targets
    print(f"   🎯 Generating optimized {method} targets...")
    targets_df = builder.build_targets(input_df, symbol=symbol)
    
    # Save and return
    return _save_classification_dataset(targets_df, input_path, symbol, method, output_dir)


def _create_classification_chunked(
    input_path: Path,
    symbol: str,
    method: str, 
    optimal_params: Dict[str, Any],
    output_dir: Path,
    chunk_size: int
) -> Path:
    """Create classification dataset using chunked processing for memory efficiency."""
    
    # Create target generator with optimized parameters
    try:
        generator = TargetGeneratorFactory.create(method, **optimal_params)
    except Exception as e:
        print(f"   ❌ Failed to create generator: {e}")
        raise
        
    # Create dataset builder
    builder = ModularDatasetBuilder([generator], verbose=False)  # Disable verbose for chunks
    
    # Load full dataset to get size
    input_df = pl.read_parquet(input_path)
    n_samples = len(input_df)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    print(f"   🔄 Processing {n_chunks} chunks of up to {chunk_size:,} samples each")
    
    # Process chunks and collect results
    all_chunks = []
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_samples)
        
        print(f"   🔄 Processing chunk {chunk_idx + 1}/{n_chunks} (samples {start_idx:,}-{end_idx:,})")
        
        # Get chunk
        chunk_df = input_df.slice(start_idx, end_idx - start_idx)
        
        # Generate targets for chunk
        try:
            chunk_targets = builder.build_targets(chunk_df, symbol=symbol)
            all_chunks.append(chunk_targets)
            print(f"      ✅ Chunk {chunk_idx + 1} complete ({len(chunk_targets):,} samples)")
        except Exception as e:
            print(f"      ❌ Chunk {chunk_idx + 1} failed: {e}")
            # Create empty targets for this chunk to maintain alignment
            chunk_targets = chunk_df.with_row_index("row_idx").select(["row_idx"])
            if symbol:
                chunk_targets = chunk_targets.with_columns(pl.lit(symbol).alias("symbol"))
            all_chunks.append(chunk_targets)
    
    # Combine all chunks
    print(f"   🔗 Combining {len(all_chunks)} chunks...")
    combined_df = pl.concat(all_chunks, how="vertical")
    
    # Save and return
    return _save_classification_dataset(combined_df, input_path, symbol, method, output_dir)


def _save_classification_dataset(
    targets_df: pl.DataFrame,
    input_path: Path,
    symbol: str,
    method: str, 
    output_dir: Path
) -> Path:
    """Save classification dataset and report statistics."""
    # Create output filename
    timestamp = input_path.stem.split("_")[-1]  # Extract timestamp from input filename
    output_filename = f"{symbol}_{method}_optimized_{timestamp}.parquet"
    output_path = output_dir / output_filename
    
    # Save dataset
    print(f"   💾 Saving to: {output_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_df.write_parquet(output_path)
    
    # Report statistics
    target_cols = [col for col in targets_df.columns if col not in ["row_idx", "symbol", "timestamp"]]
    print(f"   ✅ Saved {len(targets_df):,} samples with {len(target_cols)} target columns")
    print(f"   🎯 Target columns: {', '.join(target_cols)}")
    
    return output_path


def create_markdown_report(results: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Create comprehensive markdown report of generated classifications."""
    from datetime import datetime
    
    report_path = output_dir / "OPTIMIZED_CLASSIFICATIONS_REPORT.md"
    
    # Organize results by symbol and method
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    symbols = sorted(set(r["symbol"] for r in results))
    methods = ["ga_labeling", "binary_ctl", "ternary_ctl", "quantile_classification"]
    method_names = {
        "ga_labeling": "GA Labeling",
        "binary_ctl": "Binary CTL", 
        "ternary_ctl": "Ternary CTL",
        "quantile_classification": "Quantile Classification"
    }
    
    with open(report_path, "w") as f:
        f.write("# Optimized Classification Datasets Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"This report summarizes the optimized classification datasets generated using symbol-specific parameters.\n\n")
        f.write(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Classifications Generated**: {len(successful)}\n")
        f.write(f"**Failed Classifications**: {len(failed)}\n")
        f.write(f"**Symbols Processed**: {len(symbols)}\n")
        f.write(f"**Methods Available**: {len(methods)}\n\n")
        
        f.write("## Generation Summary\n\n")
        f.write("| Symbol | GA Labeling | Binary CTL | Ternary CTL | Quantile Class |\n")
        f.write("|--------|-------------|------------|-------------|----------------|\n")
        
        for symbol in symbols:
            row = [symbol]
            for method in methods:
                result = next((r for r in results if r["symbol"] == symbol and r["method"] == method), None)
                if result:
                    if result["status"] == "success":
                        row.append("✅")
                    else:
                        row.append("❌")
                else:
                    row.append("⚪")
            f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n## Generated Datasets\n\n")
        if successful:
            f.write("### Successful Classifications\n\n")
            for result in successful:
                f.write(f"#### {result['symbol']} - {method_names[result['method']]}\n")
                f.write(f"- **Method**: {result['method']}\n")
                f.write(f"- **Output File**: `{Path(result['output_path']).name}`\n")
                f.write(f"- **Input Dataset**: `{Path(result['input_path']).name}`\n")
                f.write(f"- **Status**: ✅ Successfully generated\n\n")
        
        if failed:
            f.write("### Failed Classifications\n\n")
            for result in failed:
                f.write(f"#### {result['symbol']} - {method_names[result['method']]}\n")
                f.write(f"- **Method**: {result['method']}\n") 
                f.write(f"- **Input Dataset**: `{Path(result['input_path']).name}`\n")
                f.write(f"- **Status**: ❌ Failed\n")
                f.write(f"- **Error**: {result['error']}\n\n")
        
        f.write("## Classification Methods\n\n")
        f.write("### GA Labeling (Genetic Algorithm)\n")
        f.write("- **Type**: Evolutionary optimization-based classification\n")
        f.write("- **Parameters**: Population size, generations, lookforward window, mutation rate\n")
        f.write("- **Output**: Binary/multi-class labels optimized for trading performance\n\n")
        
        f.write("### Binary CTL (Continuous Time Labels)\n")
        f.write("- **Type**: Academic binary trend classification\n") 
        f.write("- **Parameters**: Omega (filtering threshold)\n")
        f.write("- **Output**: Binary trend direction labels (Up/Down)\n\n")
        
        f.write("### Ternary CTL (Continuous Time Labels)\n")
        f.write("- **Type**: Academic ternary trend classification\n")
        f.write("- **Parameters**: Marginal change threshold, window size\n")
        f.write("- **Output**: Ternary labels (Up/Neutral/Down)\n\n")
        
        f.write("### Quantile Classification\n")
        f.write("- **Type**: Balanced percentile-based labeling\n")
        f.write("- **Parameters**: Number of quantile bins\n")
        f.write("- **Output**: Multi-class labels with equal distribution\n\n")
        
        f.write("## File Locations\n\n")
        f.write(f"**Classification Datasets**: `{output_dir}/`\n")
        f.write(f"**Input Datasets**: Available in inputs directory\n")
        f.write(f"**Optimized Parameters**: `optimization_results/optimized_parameters/`\n\n")
        
        f.write("## Usage\n\n")
        f.write("The generated classification datasets can be used for:\n")
        f.write("- Machine learning model training\n")
        f.write("- Backtesting trading strategies\n") 
        f.write("- Performance comparison between methods\n")
        f.write("- Research and analysis\n\n")
        
        f.write("Each dataset contains:\n")
        f.write("- All original market microstructure columns (76 columns)\n")
        f.write("- Optimized target labels specific to each method\n")
        f.write("- Row indexing for joining with other datasets\n")
        f.write("- Symbol and timestamp information\n\n")
        
        f.write("---\n")
        f.write("*Generated by Represent Optimized Classification System*\n")
    
    print(f"📄 Classification report saved: {report_path}")
    return report_path


def generate_all_classifications(
    input_dir: Path, 
    params_dir: Path, 
    output_dir: Path,
    symbol_filter: str = None,
    method_filter: str = None
) -> List[Dict[str, Any]]:
    """Generate optimized classifications for all symbols and methods."""
    
    print("🚀 OPTIMIZED CLASSIFICATION GENERATION")
    print("=" * 60)
    
    # Load optimized parameters
    print(f"\n📂 Loading optimized parameters from: {params_dir}")
    all_params = load_optimized_parameters(params_dir)
    
    if not all_params:
        raise ValueError("No optimized parameters found!")
    
    print(f"✅ Found parameters for {len(all_params)} symbols")
    
    # Discover input datasets
    print(f"\n📂 Discovering input datasets in: {input_dir}")
    input_files = list(input_dir.glob("*_inputs_only_dataset_*.parquet"))
    
    if not input_files:
        raise ValueError(f"No input datasets found in {input_dir}")
    
    print(f"✅ Found {len(input_files)} input datasets")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Available classification methods
    classification_methods = ["ga_labeling", "binary_ctl", "ternary_ctl", "quantile_classification"]
    
    # Generate classifications
    results = []
    total_generated = 0
    
    for input_file in input_files:
        # Extract symbol from filename
        symbol = input_file.stem.split("_")[0]
        
        # Apply symbol filter
        if symbol_filter and symbol != symbol_filter:
            print(f"⏭️  Skipping {symbol} (filtered)")
            continue
        
        print(f"\n🔍 Processing symbol: {symbol}")
        
        if symbol not in all_params:
            print(f"   ❌ No optimized parameters found for {symbol}")
            continue
        
        symbol_params = all_params[symbol]
        
        # Generate for each available method
        for method in classification_methods:
            # Apply method filter
            if method_filter and method != method_filter:
                continue
                
            if method not in symbol_params:
                print(f"   ⚠️  No {method} parameters for {symbol}")
                continue
            
            try:
                output_path = create_classification_dataset(
                    input_file, symbol, method, symbol_params[method], output_dir
                )
                
                results.append({
                    "symbol": symbol,
                    "method": method,
                    "input_path": str(input_file),
                    "output_path": str(output_path),
                    "status": "success"
                })
                total_generated += 1
                
            except Exception as e:
                print(f"   ❌ Failed to generate {method} for {symbol}: {e}")
                results.append({
                    "symbol": symbol, 
                    "method": method,
                    "input_path": str(input_file),
                    "output_path": None,
                    "status": "failed",
                    "error": str(e)
                })
    
    print(f"\n🎉 GENERATION COMPLETE!")
    print(f"✅ Successfully generated {total_generated} optimized classification datasets")
    print(f"📁 Output directory: {output_dir}")
    
    # Generate markdown report
    print(f"\n📄 Generating classification report...")
    create_markdown_report(results, output_dir)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized classification outputs for all symbols",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all classifications for all symbols
  python scripts/generate_optimized_classifications.py

  # Generate only for specific symbol
  python scripts/generate_optimized_classifications.py --symbol M6AH5
  
  # Generate only specific method
  python scripts/generate_optimized_classifications.py --method ga_labeling
  
  # Custom directories
  python scripts/generate_optimized_classifications.py \\
    --input-dir /path/to/inputs \\
    --params-dir /path/to/params \\
    --output-dir /path/to/outputs
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/danielfisher/data/databento/symbol_datasets/inputs"),
        help="Directory containing input datasets"
    )
    parser.add_argument(
        "--params-dir", 
        type=Path,
        default=Path("optimization_results/optimized_parameters"),
        help="Directory containing optimized parameters"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/danielfisher/data/databento/symbol_datasets/optimized_classifications"),
        help="Output directory for classification datasets"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Generate only for specific symbol (e.g., M6AH5)"
    )
    parser.add_argument(
        "--method",
        choices=["ga_labeling", "binary_ctl", "ternary_ctl", "quantile_classification"],
        help="Generate only for specific method"
    )
    
    args = parser.parse_args()
    
    try:
        results = generate_all_classifications(
            args.input_dir,
            args.params_dir, 
            args.output_dir,
            args.symbol,
            args.method
        )
        
        # Summary report
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]
        
        print(f"\n📊 SUMMARY REPORT")
        print("=" * 30)
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        if successful:
            print(f"\n🎯 Generated Classifications:")
            for result in successful:
                print(f"   {result['symbol']}: {result['method']}")
        
        if failed:
            print(f"\n❌ Failed Classifications:")
            for result in failed:
                print(f"   {result['symbol']}: {result['method']} - {result['error']}")
                
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())