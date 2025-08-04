#!/usr/bin/env python3
"""
DBN to Parquet Conversion Script

Command-line utility to convert DBN files to labeled parquet datasets
for machine learning workflows.

Usage:
    python convert_dbn_to_parquet.py input.dbn output.parquet --currency AUDUSD
    python convert_dbn_to_parquet.py data/ output/ --batch --currency GBPUSD
"""
import argparse
import sys
from pathlib import Path

from represent.converter import convert_dbn_file, batch_convert_dbn_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert DBN files to labeled parquet datasets"
    )
    
    parser.add_argument(
        "input",
        help="Input DBN file or directory containing DBN files"
    )
    
    parser.add_argument(
        "output", 
        help="Output parquet file or directory for batch conversion"
    )
    
    parser.add_argument(
        "--currency",
        default="AUDUSD",
        help="Currency pair for classification configuration (default: AUDUSD)"
    )
    
    parser.add_argument(
        "--symbol",
        help="Symbol to filter (e.g., M6AM4)"
    )
    
    parser.add_argument(
        "--features",
        nargs="+",
        default=["volume"],
        choices=["volume", "variance", "trade_counts"],
        help="Features to extract (default: volume)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch convert all DBN files in input directory"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Chunk size for processing (default: 100000)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"❌ Input path does not exist: {input_path}")
        sys.exit(1)
    
    try:
        if args.batch:
            if not input_path.is_dir():
                print("❌ Batch mode requires input to be a directory")
                sys.exit(1)
            
            print(f"🔄 Batch converting DBN files from {input_path} to {output_path}")
            
            results = batch_convert_dbn_files(
                input_directory=input_path,
                output_directory=output_path,
                currency=args.currency,
                symbol_filter=args.symbol,
                features=args.features,
                chunk_size=args.chunk_size
            )
            
            print(f"✅ Batch conversion complete! Processed {len(results)} files")
            
        else:
            if input_path.is_dir():
                print("❌ Single file mode requires input to be a file, not directory")
                print("    Use --batch flag for directory processing")
                sys.exit(1)
            
            print(f"🔄 Converting {input_path} to {output_path}")
            
            stats = convert_dbn_file(
                dbn_path=input_path,
                output_path=output_path,
                currency=args.currency,
                symbol_filter=args.symbol,
                features=args.features,
                chunk_size=args.chunk_size
            )
            
            print("✅ Conversion complete!")
            print(f"📊 {stats['labeled_samples']:,} samples generated")
            print(f"📊 Output: {stats['output_file_size_mb']:.1f}MB")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()