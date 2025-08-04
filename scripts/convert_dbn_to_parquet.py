#!/usr/bin/env python3
"""
DBN to Parquet Conversion Script

Convert DBN files to labeled parquet datasets for machine learning workflows.
Supports currency-specific configurations and custom YAML config files.

Examples:
    # Convert with predefined currency config
    python convert_dbn_to_parquet.py data.dbn output.parquet --currency AUDUSD

    # Convert with custom YAML config
    python convert_dbn_to_parquet.py data.dbn output.parquet --config my_config.yaml

    # Batch convert directory
    python convert_dbn_to_parquet.py input_dir/ output_dir/ --currency GBPUSD --batch

    # Convert with symbol filter
    python convert_dbn_to_parquet.py data.dbn output.parquet --currency EURJPY --symbol M6EM4
"""

import argparse
import sys
from pathlib import Path

from represent.converter import convert_dbn_file, batch_convert_dbn_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert DBN files to labeled parquet datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file conversion:
    %(prog)s data.dbn output.parquet --currency AUDUSD
    %(prog)s data.dbn output.parquet --config my_config.yaml
    
  Batch directory conversion:
    %(prog)s input_dir/ output_dir/ --batch --currency GBPUSD
    %(prog)s input_dir/ output_dir/ --batch --config custom.yaml
    
  With filtering and features:
    %(prog)s data.dbn out.parquet --currency EURJPY --symbol M6EM4 --features volume variance
        """,
    )

    # Input/Output arguments
    parser.add_argument("input", help="Input DBN file or directory (if --batch)")
    parser.add_argument("output", help="Output parquet file or directory (if --batch)")

    # Configuration arguments (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--currency",
        "-c",
        choices=["AUDUSD", "GBPUSD", "EURJPY", "EURUSD", "USDJPY"],
        help="Use predefined currency configuration",
    )
    config_group.add_argument("--config", "-f", help="Path to custom YAML/JSON configuration file")

    # Processing arguments
    parser.add_argument("--symbol", "-s", help="Filter by symbol (e.g., M6AM4)")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["volume"],
        choices=["volume", "variance", "trade_counts"],
        help="Features to extract (default: volume)",
    )
    parser.add_argument(
        "--batch", "-b", action="store_true", help="Batch process directory of DBN files"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100000, help="Processing chunk size (default: 100000)"
    )
    parser.add_argument(
        "--pattern", default="*.dbn*", help="File pattern for batch processing (default: *.dbn*)"
    )

    # Output control
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    parser.add_argument("--stats", action="store_true", help="Show detailed conversion statistics")

    args = parser.parse_args()

    # Validate arguments
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Error: Input path does not exist: {input_path}")
        sys.exit(1)

    if args.batch and not input_path.is_dir():
        print("❌ Error: --batch requires input to be a directory")
        sys.exit(1)

    if args.batch and output_path.exists() and not output_path.is_dir():
        print("❌ Error: --batch requires output to be a directory")
        sys.exit(1)

    # Set default configuration if none provided
    if not args.currency and not args.config:
        args.currency = "AUDUSD"
        if not args.quiet:
            print("⚠️  No configuration specified, using default AUDUSD")

    try:
        if args.batch:
            # Batch conversion
            if not args.quiet:
                print(f"🔄 Batch converting DBN files from {input_path}")
                if args.currency:
                    print(f"   Using currency config: {args.currency}")
                if args.config:
                    print(f"   Using config file: {args.config}")

            results = batch_convert_dbn_files(
                input_directory=input_path,
                output_directory=output_path,
                currency=args.currency,
                config_file=args.config,
                pattern=args.pattern,
                symbol_filter=args.symbol,
                features=args.features,
                chunk_size=args.chunk_size,
                include_metadata=True,
            )

            if not args.quiet:
                print(f"✅ Batch conversion completed: {len(results)} files processed")

            if args.stats:
                total_samples = sum(r["labeled_samples"] for r in results)
                total_time = sum(r["conversion_time_seconds"] for r in results)
                avg_throughput = total_samples / total_time if total_time > 0 else 0
                print("\n📊 Batch Statistics:")
                print(f"   Total labeled samples: {total_samples:,}")
                print(f"   Total processing time: {total_time:.1f}s")
                print(f"   Average throughput: {avg_throughput:.0f} samples/sec")

        else:
            # Single file conversion
            if input_path.is_dir():
                print("❌ Error: Single file mode requires input to be a file")
                print("    Use --batch flag for directory processing")
                sys.exit(1)

            if not args.quiet:
                print(f"🔄 Converting {input_path.name}")
                if args.currency:
                    print(f"   Using currency config: {args.currency}")
                if args.config:
                    print(f"   Using config file: {args.config}")

            stats = convert_dbn_file(
                dbn_path=input_path,
                output_path=output_path,
                currency=args.currency,
                config_file=args.config,
                symbol_filter=args.symbol,
                features=args.features,
                chunk_size=args.chunk_size,
                include_metadata=True,
            )

            if not args.quiet:
                print(f"✅ Conversion completed: {stats['labeled_samples']:,} samples")

            if args.stats:
                print("\n📊 Conversion Statistics:")
                for key, value in stats.items():
                    if key == "classification_config":
                        continue  # Skip complex nested dict
                    print(f"   {key}: {value}")

    except KeyboardInterrupt:
        print("\n⚠️  Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
