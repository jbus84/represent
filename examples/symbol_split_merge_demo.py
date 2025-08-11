#!/usr/bin/env python3
"""
Symbol Split-Merge Demo
======================

Demonstrates the complete workflow:
1. Build comprehensive datasets from 10+ DBN files
2. Verify minimum sample requirements are met (60,500+ samples per symbol)
3. Show memory-efficient processing capabilities

This example uses the new symbol-split-merge architecture with automatic
sample requirement calculation and memory-efficient processing.
"""
# Add represent package to path
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from represent import (
    DatasetBuildConfig,
    build_datasets_from_dbn_files,
    create_represent_config,
)


def create_mock_dbn_files_for_demo(input_dir: Path, num_files: int, symbols: list, records_per_file: int):
    """
    Creates a set of mock DBN files for demonstration.

    Each file contains sufficient data for the new minimum requirements:
    - Base samples: 50,000 (configurable via config.samples)
    - Lookback rows: 5,000 (default)
    - Lookforward input: 5,000 (default)
    - Lookforward offset: 500 (default)
    - Total minimum: 60,500 samples per symbol
    """
    dbn_files = []
    for i in range(num_files):
        file_path = input_dir / f"AUDUSD-mock-{i:02d}.dbn.zst"
        # Distribute symbols across files to demonstrate merging
        symbol = symbols[i % len(symbols)]

        # Create more realistic market data with sufficient volume
        base_price = 100.0 + (i * 0.01)  # Slight price drift across files
        timestamps = np.arange(records_per_file) * 1_000_000_000 + 1609459200000000000 + (i * records_per_file * 1_000_000_000)

        # Generate price movements
        price_noise = np.random.normal(0, 0.001, records_per_file)  # Small price variations
        ask_prices = base_price + 0.00005 + price_noise  # Spread of ~0.5 pips
        bid_prices = base_price - 0.00005 + price_noise

        df = pl.DataFrame({
            'ask_px_00': ask_prices,
            'bid_px_00': bid_prices,
            'ask_sz_00': np.random.randint(10, 100, records_per_file),  # Variable sizes
            'bid_sz_00': np.random.randint(10, 100, records_per_file),
            'ask_ct_00': np.random.randint(1, 10, records_per_file),    # Variable counts
            'bid_ct_00': np.random.randint(1, 10, records_per_file),
            'symbol': [symbol] * records_per_file,
            'ts_event': timestamps,
            'variance': np.random.exponential(0.01, records_per_file)   # Market variance
        })
        df.write_parquet(file_path)
        dbn_files.append(str(file_path))
    return dbn_files


def main():
    """
    Main function demonstrating the symbol-split-merge architecture with
    automatic minimum sample requirement calculation.
    """
    print("🏗️  Symbol Split-Merge Architecture Demo")
    print("=" * 70)
    print("Demonstrates:")
    print("  - Automatic minimum sample calculation (60,500+ per symbol)")
    print("  - Memory-efficient processing of multiple DBN files")
    print("  - Comprehensive symbol dataset creation")
    print("  - Uniform classification distribution")

    with tempfile.TemporaryDirectory() as tempdir:
        input_dir = Path(tempdir) / "dbn_input"
        output_dir = Path(tempdir) / "parquet_output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Step 1: Configure with proper minimum requirements
        print("\n📋 Step 1: Configuration")

        features = ['volume', 'variance']
        config = create_represent_config(
            currency="AUDUSD",
            features=features,
            samples=50000,          # Base samples needed
            lookback_rows=5000,     # Historical data required
            lookforward_input=5000, # Future data required
            lookforward_offset=500  # Offset before future window
        )

        # DatasetBuildConfig will auto-calculate minimum required samples
        dataset_config = DatasetBuildConfig(
            currency="AUDUSD",
            features=features,
            force_uniform=True,
            keep_intermediate=False
        )

        # Calculate minimum required samples
        min_required = config.samples + config.lookback_rows + config.lookforward_input + config.lookforward_offset
        print(f"   💱 Currency: {config.currency}")
        print(f"   📊 Features: {features}")
        print(f"   🎯 Minimum samples per symbol: {min_required:,}")
        print(f"      Base samples: {config.samples:,}")
        print(f"      + Lookback: {config.lookback_rows:,}")
        print(f"      + Lookforward: {config.lookforward_input:,}")
        print(f"      + Offset: {config.lookforward_offset:,}")

        # Step 2: Create mock data with sufficient volume
        print("\n🔄 Step 2: Creating mock DBN files")

        # Create 12 files to demonstrate the need for multiple files
        # Each file has 10K records, each symbol appears in multiple files
        symbols = ["M6AM4", "M6AU4"]  # Realistic symbol names
        num_files = 12
        records_per_file = 10000

        print(f"   📁 Creating {num_files} mock DBN files")
        print(f"   📊 {records_per_file:,} records per file")
        print(f"   💱 Symbols: {symbols}")
        print("   🔄 Each symbol will appear across multiple files for merging")

        start_time = time.perf_counter()
        dbn_files = create_mock_dbn_files_for_demo(
            input_dir,
            num_files=num_files,
            symbols=symbols,
            records_per_file=records_per_file
        )
        creation_time = time.perf_counter() - start_time

        print(f"   ✅ Created {len(dbn_files)} files in {creation_time:.1f}s")

        # Calculate expected data per symbol (each symbol appears in half the files)
        files_per_symbol = num_files // len(symbols) + (num_files % len(symbols))
        expected_samples_per_symbol = files_per_symbol * records_per_file
        print(f"   📊 Expected samples per symbol: ~{expected_samples_per_symbol:,}")
        print(f"   ✅ Sufficient for minimum requirement: {expected_samples_per_symbol >= min_required}")

        # Step 3: Run symbol-split-merge processing
        print("\n🏗️  Step 3: Symbol-Split-Merge Processing")
        print("   Processing phases:")
        print("   1. Split each DBN file by symbol")
        print("   2. Merge all instances of each symbol")
        print("   3. Apply uniform classification")
        print("   4. Generate feature representations")

        processing_start = time.perf_counter()

        try:
            results = build_datasets_from_dbn_files(
                config=config,
                dbn_files=dbn_files,
                output_dir=output_dir,
                dataset_config=dataset_config,
                verbose=True,
            )

            processing_time = time.perf_counter() - processing_start

            # Step 4: Verify results and minimum requirements
            print("\n✅ Step 4: Results Verification")
            print(f"   ⏱️  Total processing time: {processing_time:.2f}s")
            print(f"   🚀 Processing rate: {results['samples_per_second']:.0f} samples/sec")
            print(f"   📁 Datasets created: {results['phase_2_stats']['datasets_created']}")
            print(f"   📊 Total samples: {results['phase_2_stats']['total_samples']:,}")

            # Verify each dataset meets minimum requirements
            datasets_meeting_requirements = 0
            print("\n   🔍 Dataset Analysis:")

            for symbol, info in results['dataset_files'].items():
                meets_requirement = info['samples'] >= min_required
                if meets_requirement:
                    datasets_meeting_requirements += 1

                status = "✅" if meets_requirement else "❌"
                print(f"      {status} {symbol}:")
                print(f"         📊 Samples: {info['samples']:,} (required: {min_required:,})")
                print(f"         💾 Size: {info['file_size_mb']:.1f} MB")
                print(f"         📁 Source files: {info['source_files']}")

                # Verify classification and features
                df = pl.read_parquet(info['file_path'])
                has_classification = 'classification_label' in df.columns
                has_features = all(f"{feat}_representation" in df.columns for feat in features)

                print(f"         🏷️  Classification: {'✅' if has_classification else '❌'}")
                print(f"         📊 Features: {'✅' if has_features else '❌'}")

                if has_classification:
                    class_dist = df['classification_label'].value_counts().sort('classification_label')
                    uniform = class_dist['count'].std() < 0.1 * class_dist['count'].mean()
                    print(f"         ⚖️  Uniform distribution: {'✅' if uniform else '❌'}")

            # Final summary
            print("\n🎉 DEMO COMPLETE!")
            print(f"   📊 Datasets meeting requirements: {datasets_meeting_requirements}/{len(results['dataset_files'])}")

            if datasets_meeting_requirements == len(results['dataset_files']):
                print("   ✅ All datasets ready for ML training!")
                print("   🚀 Each dataset contains comprehensive symbol data from multiple files")
                print("   ⚖️  Uniform classification ensures balanced training")
            else:
                print("   ⚠️  Some datasets below minimum requirements")
                print("   💡 In real usage, use more DBN files or adjust configuration")

        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            raise

    print("\n📚 Next Steps:")
    print("   1. Implement custom dataloader for ML training")
    print("   2. Use comprehensive datasets for robust model training")
    print("   3. See DATALOADER_MIGRATION_GUIDE.md for implementation details")


if __name__ == "__main__":
    # Mock the databento.read_dbn function to read from our mock Parquet files
    with patch('represent.dataset_builder.db.read_dbn') as mock_read_dbn:
        def side_effect(path):
            mock_store = Mock()
            df = pl.read_parquet(path)
            mock_store.to_df.return_value = df
            return mock_store
        mock_read_dbn.side_effect = side_effect

        main()
