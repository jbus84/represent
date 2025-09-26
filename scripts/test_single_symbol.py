#!/usr/bin/env python3
"""
Test single symbol processing to debug issues
"""

import sys
from pathlib import Path
import polars as pl
from tqdm import tqdm

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent.target_generators.factory import TargetGeneratorFactory
from represent.proven_parameters import PROVEN_PARAMETERS

def test_single_symbol():
    """Test processing a single symbol to debug issues."""

    # Use smallest dataset for testing
    input_file = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AU5_inputs_only_dataset_20250909_140950.parquet")

    print(f"🔧 TESTING: {input_file.stem}")

    # Load small sample first
    df = pl.read_parquet(input_file, n_rows=1000)
    print(f"📊 Loaded {len(df):,} samples for testing")
    print(f"📋 Columns: {df.columns}")

    # Test triple_barrier with PROVEN parameters first
    try:
        params = PROVEN_PARAMETERS['triple_barrier']
        print(f"⚙️  Testing with PROVEN params: {params}")

        generator = TargetGeneratorFactory.create('triple_barrier', **params)
        targets_df = generator.generate_targets(df)

        print(f"✅ Success! Generated {len(targets_df)} targets")
        print(f"🏷️  Target columns: {targets_df.columns}")

        # Test combination
        combined = pl.concat([df, targets_df], how="horizontal")
        print(f"🔗 Combined successfully: {len(combined)} rows, {len(combined.columns)} columns")

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_symbol()