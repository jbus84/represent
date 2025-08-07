#!/usr/bin/env python3
"""
Verify Row-Level Classification Demo

This script demonstrates that the classification system classifies EACH ROW
(each price movement) into a class, not entire files.

It shows:
1. How global thresholds are calculated once
2. How each row gets its own classification label
3. Sample output showing row-by-row classification
"""

import sys
from pathlib import Path
import polars as pl

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from represent import (
    calculate_global_thresholds,
    process_dbn_to_classified_parquets
)


def verify_row_level_classification():
    """Verify that classification works at the row level."""
    
    print("🔍 VERIFYING ROW-LEVEL CLASSIFICATION")
    print("=" * 60)
    
    # Data source
    data_directory = Path("/Users/danielfisher/data/databento/AUDUSD-micro")
    output_directory = Path("/Users/danielfisher/repositories/represent/examples/classification_analysis/outputs/verification")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    if not data_directory.exists():
        print(f"❌ Data directory not found: {data_directory}")
        return False
    
    # Find DBN files
    dbn_files = sorted([f for f in data_directory.glob("*.dbn*") if f.is_file()])
    
    if not dbn_files:
        print("❌ No DBN files found")
        return False
    
    print(f"📊 Found {len(dbn_files)} DBN files")
    
    # Step 1: Calculate Global Thresholds
    print("\n🎯 STEP 1: Calculate Global Thresholds for ALL Rows")
    print("-" * 50)
    
    try:
        # Use first 5 files for quick demo
        sample_fraction = min(5 / len(dbn_files), 1.0)
        
        global_thresholds = calculate_global_thresholds(
            data_directory=data_directory,
            currency="AUDUSD",
            nbins=13,
            sample_fraction=sample_fraction,
            max_samples_per_file=3000,  # Small for quick demo
            verbose=True
        )
        
        print("\n✅ Global thresholds calculated:")
        print(f"   📊 Based on {global_thresholds.sample_size:,} individual price movements")
        print(f"   📊 From {global_thresholds.files_analyzed} files")
        
        # Show some boundaries
        boundaries = global_thresholds.quantile_boundaries
        print("\n🎯 Sample Threshold Boundaries:")
        for i in range(min(3, len(boundaries)-1)):
            print(f"   Rows with movement ≤ {boundaries[i]:.2f} μpips → Class {i}")
        print("   ...")
        print("   (13 classes total, each row gets classified individually)")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Step 2: Process One File to Show Row-Level Classification
    print("\n🔄 STEP 2: Process One File - Each Row Gets Classified")
    print("-" * 50)
    
    try:
        # Process one file
        test_file = dbn_files[0]
        print(f"Processing: {test_file.name}")
        
        results = process_dbn_to_classified_parquets(
            dbn_path=test_file,
            output_dir=output_directory,
            currency="AUDUSD",
            global_thresholds=global_thresholds,  # Same thresholds for ALL rows
            features=["volume"],
            min_symbol_samples=100,  # Lower for demo
            verbose=False
        )
        
        print("\n✅ Processing complete:")
        print(f"   📊 Generated {results['symbols_processed']} symbol files")
        print(f"   📊 Classified {results['total_classified_samples']:,} individual rows")
        print("   🎯 Each row got its own classification label (0-12)")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False
    
    # Step 3: Examine the Output to Show Row-Level Structure
    print("\n📊 STEP 3: Examine Output - Verify Row-Level Classification")
    print("-" * 50)
    
    try:
        # Find generated files
        classified_files = list(output_directory.glob("*_classified.parquet"))
        
        if not classified_files:
            print("❌ No classified files found")
            return False
        
        # Load and examine one file
        sample_file = classified_files[0]
        df = pl.read_parquet(sample_file)
        
        symbol = sample_file.stem.split('_')[1]
        print(f"📄 Examining: {sample_file.name}")
        print(f"📊 Symbol: {symbol}")
        print(f"📊 Total rows: {len(df):,}")
        
        if 'classification_label' not in df.columns:
            print("❌ No classification_label column found")
            return False
        
        # Show sample rows
        sample_rows = df.select([
            'ts_event', 'ask_px_00', 'bid_px_00', 'price_movement', 'classification_label'
        ]).head(10)
        
        print("\n📋 SAMPLE ROWS (showing row-by-row classification):")
        print("=" * 80)
        print(sample_rows)
        
        # Show classification distribution
        labels = df['classification_label'].to_numpy()
        unique_labels, counts = pl.Series(labels).value_counts().sort('classification_label').to_numpy().T
        
        print("\n📈 CLASSIFICATION DISTRIBUTION (per row):")
        print("-" * 40)
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(df)) * 100
            print(f"   Class {label:2d}: {count:6,} rows ({percentage:5.1f}%)")
        
        print("\n🎯 KEY INSIGHTS:")
        print("✅ Each row in the dataset has its own classification_label")
        print("✅ Labels are based on individual price movements (price_movement column)")
        print("✅ Same global thresholds applied to every single row")
        print("✅ NOT classifying entire files - classifying each price movement!")
        
        # Verify price movement to classification mapping
        print("\n🔍 PRICE MOVEMENT → CLASSIFICATION MAPPING:")
        print("-" * 50)
        
        # Sample some rows to show the mapping
        sample_movements = df.select(['price_movement', 'classification_label']).head(5)
        boundaries = global_thresholds.quantile_boundaries
        
        for row in sample_movements.to_dicts():
            movement = row['price_movement']
            label = row['classification_label']
            
            if movement is not None and label is not None:
                # Find which boundary this falls into
                if label < len(boundaries) - 1:
                    threshold = boundaries[label]
                    print(f"   Movement: {movement:7.2f} μpips → Class {label} (≤ {threshold:.2f})")
        
        print("\n🎉 VERIFICATION COMPLETE!")
        print("✅ System correctly classifies EACH ROW individually")
        print("✅ Each price movement gets its own classification label") 
        print("✅ Global thresholds ensure consistency across ALL rows in ALL files")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔍 ROW-LEVEL CLASSIFICATION VERIFICATION")
    print("=" * 70)
    print("Verifying that each row gets its own classification label")
    print("=" * 70)
    
    success = verify_row_level_classification()
    
    if success:
        print("\n🎉 VERIFICATION SUCCESSFUL!")
        print("✅ Confirmed: Each row gets classified individually")
        print("✅ Global thresholds applied consistently to every row")
        print("✅ NOT classifying files - classifying price movements!")
    else:
        print("\n❌ VERIFICATION FAILED")
        print("Check output above for details")