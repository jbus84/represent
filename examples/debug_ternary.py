#!/usr/bin/env python3
"""Debug script to analyze ternary generator output"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent.target_generators.tstrends_labeling import (
    TSTRENDS_AVAILABLE,
    OracleTernaryTrendGenerator,
    TernaryCTLGenerator,
)


def debug_ternary_generators():
    if not TSTRENDS_AVAILABLE:
        print("❌ TStrends not available")
        return

    print("🔍 DEBUGGING TERNARY GENERATORS")
    print("=" * 50)

    # Load sample data
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ No data directory found")
        return

    # Find first parquet file
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        print("❌ No parquet files found")
        return

    df = pl.read_parquet(parquet_files[0])
    print(f"📊 Loaded {len(df)} samples from {parquet_files[0].name}")

    # Take subset for faster testing
    df_test = df.head(10000)
    print(f"🎯 Testing with {len(df_test)} samples")

    # Test different parameter combinations for ternary
    ternary_params = [
        {"marginal_change_thres": 0.001, "window_size": 5, "name": "Very Sensitive"},
        {"marginal_change_thres": 0.002, "window_size": 5, "name": "Sensitive"},
        {"marginal_change_thres": 0.003, "window_size": 10, "name": "Moderate"},
        {"marginal_change_thres": 0.004, "window_size": 10, "name": "Balanced"},
        {"marginal_change_thres": 0.005, "window_size": 15, "name": "Conservative"},
        {"marginal_change_thres": 0.010, "window_size": 20, "name": "Very Conservative"},
    ]

    oracle_params = [
        {"transaction_cost": 0.0001, "neutral_reward_factor": 0.3, "name": "Low Cost, Low Neutral"},
        {"transaction_cost": 0.0002, "neutral_reward_factor": 0.5, "name": "Low Cost, Med Neutral"},
        {
            "transaction_cost": 0.0002,
            "neutral_reward_factor": 0.7,
            "name": "Low Cost, High Neutral",
        },
        {"transaction_cost": 0.0005, "neutral_reward_factor": 0.5, "name": "Med Cost, Med Neutral"},
        {"transaction_cost": 0.001, "neutral_reward_factor": 0.5, "name": "High Cost, Med Neutral"},
    ]

    print("\n🧪 TESTING TERNARY CTL GENERATORS")
    print("-" * 40)
    for params in ternary_params:
        try:
            gen = TernaryCTLGenerator(
                marginal_change_thres=params["marginal_change_thres"],
                window_size=params["window_size"],
                target_name="test_ternary",
            )

            targets = gen.generate_targets(df_test)
            labels = targets["test_ternary"]

            # Analyze distribution
            unique_labels = np.unique(labels)
            label_counts = Counter(labels)

            print(
                f"  {params['name']} (thres={params['marginal_change_thres']:.3f}, win={params['window_size']}):"
            )
            print(f"    Unique classes: {unique_labels}")
            print(f"    Distribution: {dict(label_counts)}")
            print(f"    Total classes: {len(unique_labels)}")

        except Exception as e:
            print(f"  ❌ {params['name']}: {e}")

    print("\n🔮 TESTING ORACLE TERNARY GENERATORS")
    print("-" * 40)
    for params in oracle_params:
        try:
            gen = OracleTernaryTrendGenerator(
                transaction_cost=params["transaction_cost"],
                neutral_reward_factor=params["neutral_reward_factor"],
                target_name="test_oracle",
            )

            targets = gen.generate_targets(df_test)
            labels = targets["test_oracle"]

            # Analyze distribution
            unique_labels = np.unique(labels)
            label_counts = Counter(labels)

            print(
                f"  {params['name']} (cost={params['transaction_cost']:.4f}, neutral={params['neutral_reward_factor']:.1f}):"
            )
            print(f"    Unique classes: {unique_labels}")
            print(f"    Distribution: {dict(label_counts)}")
            print(f"    Total classes: {len(unique_labels)}")

        except Exception as e:
            print(f"  ❌ {params['name']}: {e}")


if __name__ == "__main__":
    debug_ternary_generators()
