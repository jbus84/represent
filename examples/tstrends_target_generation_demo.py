#!/usr/bin/env python3
"""
TStrends Target Generation Demo

This demo shows how to use the tstrends-based target generators integrated
into the modular target generation system.
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from represent.target_generators.tstrends_labeling import (
        BinaryCTLGenerator,
        TernaryCTLGenerator,
        OracleBinaryTrendGenerator,
        OracleTernaryTrendGenerator,
        TSTRENDS_AVAILABLE
    )
    from represent import ModularDatasetBuilder, TargetGeneratorFactory
    
    if not TSTRENDS_AVAILABLE:
        print("❌ TStrends library not available. Install with:")
        print("   uv add git+https://github.com/agpenas/tstrends.git")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure tstrends is installed and represent package is available")
    sys.exit(1)


def create_sample_market_data(n_rows: int = 1000) -> pl.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic-looking price data with trends
    base_price = 0.6500  # AUDUSD-like price
    
    # Create trending price series
    trend_changes = np.random.choice([-1, 0, 1], n_rows, p=[0.3, 0.4, 0.3])
    trend_strength = np.random.normal(0, 0.0001, n_rows)
    price_changes = trend_changes * 0.0002 + trend_strength
    
    prices = base_price + np.cumsum(price_changes)
    
    # Ensure prices stay positive
    prices = np.maximum(prices, 0.5000)
    
    # Create timestamps
    timestamps = np.arange(n_rows) * 1000  # Millisecond timestamps
    
    return pl.DataFrame({
        "timestamp": timestamps,
        "mid_price": prices,
        "volume": np.random.exponential(1000, n_rows),  # Volume data
    })


def demo_binary_ctl():
    """Demo: Binary Cumulative Trend Labelling."""
    print("🎯 DEMO 1: Binary CTL (Cumulative Trend Labelling)")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_market_data()
    
    # Test different omega values
    omega_values = [0.01, 0.02, 0.05]
    
    for omega in omega_values:
        print(f"\n📊 Testing Binary CTL with omega={omega}")
        
        try:
            generator = BinaryCTLGenerator(omega=omega, target_name=f"binary_ctl_{omega}")
            builder = ModularDatasetBuilder([generator])
            
            dataset = builder.build_dataset(sample_data)
            
            labels = dataset[f"binary_ctl_{omega}"].to_numpy()
            unique_labels = np.unique(labels[~np.isnan(labels)])
            
            print(f"   ✅ Generated {len(labels):,} labels")
            print(f"   📈 Unique labels: {unique_labels}")
            print(f"   📊 Label distribution: {np.bincount(labels.astype(int))}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


def demo_ternary_ctl():
    """Demo: Ternary Cumulative Trend Labelling."""
    print("\n\n🎯 DEMO 2: Ternary CTL")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_market_data()
    
    try:
        generator = TernaryCTLGenerator(
            marginal_change_thres=0.02,
            window_size=10,
            target_name="ternary_ctl"
        )
        builder = ModularDatasetBuilder([generator])
        
        dataset = builder.build_dataset(sample_data)
        
        labels = dataset["ternary_ctl"].to_numpy()
        unique_labels = np.unique(labels[~np.isnan(labels)])
        
        print(f"   ✅ Generated {len(labels):,} labels")
        print(f"   📈 Unique labels: {unique_labels}")
        print(f"   📊 Label distribution: {np.bincount(labels.astype(int))}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_oracle_labelling():
    """Demo: Oracle trend labelling."""
    print("\n\n🎯 DEMO 3: Oracle Trend Labelling")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_market_data()
    
    print("\n📊 Oracle Binary Trend Labelling:")
    try:
        generator = OracleBinaryTrendGenerator(
            transaction_cost=0.001,
            target_name="oracle_binary"
        )
        builder = ModularDatasetBuilder([generator])
        
        dataset = builder.build_dataset(sample_data)
        
        labels = dataset["oracle_binary"].to_numpy()
        unique_labels = np.unique(labels[~np.isnan(labels)])
        
        print(f"   ✅ Generated {len(labels):,} labels")
        print(f"   📈 Unique labels: {unique_labels}")
        print(f"   📊 Label distribution: {np.bincount(labels.astype(int))}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n📊 Oracle Ternary Trend Labelling:")
    try:
        generator = OracleTernaryTrendGenerator(
            transaction_cost=0.001,
            neutral_reward_factor=0.5,
            target_name="oracle_ternary"
        )
        builder = ModularDatasetBuilder([generator])
        
        dataset = builder.build_dataset(sample_data)
        
        labels = dataset["oracle_ternary"].to_numpy()
        unique_labels = np.unique(labels[~np.isnan(labels)])
        
        print(f"   ✅ Generated {len(labels):,} labels")
        print(f"   📈 Unique labels: {unique_labels}")
        print(f"   📊 Label distribution: {np.bincount(labels.astype(int))}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_mixed_targets():
    """Demo: Mixing tstrends with other target generators."""
    print("\n\n🎯 DEMO 4: Mixed Target Generation")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_market_data()
    
    try:
        # Mix tstrends generators with existing ones
        from represent import QuantileClassificationGenerator, DirectionalMFEGenerator
        
        generators = [
            # Traditional represent generators
            QuantileClassificationGenerator(nbins=13, target_name="quantile_class"),
            DirectionalMFEGenerator(lookforward_horizon=100, target_names=("mfe_buy", "mfe_sell")),
            
            # TStrends generators
            BinaryCTLGenerator(omega=0.02, target_name="binary_ctl"),
            TernaryCTLGenerator(marginal_change_thres=0.02, window_size=10, target_name="ternary_ctl"),
            OracleBinaryTrendGenerator(transaction_cost=0.001, target_name="oracle_binary"),
        ]
        
        builder = ModularDatasetBuilder(generators)
        dataset = builder.build_dataset(sample_data)
        
        print(f"\n📊 Mixed target dataset created:")
        print(f"   Total columns: {len(dataset.columns)}")
        
        target_columns = [col for col in dataset.columns if col not in sample_data.columns]
        print(f"   Target columns: {target_columns}")
        
        # Show statistics for each target
        for target_col in target_columns:
            if target_col in dataset.columns:
                values = dataset[target_col].to_numpy()
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    unique_vals = np.unique(valid_values)
                    print(f"   📈 {target_col}: {len(valid_values):,} valid, unique: {unique_vals}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


def demo_factory_registration():
    """Demo: Register tstrends generators with factory."""
    print("\n\n🎯 DEMO 5: Factory Registration")
    print("=" * 60)
    
    try:
        # Register tstrends generators with factory
        TargetGeneratorFactory.register("binary_ctl", BinaryCTLGenerator)
        TargetGeneratorFactory.register("ternary_ctl", TernaryCTLGenerator)
        TargetGeneratorFactory.register("oracle_binary", OracleBinaryTrendGenerator)
        TargetGeneratorFactory.register("oracle_ternary", OracleTernaryTrendGenerator)
        
        print("✅ Registered tstrends generators with factory")
        
        # List all available generators
        available = TargetGeneratorFactory.list_available()
        print(f"📋 Available generators: {list(available.keys())}")
        
        # Create generators using factory
        sample_data = create_sample_market_data()
        
        generators = [
            TargetGeneratorFactory.create("binary_ctl", omega=0.02),
            TargetGeneratorFactory.create("oracle_binary", transaction_cost=0.001),
        ]
        
        builder = ModularDatasetBuilder(generators)
        dataset = builder.build_dataset(sample_data)
        
        print(f"✅ Factory-created dataset with {len(dataset.columns)} columns")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all demos."""
    print("🚀 TSTRENDS TARGET GENERATION DEMO")
    print("=" * 70)
    print("This demo shows integration of tstrends labelling approaches")
    print("into the modular target generation system.")
    print()
    
    try:
        demo_binary_ctl()
        demo_ternary_ctl()
        demo_oracle_labelling()
        demo_mixed_targets()
        demo_factory_registration()
        
        print("\n\n🎉 ALL TSTRENDS DEMOS COMPLETED!")
        print("\n💡 TStrends Integration Benefits:")
        print("   ✅ Academic trend labelling approaches")
        print("   ✅ Optimal oracle labelling")
        print("   ✅ Cumulative trend detection")
        print("   ✅ Seamless integration with existing generators")
        print("   ✅ Factory pattern support")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()