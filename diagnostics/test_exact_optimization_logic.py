#!/usr/bin/env python3
"""
Test Exact Optimization Logic

Test our enhanced output calculation using the EXACT same logic as the optimization
to understand why we're getting different results.
"""

import numpy as np
import polars as pl
from pathlib import Path

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    from tstrends.returns_estimation import ReturnsEstimatorWithFees
    from tstrends.returns_estimation.fees_config import FeesConfig
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


def calculate_pnl_optimization_exact(prices: np.ndarray, labels: np.ndarray, 
                                    transaction_cost: float = 0.00007) -> dict:
    """
    Calculate PnL using EXACT optimization logic from large_scale_optimization.py lines 882-915.
    """
    if not LIBRARIES_AVAILABLE:
        return {"error": "Libraries not available"}
    
    try:
        # EXACT replication of optimization logic
        fee_decimal = transaction_cost
        fees_config = FeesConfig(
            lp_transaction_fees=fee_decimal,
            sp_transaction_fees=fee_decimal,
        )
        returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
        
        # Convert labels to proper format for returns estimation
        labels_int = labels.astype(int)
        
        # Handle different label formats for ReturnsEstimatorWithFees
        # ReturnsEstimatorWithFees expects {-1, 0, 1} format:
        # -1 = short position, 0 = no position, 1 = long position
        
        unique_labels = np.unique(labels_int[~np.isnan(labels_int)])
        
        if set(unique_labels).issubset({0, 1, 2}):
            # Ternary labels from CTL generators: {0, 1, 2} → convert to {-1, 0, 1}
            # 0 (Down/Sell) → -1 (Short), 1 (Neutral) → 0 (Hold), 2 (Up/Buy) → 1 (Long)
            labels_tstrends = labels_int - 1  # {0,1,2} → {-1,0,1}
        elif len(unique_labels) == 2 and set(unique_labels).issubset({0, 1}):
            # Binary labels {0, 1} → convert to {0, 1} (0=no position, 1=long position)  
            # Note: This assumes binary is long-only strategy
            labels_tstrends = labels_int
        else:
            # Assume already in correct format or handle as-is
            labels_tstrends = labels_int
        
        returns = returns_estimator.estimate_return(
            prices.tolist(),
            labels_tstrends.tolist()
        )
        
        # Count trades
        num_trades = 0
        current_position = 0
        for target_position in labels_tstrends:
            if target_position != current_position:
                num_trades += 1
                current_position = target_position
        
        return {
            "method": "EXACT Optimization Logic",
            "total_pnl": returns,
            "num_trades": num_trades,
            "mean_return_per_trade": returns / num_trades if num_trades > 0 else 0,
            "labels_used": f"Original: {set(unique_labels)} -> TStrends: {set(labels_tstrends)}",
        }
    
    except Exception as e:
        return {"error": f"Calculation failed: {e}"}


def test_exact_optimization_logic():
    """Test using the exact same logic as optimization."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Required libraries not available")
        return
    
    print("🔬 Testing EXACT Optimization Logic")
    print("=" * 50)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Use same test sample
    test_df = df.head(10000)
    prices = test_df["mid_price"].to_numpy()
    
    print(f"Testing with {len(test_df):,} samples")
    print(f"Price range: {prices.min():.6f} to {prices.max():.6f}")
    print()
    
    # Test Binary CTL with optimized parameters
    omega = 6.339289260816091e-05  # From optimization result
    
    # Generate labels using our Binary CTL
    generator = TargetGeneratorFactory.create("binary_ctl", omega=omega)
    targets_df = generator.generate_targets(test_df)
    target_info = generator.get_target_info()
    target_col = target_info['target_names'][0]
    labels = targets_df[target_col].to_numpy()
    
    print(f"Labels distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print()
    
    # Test using EXACT optimization logic
    result = calculate_pnl_optimization_exact(prices, labels)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"🎯 {result['method']}:")
        print(f"   Total PnL: {result['total_pnl']:.6f}")
        print(f"   Num trades: {result['num_trades']:,}")
        print(f"   Mean return/trade: {result['mean_return_per_trade']:.8f}")
        print(f"   Labels mapping: {result['labels_used']}")
        
        # Convert to percentage for comparison
        pnl_pct = result['total_pnl'] * 100
        print(f"   PnL as percentage: {pnl_pct:.2f}%")
    
    print()
    print("💡 This should match the optimization's PnL calculation exactly!")


if __name__ == "__main__":
    test_exact_optimization_logic()