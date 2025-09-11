#!/usr/bin/env python3
"""
Diagnose Binary CTL Return Calculation Issue

This script investigates the discrepancy between:
1. High overall returns (19.24%) from optimization
2. Negative mean return per trade (-0.000028) from our enhanced output

Key questions:
- How does Binary CTL calculate returns in optimization vs our calculation?
- Is there a label mapping issue between {-1,1} and {0,1}?
- Are we using different PnL calculation methods?
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    from tstrends.trend_labelling import BinaryCTL
    from tstrends.returns_estimation import ReturnsEstimatorWithFees
    from tstrends.returns_estimation.fees_config import FeesConfig
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


def calculate_pnl_method_a(prices: np.ndarray, labels: np.ndarray, 
                          transaction_cost: float = 0.00007) -> Dict[str, Any]:
    """
    Method A: Our enhanced output calculation (the one showing negative returns).
    
    This uses the remapped {0, 1} labels from our Binary CTL generator.
    """
    total_pnl = 0.0
    current_position = 0
    num_trades = 0
    
    # Normalize labels to trading positions {-1, 0, 1}
    positions = np.array(labels, dtype=np.float64)
    unique_labels_set = set(np.unique(labels))
    
    if unique_labels_set <= {0, 1}:  # Binary {0, 1} -> {-1, 1}
        positions = np.where(labels == 0, -1, 1)
    elif unique_labels_set <= {0, 1, 2}:  # Ternary {0, 1, 2} -> {-1, 0, 1}
        positions = labels - 1
    
    # Calculate trades and PnL
    for i in range(1, len(prices)):
        target_position = positions[i-1]  # Use previous label for current period
        
        # Count position changes as trades
        if target_position != current_position:
            num_trades += 1
            total_pnl -= transaction_cost  # Transaction cost
            current_position = target_position
        
        # Calculate return for current position
        if current_position != 0:
            price_return = (prices[i] - prices[i-1]) / prices[i-1]
            total_pnl += current_position * price_return
    
    return {
        "method": "Enhanced Output Method (Binary {0,1} -> {-1,1})",
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "mean_return_per_trade": total_pnl / num_trades if num_trades > 0 else 0,
        "positions_used": f"Remapped to {set(np.unique(positions))}"
    }


def calculate_pnl_method_b(prices: np.ndarray, raw_tstrends_labels: np.ndarray,
                          transaction_cost: float = 0.00007) -> Dict[str, Any]:
    """
    Method B: Direct TStrends calculation using original {-1, 1} labels.
    
    This is likely what the optimization is using.
    """
    if not LIBRARIES_AVAILABLE:
        return {"error": "TStrends not available"}
    
    try:
        # Convert to format expected by TStrends
        price_list = [float(p) for p in prices.tolist()]
        label_list = [int(p) for p in raw_tstrends_labels.tolist()]
        
        # Configure fees
        fees_config = FeesConfig(transaction_cost=transaction_cost)
        estimator = ReturnsEstimatorWithFees(fees_config)
        
        # Calculate returns using TStrends method
        total_return = estimator.estimate_returns(price_list, label_list)
        
        # Count trades manually for comparison
        num_trades = 0
        current_position = 0
        for target_position in raw_tstrends_labels:
            if target_position != current_position:
                num_trades += 1
                current_position = target_position
        
        return {
            "method": "TStrends Direct Method (Original {-1,1})",
            "total_pnl": total_return,
            "num_trades": num_trades,
            "mean_return_per_trade": total_return / num_trades if num_trades > 0 else 0,
            "positions_used": f"Original {set(np.unique(raw_tstrends_labels))}"
        }
        
    except Exception as e:
        return {"error": f"TStrends calculation failed: {e}"}


def calculate_pnl_method_c(prices: np.ndarray, labels: np.ndarray,
                          transaction_cost: float = 0.00007) -> Dict[str, Any]:
    """
    Method C: Alternative calculation keeping {0, 1} as {0, 1} (not remapping).
    
    Maybe the issue is our remapping assumption.
    """
    total_pnl = 0.0
    current_position = 0
    num_trades = 0
    
    # Use labels as-is: 0 = short, 1 = long (no remapping)
    positions = np.where(labels == 0, -1, 1)  # 0->-1 (short), 1->1 (long)
    
    # Calculate trades and PnL
    for i in range(1, len(prices)):
        target_position = positions[i-1]  # Use previous label for current period
        
        # Count position changes as trades
        if target_position != current_position:
            num_trades += 1
            total_pnl -= transaction_cost  # Transaction cost
            current_position = target_position
        
        # Calculate return for current position
        if current_position != 0:
            price_return = (prices[i] - prices[i-1]) / prices[i-1]
            total_pnl += current_position * price_return
    
    return {
        "method": "No Remapping Method (0=-1, 1=1)",
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "mean_return_per_trade": total_pnl / num_trades if num_trades > 0 else 0,
        "positions_used": f"0->-1, 1->1: {set(np.unique(positions))}"
    }


def run_comprehensive_diagnosis():
    """Run comprehensive diagnosis of Binary CTL return calculation."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Required libraries not available")
        return
    
    print("🔬 Binary CTL Return Calculation Diagnosis")
    print("=" * 70)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Use a reasonable test sample
    test_df = df.head(50000)  # 50K samples
    prices = test_df["mid_price"].to_numpy()
    
    print(f"Testing with {len(test_df):,} samples")
    print(f"Price range: {prices.min():.6f} to {prices.max():.6f}")
    print()
    
    # Test with optimized Binary CTL parameters
    omega = 6.339289260816091e-05  # From your optimization result
    print(f"Using optimized omega: {omega:.8f}")
    print()
    
    # Generate labels using our wrapped Binary CTL (remapped to {0,1})
    generator = TargetGeneratorFactory.create("binary_ctl", omega=omega)
    targets_df = generator.generate_targets(test_df)
    target_info = generator.get_target_info()
    target_col = target_info['target_names'][0]
    our_labels = targets_df[target_col].to_numpy()
    
    # Generate labels using raw TStrends Binary CTL (original {-1,1})
    price_list = [float(p) for p in prices.tolist()]
    raw_labeller = BinaryCTL(omega=omega)
    raw_tstrends_labels = np.array(raw_labeller.get_labels(price_list), dtype=np.int32)
    
    print(f"Our labels distribution: {dict(zip(*np.unique(our_labels, return_counts=True)))}")
    print(f"Raw TStrends labels distribution: {dict(zip(*np.unique(raw_tstrends_labels, return_counts=True)))}")
    print()
    
    # Test all three PnL calculation methods
    methods = [
        calculate_pnl_method_a(prices, our_labels),
        calculate_pnl_method_b(prices, raw_tstrends_labels),
        calculate_pnl_method_c(prices, our_labels),
    ]
    
    print("📊 PnL CALCULATION COMPARISON:")
    print("=" * 70)
    
    for i, result in enumerate(methods, 1):
        if "error" in result:
            print(f"{i}. ❌ {result.get('method', 'Unknown')}: {result['error']}")
        else:
            print(f"{i}. {result['method']}:")
            print(f"   Total PnL: {result['total_pnl']:.6f}")
            print(f"   Num trades: {result['num_trades']:,}")
            print(f"   Mean return/trade: {result['mean_return_per_trade']:.8f}")
            print(f"   Positions used: {result['positions_used']}")
        print()
    
    # Check if we have a sign flip issue
    if len(methods) >= 2 and "error" not in methods[0] and "error" not in methods[1]:
        method_a_pnl = methods[0]['total_pnl']
        method_b_pnl = methods[1]['total_pnl']
        
        if np.sign(method_a_pnl) != np.sign(method_b_pnl):
            print("🚨 SIGN FLIP DETECTED!")
            print(f"   Method A (our calculation): {method_a_pnl:.6f}")
            print(f"   Method B (TStrends direct): {method_b_pnl:.6f}")
            print("   This suggests a label mapping issue!")
        
        ratio = abs(method_a_pnl / method_b_pnl) if method_b_pnl != 0 else float('inf')
        print(f"📈 PnL Ratio (A/B): {ratio:.2f}")
    
    print("\n💡 DIAGNOSIS SUMMARY:")
    print("If Method A (our calculation) shows negative returns while")
    print("Method B (TStrends direct) shows positive returns, then we have")
    print("a label interpretation issue in our enhanced output calculation.")


def main():
    """Run the diagnosis."""
    try:
        run_comprehensive_diagnosis()
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()