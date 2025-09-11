#!/usr/bin/env python3
"""
PnL Calculation Consistency Verification

This script verifies that PnL calculations are consistent across all target generation
methods and identifies any discrepancies in return calculation methodologies.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    REPRESENT_AVAILABLE = True
except ImportError:
    REPRESENT_AVAILABLE = False

try:
    from tstrends.returns_estimation import ReturnsEstimatorWithFees
    from tstrends.returns_estimation.fees_config import FeesConfig
    TSTRENDS_AVAILABLE = True
except ImportError:
    TSTRENDS_AVAILABLE = False


def calculate_pnl_baseline(prices: np.ndarray, labels: np.ndarray, 
                          transaction_cost: float = 0.00007) -> Dict[str, Any]:
    """
    Baseline PnL calculation methodology for consistency verification.
    
    This is our reference implementation that all other calculations should match.
    
    Args:
        prices: Price array
        labels: Trading labels (any format)
        transaction_cost: Transaction cost per trade
        
    Returns:
        Detailed PnL breakdown for verification
    """
    if len(prices) <= 1 or len(labels) == 0:
        return {"total_pnl": 0.0, "num_trades": 0, "total_fees": 0.0, "gross_pnl": 0.0}
    
    # Ensure same length
    min_len = min(len(prices), len(labels))
    prices = prices[:min_len]
    labels = labels[:min_len]
    
    # Normalize labels to {-1, 0, 1} trading positions
    positions = np.array(labels, dtype=np.float64)
    
    # Handle different label formats
    unique_labels = set(np.unique(labels))
    if unique_labels <= {0, 1}:  # Binary {0, 1} -> {-1, 1}
        positions = np.where(labels == 0, -1, 1)
    elif unique_labels <= {0, 1, 2}:  # Ternary {0, 1, 2} -> {-1, 0, 1}
        positions = labels - 1
    # Otherwise assume already in {-1, 0, 1} format
    
    total_pnl = 0.0
    gross_pnl = 0.0
    current_position = 0
    num_trades = 0
    total_fees = 0.0
    position_changes = []
    
    for i in range(1, len(prices)):
        target_position = positions[i-1]  # Use previous label for current period
        
        # Track position changes and fees
        if target_position != current_position:
            num_trades += 1
            total_fees += transaction_cost
            position_changes.append({
                "step": i,
                "from_position": current_position,
                "to_position": target_position,
                "price": prices[i],
                "fee": transaction_cost
            })
            current_position = target_position
        
        # Calculate return for current position
        if current_position != 0:
            price_return = (prices[i] - prices[i-1]) / prices[i-1]
            period_pnl = current_position * price_return
            gross_pnl += period_pnl
            total_pnl += period_pnl
        
    # Subtract total fees
    total_pnl -= total_fees
    
    return {
        "total_pnl": total_pnl,
        "gross_pnl": gross_pnl,
        "total_fees": total_fees,
        "num_trades": num_trades,
        "avg_fee_per_trade": total_fees / num_trades if num_trades > 0 else 0,
        "fee_ratio": total_fees / abs(gross_pnl) if gross_pnl != 0 else 0,
        "position_changes": position_changes[:5],  # First 5 for verification
        "total_periods": min_len - 1,
        "trading_periods": np.sum(positions != 0)
    }


def calculate_pnl_tstrends(prices: np.ndarray, labels: np.ndarray,
                          transaction_cost: float = 0.00007) -> Dict[str, Any]:
    """
    Calculate PnL using TStrends ReturnsEstimatorWithFees for comparison.
    
    This verifies our calculations match the TStrends library methodology.
    """
    if not TSTRENDS_AVAILABLE:
        return {"error": "TStrends not available"}
    
    try:
        # Convert to format expected by TStrends
        price_list = [float(p) for p in prices.tolist()]
        
        # Normalize labels to TStrends format {-1, 0, 1}
        positions = np.array(labels, dtype=np.float64)
        unique_labels = set(np.unique(labels))
        if unique_labels <= {0, 1}:  # Binary {0, 1} -> {-1, 1}
            positions = np.where(labels == 0, -1, 1)
        elif unique_labels <= {0, 1, 2}:  # Ternary {0, 1, 2} -> {-1, 0, 1}
            positions = labels - 1
        
        label_list = [int(p) for p in positions.tolist()]
        
        # Configure fees
        fees_config = FeesConfig(transaction_cost=transaction_cost)
        estimator = ReturnsEstimatorWithFees(fees_config)
        
        # Calculate returns
        total_return = estimator.estimate_returns(price_list, label_list)
        
        return {
            "total_pnl": total_return,
            "method": "TStrends ReturnsEstimatorWithFees",
            "transaction_cost": transaction_cost
        }
        
    except Exception as e:
        return {"error": f"TStrends calculation failed: {e}"}


def verify_method_pnl_consistency(method_name: str, method_params: Dict[str, Any],
                                 test_df: pl.DataFrame) -> Dict[str, Any]:
    """
    Verify PnL calculation consistency for a specific target generation method.
    
    Tests multiple scenarios and calculation approaches to ensure consistency.
    """
    results = {
        "method": method_name,
        "parameters": method_params,
        "tests": {},
        "consistency_score": 0,
        "issues": []
    }
    
    try:
        # Generate labels
        generator = TargetGeneratorFactory.create(method_name, **method_params)
        targets_df = generator.generate_targets(test_df)
        target_info = generator.get_target_info()
        target_col = target_info['target_names'][0]
        labels = targets_df[target_col].to_numpy()
        prices = test_df["mid_price"].to_numpy()
        
        # Test 1: Baseline PnL calculation
        baseline_result = calculate_pnl_baseline(prices, labels)
        results["tests"]["baseline"] = baseline_result
        
        # Test 2: TStrends PnL calculation (if available)
        if TSTRENDS_AVAILABLE:
            tstrends_result = calculate_pnl_tstrends(prices, labels)
            results["tests"]["tstrends"] = tstrends_result
            
            # Compare baseline vs TStrends
            if "error" not in tstrends_result:
                pnl_diff = abs(baseline_result["total_pnl"] - tstrends_result["total_pnl"])
                pnl_rel_diff = pnl_diff / abs(baseline_result["total_pnl"]) * 100 if baseline_result["total_pnl"] != 0 else 0
                
                results["tests"]["baseline_vs_tstrends"] = {
                    "baseline_pnl": baseline_result["total_pnl"],
                    "tstrends_pnl": tstrends_result["total_pnl"],
                    "absolute_difference": pnl_diff,
                    "relative_difference_pct": pnl_rel_diff,
                    "consistent": pnl_rel_diff < 1.0  # Within 1%
                }
        
        # Test 3: Different transaction costs
        transaction_costs = [0.00001, 0.00007, 0.0001]
        tc_results = {}
        for tc in transaction_costs:
            tc_result = calculate_pnl_baseline(prices, labels, tc)
            tc_results[f"tc_{tc:.5f}"] = tc_result
        results["tests"]["transaction_cost_sensitivity"] = tc_results
        
        # Test 4: Subsample consistency
        subsample_size = min(10000, len(test_df) // 2)
        subsample_df = test_df.head(subsample_size)
        subsample_targets_df = generator.generate_targets(subsample_df)
        subsample_labels = subsample_targets_df[target_col].to_numpy()
        subsample_prices = subsample_df["mid_price"].to_numpy()
        
        subsample_result = calculate_pnl_baseline(subsample_prices, subsample_labels)
        results["tests"]["subsample_consistency"] = {
            "full_dataset_samples": len(prices),
            "subsample_samples": len(subsample_prices),
            "full_pnl": baseline_result["total_pnl"],
            "subsample_pnl": subsample_result["total_pnl"],
            "pnl_per_sample_full": baseline_result["total_pnl"] / len(prices),
            "pnl_per_sample_subsample": subsample_result["total_pnl"] / len(subsample_prices)
        }
        
        # Test 5: Edge cases
        edge_case_results = {}
        
        # All same position
        uniform_labels = np.full_like(labels[:1000], labels[0])
        uniform_result = calculate_pnl_baseline(prices[:1000], uniform_labels)
        edge_case_results["uniform_labels"] = uniform_result
        
        # Alternating positions (maximum turnover)
        alternating_labels = np.tile([0, 1], len(labels[:1000])//2 + 1)[:len(labels[:1000])]
        alternating_result = calculate_pnl_baseline(prices[:1000], alternating_labels)
        edge_case_results["alternating_labels"] = alternating_result
        
        results["tests"]["edge_cases"] = edge_case_results
        
        # Calculate consistency score
        consistency_issues = 0
        total_checks = 0
        
        # Check TStrends consistency
        if "baseline_vs_tstrends" in results["tests"]:
            total_checks += 1
            if not results["tests"]["baseline_vs_tstrends"]["consistent"]:
                consistency_issues += 1
                results["issues"].append("Baseline vs TStrends PnL differs by >1%")
        
        # Check transaction cost monotonicity
        tc_pnls = [tc_results[k]["total_pnl"] for k in sorted(tc_results.keys())]
        tc_fees = [tc_results[k]["total_fees"] for k in sorted(tc_results.keys())]
        total_checks += 1
        if not all(tc_fees[i] <= tc_fees[i+1] for i in range(len(tc_fees)-1)):
            consistency_issues += 1
            results["issues"].append("Transaction costs not monotonic")
        
        # Check edge case reasonableness
        total_checks += 2
        if edge_case_results["uniform_labels"]["num_trades"] > 2:  # Should be minimal trades
            consistency_issues += 1
            results["issues"].append("Uniform labels produce too many trades")
        
        if edge_case_results["alternating_labels"]["num_trades"] < len(alternating_labels) // 3:
            consistency_issues += 1
            results["issues"].append("Alternating labels produce too few trades")
        
        results["consistency_score"] = (total_checks - consistency_issues) / total_checks * 100 if total_checks > 0 else 100
        
    except Exception as e:
        results["error"] = str(e)
        results["consistency_score"] = 0
    
    return results


def run_comprehensive_pnl_verification():
    """Run comprehensive PnL calculation consistency verification across all methods."""
    if not REPRESENT_AVAILABLE:
        print("❌ Represent library not available")
        return
    
    print("🔍 Comprehensive PnL Calculation Consistency Verification")
    print("=" * 70)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    
    # Use moderate test set for consistency testing
    test_df = df.head(25000)  # 25K samples for faster verification
    print(f"Testing with {len(test_df):,} samples")
    print()
    
    # Test methods with optimized parameters
    test_methods = [
        ("binary_ctl", {"omega": 0.00001}),
        ("ternary_ctl", {"marginal_change_thres": 0.00002, "window_size": 500}),
        ("oracle_binary", {"transaction_cost": 0.0001}),
        ("oracle_ternary", {"transaction_cost": 0.0001, "neutral_reward_factor": 0.5}),
    ]
    
    all_results = {}
    
    for method_name, params in test_methods:
        print(f"🧪 Testing {method_name.upper()} PnL consistency...")
        start_time = time.time()
        
        result = verify_method_pnl_consistency(method_name, params, test_df)
        all_results[method_name] = result
        
        elapsed = time.time() - start_time
        print(f"   Completed in {elapsed:.1f}s")
        
        if "error" in result:
            print(f"   ❌ ERROR: {result['error']}")
            continue
        
        # Report consistency score
        score = result["consistency_score"]
        print(f"   📊 Consistency Score: {score:.1f}%")
        
        # Report key metrics
        baseline = result["tests"]["baseline"]
        print(f"   💰 PnL: {baseline['total_pnl']:.6f} (Gross: {baseline['gross_pnl']:.6f}, Fees: {baseline['total_fees']:.6f})")
        print(f"   🔄 Trades: {baseline['num_trades']}, Fee ratio: {baseline['fee_ratio']:.1%}")
        
        # Report TStrends comparison if available
        if "baseline_vs_tstrends" in result["tests"]:
            bt = result["tests"]["baseline_vs_tstrends"]
            consistency_status = "✅" if bt["consistent"] else "❌"
            print(f"   {consistency_status} TStrends comparison: {bt['relative_difference_pct']:.2f}% difference")
        
        # Report issues
        if result["issues"]:
            print(f"   ⚠️  Issues: {', '.join(result['issues'])}")
        
        print()
    
    # Summary
    print("📋 PnL CONSISTENCY VERIFICATION SUMMARY:")
    print("=" * 50)
    
    perfect_methods = []
    good_methods = []
    problematic_methods = []
    
    for method_name, result in all_results.items():
        if "error" in result:
            problematic_methods.append((method_name, "Error"))
        else:
            score = result["consistency_score"]
            if score >= 95:
                perfect_methods.append((method_name, score))
            elif score >= 80:
                good_methods.append((method_name, score))
            else:
                problematic_methods.append((method_name, score))
    
    print(f"✅ Perfect consistency (≥95%): {len(perfect_methods)} methods")
    for method, score in perfect_methods:
        print(f"   {method}: {score:.1f}%")
    
    print(f"🟡 Good consistency (80-94%): {len(good_methods)} methods")
    for method, score in good_methods:
        print(f"   {method}: {score:.1f}%")
    
    print(f"❌ Problematic methods (<80%): {len(problematic_methods)} methods")
    for method, score in problematic_methods:
        print(f"   {method}: {score}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if problematic_methods:
        print(f"- Review PnL calculation methodology for problematic methods")
        print(f"- Ensure consistent label format handling across all methods")
        print(f"- Verify transaction cost application timing")
    else:
        print(f"- All methods show good PnL calculation consistency")
        print(f"- PnL calculations are reliable across different approaches")


def main():
    """Run the comprehensive PnL consistency verification."""
    try:
        run_comprehensive_pnl_verification()
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()