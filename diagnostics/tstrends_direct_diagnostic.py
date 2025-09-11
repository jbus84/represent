#!/usr/bin/env python3
"""
Direct TStrends Library Diagnostic

Test the underlying TStrends library methods directly to understand
why we're getting severely imbalanced label distributions.
"""

import numpy as np
import polars as pl
from pathlib import Path

try:
    from tstrends.trend_labelling import BinaryCTL, TernaryCTL, OracleBinaryTrendLabeller, OracleTernaryTrendLabeller
    TSTRENDS_AVAILABLE = True
except ImportError:
    TSTRENDS_AVAILABLE = False


def load_test_data():
    """Load a sample of our dataset for testing."""
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"Test data not found: {data_path}")
    
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    print(f"Loaded {len(df):,} samples")
    print(f"Price range: {df['mid_price'].min():.6f} to {df['mid_price'].max():.6f}")
    
    return df


def test_raw_tstrends_methods():
    """Test TStrends methods directly without our wrapper remapping."""
    if not TSTRENDS_AVAILABLE:
        print("❌ TStrends library not available")
        return
    
    print("🔍 Testing TStrends methods directly...\n")
    
    # Load test data
    df = load_test_data()
    
    # Test with small sample first
    small_sample = df.head(10000)
    prices = small_sample["mid_price"].to_numpy()
    price_list = [float(p) for p in prices.tolist()]
    
    print(f"Testing with {len(price_list):,} prices")
    print(f"Price sample: {price_list[:5]} ... {price_list[-5:]}")
    print(f"Price stats: mean={np.mean(price_list):.6f}, std={np.std(price_list):.6f}")
    print()
    
    # Test Binary CTL with different omega values
    print("=== BINARY CTL TESTS ===")
    for omega in [0.0, 0.001, 0.01, 0.02, 0.05]:
        try:
            labeller = BinaryCTL(omega=omega)
            raw_labels = labeller.get_labels(price_list)
            labels = np.array(raw_labels)
            
            unique, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            
            print(f"Omega {omega:6.3f}: Labels {unique} with counts {counts} ({percentages}%)")
            
        except Exception as e:
            print(f"Omega {omega:6.3f}: ERROR - {e}")
    
    print()
    
    # Test Ternary CTL with different parameters
    print("=== TERNARY CTL TESTS ===")
    for thres in [0.001, 0.01, 0.02, 0.05]:
        for window in [10, 50, 100, 500]:
            try:
                labeller = TernaryCTL(marginal_change_thres=thres, window_size=window)
                raw_labels = labeller.get_labels(price_list)
                labels = np.array(raw_labels)
                
                unique, counts = np.unique(labels, return_counts=True)
                percentages = counts / len(labels) * 100
                
                print(f"Thres {thres:5.3f}, Win {window:3d}: Labels {unique} with counts {counts} ({percentages}%)")
                
            except Exception as e:
                print(f"Thres {thres:5.3f}, Win {window:3d}: ERROR - {e}")
    
    print()
    
    # Test Oracle methods
    print("=== ORACLE TESTS ===")
    for tc in [0.00001, 0.00007, 0.001]:
        try:
            labeller = OracleBinaryTrendLabeller(transaction_cost=tc)
            raw_labels = labeller.get_labels(price_list)
            labels = np.array(raw_labels)
            
            unique, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            
            print(f"Oracle Binary TC {tc:7.5f}: Labels {unique} with counts {counts} ({percentages}%)")
            
        except Exception as e:
            print(f"Oracle Binary TC {tc:7.5f}: ERROR - {e}")
    
    for tc in [0.00001, 0.00007, 0.001]:
        for nrf in [0.1, 0.5, 1.0]:
            try:
                labeller = OracleTernaryTrendLabeller(transaction_cost=tc, neutral_reward_factor=nrf)
                raw_labels = labeller.get_labels(price_list)
                labels = np.array(raw_labels)
                
                unique, counts = np.unique(labels, return_counts=True)
                percentages = counts / len(labels) * 100
                
                print(f"Oracle Ternary TC {tc:7.5f}, NRF {nrf:3.1f}: Labels {unique} with counts {counts} ({percentages}%)")
                
            except Exception as e:
                print(f"Oracle Ternary TC {tc:7.5f}, NRF {nrf:3.1f}: ERROR - {e}")


def test_price_data_characteristics():
    """Analyze the price data characteristics to understand labeling behavior."""
    print("\n🔍 Analyzing price data characteristics...\n")
    
    df = load_test_data()
    prices = df["mid_price"].to_numpy()
    
    # Basic stats
    print(f"Total samples: {len(prices):,}")
    print(f"Price range: {prices.min():.6f} to {prices.max():.6f}")
    print(f"Price mean: {prices.mean():.6f}")
    print(f"Price std: {prices.std():.6f}")
    
    # Price changes
    price_changes = np.diff(prices)
    print(f"\nPrice changes:")
    print(f"  Mean change: {price_changes.mean():.8f}")
    print(f"  Std change: {price_changes.std():.8f}")
    print(f"  Min change: {price_changes.min():.8f}")
    print(f"  Max change: {price_changes.max():.8f}")
    
    # Distribution of changes
    pos_changes = price_changes > 0
    neg_changes = price_changes < 0
    zero_changes = price_changes == 0
    
    print(f"\nChange distribution:")
    print(f"  Positive: {pos_changes.sum():,} ({pos_changes.mean()*100:.2f}%)")
    print(f"  Negative: {neg_changes.sum():,} ({neg_changes.mean()*100:.2f}%)")
    print(f"  Zero: {zero_changes.sum():,} ({zero_changes.mean()*100:.2f}%)")
    
    # Percentage changes
    pct_changes = price_changes / prices[:-1] * 100
    print(f"\nPercentage changes:")
    print(f"  Mean: {pct_changes.mean():.6f}%")
    print(f"  Std: {pct_changes.std():.6f}%")
    print(f"  Min: {pct_changes.min():.6f}%")
    print(f"  Max: {pct_changes.max():.6f}%")
    
    # Check for trends
    print(f"\nTrend analysis:")
    print(f"  Overall trend: {(prices[-1] - prices[0]):.6f} ({((prices[-1] - prices[0])/prices[0]*100):.4f}%)")
    
    # Rolling volatility (100-tick windows)
    rolling_vol = []
    window = 100
    for i in range(window, len(prices)):
        window_prices = prices[i-window:i]
        vol = np.std(np.diff(window_prices))
        rolling_vol.append(vol)
    
    rolling_vol = np.array(rolling_vol)
    print(f"  Rolling volatility (100-tick): mean={rolling_vol.mean():.8f}, std={rolling_vol.std():.8f}")


def main():
    """Run comprehensive TStrends diagnostic."""
    print("🔬 TStrends Direct Library Diagnostic")
    print("=" * 50)
    
    try:
        test_price_data_characteristics()
        test_raw_tstrends_methods()
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()