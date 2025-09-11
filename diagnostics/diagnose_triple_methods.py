#!/usr/bin/env python3
"""
Diagnose Triple Barrier and Triple Exceedance Methods

Investigate why these methods are finding poor parameters:
- Triple Barrier: Max returns 0.0001 (0.01%)
- Triple Exceedance: Max returns -0.0012 (-0.12%)

Potential issues:
1. Barriers too wide/narrow for micro-volatility
2. Transaction cost assumptions
3. Label generation issues
4. PnL calculation problems
5. Optimization bounds mismatch with data characteristics
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


def analyze_micro_volatility(prices: np.ndarray) -> dict:
    """Analyze micro-volatility characteristics of the data."""
    price_changes = np.diff(prices)
    returns = price_changes / prices[:-1]
    
    return {
        'mean_abs_return': np.mean(np.abs(returns)),
        'std_return': np.std(returns),
        'zero_changes_pct': np.sum(price_changes == 0) / len(price_changes) * 100,
        'min_price_change': np.min(np.abs(price_changes[price_changes != 0])) if np.any(price_changes != 0) else 0,
        'price_tick_size': np.min(np.diff(np.unique(prices[prices > 0]))),
    }


def test_triple_barrier_parameters():
    """Test Triple Barrier with different parameter combinations."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("🔬 Triple Barrier Method Analysis")
    print("=" * 60)
    
    # Load test data
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(50000)
    prices = test_df["mid_price"].to_numpy()
    
    print(f"Test data: {len(test_df):,} samples")
    print(f"Price range: {prices.min():.6f} to {prices.max():.6f}")
    
    # Analyze micro-volatility
    vol_stats = analyze_micro_volatility(prices)
    print(f"\n📊 Micro-volatility Analysis:")
    print(f"   Mean absolute return: {vol_stats['mean_abs_return']:.8f}")
    print(f"   Return std: {vol_stats['std_return']:.8f}")
    print(f"   Zero price changes: {vol_stats['zero_changes_pct']:.1f}%")
    print(f"   Min price tick: {vol_stats['price_tick_size']:.8f}")
    print()
    
    # Test different barrier configurations
    test_configs = [
        {
            "name": "OPTIMIZED (from results)",
            "lookforward_window": 1768,
            "barrier_width": 0.0030348335902746558,  # 0.3% barriers (30 pips)
            "min_return_threshold": 5.9698897438064404e-05,
            "volatility_window": 145,
            "normalize_by_volatility": False,
        },
        {
            "name": "MICRO-SCALE",
            "lookforward_window": 1000,
            "barrier_width": 0.0001,  # 0.01% barriers (1 pip) - much tighter
            "min_return_threshold": 1e-6,  # Very small threshold
            "volatility_window": 100,
            "normalize_by_volatility": False,
        },
        {
            "name": "VOLATILITY-SCALED",
            "lookforward_window": 2000,
            "barrier_width": vol_stats['std_return'] * 2,  # 2x volatility barriers
            "min_return_threshold": vol_stats['mean_abs_return'] * 0.1,
            "volatility_window": 200,
            "normalize_by_volatility": True,
        },
        {
            "name": "TIGHT-BARRIERS",
            "lookforward_window": 500,
            "barrier_width": vol_stats['price_tick_size'] / prices.mean() * 5,  # 5 ticks
            "min_return_threshold": 1e-7,
            "volatility_window": 50,
            "normalize_by_volatility": False,
        },
    ]
    
    for i, config in enumerate(test_configs, 1):
        name = config.pop("name")
        print(f"{i}. {name}:")
        print(f"   Barrier width: {config['barrier_width']:.8f} ({config['barrier_width']*100:.4f}%)")
        print(f"   Lookforward: {config['lookforward_window']} ticks")
        
        try:
            # Generate labels
            generator = TargetGeneratorFactory.create("triple_barrier", **config)
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Analyze labels
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            balance_score = min(percentages) / max(percentages) * 100 if len(percentages) > 1 else 100
            
            print(f"   Label distribution: {dict(zip(unique_labels, percentages.round(1)))}")
            print(f"   Balance score: {balance_score:.1f}%")
            
            # Calculate PnL using exact optimization logic
            try:
                fees_config = FeesConfig(
                    lp_transaction_fees=0.00007,
                    sp_transaction_fees=0.00007,
                )
                returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
                
                # Convert labels to TStrends format  
                labels_int = labels.astype(int)
                unique_set = set(np.unique(labels_int))
                
                # Triple barrier uses {-1, 0, 1} already
                if unique_set.issubset({-1, 0, 1}):
                    labels_tstrends = labels_int
                else:
                    # Fallback mapping if needed
                    labels_tstrends = labels_int
                
                total_pnl = returns_estimator.estimate_return(
                    prices.tolist(),
                    labels_tstrends.tolist()
                )
                
                # Count trades
                num_trades = sum(1 for j in range(1, len(labels_tstrends)) 
                               if labels_tstrends[j] != labels_tstrends[j-1])
                
                mean_return = total_pnl / num_trades if num_trades > 0 else 0
                print(f"   PnL: {total_pnl:.6f} ({total_pnl*100:.2f}%)")
                print(f"   Trades: {num_trades:,}, Mean: {mean_return:.8f}")
                
            except Exception as e:
                print(f"   PnL calculation failed: {e}")
                
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")
        
        print()
        
        # Restore name for next iteration
        config["name"] = name


def test_triple_exceedance_parameters():
    """Test Triple Exceedance with different parameter combinations."""
    if not LIBRARIES_AVAILABLE:
        print("❌ Libraries not available")
        return
    
    print("🔬 Triple Exceedance Method Analysis")
    print("=" * 60)
    
    # Load test data (same as above)
    data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col('mid_price').is_not_null())
    test_df = df.head(50000)
    prices = test_df["mid_price"].to_numpy()
    
    vol_stats = analyze_micro_volatility(prices)
    
    # Test different configurations
    test_configs = [
        {
            "name": "OPTIMIZED (from results)",
            "lookforward_window": 1496,
            "scaling_factor": 19.302155196862056,  # Very high scaling
            "min_exceedance_threshold": 0.8642264086386979,  # High threshold
            "volatility_window": 499,
            "window_penalty_weight": 0.34978580535424886,
            "balance_weight": 0.5459550636056117,
            "target_balance_ratio": 0.32038938648717,
            "adaptive_scaling": False,
        },
        {
            "name": "CONSERVATIVE",
            "lookforward_window": 1000,
            "scaling_factor": 3.0,  # Much lower scaling
            "min_exceedance_threshold": 0.3,  # Lower threshold
            "volatility_window": 200,
            "window_penalty_weight": 0.1,
            "balance_weight": 0.5,
            "target_balance_ratio": 0.33,
            "adaptive_scaling": True,
        },
        {
            "name": "MICRO-OPTIMIZED",
            "lookforward_window": 500,
            "scaling_factor": 2.0,  # Minimal scaling
            "min_exceedance_threshold": 0.1,  # Very low threshold 
            "volatility_window": 100,
            "window_penalty_weight": 0.05,
            "balance_weight": 1.0,
            "target_balance_ratio": 0.25,
            "adaptive_scaling": True,
        },
    ]
    
    for i, config in enumerate(test_configs, 1):
        name = config.pop("name")
        print(f"{i}. {name}:")
        print(f"   Scaling factor: {config['scaling_factor']:.2f}")
        print(f"   Min threshold: {config['min_exceedance_threshold']:.2f}")
        print(f"   Lookforward: {config['lookforward_window']} ticks")
        
        try:
            # Generate labels
            generator = TargetGeneratorFactory.create("triple_exceedance", **config)
            targets_df = generator.generate_targets(test_df)
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            # Analyze labels  
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            balance_score = min(percentages) / max(percentages) * 100 if len(percentages) > 1 else 100
            
            print(f"   Label distribution: {dict(zip(unique_labels, percentages.round(1)))}")
            print(f"   Balance score: {balance_score:.1f}%")
            
            # Calculate PnL using exact optimization logic
            try:
                fees_config = FeesConfig(
                    lp_transaction_fees=0.00007,
                    sp_transaction_fees=0.00007,
                )
                returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
                
                # Convert labels to TStrends format
                labels_int = labels.astype(int)
                unique_set = set(np.unique(labels_int))
                
                # Triple exceedance uses {-1, 0, 1} already
                if unique_set.issubset({-1, 0, 1}):
                    labels_tstrends = labels_int
                else:
                    # Fallback mapping if needed
                    labels_tstrends = labels_int
                
                total_pnl = returns_estimator.estimate_return(
                    prices.tolist(),
                    labels_tstrends.tolist()
                )
                
                # Count trades
                num_trades = sum(1 for j in range(1, len(labels_tstrends)) 
                               if labels_tstrends[j] != labels_tstrends[j-1])
                
                mean_return = total_pnl / num_trades if num_trades > 0 else 0
                print(f"   PnL: {total_pnl:.6f} ({total_pnl*100:.2f}%)")
                print(f"   Trades: {num_trades:,}, Mean: {mean_return:.8f}")
                
            except Exception as e:
                print(f"   PnL calculation failed: {e}")
                
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")
            
        print()
        
        # Restore name for next iteration
        config["name"] = name


def main():
    """Run comprehensive triple methods diagnosis."""
    try:
        test_triple_barrier_parameters()
        test_triple_exceedance_parameters()
        
        print("💡 DIAGNOSIS SUMMARY:")
        print("=" * 60)
        print("Key issues to investigate:")
        print("1. Barrier scaling relative to micro-volatility")
        print("2. Optimization bounds may be too wide/inappropriate")
        print("3. Transaction cost assumptions")
        print("4. Label generation logic for extreme parameters")
        print("5. Multi-objective optimization conflicts")
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()