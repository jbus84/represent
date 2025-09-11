#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Comprehensive Test Suite for All Target Generation Methods

This script extensively tests all implemented labeling methods:
- GA Labeling (Genetic Algorithm with optimized parameters)
- Oracle Binary/Ternary (Perfect foresight with transaction costs)
- Binary/Ternary CTL (Academic trend labeling) 
- Triple Barrier Method (López de Prado barriers)
- Triple Exceedance Method (Transaction cost-scaled, multi-objective)
- Quantile Classification (Traditional balanced labeling)
- Log Return Horizons (Multi-horizon regression)

Testing includes multiple market regimes, performance metrics, and comparative analysis.
"""

import polars as pl
import numpy as np
from represent.target_generators.factory import TargetGeneratorFactory
import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time


class ComprehensiveMethodTester:
    """Comprehensive tester for all target generation methods"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        self.test_data = {}
        self.transaction_cost = 0.0001  # 1 pip standard
        
        # Load optimized parameters
        self._load_optimized_parameters()
    
    def _load_optimized_parameters(self):
        """Load optimized parameters for all methods"""
        self.optimized_params = {}
        
        # Parameter files to try loading
        param_files = {
            "ga_labeling": ["optimized_ga_params_corrected.json", "fixed_ga_params.json"],
            "oracle_binary": ["corrected_oracle_binary_params.json", "optimized_oracle_binary_params.json"],
            "oracle_ternary": ["corrected_oracle_ternary_params.json", "optimized_oracle_ternary_params.json"],
            "binary_ctl": ["corrected_binary_ctl_params.json", "optimized_binary_ctl_params.json"],
            "ternary_ctl": ["corrected_ternary_ctl_params.json", "optimized_ternary_ctl_params.json"],
            "triple_barrier": ["optimized_triple_barrier_params.json"],
            "triple_exceedance": ["optimized_triple_exceedance_multi_objective_params.json", "optimized_triple_exceedance_params.json"]
        }
        
        for method, file_list in param_files.items():
            for filename in file_list:
                if Path(filename).exists():
                    try:
                        with open(filename, 'r') as f:
                            self.optimized_params[method] = json.load(f)
                        if self.verbose:
                            print(f"✅ Loaded optimized parameters for {method} from {filename}")
                        break
                    except Exception as e:
                        if self.verbose:
                            print(f"⚠️ Failed to load {filename}: {e}")
            
            # Fallback to default optimized parameters if no file found
            if method not in self.optimized_params:
                self.optimized_params[method] = self._get_default_params(method)
                if self.verbose:
                    print(f"📝 Using default parameters for {method}")
    
    def _get_default_params(self, method: str) -> Dict[str, Any]:
        """Get default optimized parameters for methods"""
        defaults = {
            "ga_labeling": {
                "population_size": 50, "max_generations": 30, "lookforward_window": 2500,
                "transaction_cost": self.transaction_cost, "max_trade_frequency": 0.05,
                "min_trades": 20, "min_win_rate": 0.1, "max_win_rate": 0.9, "min_profit_factor": 0.5
            },
            "oracle_binary": {"transaction_cost": self.transaction_cost},
            "oracle_ternary": {"transaction_cost": self.transaction_cost, "neutral_reward_factor": 0.5},
            "binary_ctl": {"omega": 0.005},
            "ternary_ctl": {"marginal_change_thres": 0.01, "window_size": 200},
            "triple_barrier": {
                "lookforward_window": 500, "barrier_width": 0.0005, "transaction_cost": self.transaction_cost
            },
            "triple_exceedance": {
                "lookforward_window": 200, "scaling_factor": 8.0, "transaction_cost": self.transaction_cost,
                "balance_weight": 0.5, "target_balance_ratio": 0.33, "window_penalty_weight": 0.2
            }
        }
        return defaults.get(method, {})
    
    def create_test_datasets(self, n_samples: int = 1500):
        """Create multiple test datasets representing different market regimes"""
        np.random.seed(42)  # Reproducible results
        base_price = 1.1000
        
        datasets = {}
        
        # 1. Trending Market (Strong Uptrend)
        trend = np.linspace(0, 0.008, n_samples)  # 80 pip trend
        noise = np.random.normal(0, 0.0001, n_samples)
        prices_trend = base_price + trend + np.cumsum(noise)
        
        datasets["trending"] = pl.DataFrame({
            "ts_event": range(n_samples),
            "mid_price": prices_trend,
            "symbol": ["EURUSD"] * n_samples
        })
        
        # 2. Mean Reverting Market (Range-bound)
        t = np.linspace(0, 8*np.pi, n_samples)
        mean_reversion = 0.002 * np.sin(t)  # 20 pip oscillations
        noise = np.random.normal(0, 0.00005, n_samples)  # Lower noise
        prices_range = base_price + mean_reversion + np.cumsum(noise)
        
        datasets["mean_reverting"] = pl.DataFrame({
            "ts_event": range(n_samples),
            "mid_price": prices_range,
            "symbol": ["EURUSD"] * n_samples
        })
        
        # 3. High Volatility Market (Volatile with clustering)
        returns = []
        volatility = 0.0002  # Higher base volatility
        for i in range(n_samples):
            if i > 0:
                volatility = 0.85 * volatility + 0.15 * 0.0002 + 0.3 * (returns[-1]**2)
            ret = np.random.normal(0, volatility)
            returns.append(ret)
        
        prices_volatile = base_price + np.cumsum(returns)
        
        datasets["volatile"] = pl.DataFrame({
            "ts_event": range(n_samples),
            "mid_price": prices_volatile,
            "symbol": ["EURUSD"] * n_samples
        })
        
        # 4. Mixed Regime Market (Trend + Mean Reversion + Volatility)
        trend_component = 0.003 * np.sin(t * 0.3)  # Slow trending
        mean_rev_component = 0.001 * np.sin(t * 2)  # Fast oscillations
        
        mixed_returns = []
        mixed_vol = 0.0001
        for i in range(n_samples):
            if i > 0:
                mixed_vol = 0.9 * mixed_vol + 0.1 * 0.0001 + 0.2 * (mixed_returns[-1]**2)
            ret = np.random.normal(0, mixed_vol)
            mixed_returns.append(ret)
        
        prices_mixed = base_price + np.cumsum(trend_component + mean_rev_component + mixed_returns)
        
        datasets["mixed"] = pl.DataFrame({
            "ts_event": range(n_samples),
            "mid_price": prices_mixed,
            "symbol": ["EURUSD"] * n_samples
        })
        
        # 5. Low Volatility Market (Quiet market)
        quiet_noise = np.random.normal(0, 0.00003, n_samples)  # Very low volatility
        small_trend = np.linspace(0, 0.001, n_samples)  # 10 pip trend
        prices_quiet = base_price + small_trend + np.cumsum(quiet_noise)
        
        datasets["quiet"] = pl.DataFrame({
            "ts_event": range(n_samples),
            "mid_price": prices_quiet,
            "symbol": ["EURUSD"] * n_samples
        })
        
        self.test_data = datasets
        
        if self.verbose:
            print(f"✅ Created {len(datasets)} test datasets with {n_samples} samples each")
            for name, data in datasets.items():
                price_range = data["mid_price"].max() - data["mid_price"].min()
                print(f"   {name:15}: Range = {price_range * 100000:6.1f} pips")
    
    def test_method(self, method_name: str, dataset_name: str, dataset: pl.DataFrame) -> Dict[str, Any]:
        """Test a single method on a single dataset"""
        try:
            start_time = time.time()
            
            # Get parameters for this method
            params = self.optimized_params.get(method_name, {}).copy()
            
            # Ensure transaction cost is set consistently
            if "transaction_cost" in params or method_name in ["ga_labeling", "oracle_binary", "oracle_ternary", "triple_barrier", "triple_exceedance"]:
                params["transaction_cost"] = self.transaction_cost
            
            # Create generator
            generator = TargetGeneratorFactory.create(method_name, **params)
            
            # Generate targets
            targets = generator.generate_targets(dataset)
            generation_time = time.time() - start_time
            
            # Find target columns (exclude metadata columns)
            target_cols = [col for col in targets.columns 
                          if col not in ["row_idx", "symbol", "timestamp"] 
                          and not col.endswith("_return") 
                          and not col.endswith("_barrier_width")
                          and not col.endswith("_exceedance")]
            
            if not target_cols:
                return {"error": "No target columns found", "generation_time": generation_time}
            
            results = {}
            
            # Analyze each target column
            for target_col in target_cols:
                labels = targets[target_col].to_numpy()
                
                # Basic statistics
                unique_labels = sorted(np.unique(labels).tolist())
                label_counts = dict(zip(*np.unique(labels, return_counts=True)))
                total_labels = len(labels)
                
                # Class distribution
                class_distribution = {str(label): count/total_labels for label, count in label_counts.items()}
                
                # Calculate class balance metrics
                if len(unique_labels) > 1:
                    # Entropy calculation
                    entropy = 0
                    for count in label_counts.values():
                        if count > 0:
                            p = count / total_labels
                            entropy -= p * np.log2(p)
                    
                    max_entropy = np.log2(len(unique_labels))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    
                    # Balance score (for 3-class case)
                    if len(unique_labels) == 3:
                        target_prop = 1.0 / 3
                        deviations = [abs((count/total_labels) - target_prop) for count in label_counts.values()]
                        balance_score = max(0, 1 - np.mean(deviations) / target_prop)
                    else:
                        balance_score = normalized_entropy  # Use entropy for non-3-class
                else:
                    entropy = 0
                    normalized_entropy = 0
                    balance_score = 0
                
                # Trading frequency (non-neutral labels)
                if generator.target_type == "classification":
                    if 0 in label_counts:
                        # Assume 0 is neutral
                        trading_labels = total_labels - label_counts.get(0, 0)
                    else:
                        # All labels are trading labels
                        trading_labels = total_labels
                else:
                    # Regression - count non-zero values
                    trading_labels = np.sum(labels != 0)
                
                trade_frequency = trading_labels / total_labels
                
                # Calculate returns if possible
                expected_return = 0
                total_trades = 0
                win_rate = 0
                
                if generator.target_type == "classification" and len(dataset) > 1:
                    returns = []
                    prices = dataset["mid_price"].to_numpy()
                    
                    for i in range(len(labels) - 1):
                        if labels[i] != 0:  # Non-neutral label
                            entry_price = prices[i]
                            exit_price = prices[i + 1]
                            
                            # Determine position based on label
                            if isinstance(labels[i], (int, np.integer)):
                                if labels[i] > 0:
                                    # Long position
                                    ret = (exit_price - entry_price) / entry_price - self.transaction_cost
                                else:
                                    # Short position  
                                    ret = (entry_price - exit_price) / entry_price - self.transaction_cost
                            else:
                                # Continuous labels - use sign
                                if float(labels[i]) > 0:
                                    ret = (exit_price - entry_price) / entry_price - self.transaction_cost
                                else:
                                    ret = (entry_price - exit_price) / entry_price - self.transaction_cost
                            
                            returns.append(ret)
                    
                    if returns:
                        expected_return = np.mean(returns)
                        total_trades = len(returns)
                        win_rate = sum(1 for r in returns if r > 0) / len(returns)
                
                # Compile results for this target column
                results[target_col] = {
                    "unique_labels": unique_labels,
                    "class_distribution": class_distribution,
                    "entropy": entropy,
                    "normalized_entropy": normalized_entropy,
                    "balance_score": balance_score,
                    "trade_frequency": trade_frequency,
                    "expected_return": expected_return,
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                    "generation_time": generation_time,
                    "target_type": generator.target_type,
                    "parameters": params
                }
            
            return results
            
        except Exception as e:
            return {"error": str(e), "generation_time": time.time() - start_time}
    
    def run_comprehensive_test(self):
        """Run comprehensive test on all methods and all datasets"""
        if not self.test_data:
            self.create_test_datasets()
        
        # Methods to test
        methods_to_test = [
            "ga_labeling",
            "oracle_binary", 
            "oracle_ternary",
            "binary_ctl",
            "ternary_ctl", 
            "quantile_classification",
            "triple_barrier",
            "triple_exceedance",
            "log_return_horizons"
        ]
        
        print("=" * 100)
        print("COMPREHENSIVE TARGET GENERATION METHOD TESTING")
        print("=" * 100)
        print(f"Testing {len(methods_to_test)} methods on {len(self.test_data)} market regimes")
        print(f"Transaction cost: {self.transaction_cost * 100000:.1f} pips")
        print()
        
        total_tests = len(methods_to_test) * len(self.test_data)
        completed_tests = 0
        
        # Test each method on each dataset
        for method_name in methods_to_test:
            if method_name not in self.results:
                self.results[method_name] = {}
            
            print(f"Testing {method_name.upper()}...")
            
            for dataset_name, dataset in self.test_data.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = self.test_method(method_name, dataset_name, dataset)
                    
                    self.results[method_name][dataset_name] = result
                    completed_tests += 1
                    
                    if "error" in result:
                        print(f"  ❌ {dataset_name:15}: {result['error']}")
                    else:
                        # Quick summary
                        target_cols = [k for k in result.keys() if k != "generation_time"]
                        if target_cols:
                            first_target = result[target_cols[0]]
                            print(f"  ✅ {dataset_name:15}: {first_target['trade_frequency']:6.1%} freq, "
                                  f"{first_target['expected_return']:8.5f} ret, "
                                  f"{first_target['generation_time']:6.2f}s")
                        else:
                            print(f"  ⚠️ {dataset_name:15}: No targets generated")
                            
                except Exception as e:
                    print(f"  ❌ {dataset_name:15}: Exception - {e}")
                    self.results[method_name][dataset_name] = {"error": str(e)}
                    completed_tests += 1
            
            print(f"Progress: {completed_tests}/{total_tests} tests completed\n")
        
        print("✅ Comprehensive testing completed!")
    
    def generate_comparative_analysis(self):
        """Generate detailed comparative analysis of all methods"""
        print("\n" + "=" * 100)
        print("COMPARATIVE ANALYSIS OF ALL TARGET GENERATION METHODS")
        print("=" * 100)
        
        # Performance summary table
        print("\n📊 PERFORMANCE SUMMARY BY METHOD")
        print("-" * 130)
        print(f"{'Method':>20} {'Avg Return':>10} {'Avg TradeFreq':>12} {'Avg Balance':>11} {'Avg GenTime':>11} {'Success Rate':>12} {'Best Regime':>15}")
        print("-" * 130)
        
        method_summaries = {}
        
        for method_name, method_results in self.results.items():
            returns = []
            trade_freqs = []
            balance_scores = []
            gen_times = []
            errors = 0
            best_regime = ""
            best_return = float('-inf')
            
            for dataset_name, result in method_results.items():
                if "error" in result:
                    errors += 1
                    continue
                
                # Get first target column results
                target_cols = [k for k in result.keys() if k != "generation_time"]
                if target_cols:
                    target_result = result[target_cols[0]]
                    
                    returns.append(target_result["expected_return"])
                    trade_freqs.append(target_result["trade_frequency"])
                    balance_scores.append(target_result["balance_score"])
                    gen_times.append(target_result.get("generation_time", 0))
                    
                    if target_result["expected_return"] > best_return:
                        best_return = target_result["expected_return"]
                        best_regime = dataset_name
            
            if returns:
                avg_return = np.mean(returns)
                avg_trade_freq = np.mean(trade_freqs)
                avg_balance = np.mean(balance_scores)
                avg_gen_time = np.mean(gen_times)
                success_rate = (len(method_results) - errors) / len(method_results)
                
                method_summaries[method_name] = {
                    "avg_return": avg_return,
                    "avg_trade_freq": avg_trade_freq,
                    "avg_balance": avg_balance,
                    "avg_gen_time": avg_gen_time,
                    "success_rate": success_rate,
                    "best_regime": best_regime,
                    "best_return": best_return
                }
                
                print(f"{method_name:>20} {avg_return:>10.6f} {avg_trade_freq:>12.1%} {avg_balance:>11.3f} {avg_gen_time:>11.2f}s {success_rate:>12.1%} {best_regime:>15}")
            else:
                print(f"{method_name:>20} {'ERROR':>10} {'ERROR':>12} {'ERROR':>11} {'ERROR':>11} {'0.0%':>12} {'NONE':>15}")
        
        # Market regime analysis
        print(f"\n📈 PERFORMANCE BY MARKET REGIME")
        print("-" * 120)
        print(f"{'Regime':>15} {'Best Method':>20} {'Best Return':>11} {'Worst Method':>20} {'Worst Return':>12}")
        print("-" * 120)
        
        for dataset_name in self.test_data.keys():
            regime_results = []
            
            for method_name, method_results in self.results.items():
                if dataset_name in method_results and "error" not in method_results[dataset_name]:
                    result = method_results[dataset_name]
                    target_cols = [k for k in result.keys() if k != "generation_time"]
                    if target_cols:
                        target_result = result[target_cols[0]]
                        regime_results.append((method_name, target_result["expected_return"]))
            
            if regime_results:
                regime_results.sort(key=lambda x: x[1], reverse=True)
                best_method, best_return = regime_results[0]
                worst_method, worst_return = regime_results[-1]
                
                print(f"{dataset_name:>15} {best_method:>20} {best_return:>11.6f} {worst_method:>20} {worst_return:>12.6f}")
        
        # Top performers analysis
        print(f"\n🏆 TOP PERFORMERS BY METRIC")
        print("-" * 80)
        
        if method_summaries:
            # Best return
            best_return_method = max(method_summaries.items(), key=lambda x: x[1]["avg_return"])
            print(f"📈 Highest Returns: {best_return_method[0]} ({best_return_method[1]['avg_return']:.6f})")
            
            # Most balanced
            best_balance_method = max(method_summaries.items(), key=lambda x: x[1]["avg_balance"])
            print(f"⚖️  Best Balance: {best_balance_method[0]} ({best_balance_method[1]['avg_balance']:.3f})")
            
            # Fastest generation
            fastest_method = min(method_summaries.items(), key=lambda x: x[1]["avg_gen_time"])
            print(f"⚡ Fastest Generation: {fastest_method[0]} ({fastest_method[1]['avg_gen_time']:.2f}s)")
            
            # Most reliable
            most_reliable = max(method_summaries.items(), key=lambda x: x[1]["success_rate"])
            print(f"🛡️  Most Reliable: {most_reliable[0]} ({most_reliable[1]['success_rate']:.1%})")
        
        # Method-specific insights
        print(f"\n🔬 METHOD-SPECIFIC INSIGHTS")
        print("-" * 80)
        
        insights = {
            "ga_labeling": "Evolutionary optimization with transaction cost awareness",
            "oracle_binary": "Perfect foresight binary classification - theoretical maximum", 
            "oracle_ternary": "Perfect foresight ternary classification with neutral zone",
            "binary_ctl": "Academic binary trend labeling from TStrends research",
            "ternary_ctl": "Academic ternary trend labeling with configurable thresholds",
            "quantile_classification": "Traditional balanced classification approach",
            "triple_barrier": "López de Prado barrier method with profit/loss/time barriers",
            "triple_exceedance": "Novel transaction cost-scaled multi-objective method",
            "log_return_horizons": "Multi-horizon regression for diverse time scales"
        }
        
        for method_name in method_summaries.keys():
            insight = insights.get(method_name, "Method analysis")
            summary = method_summaries[method_name]
            
            print(f"\n{method_name.upper()}:")
            print(f"  {insight}")
            print(f"  Performance: {summary['avg_return']:.6f} return, {summary['avg_trade_freq']:.1%} frequency")
            print(f"  Best on: {summary['best_regime']} market ({summary['best_return']:.6f} return)")
        
        return method_summaries
    
    def export_results(self, filename: str = "comprehensive_method_test_results.json"):
        """Export detailed results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n💾 Detailed results exported to {filename}")
        except Exception as e:
            print(f"❌ Failed to export results: {e}")


def main():
    """Main testing function"""
    print("🚀 Starting Comprehensive Target Generation Method Testing")
    print("=" * 60)
    
    # Initialize tester
    tester = ComprehensiveMethodTester(verbose=True)
    
    # Create test datasets
    tester.create_test_datasets(n_samples=1200)  # Reasonable size for testing
    
    # Run comprehensive tests
    tester.run_comprehensive_test()
    
    # Generate analysis
    method_summaries = tester.generate_comparative_analysis()
    
    # Export results
    tester.export_results()
    
    print(f"\n✅ Comprehensive testing completed!")
    print(f"📊 Tested {len(method_summaries)} methods on {len(tester.test_data)} market regimes")
    print(f"💾 Results saved for further analysis")


if __name__ == "__main__":
    main()