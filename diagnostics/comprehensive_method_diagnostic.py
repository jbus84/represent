#!/usr/bin/env python3
"""
Comprehensive Diagnostic Analysis for Non-GA Target Generation Methods

This script systematically analyzes potential logical issues with all labeling approaches
to identify problems in parameter bounds, PnL calculations, label distributions, and sampling strategies.
"""

import sys
import os
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from represent.target_generators.factory import TargetGeneratorFactory
from represent.large_scale_optimization import LargeScaleParameterOptimizer

class ComprehensiveMethodDiagnostic:
    """Comprehensive diagnostic analysis for all target generation methods."""
    
    def __init__(self, data_path: str):
        """Initialize with dataset path."""
        self.data_path = Path(data_path)
        self.data = None
        self.results = {}
        
        # Load data
        if data_path.endswith('.parquet'):
            self.data = pl.read_parquet(data_path)
            
            # Handle NaN values
            nan_count = self.data['mid_price'].null_count()
            if nan_count > 0:
                print(f"⚠️ Found {nan_count} NaN values, removing...")
                self.data = self.data.filter(pl.col('mid_price').is_not_null())
            
            self.prices = self.data['mid_price'].to_numpy()
        else:
            raise ValueError("Only parquet files supported for now")
            
        print(f"📊 Loaded dataset: {len(self.data):,} samples")
        print(f"💰 Price range: {self.prices.min():.6f} to {self.prices.max():.6f}")
        
    def diagnose_all_methods(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics on all available methods."""
        
        methods_to_test = [
            'binary_ctl',
            'ternary_ctl', 
            'oracle_binary',
            'oracle_ternary',
            'triple_barrier',
            'triple_exceedance',
            'quantile_classification'
        ]
        
        print("\\n" + "="*80)
        print("🔍 COMPREHENSIVE METHOD DIAGNOSTIC ANALYSIS")
        print("="*80)
        
        for method in methods_to_test:
            print(f"\\n{'='*60}")
            print(f"📋 ANALYZING METHOD: {method.upper()}")
            print(f"{'='*60}")
            
            try:
                self.results[method] = self._diagnose_single_method(method)
            except Exception as e:
                print(f"❌ Failed to analyze {method}: {e}")
                self.results[method] = {'error': str(e)}
                
        return self.results
    
    def _diagnose_single_method(self, method_name: str) -> Dict[str, Any]:
        """Comprehensive analysis of a single method."""
        
        diagnosis = {
            'method': method_name,
            'basic_functionality': None,
            'label_distribution': None,
            'parameter_sensitivity': None,
            'pnl_validation': None,
            'sampling_analysis': None,
            'optimization_bounds': None
        }
        
        # 1. Basic Functionality Test
        print(f"\\n🧪 1. BASIC FUNCTIONALITY TEST")
        diagnosis['basic_functionality'] = self._test_basic_functionality(method_name)
        
        # 2. Label Distribution Analysis  
        print(f"\\n📊 2. LABEL DISTRIBUTION ANALYSIS")
        diagnosis['label_distribution'] = self._analyze_label_distribution(method_name)
        
        # 3. Parameter Sensitivity Analysis
        print(f"\\n🎛️ 3. PARAMETER SENSITIVITY ANALYSIS") 
        diagnosis['parameter_sensitivity'] = self._analyze_parameter_sensitivity(method_name)
        
        # 4. PnL Calculation Validation
        print(f"\\n💰 4. PNL CALCULATION VALIDATION")
        diagnosis['pnl_validation'] = self._validate_pnl_calculation(method_name)
        
        # 5. Sampling Strategy Analysis
        print(f"\\n🔀 5. SAMPLING STRATEGY ANALYSIS")
        diagnosis['sampling_analysis'] = self._analyze_sampling_strategy(method_name)
        
        # 6. Optimization Bounds Check
        print(f"\\n🎯 6. OPTIMIZATION BOUNDS VALIDATION")
        diagnosis['optimization_bounds'] = self._validate_optimization_bounds(method_name)
        
        return diagnosis
    
    def _test_basic_functionality(self, method_name: str) -> Dict[str, Any]:
        """Test if method can generate labels without errors."""
        
        try:
            # Test with default parameters
            generator = TargetGeneratorFactory.create(method_name)
            
            # Use small sample for basic test - create proper DataFrame
            test_df = self.data[:10000]  # 10K sample
            targets_df = generator.generate_targets(test_df)
            
            # Extract the actual target columns (exclude metadata columns)
            target_info = generator.get_target_info()
            target_cols = target_info['target_names']
            labels = targets_df.select(target_cols).to_numpy().flatten() if len(target_cols) == 1 else targets_df.select(target_cols).to_numpy()
            
            result = {
                'success': True,
                'input_samples': len(test_df),
                'output_samples': len(labels),
                'output_type': type(labels).__name__,
                'unique_labels': sorted(np.unique(labels).tolist()),
                'label_counts': {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
            }
            
            print(f"   ✅ Basic functionality: PASSED")
            print(f"   📊 Input: {len(test_df):,} samples → Output: {len(labels):,} labels")
            print(f"   🏷️ Unique labels: {result['unique_labels']}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Basic functionality: FAILED - {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_label_distribution(self, method_name: str) -> Dict[str, Any]:
        """Analyze label distribution patterns."""
        
        try:
            # Test with multiple parameter sets to see distribution variations
            generator = TargetGeneratorFactory.create(method_name)
            
            # Use medium sample for distribution analysis
            test_df = self.data[:50000]  # 50K sample
            targets_df = generator.generate_targets(test_df)
            
            # Extract labels from target columns
            target_info = generator.get_target_info()
            target_cols = target_info['target_names']
            labels = targets_df.select(target_cols).to_numpy().flatten() if len(target_cols) == 1 else targets_df.select(target_cols).to_numpy()
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_labels = len(labels)
            
            distribution = {
                'total_labels': total_labels,
                'unique_labels': len(unique_labels),
                'label_counts': {str(label): int(count) for label, count in zip(unique_labels, counts)},
                'label_percentages': {str(label): (count/total_labels)*100 for label, count in zip(unique_labels, counts)},
                'balance_score': self._calculate_balance_score(counts),
                'entropy': self._calculate_entropy(counts)
            }
            
            print(f"   📊 Label distribution:")
            for label, pct in distribution['label_percentages'].items():
                print(f"      Label {label}: {pct:6.2f}%")
            print(f"   ⚖️ Balance score: {distribution['balance_score']:.3f} (1.0 = perfect balance)")
            print(f"   📈 Entropy: {distribution['entropy']:.3f}")
            
            return distribution
            
        except Exception as e:
            print(f"   ❌ Label distribution analysis: FAILED - {e}")
            return {'error': str(e)}
    
    def _analyze_parameter_sensitivity(self, method_name: str) -> Dict[str, Any]:
        """Test how method responds to different parameters."""
        
        try:
            # Get parameter bounds from optimizer
            optimizer = LargeScaleParameterOptimizer()
            bounds = self._get_method_bounds(method_name, optimizer)
            
            if not bounds:
                return {'message': 'No parameters to optimize (fixed parameters method)'}
            
            # Test 3 parameter sets: low, mid, high values
            test_df = self.data[:25000]  # 25K sample for speed
            sensitivity_results = {}
            
            for param_name, (low, high) in bounds.items():
                print(f"   🎛️ Testing parameter: {param_name}")
                
                param_tests = {}
                for test_name, test_value in [('low', low), ('mid', (low+high)/2), ('high', high)]:
                    try:
                        # Create generator with test parameter
                        params = {param_name: test_value}
                        generator = TargetGeneratorFactory.create(method_name, **params)
                        
                        targets_df = generator.generate_targets(test_df)
                        target_info = generator.get_target_info()
                        target_cols = target_info['target_names']
                        labels = targets_df.select(target_cols).to_numpy().flatten() if len(target_cols) == 1 else targets_df.select(target_cols).to_numpy()
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        
                        param_tests[test_name] = {
                            'parameter_value': test_value,
                            'unique_labels': len(unique_labels),
                            'label_distribution': {str(k): int(v) for k, v in zip(unique_labels, counts)},
                            'balance_score': self._calculate_balance_score(counts)
                        }
                        
                    except Exception as e:
                        param_tests[test_name] = {'error': str(e)}
                
                sensitivity_results[param_name] = param_tests
                
                # Report sensitivity
                try:
                    low_balance = param_tests['low']['balance_score']
                    high_balance = param_tests['high']['balance_score']
                    sensitivity = abs(high_balance - low_balance)
                    print(f"      Sensitivity: {sensitivity:.3f} balance score change")
                except:
                    print(f"      Sensitivity: Could not calculate")
            
            return sensitivity_results
            
        except Exception as e:
            print(f"   ❌ Parameter sensitivity analysis: FAILED - {e}")
            return {'error': str(e)}
    
    def _validate_pnl_calculation(self, method_name: str) -> Dict[str, Any]:
        """Validate PnL calculation with known test cases."""
        
        try:
            generator = TargetGeneratorFactory.create(method_name)
            
            # Create synthetic test cases with known expected results
            test_cases = self._create_pnl_test_cases()
            validation_results = {}
            
            for test_name, (prices, expected_labels, expected_pnl_range) in test_cases.items():
                print(f"   💰 Testing PnL case: {test_name}")
                
                try:
                    # Convert prices to DataFrame for generator
                    test_df = pl.DataFrame({
                        'mid_price': prices,
                        'ts_event': range(len(prices))
                    })
                    
                    targets_df = generator.generate_targets(test_df)
                    target_info = generator.get_target_info()
                    target_cols = target_info['target_names']
                    labels = targets_df.select(target_cols).to_numpy().flatten() if len(target_cols) == 1 else targets_df.select(target_cols).to_numpy()
                    
                    # Calculate PnL using the same logic as optimizer
                    pnl = self._calculate_test_pnl(prices, labels, fee=0.00035)  # 0.35 pips (half of 0.7)
                    
                    validation_results[test_name] = {
                        'input_prices_samples': len(prices),
                        'generated_labels_samples': len(labels),
                        'calculated_pnl': pnl,
                        'expected_pnl_range': expected_pnl_range,
                        'within_expected_range': expected_pnl_range[0] <= pnl <= expected_pnl_range[1]
                    }
                    
                    print(f"      PnL: {pnl:.6f} (expected: {expected_pnl_range})")
                    
                except Exception as e:
                    validation_results[test_name] = {'error': str(e)}
            
            return validation_results
            
        except Exception as e:
            print(f"   ❌ PnL validation: FAILED - {e}")
            return {'error': str(e)}
    
    def _analyze_sampling_strategy(self, method_name: str) -> Dict[str, Any]:
        """Compare performance on sampled vs full dataset."""
        
        try:
            generator = TargetGeneratorFactory.create(method_name)
            
            # Test on full dataset vs sampled windows
            full_sample_size = min(100000, len(self.data))  # 100K max for computational feasibility
            full_df = self.data[:full_sample_size]
            
            # Generate labels for full dataset
            targets_df = generator.generate_targets(full_df)
            target_info = generator.get_target_info()
            target_cols = target_info['target_names']
            full_labels = targets_df.select(target_cols).to_numpy().flatten() if len(target_cols) == 1 else targets_df.select(target_cols).to_numpy()
            full_pnl = self._calculate_test_pnl(full_df['mid_price'].to_numpy(), full_labels, fee=0.00035)
            
            # Generate labels for 5 random windows (like optimization does)
            window_size = 20000
            window_pnls = []
            
            for i in range(5):
                start_idx = np.random.randint(0, len(self.data) - window_size)
                window_df = self.data[start_idx:start_idx + window_size]
                
                window_targets_df = generator.generate_targets(window_df)
                window_labels = window_targets_df.select(target_cols).to_numpy().flatten() if len(target_cols) == 1 else window_targets_df.select(target_cols).to_numpy()
                window_pnl = self._calculate_test_pnl(window_df['mid_price'].to_numpy(), window_labels, fee=0.00035)
                window_pnls.append(window_pnl)
            
            avg_window_pnl = np.mean(window_pnls)
            std_window_pnl = np.std(window_pnls)
            
            result = {
                'full_dataset_pnl': full_pnl,
                'full_dataset_samples': full_sample_size,
                'window_pnls': window_pnls,
                'avg_window_pnl': avg_window_pnl,
                'std_window_pnl': std_window_pnl,
                'pnl_difference': abs(full_pnl - avg_window_pnl),
                'relative_difference_pct': abs((full_pnl - avg_window_pnl) / full_pnl * 100) if full_pnl != 0 else float('inf')
            }
            
            print(f"   📊 Full dataset PnL: {full_pnl:.6f}")
            print(f"   🔀 Avg window PnL: {avg_window_pnl:.6f} ± {std_window_pnl:.6f}")
            print(f"   📈 Difference: {result['relative_difference_pct']:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Sampling analysis: FAILED - {e}")
            return {'error': str(e)}
    
    def _validate_optimization_bounds(self, method_name: str) -> Dict[str, Any]:
        """Check if optimization bounds are reasonable."""
        
        try:
            optimizer = LargeScaleParameterOptimizer()
            bounds = self._get_method_bounds(method_name, optimizer)
            
            if not bounds:
                return {'message': 'No parameters to optimize (fixed parameters method)'}
            
            bounds_analysis = {}
            
            for param_name, (low, high) in bounds.items():
                print(f"   🎯 Parameter: {param_name}")
                print(f"      Range: [{low}, {high}]")
                
                bounds_analysis[param_name] = {
                    'low': low,
                    'high': high,
                    'range_size': high - low,
                    'is_reasonable': self._assess_parameter_reasonableness(param_name, low, high)
                }
            
            return bounds_analysis
            
        except Exception as e:
            print(f"   ❌ Bounds validation: FAILED - {e}")
            return {'error': str(e)}
    
    def _get_method_bounds(self, method_name: str, optimizer: LargeScaleParameterOptimizer) -> Dict[str, Tuple[float, float]]:
        """Get optimization bounds for a method."""
        
        method_map = {
            'binary_ctl': lambda: {'omega': (0.0, 0.05)},
            'ternary_ctl': lambda: {
                'marginal_change_thres': (0.0001, 0.05),
                'window_size': (50, 1000)
            },
            'oracle_binary': lambda: {},  # No parameters
            'oracle_ternary': lambda: {'neutral_reward_factor': (0.1, 0.9)},
            'triple_barrier': lambda: {
                'lookforward_window': (1000, 10000),
                'barrier_width': (0.0001, 0.005),
                'min_return_threshold': (0.00001, 0.0001),
                'volatility_window': (100, 500),
                'normalize_by_volatility': (0, 1)
            },
            'triple_exceedance': lambda: {
                'lookforward_window': (1000, 10000),
                'scaling_factor': (2.0, 20.0),
                'min_exceedance_threshold': (0.3, 0.9),
                'volatility_window': (100, 500),
                'window_penalty_weight': (0.05, 0.5),
                'balance_weight': (0.1, 1.0),
                'target_balance_ratio': (0.25, 0.40),
                'adaptive_scaling': (0, 1)
            },
            'quantile_classification': lambda: {'nbins': (3, 10)}
        }
        
        if method_name in method_map:
            return method_map[method_name]()
        else:
            return {}
    
    def _calculate_balance_score(self, counts: np.ndarray) -> float:
        """Calculate how balanced the label distribution is (1.0 = perfect balance)."""
        if len(counts) <= 1:
            return 1.0
        
        total = np.sum(counts)
        expected_per_class = total / len(counts)
        
        # Calculate how far each class is from perfect balance
        deviations = np.abs(counts - expected_per_class) / expected_per_class
        avg_deviation = np.mean(deviations)
        
        # Convert to score where 1.0 = perfect balance, 0.0 = maximum imbalance
        return max(0.0, 1.0 - avg_deviation)
    
    def _calculate_entropy(self, counts: np.ndarray) -> float:
        """Calculate Shannon entropy of label distribution."""
        total = np.sum(counts)
        probs = counts / total
        probs = probs[probs > 0]  # Remove zero probabilities
        return -np.sum(probs * np.log2(probs))
    
    def _create_pnl_test_cases(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, Tuple[float, float]]]:
        """Create synthetic test cases with known PnL expectations."""
        
        test_cases = {}
        
        # Test case 1: Rising trend - long positions should profit
        rising_prices = np.array([100.0, 100.1, 100.2, 100.3, 100.4, 100.5])
        test_cases['rising_trend'] = (rising_prices, None, (0.0, 0.01))  # Should be positive
        
        # Test case 2: Falling trend - short positions should profit  
        falling_prices = np.array([100.0, 99.9, 99.8, 99.7, 99.6, 99.5])
        test_cases['falling_trend'] = (falling_prices, None, (-0.01, 0.01))  # Depends on method
        
        # Test case 3: Flat/sideways - should lose money to fees
        flat_prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        test_cases['flat_market'] = (flat_prices, None, (-0.01, 0.0))  # Should be negative due to fees
        
        return test_cases
    
    def _calculate_test_pnl(self, prices: np.ndarray, labels: np.ndarray, fee: float) -> float:
        """Calculate PnL using same logic as optimizer."""
        
        # Normalize labels to {-1, 0, 1} format
        normalized_labels = self._normalize_labels_for_pnl(labels)
        
        pnl = 0.0
        position = 0  # -1 short, 0 flat, 1 long
        
        for t in range(1, len(prices)):
            ret = (prices[t] - prices[t-1]) / prices[t-1]
            
            # Change position cost
            if normalized_labels[t] != position:
                if position != 0:  # exiting previous position
                    pnl -= fee
                if normalized_labels[t] != 0:  # entering new position
                    pnl -= fee
                position = normalized_labels[t]
            
            # Accrue returns
            pnl += ret * position
        
        return pnl
    
    def _normalize_labels_for_pnl(self, labels: np.ndarray) -> np.ndarray:
        """Normalize labels to {-1, 0, 1} format for PnL calculation."""
        
        unique_labels = np.unique(labels)
        
        if len(unique_labels) == 2:
            # Binary case: map to {-1, 1}
            low, high = np.min(unique_labels), np.max(unique_labels)
            normalized = np.zeros_like(labels, dtype=np.int32)
            normalized[labels == low] = -1   # Short
            normalized[labels == high] = 1   # Long
            return normalized
            
        elif len(unique_labels) == 3:
            # Ternary case: map to {-1, 0, 1}
            sorted_labels = np.sort(unique_labels)
            low, mid, high = sorted_labels
            normalized = np.zeros_like(labels, dtype=np.int32)
            normalized[labels == low] = -1   # Short
            normalized[labels == mid] = 0    # Flat  
            normalized[labels == high] = 1   # Long
            return normalized
            
        else:
            # Multi-class case: map to {-1, 0, 1} by thirds
            sorted_labels = np.sort(unique_labels)
            n_classes = len(sorted_labels)
            
            normalized = np.zeros_like(labels, dtype=np.int32)
            
            # Bottom third -> Short (-1)
            bottom_third = sorted_labels[:n_classes//3]
            for label in bottom_third:
                normalized[labels == label] = -1
            
            # Top third -> Long (1)  
            top_third = sorted_labels[-n_classes//3:]
            for label in top_third:
                normalized[labels == label] = 1
            
            # Middle -> Flat (0) - already initialized to 0
            
            return normalized
    
    def _assess_parameter_reasonableness(self, param_name: str, low: float, high: float) -> Dict[str, Any]:
        """Assess whether parameter bounds are reasonable."""
        
        assessments = {
            'omega': lambda l, h: {'reasonable': 0 <= l < h <= 0.1, 'comment': 'CTL omega should be small positive'},
            'marginal_change_thres': lambda l, h: {'reasonable': 0 < l < h <= 0.1, 'comment': 'Threshold should be small positive'},
            'window_size': lambda l, h: {'reasonable': 50 <= l < h <= 2000, 'comment': 'Window size should be reasonable'},
            'neutral_reward_factor': lambda l, h: {'reasonable': 0 < l < h < 1, 'comment': 'Factor should be between 0 and 1'},
            'lookforward_window': lambda l, h: {'reasonable': 100 <= l < h <= 20000, 'comment': 'Lookforward should be substantial'},
            'barrier_width': lambda l, h: {'reasonable': 0.00001 <= l < h <= 0.01, 'comment': 'Barrier width should be small'},
            'volatility_window': lambda l, h: {'reasonable': 50 <= l < h <= 1000, 'comment': 'Volatility window should be reasonable'},
            'nbins': lambda l, h: {'reasonable': 2 <= l < h <= 20, 'comment': 'Number of bins should be reasonable'}
        }
        
        if param_name in assessments:
            return assessments[param_name](low, high)
        else:
            return {'reasonable': True, 'comment': 'Unknown parameter - cannot assess'}
    
    def save_results(self, output_path: str):
        """Save diagnostic results to JSON file."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_types(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\\n💾 Results saved to: {output_file}")

def main():
    """Run comprehensive diagnostic analysis."""
    
    # Use the M6AM4 dataset that showed issues
    dataset_path = "/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Run comprehensive analysis
    diagnostic = ComprehensiveMethodDiagnostic(dataset_path)
    results = diagnostic.diagnose_all_methods()
    
    # Save results
    output_path = "comprehensive_method_diagnostic_results.json"
    diagnostic.save_results(output_path)
    
    print("\\n" + "="*80)
    print("🎯 DIAGNOSTIC ANALYSIS COMPLETE")
    print("="*80)
    print(f"📋 Analyzed {len(results)} methods")
    print(f"💾 Detailed results saved to: {output_path}")
    
    # Print summary of critical issues
    print("\\n🚨 CRITICAL ISSUES SUMMARY:")
    for method, data in results.items():
        if 'error' in data:
            print(f"   ❌ {method}: {data['error']}")
        elif data.get('basic_functionality', {}).get('success') == False:
            print(f"   ❌ {method}: Basic functionality failed")
    
    print("\\n✅ Analysis complete! Check the detailed JSON results for comprehensive findings.")

if __name__ == "__main__":
    main()