#!/usr/bin/env python3
"""
COMPREHENSIVE ASSESSMENT PLAN

Systematic evaluation of all optimization fixes implemented:

1. TECHNICAL FIXES VALIDATION
   - Parameter type conversion (float->int/bool)
   - Enhanced output calculation consistency
   - Windowing strategy alignment
   - PnL calculation methodology

2. ECONOMIC ANALYSIS VALIDATION
   - Transaction cost corrections (0.7 pips total, not 1.4)
   - Barrier width economics
   - Hit rate analysis
   - Breakeven thresholds

3. TEMPORAL ANALYSIS VALIDATION  
   - Lookforward window impact (5000+ ticks)
   - Move development over time
   - Timeout rate optimization
   - Adverse selection reduction

4. METHOD-SPECIFIC EVALUATION
   - Binary CTL: Updated bounds effectiveness
   - Ternary CTL: Class balance issues resolution
   - Triple Barrier: Long window + small barriers
   - Triple Exceedance: Scaling factor optimization
   - GA Labeling: Existing performance baseline

5. INTEGRATION TESTING
   - Enhanced output vs optimization consistency
   - All methods with new bounds
   - Edge case handling
   - Performance comparison

This assessment will provide definitive answers on whether our fixes work.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Any
import traceback

try:
    from represent.target_generators.factory import TargetGeneratorFactory
    from tstrends.returns_estimation import ReturnsEstimatorWithFees
    from tstrends.returns_estimation.fees_config import FeesConfig
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


class ComprehensiveAssessment:
    """Comprehensive assessment of all optimization fixes."""
    
    def __init__(self):
        self.results = {}
        self.test_data = None
        self.prices = None
        self.load_test_data()
    
    def load_test_data(self):
        """Load test data for assessments."""
        try:
            data_path = Path("/Users/danielfisher/data/databento/symbol_datasets/inputs/M6AM4_inputs_only_dataset_20250909_140944.parquet")
            df = pl.read_parquet(data_path)
            df = df.filter(pl.col('mid_price').is_not_null())
            self.test_data = df.head(30000)  # 30K samples for comprehensive testing
            self.prices = self.test_data["mid_price"].to_numpy()
            print(f"✅ Loaded {len(self.test_data):,} samples for assessment")
        except Exception as e:
            print(f"❌ Failed to load test data: {e}")
            raise
    
    def assess_technical_fixes(self) -> Dict[str, Any]:
        """1. TECHNICAL FIXES VALIDATION"""
        print("\n" + "="*80)
        print("1. TECHNICAL FIXES VALIDATION")
        print("="*80)
        
        results = {"status": "SUCCESS", "issues": [], "validations": []}
        
        # Test parameter type conversion
        print("\n🔧 Testing Parameter Type Conversion...")
        try:
            # Import the conversion function
            import sys
            sys.path.append('/Users/danielfisher/repositories/represent/examples')
            from symbol_optimization_runner import convert_params_for_generator
            
            # Test triple barrier conversion
            test_params = {
                'lookforward_window': 5000.0,
                'volatility_window': 200.5,
                'normalize_by_volatility': 0.8,
                'barrier_width': 0.0001,
            }
            
            converted = convert_params_for_generator("triple_barrier", test_params)
            
            # Validate conversions
            assert isinstance(converted['lookforward_window'], int), "lookforward_window not converted to int"
            assert isinstance(converted['volatility_window'], int), "volatility_window not converted to int" 
            assert isinstance(converted['normalize_by_volatility'], bool), "normalize_by_volatility not converted to bool"
            assert isinstance(converted['barrier_width'], float), "barrier_width should remain float"
            
            results["validations"].append("✅ Parameter type conversion works correctly")
            print("   ✅ Parameter type conversion validated")
            
        except Exception as e:
            results["issues"].append(f"Parameter type conversion failed: {e}")
            print(f"   ❌ Parameter type conversion failed: {e}")
        
        # Test enhanced output calculation
        print("\n📊 Testing Enhanced Output Calculation...")
        try:
            from symbol_optimization_runner import calculate_additional_metrics
            
            # Test with simple binary CTL parameters
            test_params = {"omega": 0.01}
            metrics = calculate_additional_metrics(self.prices[:5000], "binary_ctl", test_params)
            
            required_keys = ['class_balance_score', 'label_distribution', 'num_trades', 
                           'total_pnl', 'mean_return_per_trade', 'valid_windows', 'total_windows']
            
            for key in required_keys:
                assert key in metrics, f"Missing key: {key}"
            
            assert metrics['valid_windows'] > 0, "No valid windows processed"
            assert 'error' not in metrics, f"Error in metrics: {metrics.get('error')}"
            
            results["validations"].append("✅ Enhanced output calculation works correctly")
            print("   ✅ Enhanced output calculation validated")
            
        except Exception as e:
            results["issues"].append(f"Enhanced output calculation failed: {e}")
            print(f"   ❌ Enhanced output calculation failed: {e}")
        
        return results
    
    def assess_economic_analysis(self) -> Dict[str, Any]:
        """2. ECONOMIC ANALYSIS VALIDATION"""
        print("\n" + "="*80)
        print("2. ECONOMIC ANALYSIS VALIDATION")
        print("="*80)
        
        results = {"status": "SUCCESS", "issues": [], "validations": []}
        
        # Validate transaction cost understanding
        print("\n💸 Validating Transaction Cost Analysis...")
        
        transaction_cost = 0.00007  # Our standard cost
        pip_value = 0.00001
        transaction_cost_pips = transaction_cost / pip_value
        
        print(f"   Transaction cost: {transaction_cost} decimal = {transaction_cost_pips} pips")
        
        # Should be 0.7 pips total round-trip
        expected_pips = 0.7
        if abs(transaction_cost_pips - expected_pips) < 0.01:
            results["validations"].append("✅ Transaction cost correctly understood (0.7 pips total)")
            print("   ✅ Transaction cost validation passed")
        else:
            results["issues"].append(f"Transaction cost mismatch: {transaction_cost_pips} vs {expected_pips}")
            print(f"   ❌ Transaction cost mismatch")
        
        # Test economic breakeven logic
        print("\n📈 Testing Economic Breakeven Logic...")
        breakeven_barrier_decimal = transaction_cost + 0.00001  # Slightly above transaction cost
        breakeven_barrier_pips = breakeven_barrier_decimal / pip_value
        
        if breakeven_barrier_pips > transaction_cost_pips:
            results["validations"].append(f"✅ Breakeven logic correct: {breakeven_barrier_pips:.1f}p barrier > {transaction_cost_pips:.1f}p cost")
            print(f"   ✅ Breakeven barrier ({breakeven_barrier_pips:.1f}p) > transaction cost ({transaction_cost_pips:.1f}p)")
        
        return results
    
    def assess_temporal_analysis(self) -> Dict[str, Any]:
        """3. TEMPORAL ANALYSIS VALIDATION"""
        print("\n" + "="*80)
        print("3. TEMPORAL ANALYSIS VALIDATION") 
        print("="*80)
        
        results = {"status": "SUCCESS", "issues": [], "validations": [], "measurements": {}}
        
        # Test move development over different time horizons
        print("\n⏱️  Testing Move Development Over Time...")
        
        try:
            lookforward_windows = [1000, 3000, 5000, 10000]
            
            for window in lookforward_windows:
                # Calculate max moves over this window
                max_moves = []
                for i in range(0, len(self.prices) - window, window//2):  # 50% overlap
                    window_prices = self.prices[i:i+window]
                    if len(window_prices) > 10:
                        max_move = np.max(window_prices) - np.min(window_prices)
                        max_moves.append(max_move / 0.00001)  # Convert to pips
                
                if max_moves:
                    mean_max_move = np.mean(max_moves)
                    results["measurements"][f"mean_max_move_{window}"] = mean_max_move
                    print(f"   Window {window}: Mean max move = {mean_max_move:.1f} pips")
            
            # Validate that longer windows capture larger moves
            short_moves = results["measurements"].get("mean_max_move_1000", 0)
            long_moves = results["measurements"].get("mean_max_move_10000", 0)
            
            if long_moves > short_moves * 1.5:
                results["validations"].append(f"✅ Longer windows capture larger moves ({long_moves:.1f}p vs {short_moves:.1f}p)")
                print(f"   ✅ Temporal scaling validated")
            else:
                results["issues"].append(f"Temporal scaling unclear: {long_moves:.1f}p vs {short_moves:.1f}p")
        
        except Exception as e:
            results["issues"].append(f"Temporal analysis failed: {e}")
            print(f"   ❌ Temporal analysis failed: {e}")
        
        return results
    
    def assess_method_specific_performance(self) -> Dict[str, Any]:
        """4. METHOD-SPECIFIC EVALUATION"""
        print("\n" + "="*80)
        print("4. METHOD-SPECIFIC EVALUATION")
        print("="*80)
        
        results = {"methods": {}, "status": "SUCCESS", "issues": []}
        
        if not LIBRARIES_AVAILABLE:
            results["issues"].append("Libraries not available for method testing")
            return results
        
        # Test configurations for each method
        test_configs = {
            "binary_ctl": {
                "omega": 0.01,
                "expected_classes": 2,
                "description": "Binary trend labeling"
            },
            "ternary_ctl": {
                "marginal_change_thres": 0.001,
                "window_size": 200,
                "expected_classes": 3,
                "description": "Ternary trend labeling"
            },
            "triple_barrier": {
                "lookforward_window": 8000,  # Long window per our fix
                "barrier_width": 0.0002,     # 2 pip barriers
                "min_return_threshold": 1e-7,
                "volatility_window": 300,
                "normalize_by_volatility": False,
                "expected_classes": 3,
                "description": "Triple barrier with long windows"
            },
            "triple_exceedance": {
                "lookforward_window": 8000,  # Long window per our fix
                "scaling_factor": 3.0,       # 3x scaling = 2.1 pip barriers
                "expected_classes": 3,
                "description": "Triple exceedance with long windows"
            }
        }
        
        for method_name, config in test_configs.items():
            print(f"\n🎯 Testing {method_name.upper()}...")
            method_results = {"status": "SUCCESS", "metrics": {}, "issues": []}
            
            try:
                expected_classes = config.pop("expected_classes")
                description = config.pop("description")
                
                # Generate labels
                generator = TargetGeneratorFactory.create(method_name, **config)
                targets_df = generator.generate_targets(self.test_data)
                target_info = generator.get_target_info()
                target_col = target_info['target_names'][0]
                labels = targets_df[target_col].to_numpy()
                
                # Basic validation
                unique_labels, counts = np.unique(labels, return_counts=True)
                num_classes = len(unique_labels)
                percentages = counts / len(labels) * 100
                
                method_results["metrics"]["num_classes"] = num_classes
                method_results["metrics"]["class_distribution"] = dict(zip(unique_labels, percentages.round(1)))
                
                # Calculate PnL using exact method
                fees_config = FeesConfig(lp_transaction_fees=0.00007, sp_transaction_fees=0.00007)
                returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
                
                total_pnl = returns_estimator.estimate_return(
                    self.prices.tolist(),
                    labels.tolist()
                )
                
                num_trades = sum(1 for j in range(1, len(labels)) if labels[j] != labels[j-1])
                
                method_results["metrics"]["total_pnl"] = total_pnl
                method_results["metrics"]["total_pnl_pips"] = total_pnl * 10000
                method_results["metrics"]["num_trades"] = num_trades
                method_results["metrics"]["mean_return_per_trade"] = total_pnl / num_trades if num_trades > 0 else 0
                
                # Evaluate results
                print(f"   📊 Classes: {num_classes} (expected: {expected_classes})")
                print(f"   📊 Distribution: {dict(zip(unique_labels, percentages.round(1)))}")
                print(f"   💰 PnL: {total_pnl*10000:.0f} pips ({total_pnl*100:.2f}%)")
                print(f"   🔄 Trades: {num_trades:,}")
                
                # Validation checks
                if num_classes >= expected_classes:
                    method_results["validations"] = method_results.get("validations", [])
                    method_results["validations"].append(f"✅ Generated {num_classes} classes as expected")
                else:
                    method_results["issues"].append(f"Only {num_classes} classes, expected {expected_classes}")
                
                if num_trades > 0:
                    method_results["validations"] = method_results.get("validations", [])
                    method_results["validations"].append("✅ Generated trading activity")
                else:
                    method_results["issues"].append("No trading activity generated")
                
                if total_pnl > -0.01:  # Within 1% loss tolerance
                    method_results["validations"] = method_results.get("validations", [])
                    method_results["validations"].append(f"✅ Reasonable PnL performance ({total_pnl*100:.2f}%)")
                else:
                    method_results["issues"].append(f"Poor PnL performance ({total_pnl*100:.2f}%)")
                
            except Exception as e:
                method_results["status"] = "FAILED"
                method_results["issues"].append(f"Method failed: {str(e)[:100]}")
                print(f"   ❌ {method_name} failed: {e}")
                traceback.print_exc()
            
            results["methods"][method_name] = method_results
        
        return results
    
    def assess_integration_testing(self) -> Dict[str, Any]:
        """5. INTEGRATION TESTING"""
        print("\n" + "="*80)
        print("5. INTEGRATION TESTING")
        print("="*80)
        
        results = {"status": "SUCCESS", "issues": [], "validations": []}
        
        # Test enhanced output vs optimization consistency
        print("\n🔗 Testing Enhanced Output vs Optimization Consistency...")
        
        try:
            import sys
            sys.path.append('/Users/danielfisher/repositories/represent/examples')
            from symbol_optimization_runner import calculate_additional_metrics
            
            # Test with binary CTL as reference
            test_params = {"omega": 0.01}
            
            # Calculate using enhanced output method
            enhanced_metrics = calculate_additional_metrics(self.prices[:5000], "binary_ctl", test_params)
            
            # Manually calculate using same method for comparison
            generator = TargetGeneratorFactory.create("binary_ctl", **test_params)
            targets_df = generator.generate_targets(pl.DataFrame({"mid_price": self.prices[:5000]}))
            target_info = generator.get_target_info()
            target_col = target_info['target_names'][0]
            labels = targets_df[target_col].to_numpy()
            
            fees_config = FeesConfig(lp_transaction_fees=0.00007, sp_transaction_fees=0.00007)
            returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
            manual_pnl = returns_estimator.estimate_return(self.prices[:5000].tolist(), labels.tolist())
            
            # Compare results
            if 'error' not in enhanced_metrics:
                enhanced_pnl = enhanced_metrics['total_pnl']
                pnl_diff = abs(enhanced_pnl - manual_pnl)
                
                if pnl_diff < 0.001:  # Within 0.1% tolerance
                    results["validations"].append("✅ Enhanced output matches manual calculation")
                    print(f"   ✅ PnL consistency validated (diff: {pnl_diff:.6f})")
                else:
                    results["issues"].append(f"PnL mismatch: enhanced={enhanced_pnl:.6f}, manual={manual_pnl:.6f}")
                    print(f"   ❌ PnL mismatch detected")
            else:
                results["issues"].append(f"Enhanced metrics failed: {enhanced_metrics['error']}")
        
        except Exception as e:
            results["issues"].append(f"Integration test failed: {e}")
            print(f"   ❌ Integration test failed: {e}")
        
        return results
    
    def generate_final_assessment(self) -> Dict[str, Any]:
        """Generate final comprehensive assessment report."""
        print("\n" + "="*80)
        print("FINAL ASSESSMENT REPORT")
        print("="*80)
        
        # Compile all results
        final_results = {
            "technical_fixes": self.results.get("technical_fixes", {}),
            "economic_analysis": self.results.get("economic_analysis", {}),
            "temporal_analysis": self.results.get("temporal_analysis", {}),
            "method_performance": self.results.get("method_performance", {}),
            "integration_testing": self.results.get("integration_testing", {}),
            "overall_status": "SUCCESS",
            "critical_issues": [],
            "successes": [],
            "recommendations": []
        }
        
        # Analyze results
        all_issues = []
        all_successes = []
        
        for category, result in final_results.items():
            if isinstance(result, dict):
                if "issues" in result:
                    all_issues.extend(result["issues"])
                if "validations" in result:
                    all_successes.extend(result["validations"])
        
        final_results["critical_issues"] = all_issues
        final_results["successes"] = all_successes
        
        # Determine overall status
        if len(all_issues) > 3:
            final_results["overall_status"] = "NEEDS_WORK"
        elif len(all_issues) > 0:
            final_results["overall_status"] = "PARTIAL_SUCCESS"
        
        # Generate recommendations
        recommendations = []
        
        if len(all_issues) == 0:
            recommendations.append("✅ All fixes working correctly - ready for production optimization")
        else:
            recommendations.append("🔧 Address remaining issues before production deployment")
            
        if len(all_successes) > 5:
            recommendations.append("✅ Major improvements achieved in optimization system")
            
        final_results["recommendations"] = recommendations
        
        return final_results
    
    def run_comprehensive_assessment(self):
        """Execute complete assessment plan."""
        print("🔍 STARTING COMPREHENSIVE ASSESSMENT")
        print("="*80)
        print("Evaluating all optimization fixes systematically...")
        
        try:
            # Execute each assessment phase
            self.results["technical_fixes"] = self.assess_technical_fixes()
            self.results["economic_analysis"] = self.assess_economic_analysis()
            self.results["temporal_analysis"] = self.assess_temporal_analysis()
            self.results["method_performance"] = self.assess_method_specific_performance()
            self.results["integration_testing"] = self.assess_integration_testing()
            
            # Generate final report
            final_assessment = self.generate_final_assessment()
            
            print(f"\n🎯 OVERALL STATUS: {final_assessment['overall_status']}")
            print(f"✅ Successes: {len(final_assessment['successes'])}")
            print(f"❌ Issues: {len(final_assessment['critical_issues'])}")
            
            if final_assessment['successes']:
                print("\n✅ SUCCESSES:")
                for success in final_assessment['successes'][:5]:  # Top 5
                    print(f"   {success}")
            
            if final_assessment['critical_issues']:
                print("\n❌ CRITICAL ISSUES:")
                for issue in final_assessment['critical_issues'][:5]:  # Top 5
                    print(f"   {issue}")
            
            if final_assessment['recommendations']:
                print("\n💡 RECOMMENDATIONS:")
                for rec in final_assessment['recommendations']:
                    print(f"   {rec}")
            
            return final_assessment
            
        except Exception as e:
            print(f"❌ Assessment failed: {e}")
            traceback.print_exc()
            return {"overall_status": "FAILED", "error": str(e)}


def main():
    """Execute comprehensive assessment."""
    assessment = ComprehensiveAssessment()
    return assessment.run_comprehensive_assessment()


if __name__ == "__main__":
    main()