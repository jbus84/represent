#!/usr/bin/env python3
"""
Parameter Optimization Demo

This script demonstrates Bayesian parameter optimization for different
labeling approaches, optimizing for returns with 0.7 pip transaction fees.
"""

import sys
from pathlib import Path

import numpy as np

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.labeling_approaches_visualization import (
    create_synthetic_market_data,
    load_market_data_from_dbn,
)
from represent.parameter_optimization import ParameterOptimizer, optimize_all_methods


def create_optimization_data(n_series: int = 3, samples_per_series: int = 25000):
    """Create multiple price series for optimization."""
    print(f"📊 Creating {n_series} synthetic price series ({samples_per_series} samples each)")

    price_series = []
    for i in range(n_series):
        np.random.seed(42 + i)  # Different seed for each series
        market_data = create_synthetic_market_data(samples_per_series)
        prices = market_data['mid_price'].to_numpy()
        price_series.append(prices)

        trend = "UP" if prices[-1] > prices[0] else "DOWN"
        change = ((prices[-1] / prices[0]) - 1) * 100
        print(f"   Series {i+1}: {prices[0]:.4f} -> {prices[-1]:.4f} ({change:+.2f}% {trend})")

    return price_series


def demo_ga_optimization():
    """Demonstrate GA labeling parameter optimization."""
    print("\n🧬 GA LABELING PARAMETER OPTIMIZATION")
    print("=" * 60)

    # Create test data with realistic sample size
    price_series = create_optimization_data(n_series=2, samples_per_series=15000)

    # Initialize optimizer with 0.7 pip fees
    optimizer = ParameterOptimizer(
        fee_pips=0.7,
        initial_points=5,  # Reduced for demo
        n_calls=20,        # Reduced for demo
        verbose=True
    )

    # Custom bounds with realistic values
    custom_bounds = {
        'population_size': (30, 60),
        'max_generations': (25, 75),
        'lookforward_window': (100, 400),
        'transaction_cost': (0.00005, 0.0001),  # 0.5-1.0 pips around 0.7 pips
    }

    # Optimize GA parameters
    result = optimizer.optimize_ga_labeling(price_series, custom_bounds=custom_bounds)

    print("\n🎯 GA OPTIMIZATION RESULTS:")
    print(f"   Optimal parameters: {result['optimal_params']}")
    print(f"   Maximum returns: {result['maximum_returns']:.4f}")
    print(f"   Transaction fee: {result['fee_pips']} pips")

    return result


def demo_comparison_optimization():
    """Demonstrate optimization across multiple methods."""
    print("\n🔍 MULTI-METHOD PARAMETER OPTIMIZATION")
    print("=" * 60)

    # Create test data with realistic sample size
    price_series = create_optimization_data(n_series=2, samples_per_series=12000)

    # Optimize only available methods (GA + any tstrends methods available)
    try:
        results = optimize_all_methods(
            prices=price_series,
            methods=['ga_labeling'],  # Start with GA only
            fee_pips=0.7,
            initial_points=3,
            n_calls=10,
            verbose=True
        )

        print("\n📊 OPTIMIZATION COMPARISON:")
        print("=" * 60)

        best_method = None
        best_returns = float('-inf')

        for method, result in results.items():
            returns = result['maximum_returns']
            print(f"{method.upper()}: {returns:.4f} returns")

            if returns > best_returns:
                best_returns = returns
                best_method = method

        if best_method:
            print(f"\n🏆 Best method: {best_method.upper()} ({best_returns:.4f} returns)")
            print(f"   Parameters: {results[best_method]['optimal_params']}")

        return results

    except Exception as e:
        print(f"❌ Multi-method optimization failed: {e}")
        return {}


def demo_real_data_optimization():
    """Demonstrate optimization on real market data."""
    print("\n📈 REAL DATA OPTIMIZATION")
    print("=" * 60)

    try:
        # Load real market data
        market_data = load_market_data_from_dbn(max_samples=5000, max_files=2)

        if len(market_data) < 1000:
            print("⚠️  Insufficient real data, using synthetic data")
            market_data = create_synthetic_market_data(2000)

        prices = market_data['mid_price'].to_numpy()
        print(f"📊 Using real market data: {len(prices)} samples")
        print(f"   Price range: {prices.min():.6f} to {prices.max():.6f}")

        # Optimize GA on real data
        optimizer = ParameterOptimizer(
            fee_pips=0.7,
            initial_points=5,
            n_calls=15,
            verbose=True
        )

        result = optimizer.optimize_ga_labeling([prices])

        print("\n🎯 REAL DATA GA OPTIMIZATION:")
        print(f"   Optimal parameters: {result['optimal_params']}")
        print(f"   Maximum returns: {result['maximum_returns']:.4f}")

        return result

    except Exception as e:
        print(f"❌ Real data optimization failed: {e}")
        return None


def main():
    """Main demonstration function."""
    print("🚀 PARAMETER OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    print("This script demonstrates Bayesian parameter optimization")
    print("for labeling approaches using 0.7 pip transaction fees.")
    print()

    try:
        # Demo 1: GA optimization
        ga_result = demo_ga_optimization()

        # Demo 2: Multi-method comparison
        demo_comparison_optimization()

        # Demo 3: Real data optimization
        demo_real_data_optimization()

        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("Key takeaways:")
        print("• Bayesian optimization can significantly improve labeling performance")
        print("• Different methods have different optimal parameter ranges")
        print("• Real market data may require different parameters than synthetic data")
        print("• 0.7 pip transaction costs are properly accounted for in optimization")

        if ga_result:
            print("\nTo use optimized GA parameters:")
            print("```python")
            print("from represent import GALabelingGenerator")
            print("")
            print("generator = GALabelingGenerator(")
            for param, value in ga_result['optimal_params'].items():
                if isinstance(value, float):
                    print(f"    {param}={value:.4f},")
                else:
                    print(f"    {param}={value},")
            print(")")
            print("```")

    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("\nTo use parameter optimization, install:")
        print("  pip install scikit-optimize")
        print("  pip install git+https://github.com/agpenas/tstrends.git")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
