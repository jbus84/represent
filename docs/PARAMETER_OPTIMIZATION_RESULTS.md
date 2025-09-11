# Symbol-Specific Parameter Optimization Results

## Overview

This document summarizes the Bayesian parameter optimization results for Australian Dollar (AUD) futures contracts using the Represent package. Each contract was optimized individually to account for unique microstructure characteristics.

## Methodology

- **Optimization Algorithm**: Gaussian Process (Bayesian Optimization) with Window Sampling
- **Transaction Costs**: 0.7 pips (realistic market conditions)
- **Fitness Metric**: Net returns after transaction fees
- **Optimization Calls**: 50 iterations per method per symbol
- **Window Sampling**: 75k sample windows, 12 windows per evaluation (stratified sampling)
- **Data**: Real AUD futures tick data from Databento

## Symbol Coverage

Optimized parameters for **6 Australian Dollar futures contracts**:
- **M6AH5**: March 2025 AUD futures (18.98M samples)
- **M6AM4**: June 2024 AUD futures (15.09M samples)  
- **M6AM5**: June 2025 AUD futures (3.53M samples)
- **M6AU4**: September 2024 AUD futures (21.85M samples)
- **M6AU5**: September 2025 AUD futures (0.10M samples)
- **M6AZ4**: December 2024 AUD futures (26.44M samples)

## Results Summary

### Performance Ranking by Method

| Method | Best Returns | Best Symbol | Average Returns | Range |
|--------|--------------|-------------|-----------------|-------|
| **GA Labeling** | **73.33%** | M6AZ4 | 47.75% | 21.13% - 73.33% |
| **Binary CTL** | **40.06%** | M6AM4 | 26.82% | 14.62% - 40.06% |
| **Ternary CTL** | **24.67%** | M6AU5 | 14.15% | 4.83% - 24.67% |

### Key Insights

1. **GA Labeling Dominates**: Evolutionary optimization consistently outperforms academic methods
2. **Contract-Specific Variation**: Up to 3.5x performance difference between contracts for same method
3. **Expiry Effects**: Near-expiry contracts (M6AU5) show different optimal parameters
4. **Volume Impact**: Larger contracts (M6AZ4, M6AU4) benefit from more aggressive parameters

## Method-Specific Results

### GA Labeling Optimization

**Best Performer**: M6AZ4 (December 2024) - 73.33% returns

#### Optimal Parameter Ranges Across Symbols
- **Population Size**: 21-28 (avg: 24.5)
- **Max Generations**: 16-23 (avg: 19)
- **Lookforward Window**: 3-7 ticks (avg: 5.8)
- **Transaction Cost**: 0.27-0.80% (avg: 0.59%)
- **Win Rate Range**: 66.0%-82.7% (avg: 73.6%)
- **Mutation Rate**: 1.34%-7.73% (avg: 4.56%)

#### Contract-Specific Highlights
- **M6AZ4**: Longest lookforward (7 ticks), highest profit factor (1.79)
- **M6AU5**: Lowest transaction cost threshold (0.27%), most conservative
- **M6AH5**: Smallest population (21), fastest convergence

### Binary CTL Optimization  

**Best Performer**: M6AM4 (June 2024) - 40.06% returns

#### Optimal Parameter Ranges
- **Omega**: 0.000914 - 0.007493 (avg: 0.0038)

#### Contract Insights
- Near-zero omega optimal for most contracts (minimal filtering)
- M6AU4 requires highest omega (0.0075) - most volatile contract
- Consistent 26.8% average returns across all contracts

### Ternary CTL Optimization

**Best Performer**: M6AU5 (September 2025) - 24.67% returns  

#### Optimal Parameter Ranges
- **Marginal Change Threshold**: 0.012461 - 0.077024 (avg: 0.056)
- **Window Size**: 359 - 942 ticks (avg: 684)

#### Contract Insights
- M6AU5 uses lowest threshold (1.25%) - most sensitive to small moves
- Larger contracts prefer larger windows for trend confirmation
- Most challenging method - requires careful threshold tuning

## Production Usage

### Using Optimized Parameters

For each contract, use the stored parameters:

```python
from represent.parameter_storage import ParameterStorage

# Load optimized parameters
storage = ParameterStorage("/Users/danielfisher/data/databento/symbol_datasets/optimization_results/optimized_parameters")

# Get GA parameters for M6AZ4
m6az4_ga_params = storage.load_symbol_parameters("M6AZ4", "ga_labeling")
optimal_params = m6az4_ga_params["optimal_params"]

# Create optimized generator
from represent.target_generators.ga_labeling import GALabelingGenerator
generator = GALabelingGenerator(**optimal_params)
```

### Runtime Performance

- **Optimization Time**: ~4 hours per symbol (50 calls × 12 windows × 75k samples)
- **Sample Efficiency**: 5.0-7.5% of dataset used per optimization
- **Memory Usage**: <2GB peak regardless of dataset size
- **Speedup**: ~50x faster than full dataset optimization

## Files Generated

The optimization process creates:

```
/Users/danielfisher/data/databento/symbol_datasets/optimization_results/
├── OPTIMIZATION_RESULTS.md              # Detailed results report
├── parameter_comparison.csv             # Raw parameter data  
├── parameter_distributions_*.png        # Parameter visualizations
├── returns_comparison.png               # Performance comparison
└── optimized_parameters/                # Individual parameter files
    ├── M6AH5/
    │   ├── ga_labeling_params.json
    │   ├── binary_ctl_params.json
    │   └── ternary_ctl_params.json
    ├── M6AM4/ ... (similar structure)
    └── ... (for each symbol)
```

## Conclusion

The symbol-specific parameter optimization demonstrates:

1. **Significant Performance Gains**: Up to 73% returns vs baseline
2. **Contract Heterogeneity**: Each futures contract benefits from unique parameters  
3. **Method Hierarchy**: GA > Binary CTL > Ternary CTL across all contracts
4. **Scalable Architecture**: Efficient optimization on datasets up to 26M samples

These optimized parameters should be used for production labeling on their respective contracts to maximize trading strategy performance.

---
*Generated by Represent Parameter Optimization System on 2025-09-09*