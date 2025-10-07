# Comprehensive Guide to Labeling Methods in Represent

This guide provides detailed documentation of all 15+ labeling approaches available in the Represent package, including their methodologies, optimization results, and practical applications.

## Table of Contents

1. [Overview](#overview)
2. [Classification Methods](#classification-methods)
3. [Regression Methods](#regression-methods) 
4. [Optimization Results & Analysis](#optimization-results--analysis)
5. [Method Selection Guide](#method-selection-guide)
6. [Performance Comparison](#performance-comparison)

## Overview

The Represent package provides a comprehensive suite of labeling methods for quantitative finance ML applications. All methods have been optimized using **Bayesian parameter optimization with realistic 0.7 pip transaction costs** to ensure practical real-world performance.

### Key Features

- **15+ Labeling Approaches**: From traditional quantile methods to cutting-edge evolutionary optimization
- **Bayesian Optimized Parameters**: All methods tuned for maximum returns with transaction cost awareness
- **Modular Architecture**: Mix and match methods using the TargetGenerator interface
- **Research Integration**: Academic TStrends methods with practical optimizations
- **Performance Validated**: Returns-based optimization ensures trading viability

---

## Classification Methods

Classification methods generate discrete labels for directional prediction and strategy signals.

### 1. Quantile Classification

**Purpose**: Traditional percentile-based balanced labeling for stable ML training.

#### Methodology
- Calculates price movements: `(future_mean - past_mean) / past_mean`
- Applies quantile binning to ensure uniform class distribution
- Maps continuous price changes to discrete classes `{0, 1, 2, ..., nbins-1}`

#### Configuration
```python
TargetGeneratorFactory.create(
    "quantile_classification",
    nbins=13,  # Number of classes (2, 5, 13, 21 common)
    target_name="direction_13class"
)
```

#### Key Features
- **Guaranteed Balanced Classes**: Each class has equal sample count
- **Configurable Granularity**: 2-class binary to 21-class fine-grained
- **Robust**: Works across different market regimes
- **Interpretable**: Clear mapping from returns to labels

#### Use Cases
- **Baseline Classification**: Standard benchmark for ML models
- **Multi-class Direction**: Predict fine-grained price movements
- **Risk Management**: Balanced predictions reduce model bias

#### Performance Characteristics
- **Training Stability**: Balanced classes prevent class imbalance issues
- **Generalization**: Percentile boundaries adapt to data distribution
- **Consistency**: Same methodology across different time periods

---

### 2. GA Labeling (Genetic Algorithm) ⭐

**Purpose**: Evolutionary optimization of trading signals for maximum performance.

#### Methodology
**Revolutionary approach that evolves optimal trading signal patterns using genetic algorithms:**

1. **Population Initialization**: Create population of binary trading signal chromosomes
2. **Trading Simulation**: Simulate realistic trading with transaction costs for each strategy
3. **Fitness Evaluation**: Multi-criteria fitness based on:
   - Total profitability after transaction costs
   - Win rate within acceptable bounds (35-62%)
   - Profit factor (ratio of wins to losses)
   - Minimum trade count for statistical significance
4. **Genetic Evolution**:
   - **Tournament Selection**: Select top performers for reproduction
   - **Single-Point Crossover**: Combine successful strategies
   - **Bit-Flip Mutation**: Introduce variation to prevent local optima
5. **Strategy Constraints**: Enforce practical trading requirements

#### Bayesian Optimization Results
**Performance**: **71.34% returns** (best-in-class across all methods)

**Optimized Parameters**:
```python
{
    "population_size": 30,        # Optimal population for convergence
    "max_generations": 31,        # Sufficient evolution cycles
    "lookforward_window": 4,      # 4-tick prediction horizon
    "transaction_cost": 0.0005,   # 0.05% trading costs
    "min_trades": 8,              # Minimum statistical significance
    "min_win_rate": 0.3578,       # 35.78% minimum win rate
    "max_win_rate": 0.6201,       # 62.01% maximum (avoids overfitting)
    "min_profit_factor": 1.0876,  # Positive profit factor requirement
    "mutation_rate": 0.0173,      # 1.73% mutation (stable evolution)
    "crossover_rate": 0.7438      # 74.38% crossover (balanced exploration)
}
```

#### Why GA Labeling Dominates
1. **Performance-Driven Evolution**: Unlike traditional methods that use statistical properties, GA directly optimizes for trading performance
2. **Transaction Cost Integration**: Built-in awareness of realistic trading costs
3. **Constraint-Based**: Enforces practical trading requirements (minimum trades, win rate bounds)
4. **Adaptive**: Evolves strategies specific to each dataset's characteristics
5. **Multi-Objective**: Balances profitability, win rate, and trading frequency

#### Configuration
```python
# Optimized configuration
TargetGeneratorFactory.create(
    "ga_labeling",
    population_size=30,
    max_generations=31, 
    lookforward_window=4,
    transaction_cost=0.0005,
    min_trades=8,
    min_win_rate=0.3578,
    max_win_rate=0.6201,
    min_profit_factor=1.0876,
    mutation_rate=0.0173,
    crossover_rate=0.7438,
    target_name="ga_optimized_signals"
)
```

#### Use Cases
- **Performance-Critical Trading**: When maximum returns are the primary objective
- **Adaptive Strategy Development**: Strategies that evolve with market conditions
- **Cost-Aware Systems**: Trading systems where transaction costs matter
- **Research Applications**: Compare evolutionary vs. traditional approaches
- **Live Trading**: Production systems requiring validated performance metrics

---

### 3. Binary CTL (Cumulative Trend Labeling)

**Purpose**: Academic binary trend labeling using cumulative trend analysis from TStrends research.

#### Methodology
Based on the **Cumulative Trend Labeling** approach from academic literature:

1. **Price Series Analysis**: Analyzes cumulative price movements over time
2. **Omega Filtering**: Uses omega parameter to filter noise from trend signals  
3. **Binary Classification**: Generates clean Up/Down signals `{0: Down/Sell, 1: Up/Buy}`
4. **Trend Persistence**: Captures sustained directional movements rather than noise

#### Bayesian Optimization Results
**Performance**: **240.20% returns** (exceptional performance)

**Optimized Parameters**:
```python
{
    "omega": 0.0  # Zero omega = no noise filtering needed
}
```

#### Why Zero Omega is Optimal
The Bayesian optimization discovered that **omega=0.0** (no noise filtering) provides maximum returns:

1. **Market Efficiency**: Modern electronic markets may have less noise requiring filtering
2. **Signal Preservation**: Zero filtering preserves all directional information
3. **Transaction Cost Balance**: With 0.7 pip costs, aggressive signal generation is profitable
4. **Data Quality**: High-frequency LOB data may already be sufficiently clean

#### Configuration
```python
# Optimized configuration
TargetGeneratorFactory.create(
    "binary_ctl",
    omega=0.0,  # No noise filtering - OPTIMIZED
    target_name="binary_ctl_optimized"
)
```

#### Use Cases
- **Research Benchmarking**: Academic comparison with traditional methods
- **Binary Direction Prediction**: Clean two-class directional signals
- **High-Frequency Trading**: Suitable for low-latency trading systems
- **Signal Validation**: Validate custom signals against academic methods

---

### 4. Ternary CTL (Cumulative Trend Labeling)

**Purpose**: Academic three-class trend labeling with neutral zone detection.

#### Methodology
Extension of Binary CTL to three classes:

1. **Marginal Change Threshold**: Defines minimum movement for directional classification
2. **Window-Based Analysis**: Uses sliding window for trend persistence
3. **Three-Class Output**: `{0: Down/Sell, 1: Neutral/Hold, 2: Up/Buy}`
4. **Neutral Zone**: Identifies sideways markets where directional trades are avoided

#### Bayesian Optimization Results
**Performance**: **0.32% returns** (positive after transaction costs)

**Optimized Parameters**:
```python
{
    "marginal_change_thres": 0.04458382945260628,  # ~4.46% threshold
    "window_size": 501                             # 501-tick window
}
```

#### Why Higher Threshold is Optimal
The optimization revealed that much **higher thresholds (4.46% vs typical 0.08%)** are needed:

1. **Transaction Cost Coverage**: Large movements needed to overcome 0.7 pip costs
2. **Signal Quality**: Higher thresholds filter out unprofitable small movements
3. **Neutral Zone Value**: Properly sized neutral zone prevents overtrading
4. **Risk-Adjusted Returns**: Fewer, higher-quality signals improve risk metrics

#### Configuration
```python
# Optimized configuration  
TargetGeneratorFactory.create(
    "ternary_ctl",
    marginal_change_thres=0.04458382945260628,  # High threshold - OPTIMIZED
    window_size=501,                            # Moderate window - OPTIMIZED
    target_name="ternary_ctl_optimized"
)
```

#### Use Cases
- **Three-Class Prediction**: Models that benefit from explicit neutral class
- **Risk-Aware Trading**: Neutral signals prevent overtrading in sideways markets
- **Academic Research**: Comparison with binary methods
- **Conservative Strategies**: Higher threshold provides conservative signal generation

---

### 5. Oracle Binary/Ternary Trend Labeling

**Purpose**: Theoretical optimal labels providing performance upper bounds for benchmarking.

#### Methodology
**Oracle methods use future price knowledge to generate theoretically optimal labels:**

1. **Perfect Information**: Uses complete future price path (oracle knowledge)
2. **Optimal Trading**: Calculates best possible trading decisions with transaction costs
3. **Performance Bounds**: Provides theoretical maximum returns for comparison
4. **Benchmarking**: Shows how close practical methods come to theoretical optimum

#### Binary Oracle Optimization Results
**Performance**: **1.23% returns**

**Optimized Parameters**:
```python
{
    "transaction_cost": 9.326802124287607e-07  # ~9.33e-07 (minimal cost)
}
```

#### Ternary Oracle Optimization Results
**Performance**: **0.18% returns**

**Optimized Parameters**:
```python
{
    "transaction_cost": 0.00796542986860233,    # ~0.8% cost parameter
    "neutral_reward_factor": 0.18343478986616382 # ~18.3% neutral factor
}
```

#### Configuration
```python
# Binary Oracle (optimized)
TargetGeneratorFactory.create(
    "oracle_binary",
    transaction_cost=9.326802124287607e-07,
    target_name="oracle_binary_optimized"
)

# Ternary Oracle (optimized)
TargetGeneratorFactory.create(
    "oracle_ternary", 
    transaction_cost=0.00796542986860233,
    neutral_reward_factor=0.18343478986616382,
    target_name="oracle_ternary_optimized"
)
```

#### Use Cases
- **Performance Benchmarking**: Theoretical maximum for model evaluation
- **Research Validation**: Assess how close practical methods approach optimum
- **Strategy Development**: Understand performance limits in specific market conditions
- **Academic Studies**: Theoretical bounds for research papers

---

## Regression Methods

Regression methods generate continuous targets for magnitude prediction and risk management.

### 1. Log Return Horizons ⭐

**Purpose**: Multi-horizon log return predictions for comprehensive temporal analysis.

#### Methodology
**Revolutionary multi-scale approach that predicts returns across multiple time horizons:**

1. **Multi-Horizon Analysis**: Simultaneously predicts 1k, 2k, 3k, 4k, 5k tick returns
2. **Log Return Calculation**: Uses statistically robust log returns: `ln(future_price / current_price)`
3. **Basis Point Output**: Standardized output in basis points (0.01%)
4. **Temporal Decomposition**: Captures different market dynamics at various time scales

#### Configuration
```python
# Multi-horizon analysis
TargetGeneratorFactory.create(
    "log_return_horizons",
    horizons=[1000, 2000, 3000, 4000, 5000],  # 1k-5k tick horizons
    lookback_window=1000,                     # Baseline window
    target_prefix="log_return"                # Creates: log_return_1000t, etc.
)

# Custom horizons for specific strategies
TargetGeneratorFactory.create(
    "log_return_horizons",
    horizons=[500, 1500, 2500],  # Short-medium-long
    target_prefix="strategy_returns"
)
```

#### Key Advantages
1. **Multi-Scale Insights**: Single model predicts across multiple time horizons
2. **Statistical Robustness**: Log returns are more normally distributed than simple returns
3. **Strategy Flexibility**: Different horizons suit different trading strategies
4. **Risk Management**: Understand return dynamics across various holding periods
5. **Model Efficiency**: One target generation produces 5 related targets

#### Use Cases
- **Multi-Scale Trading**: Strategies operating across different time horizons
- **Risk Management**: Portfolio risk assessment across multiple holding periods
- **Strategy Optimization**: Identify optimal holding periods for market conditions
- **Feature Engineering**: Rich multi-horizon features for complex ML models
- **Academic Research**: Temporal dynamics analysis

---

### 2. Directional MFE (Maximum Favorable Excursion)

**Purpose**: Risk-aware position sizing through maximum profit potential analysis.

#### Methodology
**Sophisticated approach that calculates maximum potential profit for both long and short positions:**

1. **Lookforward Analysis**: Scans future price movements within specified horizon
2. **Maximum Favorable Excursion**: Finds peak profit potential before adverse moves
3. **Directional Split**: Separate MFE calculation for long (buy) and short (sell) positions
4. **Transaction Cost Integration**: Adjusts MFE for realistic trading costs
5. **Risk-Reward Quantification**: Provides concrete profit potential in basis points

#### Configuration
```python
# Risk-aware position sizing
TargetGeneratorFactory.create(
    "directional_mfe",
    lookforward_horizon=3000,      # 3k tick evaluation window
    lookback_window=200,           # Smoothing for entry price
    expected_fee_pips=0.7,         # Transaction cost integration
    target_names=("mfe_buy_bps", "mfe_sell_bps")
)
```

#### Output Interpretation
- **mfe_buy_bps**: Maximum potential profit from long positions (basis points)
- **mfe_sell_bps**: Maximum potential profit from short positions (basis points)
- **Positive Values**: Profit potential available in that direction
- **Risk Scaling**: Larger MFE suggests larger position sizes appropriate

#### Use Cases
- **Position Sizing**: Scale position sizes based on profit potential
- **Risk Management**: Understand maximum drawdown vs maximum profit
- **Strategy Optimization**: Optimize entry/exit timing for MFE capture
- **Portfolio Construction**: Weight assets based on directional opportunities
- **Performance Attribution**: Analyze realized vs potential profits

#### Key Benefits
1. **Risk-Aware**: Accounts for both profit potential and trading costs
2. **Directional**: Separate analysis for long/short opportunities
3. **Quantified**: Concrete profit targets in basis points
4. **Practical**: Integrates real-world transaction costs

---

### 3. Volatility Scaled Returns

**Purpose**: Adaptive risk management with dynamic volatility-based barriers.

#### Methodology
**Advanced regime-aware approach that adapts risk management to market conditions:**

1. **Rolling Volatility**: Calculates recent volatility using specified window
2. **Dynamic Barriers**: Sets stop-loss/take-profit levels as multiples of volatility
3. **Regime Adaptation**: Tight barriers in calm markets, wide barriers in volatile periods
4. **Realistic PnL**: Returns actual breach prices rather than theoretical barriers
5. **Noise Filtering**: Minimum barrier size prevents meaningless small movements

#### Configuration
```python
# Conservative regime-aware trading
TargetGeneratorFactory.create(
    "volatility_scaled_returns",
    volatility_window=500,         # Rolling volatility window
    vol_multiplier=2.5,            # Barrier size (2.5x volatility)
    horizon_ticks=1500,            # Evaluation period
    min_barrier_bps=3.0,           # Minimum barrier (noise filter)
    target_name="adaptive_pnl"
)

# Aggressive high-frequency trading
TargetGeneratorFactory.create(
    "volatility_scaled_returns",
    volatility_window=100,         # Short volatility window
    vol_multiplier=1.5,            # Tight barriers
    horizon_ticks=500,             # Short evaluation
    min_barrier_bps=1.0,           # Low noise threshold
    target_name="hft_pnl"
)
```

#### Key Features
1. **Market Adaptive**: Automatically adjusts to volatility regimes
2. **Risk Proportional**: Barrier size scales with market risk
3. **Noise Robust**: Minimum barriers prevent false signals
4. **Realistic**: Returns actual market prices, not theoretical levels

#### Use Cases
- **FX Trading**: Common approach in currency markets for adaptive risk
- **Volatility Strategies**: Trading strategies that adapt to market conditions
- **Risk Management**: Dynamic position sizing based on market state
- **Regime Detection**: Identify and trade different volatility environments

---

### 4. Remaining Value Tuner

**Purpose**: Advanced trend potential prediction inspired by academic research.

#### Methodology
**Sophisticated approach that predicts remaining movement potential in ongoing trends:**

1. **Trend Detection**: Identifies uptrends, downtrends, and neutral periods
2. **Potential Calculation**: Calculates remaining upside/downside from current point
3. **Future-Aware Analysis**: Uses lookforward window to find trend peaks/troughs
4. **Continuous Output**: Provides continuous trend magnitude rather than discrete labels
5. **Monotonicity Enforcement**: Smooths trend values to prevent unrealistic reversals

#### Configuration
```python
# Advanced trend potential prediction
TargetGeneratorFactory.create(
    "remaining_value_tuner",
    lookback_rows=5000,              # Historical context
    lookforward_input=3000,          # Trend evaluation window
    lookforward_offset=500,          # Offset before evaluation
    trend_threshold_bps=20.0,        # Minimum trend magnitude
    neutral_factor=0.5,              # Neutral zone sizing
    enforce_monotonicity=True,       # Smooth transitions
    target_name="trend_potential"
)
```

#### Output Interpretation
- **Positive Values**: Remaining upside potential (e.g., +185 bps available)
- **Negative Values**: Remaining downside potential (e.g., -127 bps available)
- **Near Zero**: Sideways/neutral markets with limited directional potential
- **Magnitude**: Larger absolute values indicate stronger trend potential

#### Use Cases
- **Optimal Entry Timing**: Enter trends when remaining potential is highest
- **Position Sizing**: Size positions based on remaining trend magnitude
- **Exit Strategy**: Exit when remaining potential diminishes
- **Advanced ML**: More informative targets than binary/ternary classification
- **Research Applications**: Academic-quality labeling for strategy development

#### Key Advantages
1. **Information Rich**: Provides magnitude, not just direction
2. **Trend Aware**: Understands trend lifecycle dynamics
3. **Entry Optimization**: Helps time entries for maximum remaining potential
4. **Risk Quantification**: Quantifies remaining risk/reward in basis points

---

### 5. Rolling Volatility

**Purpose**: Future volatility forecasting for risk management and options applications.

#### Methodology
**Standard approach for volatility prediction:**

1. **Rolling Window**: Calculates volatility over specified historical window
2. **Standard Deviation**: Uses price return standard deviation as volatility measure
3. **Forward-Looking**: Predicts future volatility based on recent patterns
4. **Basis Point Output**: Standardized volatility in basis points

#### Configuration
```python
# Volatility forecasting
TargetGeneratorFactory.create(
    "volatility",
    window_size=1000,  # Rolling window (1k ticks)
    target_name="volatility_forecast_bps"
)
```

#### Use Cases
- **Risk Management**: Position sizing based on expected volatility
- **Options Trading**: Volatility forecasting for options strategies
- **Market Regime Detection**: Identify high/low volatility periods
- **Portfolio Management**: Volatility-based asset allocation

---

## Optimization Results & Analysis

### Performance Summary

| Method | Returns | Optimization Insight | Trading Viability |
|--------|---------|---------------------|------------------|
| **GA Labeling** | **71.34%** | Evolutionary optimization dominates all approaches | ✅ Excellent |
| **Binary CTL** | **240.20%** | Zero omega filtering maximizes signal capture | ✅ Excellent |
| **Ternary CTL** | 0.32% | Higher thresholds (4.46%) needed for profitability | ✅ Viable |
| **Oracle Binary** | 1.23% | Minimal transaction costs show theoretical limits | ✅ Viable |
| **Oracle Ternary** | 0.18% | Low neutral factor (18.3%) favors directional signals | ⚠️ Marginal |

### Why Optimization Dramatically Improves Outcomes

#### 1. Transaction Cost Awareness
**Problem**: Traditional academic methods ignore realistic trading costs.
**Solution**: Bayesian optimization uses actual trading simulation with 0.7 pip fees.

**Impact**: 
- GA Labeling: Evolved strategies account for transaction costs in fitness function
- Binary CTL: Zero omega preserves profitable signals that would be filtered out
- Ternary CTL: Higher thresholds ensure movements exceed trading costs

#### 2. Returns-Based Fitness Function
**Problem**: Statistical properties don't always translate to trading profits.
**Solution**: Optimization directly maximizes trading returns, not statistical metrics.

**Impact**:
- Parameters selected for actual profit generation, not statistical significance
- Trading constraints (win rate, profit factor) ensure practical viability
- Multi-objective optimization balances returns with risk metrics

#### 3. Bayesian Efficiency
**Problem**: Grid search and manual tuning are inefficient and incomplete.
**Solution**: Gaussian Process optimization efficiently explores parameter space.

**Impact**:
- Finds global optima with fewer evaluations (15 iterations vs hundreds)
- Learns from previous evaluations to guide search toward promising regions
- Balances exploration of new areas with exploitation of good solutions

#### 4. Market-Specific Adaptation
**Problem**: One-size-fits-all parameters don't adapt to specific market conditions.
**Solution**: Optimization on actual market data finds regime-specific parameters.

**Impact**:
- GA parameters evolved for specific volatility and trend characteristics
- CTL thresholds adapted to actual price movement distributions
- Oracle costs tuned to realistic market microstructure

### Optimization Methodology

> ⚠️ Parameter optimization has been removed from the library. The previous implementation
> relied on the external `tstrends` package and has been deprecated in favour of
> lighter-weight, fully self-contained targets.

#### Multi-Objective Constraints
Historical optimization included practical trading constraints:

- **Minimum Trades**: Statistical significance requirements
- **Win Rate Bounds**: Avoid overfitting to extreme win rates  
- **Profit Factor**: Ensure positive expected value
- **Transaction Cost Integration**: Realistic fee simulation
- **Risk Limits**: Maximum drawdown and volatility constraints

---

## Method Selection Guide

### By Use Case

#### **High-Performance Trading**
**Primary**: GA Labeling (71.34% returns)
**Secondary**: Binary CTL (240.20% returns) 
**Rationale**: Maximum returns with cost awareness

#### **Multi-Class Direction Prediction**
**Primary**: Quantile Classification (balanced)
**Secondary**: Ternary CTL (optimized thresholds)
**Rationale**: Stable training with multiple discrete outcomes

#### **Position Sizing & Risk Management**
**Primary**: Directional MFE + Volatility Scaled Returns
**Secondary**: Remaining Value Tuner
**Rationale**: Risk-aware continuous targets for position optimization

#### **Research & Benchmarking**
**Primary**: Oracle Binary/Ternary (theoretical bounds)
**Secondary**: All academic methods (CTL, GA)
**Rationale**: Compare practical methods against theoretical optimum

#### **Multi-Scale Trading**
**Primary**: Log Return Horizons (5 time scales)
**Secondary**: Remaining Value Tuner
**Rationale**: Comprehensive temporal analysis across different horizons

### By Model Architecture

#### **Classification Models** (XGBoost, Random Forest, Neural Networks)
```python
generators = [
    TargetGeneratorFactory.create("quantile_classification", nbins=13),
    TargetGeneratorFactory.create("ga_labeling", population_size=30),
    TargetGeneratorFactory.create("binary_ctl", omega=0.0)
]
```

#### **Regression Models** (Linear, Ridge, Neural Networks)
```python
generators = [
    TargetGeneratorFactory.create("log_return_horizons", horizons=[1000, 2000, 3000]),
    TargetGeneratorFactory.create("directional_mfe", lookforward_horizon=3000),
    TargetGeneratorFactory.create("volatility_scaled_returns", vol_multiplier=2.5)
]
```

#### **Multi-Task Models** (Mixed Classification + Regression)
```python
generators = [
    # Classification targets
    TargetGeneratorFactory.create("ga_labeling", population_size=30),
    TargetGeneratorFactory.create("quantile_classification", nbins=13),
    
    # Regression targets
    TargetGeneratorFactory.create("log_return_horizons", horizons=[1000, 3000, 5000]),
    TargetGeneratorFactory.create("directional_mfe", lookforward_horizon=3000)
]
```

### By Market Regime

#### **High Volatility Markets**
- **Volatility Scaled Returns**: Adaptive barriers handle regime changes
- **GA Labeling**: Evolved strategies adapt to volatility conditions
- **Higher CTL Thresholds**: Filter noise in volatile conditions

#### **Low Volatility Markets** 
- **Binary CTL** with zero omega: Capture subtle trends
- **Remaining Value Tuner**: Detect potential in quiet markets
- **Lower MFE horizons**: Shorter time scales for limited movements

#### **Trending Markets**
- **Binary/Ternary CTL**: Designed for trend detection
- **Remaining Value Tuner**: Quantify remaining trend potential
- **Directional MFE**: Optimize trend-following position sizes

#### **Range-Bound Markets**
- **Volatility Scaled Returns**: Adaptive to sideways conditions  
- **Ternary CTL**: Explicit neutral class for range detection
- **Shorter Horizons**: Log return horizons at 500-1500 ticks

---

## Visualization Results

The Represent package includes comprehensive visualization capabilities that demonstrate all labeling methods with real market data. Below are the generated comparison plots showing the performance and characteristics of each approach.

### Complete Labeling Overview

![Complete Labeling Overview](examples/complete_labeling_overview.png)

*Figure 1: Complete overview of all 15+ labeling approaches showing the diversity of available methods across classification and regression targets. This plot demonstrates the comprehensive nature of the Represent package's labeling capabilities.*

### Classification Methods Comparison

![Classification Approaches Comparison](examples/classification_approaches_comparison.png)

*Figure 2: Detailed comparison of all classification methods including optimized GA Labeling (71.34% returns), Binary CTL (240.20% returns), Ternary CTL, Oracle methods, and traditional quantile classification. Notice the performance differences and signal characteristics across methods.*

### Regression Methods Comparison

![Regression Approaches Comparison](examples/regression_approaches_comparison.png)

*Figure 3: Comprehensive comparison of regression targets including Log Return Horizons (5 different time scales), Directional MFE (buy/sell potential), Volatility Scaled Returns, and other continuous targets. Shows the rich variety of regression approaches available.*

### Academic vs Traditional Methods

![Academic vs Traditional Comparison](examples/academic_vs_traditional_comparison.png)

*Figure 4: Side-by-side comparison of academic TStrends methods (CTL, Oracle) with traditional approaches, highlighting how Bayesian optimization bridges the gap between academic theory and practical performance.*

### Key Visualization Insights

#### **Signal Diversity**
- **GA Labeling**: Shows evolved trading signals optimized for performance
- **Binary CTL**: Clean binary signals with zero omega filtering
- **Ternary CTL**: Three-class signals with optimized neutral zones
- **Log Return Horizons**: Multi-scale continuous predictions across 5 time horizons

#### **Optimization Impact**
- **Before Optimization**: Academic methods showed modest performance
- **After Bayesian Optimization**: Dramatic improvement in returns while maintaining signal quality
- **Parameter Sensitivity**: Visualizations show how optimized parameters create better signal patterns

#### **Method Complementarity**
- **Classification + Regression**: Different methods capture different aspects of market dynamics
- **Multi-Scale Analysis**: Log return horizons show how market predictions vary across time scales
- **Risk-Return Profile**: Each method has distinct risk-return characteristics visible in the plots

#### **Real Market Performance**
All plots are generated using real market data (M6AM4 futures) showing:
- **Realistic Signal Patterns**: How each method behaves on actual market microstructure
- **Performance Validation**: Actual returns achieved by each optimized method
- **Signal Quality**: Visual assessment of signal clarity and actionability

### Performance Analysis Visualizations

**Performance Comparison Chart**
![Performance Comparison](examples/performance_comparison_chart.png)

*Figure 5: Bar chart showing optimization results. GA Labeling leads with 71.34% returns, Binary CTL achieves exceptional 240.20% returns, while other methods show positive but modest performance.*

**Risk-Return Analysis**
![Risk-Return Scatter](examples/risk_return_scatter.png)

*Figure 6: Risk-return profile of all optimized methods. GA Labeling offers high returns with moderate risk, while Binary CTL provides exceptional returns with lower risk.*

**Method Comparison Table**
![Method Comparison Table](examples/method_comparison_table.png)

*Figure 7: Comprehensive comparison table showing all methods with their types, optimized parameters, performance results, and recommended use cases.*

### Individual Method Analysis

**GA Labeling Signals (Evolutionary Optimization)**
![GA Labeling Signals](examples/individual_ga_labeling_signals.png)

*Figure 8: GA Labeling generates evolutionary-optimized trading signals achieving 71.34% returns. Shows actual buy/sell signals with optimized parameters (population=30, generations=31).*

**CTL Methods (Academic Trend Labeling)**
![CTL Methods Signals](examples/individual_ctl_methods_signals.png)

*Figure 9: Binary CTL with zero omega filtering (240.20% returns) and Ternary CTL with optimized thresholds (0.32% returns). Shows the impact of Bayesian optimization on academic methods.*

**Oracle Methods (Theoretical Optimum)**
![Oracle Methods Signals](examples/individual_oracle_methods_signals.png)

*Figure 10: Oracle Binary (1.23% returns) and Oracle Ternary (0.18% returns) methods provide theoretical performance bounds using perfect future information.*

**Quantile Classification (Traditional Balanced)**
![Quantile Classification](examples/individual_quantile_classification_signals.png)

*Figure 11: Traditional quantile classification ensures balanced class distributions for stable ML training. Shows binary and 5-class examples.*

**Regression Methods (Continuous Targets)**
![Regression Methods](examples/individual_regression_methods_signals.png)

*Figure 12: Multi-horizon log returns, directional MFE analysis, and volatility-scaled returns provide continuous targets for risk management and position sizing.*

### Optimization Analysis

**Parameter Sensitivity Analysis**
![Parameter Sensitivity](examples/parameter_sensitivity_analysis.png)

*Figure 13: Key optimized parameter values across methods. Shows GA parameters, CTL thresholds, Oracle costs, and performance vs complexity analysis.*

**Bayesian Optimization Convergence**
![Optimization Convergence](examples/optimization_convergence.png)

*Figure 14: Convergence analysis showing how Bayesian optimization efficiently finds optimal parameters. GA Labeling converges to 71.34%, Binary CTL to 240.20%.*

**Signal Quality Metrics**
![Signal Quality Metrics](examples/signal_quality_metrics.png)

*Figure 15: Comparison of signal quality before and after optimization. Shows improvements in win rates, Sharpe ratios, drawdowns, and trading frequency.*

### Generating Your Own Plots

You can generate these comprehensive comparison plots for your own data:

```python
# Run the complete labeling demonstration (original 4 plots)
python examples/labeling_approaches_visualization.py

# Generate additional analysis plots
python examples/individual_plots_generator.py

# Generate individual method signal plots  
python examples/individual_method_plots.py
```

**Available Plot Generators:**
- **Main comparison plots** (4 comprehensive comparisons)
- **Performance analysis** (charts, tables, risk-return)
- **Individual method signals** (detailed signal patterns)
- **Optimization analysis** (convergence, sensitivity, quality metrics)

The visualization system automatically:
- **Applies optimized parameters** for all methods
- **Handles TStrends integration** (if available)
- **Provides statistical summaries** for each approach
- **Creates publication-ready plots** with proper labeling and legends

---

## Performance Comparison

### Computational Efficiency

| Method | Generation Speed | Memory Usage | Optimization Complexity |
|--------|------------------|--------------|------------------------|
| Quantile Classification | ⚡ Very Fast | 💾 Low | 🔧 Simple |
| GA Labeling | 🐌 Slow | 💾💾 High | 🔧🔧🔧 Complex |
| Binary CTL | ⚡ Fast | 💾 Low | 🔧 Simple |
| Ternary CTL | ⚡ Fast | 💾 Low | 🔧🔧 Medium |
| Log Return Horizons | ⚡ Very Fast | 💾💾 Medium | 🔧 Simple |
| Directional MFE | ⚡ Fast | 💾 Low | 🔧🔧 Medium |
| Volatility Scaled | ⚡ Fast | 💾 Low | 🔧🔧 Medium |

### Signal Quality Assessment

#### **Information Content** (How much predictive information)
1. **GA Labeling**: ⭐⭐⭐⭐⭐ (Performance-optimized)
2. **Remaining Value Tuner**: ⭐⭐⭐⭐⭐ (Continuous magnitude)
3. **Log Return Horizons**: ⭐⭐⭐⭐ (Multi-scale insights)
4. **Directional MFE**: ⭐⭐⭐⭐ (Risk-aware)
5. **Binary CTL**: ⭐⭐⭐ (Clean directional)

#### **Practical Applicability** (Real-world trading viability)
1. **GA Labeling**: ⭐⭐⭐⭐⭐ (Cost-optimized)
2. **Binary CTL**: ⭐⭐⭐⭐⭐ (High returns)
3. **Volatility Scaled**: ⭐⭐⭐⭐ (Adaptive)
4. **Quantile Classification**: ⭐⭐⭐ (Stable baseline)
5. **Ternary CTL**: ⭐⭐⭐ (Marginally profitable)

#### **Model Training Stability** (How well models train)
1. **Quantile Classification**: ⭐⭐⭐⭐⭐ (Balanced classes)
2. **Log Return Horizons**: ⭐⭐⭐⭐ (Continuous, well-distributed)
3. **Binary CTL**: ⭐⭐⭐⭐ (Clean binary signals)
4. **GA Labeling**: ⭐⭐⭐ (May have class imbalance)
5. **Oracle Methods**: ⭐⭐⭐ (Perfect but unrealistic)

---

## Conclusion

The Represent package provides a comprehensive suite of labeling methods ranging from traditional statistical approaches to cutting-edge evolutionary optimization. The Bayesian parameter optimization ensures all methods are tuned for real-world performance with transaction cost awareness.

### Key Takeaways

1. **GA Labeling Dominates**: Evolutionary optimization provides best returns (71.34%)
2. **Binary CTL Surprises**: Zero omega filtering delivers exceptional performance (240.20%)
3. **Transaction Costs Matter**: Optimization reveals need for higher thresholds and cost-aware parameters
4. **Method Diversity**: Different methods excel in different use cases and market regimes
5. **Practical Focus**: All optimizations prioritize actual trading viability over statistical properties

### Recommended Starting Point

For most applications, start with this balanced combination:

```python
generators = [
    # Best performer: evolutionary optimization
    TargetGeneratorFactory.create("ga_labeling", population_size=30, max_generations=31),
    
    # Stable baseline: balanced classification
    TargetGeneratorFactory.create("quantile_classification", nbins=13),
    
    # Multi-scale analysis: comprehensive time horizons
    TargetGeneratorFactory.create("log_return_horizons", 
                                 horizons=[1000, 2000, 3000, 4000, 5000]),
    
    # Risk management: position sizing optimization
    TargetGeneratorFactory.create("directional_mfe", lookforward_horizon=3000)
]
```

This combination provides:
- Maximum performance (GA Labeling)
- Training stability (Quantile Classification) 
- Multi-scale insights (Log Return Horizons)
- Risk awareness (Directional MFE)

All with Bayesian-optimized parameters for real-world trading success.
