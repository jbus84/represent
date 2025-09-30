#!/usr/bin/env python3
"""
Recommended Labeling Methods for FX Microstructure Prediction

This example demonstrates the BETTER alternatives to triple barrier method
for predicting FX price movements at tick level.

Based on diagnostic findings showing triple barrier creates unpredictable
balanced classes (33/32/34) with 642-tick horizon that exceeds FX
predictability range.
"""

import sys
from pathlib import Path

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from represent.target_generators.factory import TargetGeneratorFactory

# ============================================================================
# OPTION 1: MULTI-HORIZON REGRESSION (RECOMMENDED)
# ============================================================================
print("=" * 80)
print("OPTION 1: LOG RETURN HORIZONS (Multi-Scale Regression)")
print("=" * 80)

# Why this is better than triple barrier:
# 1. Continuous targets (more information than 3 classes)
# 2. Multiple time scales (captures different market dynamics)
# 3. No arbitrary barriers or timeout class
# 4. Shorter horizons within FX predictability range (50-500 ticks)
# 5. Can threshold predictions for trading decisions

generator_regression = TargetGeneratorFactory.create(
    "log_return_horizons",
    horizons=[50, 100, 250, 500, 1000],  # Multiple prediction windows
    include_mean_returns=True,  # NEW: Also calculate mean returns (default)
    # Output: 10 targets per sample (5 endpoint + 5 mean)
)

print("\nGenerator Info:")
info = generator_regression.get_target_info()
print(f"  Type: {info['target_type']}")
print(f"  Targets: {info['target_names']}")
print(f"  Description: {info['description']}")

print("\nAdvantages over Triple Barrier:")
print("  ✓ Preserves all price movement information (not just barrier hits)")
print("  ✓ Multiple horizons: 50 ticks (10s) to 1000 ticks (3min)")
print("  ✓ No 32% timeout class wasting data")
print("  ✓ Continuous predictions enable confidence thresholds")
print("  ✓ Better for gradient-based learning (smooth loss landscape)")
print("  ✓ NEW: Mean returns capture path info (50-100x smoother signal!)")

print("\nExpected Performance:")
print("  • R² > 0.001: Statistically significant")
print("  • R² > 0.01: Potentially tradeable")
print("  • R² > 0.05: Excellent for FX microstructure")

print("\nUsage with ML:")
print("""
  # Train regression model (e.g., LightGBM, XGBoost, Neural Net)
  model.fit(X_train, y_multi_horizon)

  # Predict returns at multiple horizons
  predictions = model.predict(X_test)  # Shape: (N, 10) - 5 endpoint + 5 mean

  # Trading decisions using BOTH endpoint and mean returns
  # Endpoint: Final P&L prediction
  # Mean: Trend/drift direction
  endpoint_250 = predictions[:, 2]  # log_return_250t
  mean_250 = predictions[:, 7]      # log_return_mean_250t

  # Long: Positive endpoint AND positive mean (trending up)
  long_signal = (endpoint_250 > 0.0002) & (mean_250 > 0)
  # Short: Negative endpoint AND negative mean (trending down)
  short_signal = (endpoint_250 < -0.0002) & (mean_250 < 0)
""")

# ============================================================================
# OPTION 2: DIRECTIONAL MFE (Trading-Focused Regression)
# ============================================================================
print("\n" + "=" * 80)
print("OPTION 2: DIRECTIONAL MFE (Maximum Favorable Excursion)")
print("=" * 80)

# Why this is better than triple barrier:
# 1. Measures actual profit potential in both directions
# 2. Trading-aware (considers directional strategies)
# 3. Captures peak opportunity, not just first barrier hit
# 4. Useful for position sizing and risk management

generator_mfe = TargetGeneratorFactory.create(
    "directional_mfe",
    lookforward_horizon=500,  # Shorter than triple barrier's 1063
    # Output: 2 targets - buy_mfe and sell_mfe
)

print("\nGenerator Info:")
info = generator_mfe.get_target_info()
print(f"  Type: {info['target_type']}")
print(f"  Targets: {info['target_names']}")

print("\nWhat MFE Measures:")
print("  • buy_mfe: Best possible profit from long position (in bps)")
print("  • sell_mfe: Best possible profit from short position (in bps)")
print("  → Models learn: 'How much could I make if I timed exit perfectly?'")

print("\nAdvantages over Triple Barrier:")
print("  ✓ Captures peak profit potential, not arbitrary barrier")
print("  ✓ Two targets: learn both long and short opportunities")
print("  ✓ Useful for position sizing (higher MFE = bigger position)")
print("  ✓ Natural for ensemble: combine with other methods")

print("\nUsage with ML:")
print("""
  # Predict profit potential in both directions
  buy_mfe_pred, sell_mfe_pred = model.predict(X_test).T

  # Trade based on relative opportunity
  long_signal = (buy_mfe_pred > sell_mfe_pred) & (buy_mfe_pred > threshold)
  short_signal = (sell_mfe_pred > buy_mfe_pred) & (sell_mfe_pred > threshold)

  # Position sizing based on confidence
  position_size = np.clip(buy_mfe_pred / max_position_mfe, 0, 1)
""")

# ============================================================================
# OPTION 3: ORACLE BINARY (Optimal Classification)
# ============================================================================
print("\n" + "=" * 80)
print("OPTION 3: ORACLE BINARY (Optimal Trading Signals)")
print("=" * 80)

# Why this is better than triple barrier:
# 1. Labels based on ACTUAL profitability (not arbitrary barriers)
# 2. Includes transaction costs in label generation
# 3. Naturally imbalanced (most samples = no trade)
# 4. Represents theoretical best performance

generator_oracle = TargetGeneratorFactory.create(
    "oracle_binary",
    transaction_cost=0.00007,  # 0.7 pips for FX
    # Output: Binary {0, 1} where 1 = profitable trade opportunity
    # Note: Oracle methods use entire lookforward from data
)

print("\nGenerator Info:")
info = generator_oracle.get_target_info()
print(f"  Type: {info['target_type']}")
print(f"  Parameters: {info['parameters']}")

print("\nHow Oracle Labels Work:")
print("  • Looks ahead 250 ticks")
print("  • Finds best possible entry/exit")
print("  • Label = 1 if trade would be profitable after transaction costs")
print("  • Label = 0 otherwise")
print("  → Represents upper bound on performance")

print("\nAdvantages over Triple Barrier:")
print("  ✓ Labels align with trading objective (profit)")
print("  ✓ Transaction cost aware (realistic)")
print("  ✓ Naturally imbalanced (~10-30% positive class)")
print("  ✓ Shorter horizon = more predictable")
print("  ✓ Binary classification simpler than 3-class")

print("\nExpected Class Distribution:")
print("  • Positive class: 10-30% (tradeable opportunities)")
print("  • Negative class: 70-90% (no clear edge)")
print("  → Much more realistic than 33/33/33 balance!")

# ============================================================================
# OPTION 4: QUANTILE CLASSIFICATION (Better Balanced Classes)
# ============================================================================
print("\n" + "=" * 80)
print("OPTION 4: QUANTILE CLASSIFICATION (Informative Bins)")
print("=" * 80)

# Why this is better than triple barrier:
# 1. Balanced by construction (equal samples per class)
# 2. But preserves RANKING information (unlike triple barrier)
# 3. No timeout class
# 4. Simpler interpretation

generator_quantile = TargetGeneratorFactory.create(
    "quantile_classification",
    nbins=5,  # Top 20%, next 20%, ..., bottom 20%
    # Output: Classes {0, 1, 2, 3, 4} representing return quintiles
    # Note: Uses default lookforward parameters from generator
)

print("\nGenerator Info:")
info = generator_quantile.get_target_info()
print(f"  Type: {info['target_type']}")
print(f"  Bins: {info['parameters']['nbins']}")

print("\nHow Quantile Labels Work:")
print("  • Compute 250-tick forward return for each sample")
print("  • Bin into quintiles (20% per bin)")
print("  • Class 0: Bottom 20% (worst returns)")
print("  • Class 2: Middle 20% (neutral)")
print("  • Class 4: Top 20% (best returns)")

print("\nAdvantages over Triple Barrier:")
print("  ✓ Balanced classes (20% each) but preserves ranking")
print("  ✓ No arbitrary barriers")
print("  ✓ No timeout class wasting 32% of data")
print("  ✓ Clear interpretation: relative strength")
print("  ✓ Can collapse to binary: {0,1,2} vs {3,4}")

print("\nUsage with ML:")
print("""
  # Train 5-class classifier
  model.fit(X_train, y_quintile)

  # Predict return quintile
  pred_class = model.predict(X_test)

  # Trading decisions
  long_signal = pred_class >= 3   # Top 40%
  short_signal = pred_class <= 1  # Bottom 40%
  neutral = pred_class == 2       # Middle 20%
""")

# ============================================================================
# COMPLETE EXAMPLE: BUILD DATASET WITH RECOMMENDED METHODS
# ============================================================================
print("\n" + "=" * 80)
print("COMPLETE EXAMPLE: Multi-Target Dataset")
print("=" * 80)

print("\nBuilding dataset with ALL recommended methods...")

# Create all recommended generators
generators = [
    # 1. Multi-horizon regression (PRIMARY RECOMMENDATION)
    TargetGeneratorFactory.create(
        "log_return_horizons",
        horizons=[50, 100, 250, 500, 1000],
    ),

    # 2. Directional MFE (profit potential)
    TargetGeneratorFactory.create(
        "directional_mfe",
        lookforward_horizon=500,
    ),

    # 3. Oracle binary (optimal classification)
    TargetGeneratorFactory.create(
        "oracle_binary",
        transaction_cost=0.00007,
    ),

    # 4. Quantile classification (balanced but informative)
    TargetGeneratorFactory.create(
        "quantile_classification",
        nbins=5,
    ),
]

print("\nGenerators created:")
for i, gen in enumerate(generators, 1):
    info = gen.get_target_info()
    print(f"  {i}. {info['description']}")

print("\n" + "=" * 80)
print("COMPARISON: Triple Barrier vs Recommended Methods")
print("=" * 80)

comparison = """
┌─────────────────────────┬──────────────────┬──────────────────────┐
│ Metric                  │ Triple Barrier   │ Recommended Methods  │
├─────────────────────────┼──────────────────┼──────────────────────┤
│ Class Balance           │ 33/32/34 %       │ Varies (realistic)   │
│ Information Loss        │ High (32% waste) │ Low (all data used)  │
│ Prediction Horizon      │ 642 ticks avg    │ 50-500 ticks         │
│ FX Predictability       │ Beyond range     │ Within range         │
│ Transaction Cost Aware  │ Post-hoc         │ Built-in             │
│ Target Type             │ 3-class discrete │ Continuous/adaptive  │
│ Trading Alignment       │ Poor             │ Excellent            │
│ Expected ML Performance │ 34% (random)     │ R² > 0.01            │
│ Interpretability        │ Confusing        │ Clear                │
│ Timeout Class           │ 32% wasted       │ No timeouts          │
└─────────────────────────┴──────────────────┴──────────────────────┘

KEY INSIGHT: Triple barrier creates perfectly balanced classes (33/32/34)
because FX markets have NO directional edge at 642-tick horizon. This makes
ML impossible. Recommended methods work because they either:
  1. Use shorter horizons (50-250 ticks, within predictability range)
  2. Use continuous targets (preserve all information)
  3. Align with trading objective (profit, not arbitrary barriers)
"""

print(comparison)

# ============================================================================
# USAGE EXAMPLE WITH ACTUAL DATA
# ============================================================================
print("\n" + "=" * 80)
print("USAGE EXAMPLE")
print("=" * 80)

example_code = """
# Load your input data
from represent.modular_dataset_builder import ModularDatasetBuilder
from represent.target_generators.factory import TargetGeneratorFactory

# Option 1: Multi-Horizon Regression (RECOMMENDED)
generators_regression = [
    TargetGeneratorFactory.create(
        "log_return_horizons",
        horizons=[50, 100, 250, 500, 1000]
    ),
    TargetGeneratorFactory.create(
        "directional_mfe",
        lookforward_window=500
    ),
]

builder = ModularDatasetBuilder(generators_regression)
dataset = builder.build_from_parquet("M6AH5_inputs_only.parquet")

# Result: 7 regression targets per sample
# - log_return_50, log_return_100, ..., log_return_1000 (5 targets)
# - directional_mfe_buy, directional_mfe_sell (2 targets)

# Option 2: Oracle Binary Classification
generators_classification = [
    TargetGeneratorFactory.create(
        "oracle_binary",
        transaction_cost=0.00007
    )
]

builder = ModularDatasetBuilder(generators_classification)
dataset = builder.build_from_parquet("M6AH5_inputs_only.parquet")

# Result: Binary target with natural imbalance (10-30% positive)
# More realistic than triple barrier's 33/32/34 split!

# Option 3: Ensemble (Use Multiple Methods)
generators_ensemble = [
    TargetGeneratorFactory.create("log_return_horizons", horizons=[100, 250, 500]),
    TargetGeneratorFactory.create("oracle_binary", transaction_cost=0.00007),
    TargetGeneratorFactory.create("quantile_classification", nbins=5),
]

builder = ModularDatasetBuilder(generators_ensemble)
dataset = builder.build_from_parquet("M6AH5_inputs_only.parquet")

# Result: Multiple complementary targets
# Train separate models, combine predictions for robust trading signals
"""

print(example_code)

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print("""
1. START WITH REGRESSION (log_return_horizons)
   → Easiest to validate, preserves most information

2. EVALUATE PREDICTIVE POWER
   → Compute feature-target correlations
   → Train simple baseline (linear regression)
   → Check R² > 0.001 before complex models

3. ADD MICROSTRUCTURE FEATURES
   → Order flow imbalance
   → Bid-ask spread dynamics
   → Volume-weighted features
   → Recent momentum (10-50 ticks)

4. REALISTIC EXPECTATIONS
   → R² = 0.01-0.05 is GOOD for FX microstructure
   → Don't expect high accuracy (markets are efficient)
   → Small edge compounds over many trades

5. VALIDATE WITH CROSS-VALIDATION
   → Time-series aware splits
   → Walk-forward validation
   → Out-of-sample testing

BOTTOM LINE: Stop using triple barrier. It creates unpredictable balanced
classes. Use regression with shorter horizons or oracle methods instead.
""")

print("=" * 80)
