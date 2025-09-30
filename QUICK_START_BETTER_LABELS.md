# Quick Start: Better Labels for FX Microstructure

## The Problem with Triple Barrier

Your ML models fail because triple barrier creates **perfectly balanced classes (33/32/34%)** in efficient FX markets with a **642-tick prediction horizon** that's 10-20x beyond the FX predictability range (10-50 ticks).

**Result**: 34% baseline accuracy (random guessing), no learning possible.

## The Solution: Use These Instead

### 1. ⭐ **Multi-Horizon Regression** (RECOMMENDED)

```python
from represent.target_generators.factory import TargetGeneratorFactory

generator = TargetGeneratorFactory.create(
    "log_return_horizons",
    horizons=[50, 100, 250, 500, 1000],  # Multiple time scales
    include_mean_returns=True  # NEW: Also get mean returns (default)
)
```

**Output**: 10 targets per sample
- 5 endpoint returns: `log_return_50t`, `log_return_100t`, ..., `log_return_1000t`
- 5 mean returns: `log_return_mean_50t`, `log_return_mean_100t`, ..., `log_return_mean_1000t`

**Why this works:**
- ✅ Continuous targets (more info than 3 classes)
- ✅ Shorter horizons within FX predictability
- ✅ No 32% timeout class wasting data
- ✅ Can threshold predictions for trading
- ✅ **NEW**: Mean returns capture path/trend info (50-100x smoother!)

**What's the difference?**
- **Endpoint return** (`log_return_Nt`): Entry price → Exit price (final P&L)
- **Mean return** (`log_return_mean_Nt`): Average tick-to-tick return over horizon (trend/drift)
  - Smoother signal (less noise from exit tick)
  - Better for shorter horizons
  - Captures path information

**Expected**: R² = 0.01-0.05 (good for FX!)

---

### 2. **Directional MFE** (Profit Potential)

```python
generator = TargetGeneratorFactory.create(
    "directional_mfe",
    lookforward_horizon=500  # Peak profit opportunity
)
```

**Output**: `buy_mfe`, `sell_mfe` (best possible profit in each direction)

**Why this works:**
- ✅ Trading-focused (actual profit potential)
- ✅ Useful for position sizing
- ✅ Two targets better than one

---

### 3. **Oracle Binary** (Optimal Classification)

```python
generator = TargetGeneratorFactory.create(
    "oracle_binary",
    transaction_cost=0.00007  # 0.7 pips
)
```

**Output**: Binary {0, 1} where 1 = profitable trade opportunity

**Why this works:**
- ✅ Labels based on actual profitability
- ✅ Naturally imbalanced (10-30% positive)
- ✅ Transaction cost aware
- ✅ Simpler than 3-class

---

### 4. **Quantile Classification** (Balanced but Informative)

```python
generator = TargetGeneratorFactory.create(
    "quantile_classification",
    nbins=5  # Top 20%, next 20%, etc.
)
```

**Output**: Classes {0, 1, 2, 3, 4} representing return quintiles

**Why this works:**
- ✅ Balanced but preserves ranking
- ✅ No timeout class
- ✅ Clear interpretation

---

## Complete Example

```python
from represent.modular_dataset_builder import ModularDatasetBuilder
from represent.target_generators.factory import TargetGeneratorFactory

# Create multiple generators
generators = [
    # PRIMARY: Multi-horizon regression
    TargetGeneratorFactory.create(
        "log_return_horizons",
        horizons=[50, 100, 250, 500, 1000]
    ),

    # SECONDARY: Profit potential
    TargetGeneratorFactory.create(
        "directional_mfe",
        lookforward_horizon=500
    ),

    # TERTIARY: Binary classification
    TargetGeneratorFactory.create(
        "oracle_binary",
        transaction_cost=0.00007
    ),
]

# Build dataset
builder = ModularDatasetBuilder(generators)
dataset = builder.build_from_parquet("M6AH5_inputs_only.parquet")

# Result: 8 targets per sample
# - 5 regression targets (multi-horizon returns)
# - 2 regression targets (buy/sell MFE)
# - 1 binary classification (oracle optimal)
```

---

## Comparison

| Metric | Triple Barrier | Recommended |
|--------|---------------|-------------|
| **Class Balance** | 33/32/34% (no edge) | Varies (realistic) |
| **Horizon** | 642 ticks (too far) | 50-500 ticks (predictable) |
| **Information Loss** | 32% timeout waste | All data used |
| **ML Performance** | 34% (random) | R² > 0.01 |
| **Transaction Costs** | Post-hoc | Built-in |

---

## Expected Results

**Triple Barrier:**
- Accuracy: ~34% (random baseline)
- Learning: None (no patterns exist)
- Trading: Impossible

**Regression (log_return_horizons):**
- R² > 0.001: Statistically significant
- R² > 0.01: Potentially tradeable
- R² > 0.05: Excellent for FX

**Classification (oracle_binary):**
- Accuracy: 70-90% (imbalanced baseline)
- Precision/Recall: Measure real performance
- F1 Score: Better metric than accuracy

---

## Next Steps

1. **Replace** triple barrier with `log_return_horizons`
2. **Train** simple baseline (linear regression)
3. **Validate** R² > 0.001 before complex models
4. **Add** microstructure features (order flow, spread)
5. **Test** with walk-forward cross-validation

---

## Full Example Script

See: `examples/recommended_labeling_methods.py`

Run: `python examples/recommended_labeling_methods.py`

---

## Bottom Line

❌ **Stop using triple barrier** - it creates unpredictable balanced classes

✅ **Use regression** (`log_return_horizons`) with shorter horizons

✅ **Or use oracle** (`oracle_binary`) for classification with realistic imbalance

✅ **Expect R² = 0.01-0.05** (this is GOOD for FX microstructure!)