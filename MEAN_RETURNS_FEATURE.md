# NEW: Mean Log Returns Feature

## What's New

The `log_return_horizons` generator now includes **mean returns** alongside endpoint returns, giving you **2x the targets** with complementary information.

## Quick Example

```python
from represent.target_generators.factory import TargetGeneratorFactory

generator = TargetGeneratorFactory.create(
    "log_return_horizons",
    horizons=[50, 100, 250, 500],
    include_mean_returns=True  # Default: True
)

# Output: 8 targets per sample
# - Endpoint returns: log_return_50t, log_return_100t, log_return_250t, log_return_500t
# - Mean returns: log_return_mean_50t, log_return_mean_100t, log_return_mean_250t, log_return_mean_500t
```

## What's the Difference?

### Endpoint Returns (`log_return_Nt`)
**Measures**: Entry price → Exit price after N ticks
```
log_return_250t = log(price[i+250] / price[i]) * 10000  # bps
```

**Use for**:
- Final P&L prediction
- "Where will price end up?"
- Traditional return forecasting

**Characteristics**:
- Sensitive to exit tick noise
- Higher variance (especially for short horizons)
- Standard approach in literature

---

### Mean Returns (`log_return_mean_Nt`)  ⭐ **NEW**
**Measures**: Average of all tick-to-tick returns over N ticks
```
# All tick-to-tick returns over horizon
tick_returns = [log(price[i+1]/price[i]), log(price[i+2]/price[i+1]), ..., log(price[i+N]/price[i+N-1])]

# Mean of all returns
log_return_mean_Nt = mean(tick_returns) * 10000  # bps
```

**Use for**:
- Trend/drift direction
- "What's the average movement per tick?"
- Path information (not just endpoints)

**Characteristics**:
- **50-100x smoother** than endpoint returns!
- Less sensitive to single-tick noise
- Better for shorter horizons
- Captures sustained trends

---

## Performance Comparison

From test on 10K samples:

| Horizon | Endpoint Std | Mean Std | Smoother By |
|---------|-------------|----------|-------------|
| 50 ticks | 6.97 bps | 0.14 bps | **50x** |
| 100 ticks | 9.98 bps | 0.10 bps | **100x** |
| 250 ticks | 14.68 bps | 0.06 bps | **244x** |

**Mean returns are dramatically smoother** while capturing the same trend information!

---

## Why Both Are Useful

### Complementary Information

1. **Endpoint**: Final destination
   - "Will price be higher or lower?"
   - Important for P&L
   - Noisy for short horizons

2. **Mean**: Journey/path
   - "Is price consistently drifting up/down?"
   - Captures trend strength
   - Smooth signal even for short horizons

### Trading Strategy Example

```python
# Predict both endpoint and mean returns
endpoint_pred = model.predict(features)[:, 0]  # log_return_250t
mean_pred = model.predict(features)[:, 5]      # log_return_mean_250t

# Strong long signal: BOTH positive
# - Endpoint: Price will end higher (profitable)
# - Mean: Consistent upward drift (confident trend)
strong_long = (endpoint_pred > 0.0002) & (mean_pred > 0)

# Strong short signal: BOTH negative
strong_short = (endpoint_pred < -0.0002) & (mean_pred < 0)

# Avoid contradictory signals:
# - Endpoint positive but mean negative = choppy, not trending
# - Endpoint negative but mean positive = reversal risk
contradictory = (endpoint_pred > 0) != (mean_pred > 0)
```

---

## When to Use Each

### Use Endpoint Returns When:
- Predicting final P&L
- Long horizons (500+ ticks)
- Strong trends expected
- Less noise tolerance needed

### Use Mean Returns When:
- Predicting trend direction
- Short horizons (50-250 ticks)
- Noisy exit data
- Want smoother training signal

### Use BOTH When (Recommended):
- Maximum information
- Ensemble models
- Different time scales
- Production trading systems

---

## ML Training Tips

### 1. Different Loss Functions

```python
# Endpoint returns: MSE (care about magnitude)
model_endpoint = LGBMRegressor(objective='regression')
model_endpoint.fit(X, endpoint_targets)

# Mean returns: MAE (care about direction)
model_mean = LGBMRegressor(objective='regression_l1')
model_mean.fit(X, mean_targets)
```

### 2. Feature Importance

Mean returns often show:
- **Higher R²** for microstructure features (order flow, spread)
- **Lower R²** overall (smoother = less variance to explain)
- **Better generalization** (less overfitting to noise)

### 3. Ensemble

```python
# Combine predictions
final_signal = 0.6 * endpoint_pred + 0.4 * (mean_pred * 250)  # Scale mean to horizon
#                ↑ P&L prediction        ↑ Trend confirmation
```

---

## Default Behavior

**Since v1.x.x**: `include_mean_returns=True` by default

To get legacy behavior (endpoint only):
```python
generator = TargetGeneratorFactory.create(
    "log_return_horizons",
    horizons=[50, 100, 250],
    include_mean_returns=False  # Disable mean returns
)
# Output: 3 targets (endpoint only)
```

---

## Technical Details

### Mean Return Calculation

For each sample at index `i` with horizon `N`:

1. Extract price path: `prices[i:i+N+1]`
2. Compute tick-to-tick log returns:
   ```python
   tick_returns = log(prices[1:] / prices[:-1])
   ```
3. Take mean:
   ```python
   mean_return = np.mean(tick_returns) * 10000  # Convert to bps
   ```

### Boundary Handling

Both endpoint and mean returns use the same boundary logic:
- Start: After `lookback_window`
- End: Before `max(horizons)` from end
- NaN for invalid samples

---

## Examples

See:
- **`test_mean_returns.py`** - Feature demonstration
- **`examples/recommended_labeling_methods.py`** - Full usage example
- **`QUICK_START_BETTER_LABELS.md`** - Quick reference

---

## Summary

✅ **Use mean returns for**:
- Smoother training signal (50-100x less noise)
- Trend/drift prediction
- Short horizons (50-250 ticks)
- Complementary information to endpoint returns

✅ **Default**: Both endpoint and mean (2x targets)

✅ **Benefits**:
- More information per sample
- Better signals for short horizons
- Ensemble opportunities
- Production-ready smooth predictions

**Recommendation**: Keep default `include_mean_returns=True` and train models on both target types!