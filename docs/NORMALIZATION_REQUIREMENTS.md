# Critical Normalization Requirements

## ⚠️ NEVER CHANGE THE SIGNED NORMALIZATION ⚠️

This document explains the **CRITICAL** normalization requirements that must **NEVER** be changed without understanding the severe consequences.

## The Correct Normalization (Current Implementation)

The normalization in `represent/data_structures.py` **MUST** follow this exact approach from the notebook:

```python
def prepare_output(self, ask_grid: np.ndarray, bid_grid: np.ndarray) -> np.ndarray:
    """Prepare normalized combined output using notebook approach."""
    # Calculate combined volume (ask - bid)
    np.subtract(ask_grid, bid_grid, out=self._temp_combined)
    
    # Create mask for negative values (bid > ask)
    neg_mask = self._temp_combined < 0
    
    # Take absolute value 
    np.abs(self._temp_combined, out=self._temp_abs)
    
    # Normalize: (abs_combined - 0) / (abs_combined.max() - 0)
    # min is always 0 volume
    max_val = np.max(self._temp_abs)
    if max_val > 0:
        np.divide(self._temp_abs, max_val, out=self._buffer)
    else:
        self._buffer.fill(0)
        
    # CRITICAL: Restore negative sign for values where bid > ask
    self._buffer[neg_mask] *= -1
    
    return self._buffer
```

## Why This Approach is Critical

### ✅ **Correct Signed [-1, 1] Range:**
- **-1.0**: Maximum bid dominance (selling pressure)
- **0.0**: Perfect bid/ask balance  
- **+1.0**: Maximum ask dominance (buying pressure)

### ✅ **Preserves Market Directional Information:**
- **Negative values**: Bid side has more volume (market selling pressure)
- **Positive values**: Ask side has more volume (market buying pressure)
- **Zero values**: Perfectly balanced market

### ✅ **CNN Training Optimized:**
- Zero-centered data improves gradient flow
- Signed range provides semantic meaning to the model
- Standard practice for CNN input normalization

## The WRONG Approach (DO NOT IMPLEMENT)

❌ **NEVER implement unsigned [0,1] normalization like this:**

```python
# WRONG - DO NOT USE
def wrong_normalization(ask_grid, bid_grid):
    combined = ask_grid - bid_grid
    abs_combined = np.abs(combined)  # ❌ Loses sign information
    return abs_combined / abs_combined.max()  # ❌ Only [0,1] range
```

### Why Unsigned [0,1] is Catastrophic:

❌ **Loses directional information** - Cannot distinguish ask vs bid dominance
❌ **Breaks CNN training** - Model cannot learn market dynamics  
❌ **Not zero-centered** - Poor gradient flow during training
❌ **Inconsistent with notebook** - Doesn't match reference implementation

## Tests That Prevent Regression

The following tests in `tests/unit/test_signed_normalization.py` and `tests/unit/test_data_structures.py` **MUST ALWAYS PASS**:

1. **`test_correct_signed_normalization`** - Ensures [-1,1] range
2. **`test_regression_prevention_unsigned_normalization`** - Prevents [0,1] regression  
3. **`test_directional_information_preservation`** - Ensures semantic meaning
4. **`test_notebook_compliance`** - Matches reference implementation exactly

### Running the Tests

```bash
# These tests MUST always pass
uv run pytest tests/unit/test_signed_normalization.py -v
uv run pytest tests/unit/test_data_structures.py::TestOutputBuffer::test_correct_signed_normalization -v
```

## Real Data Verification

The correct normalization produces these characteristics on real AUDUSD data:

```python
volume_result = process_market_data(df_polars, features=["volume"])

# Correct output characteristics:
print(f"Range: [{volume_result.min():.6f}, {volume_result.max():.6f}]")
# Expected: Range: [-1.000000, 0.954310] (signed range)

print(f"Negative values: {np.sum(volume_result < 0)}")  # ~37% bid dominance
print(f"Positive values: {np.sum(volume_result > 0)}")  # ~61% ask dominance  
print(f"Zero values: {np.sum(volume_result == 0)}")     # ~2% perfect balance
```

## Visualization Impact

The signed normalization enables proper market depth visualization:

- **RdBu colormap with vmin=-1, vmax=1**
- **Red regions**: Bid dominance (selling pressure)
- **Blue regions**: Ask dominance (buying pressure)  
- **White/gray regions**: Market balance

## If You Need to Change This

**DON'T.** The normalization is based on extensive analysis in the Jupyter notebook at:
`/Users/danielfisher/repositories/represent/notebooks/market_depth_extraction_micro_pips.ipynb`

If you absolutely must change it:

1. **Update the notebook first** and verify the new approach
2. **Update ALL tests** to match the new requirements  
3. **Verify CNN training** still works with the new range
4. **Update visualization code** to handle the new range
5. **Update this documentation** to explain the new approach

## Summary

The signed [-1, 1] normalization is **fundamental** to the represent library's correctness. It:

- ✅ Preserves critical market directional information
- ✅ Optimizes CNN training with zero-centered data  
- ✅ Matches the reference notebook implementation exactly
- ✅ Enables proper market depth visualization

**Changing this breaks the entire library's market analysis capabilities.**