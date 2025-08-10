# Bug Report: represent v1.11.1 Complete Pipeline Issues

## Summary

The `api.run_complete_pipeline()` function in represent v1.11.1 fails to create classified files due to missing column and insufficient sample issues during Stage 2 classification.

## Environment

- **represent version**: v1.11.1
- **Python version**: 3.12+
- **Platform**: macOS (Darwin 24.5.0)
- **Data source**: Databento DBN files (AUDUSD micro futures)

## Issues Identified

### 1. Missing `ts_event` Column

**Error Message:**
```
❌ Failed to process AUDUSD_M6AM4.parquet: unable to find column "ts_event"; valid columns: ["symbol", "market_depth_features", "feature_shape", "start_timestamp", "end_timestamp", "global_start_idx", "global_end_idx", "sample_id", "source_file", "date", "hour", "start_mid_price", "end_mid_price"]
```

**Issue:** 
- Stage 1 (UnlabeledDBNConverter) creates parquet files without `ts_event` column
- Stage 2 (ParquetClassifier) expects `ts_event` column to exist
- This causes all classification to fail

**Expected Behavior:**
- Stage 1 should create the `ts_event` column that Stage 2 requires
- Or Stage 2 should use available timestamp columns (`start_timestamp`, `end_timestamp`)

### 2. Insufficient Samples Threshold

**Error Messages:**
```
❌ Insufficient samples: 403 < 1000
❌ Insufficient samples: 491 < 1000  
❌ Insufficient samples: 324 < 1000
```

**Issue:**
- Hard-coded minimum 1000 samples per symbol requirement
- Many DBN files contain <1000 samples per symbol, especially for smaller timeframes
- This results in most symbols being rejected even when they have hundreds of valid samples

**Current Impact:**
- 10 DBN files processed: ~25,000 total samples generated in Stage 1
- 0 classified samples produced in Stage 2 (100% rejection rate)

**Suggested Solutions:**
1. **Lower default threshold** to 100 or 500 samples per symbol
2. **Make threshold configurable** via RepresentConfig
3. **Aggregate across symbols** instead of per-symbol requirements

### 3. Configuration Issues

**Feature Name Mismatch:**
- Config uses `"trade_count"` but represent expects `"trade_counts"` (plural)
- This causes immediate validation failure

## Reproduction Steps

1. Install represent v1.11.1 from GitHub
2. Create RepresentConfig for AUDUSD with features `["volume"]`
3. Run complete pipeline on any DBN file:
   ```python
   api.run_complete_pipeline(
       config=config,
       dbn_path="path/to/file.dbn.zst", 
       output_base_dir="output",
       verbose=True
   )
   ```
4. Observe Stage 1 completes successfully
5. Observe Stage 2 fails with `ts_event` column error

## Expected vs Actual Behavior

**Expected:**
- Complete pipeline produces classified parquet files ready for ML training
- Reasonable sample thresholds allow most symbols to be processed
- Column schema consistent between stages

**Actual:**
- Stage 1 completes but creates incompatible schema
- Stage 2 fails completely due to missing column
- Zero classified files produced despite thousands of input samples

## Suggested Fixes

### Fix 1: Column Schema Consistency
```python
# In UnlabeledDBNConverter, ensure ts_event column is created
df['ts_event'] = df['start_timestamp']  # or appropriate mapping
```

### Fix 2: Configurable Sample Threshold  
```python
# In RepresentConfig, add configurable threshold
class RepresentConfig:
    min_samples_per_symbol: int = 100  # Default lowered from 1000
```

### Fix 3: Feature Name Standardization
Update documentation to clarify:
- `"trade_counts"` (plural) is the correct feature name
- Or support both `"trade_count"` and `"trade_counts"`

## Test Data

The issue can be reproduced with any Databento AUDUSD micro futures DBN files. Sample files used:
- `glbx-mdp3-20240403.mbp-10.dbn.zst` (282k rows → 2,739 samples → 0 classified)
- `glbx-mdp3-20240404.mbp-10.dbn.zst` (295k rows → 2,856 samples → 0 classified)
- Multiple other files with similar pattern

## Impact

This bug prevents the complete pipeline from being usable for ML training workflows, as it produces zero classified training data despite successfully processing the raw market data.

## Urgency

High - Core functionality is broken for the primary use case of the complete pipeline API.