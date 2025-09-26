# Duplicate Row Removal Integration Plan

## Problem Statement
`process_symbols_with_triple_methods.py` lacks duplicate row handling when processing batched datasets, while `build_symbol_datasets_from_dbn.py` has proven deduplication logic. Need to integrate this into `ModularDatasetBuilder` while preserving target-only output.

## Current Situation Analysis

### 🔍 Duplicate Handling Gap

**`process_symbols_with_triple_methods.py`** (OLD - No duplicate handling):
- Uses `ModularDatasetBuilder.build_targets_from_parquet_chunked()`
- Processes data in 500K chunks
- **ISSUE**: No duplicate removal when merging chunked results
- **Risk**: Duplicated row indices can cause inconsistent joins with input data

**`build_symbol_datasets_from_dbn.py`** (NEW - Has duplicate handling):
- Sophisticated deduplication logic after merging chunks:
  ```python
  # Smart column-based deduplication
  if "seqnum" in merged_df.columns:
      dedup_subset = ["ts_event", "seqnum"]
  elif "ts_recv" in merged_df.columns:
      dedup_subset = ["ts_event", "ts_recv"]
  elif "symbol" in merged_df.columns:
      dedup_subset = ["ts_event", "symbol"]
  else:
      dedup_subset = ["ts_event"]

  merged_df = merged_df.unique(subset=dedup_subset, maintain_order=True)
  ```

## Solution Overview
Add deduplication logic to `ModularDatasetBuilder.build_targets_from_parquet_chunked()` that **always runs when needed** (when duplicates could exist), using the smart column-based approach from `build_symbol_datasets_from_dbn.py`.

## 🚀 Implementation Plan

### Phase 1: Add Deduplication to ModularDatasetBuilder

**File**: `represent/modular_dataset_builder.py`

#### 1.1 Add Smart Deduplication Method
```python
def _remove_duplicates(self, df: pl.DataFrame, verbose_prefix: str = "") -> pl.DataFrame:
    """Remove duplicate rows using smart column-based deduplication."""
    before_len = len(df)

    # Smart column detection (most granular first)
    if "seqnum" in df.columns:
        dedup_subset = ["ts_event", "seqnum"]
    elif "ts_recv" in df.columns:
        dedup_subset = ["ts_event", "ts_recv"]
    elif "symbol" in df.columns:
        dedup_subset = ["ts_event", "symbol"]
    elif "ts_event" in df.columns:
        dedup_subset = ["ts_event"]
    else:
        # Fallback: drop exact duplicate rows
        deduplicated = df.unique(maintain_order=True)
        after_len = len(deduplicated)
        if self.verbose and before_len != after_len:
            print(f"{verbose_prefix}🧹 Removed {before_len - after_len:,} duplicate rows")
        return deduplicated

    deduplicated = df.unique(subset=dedup_subset, maintain_order=True)
    after_len = len(deduplicated)

    if self.verbose and before_len != after_len:
        print(f"{verbose_prefix}🧹 Removed {before_len - after_len:,} duplicate rows using {dedup_subset}")

    return deduplicated
```

#### 1.2 Integrate into Chunked Processing
**Location**: After line 227 in `build_targets_from_parquet_chunked()`

```python
# Concatenate all chunks for this generator
generator_targets = pl.concat(generator_chunks)

# ALWAYS remove duplicates after chunk concatenation
generator_targets = self._remove_duplicates(
    generator_targets,
    verbose_prefix=f"      "
)
```

**Rationale**:
- Always runs after chunk concatenation where duplicates could exist
- No configuration needed - automatic and intelligent
- Preserves chronological order with `maintain_order=True`
- Uses most granular unique key available

#### 1.3 Integration Point Details
The deduplication will be inserted in `build_targets_from_parquet_chunked()` method:

**Before** (line 226-228):
```python
# Concatenate all chunks for this generator
generator_targets = pl.concat(generator_chunks)
all_target_chunks.append(generator_targets)
```

**After**:
```python
# Concatenate all chunks for this generator
generator_targets = pl.concat(generator_chunks)

# Remove duplicates after chunk concatenation
generator_targets = self._remove_duplicates(
    generator_targets,
    verbose_prefix=f"      "
)

all_target_chunks.append(generator_targets)
```

### Phase 2: Update Scripts

**File**: `scripts/process_symbols_with_triple_methods.py`

**No changes needed** - deduplication now runs automatically when processing chunks.

The existing call at line 116-120:
```python
targets_df = builder.build_targets_from_parquet_chunked(
    input_file,
    symbol=symbol_name,
    chunk_size=500_000
)
```

Will now automatically include deduplication.

## 🎯 Expected Benefits

1. **Always runs**: Deduplication happens automatically when needed
2. **Smart logic**: Uses proven column-based approach from `build_symbol_datasets_from_dbn.py`
3. **Zero config**: No parameters needed, works out of the box
4. **Target-only preserved**: Output format unchanged (`row_idx`, `timestamp`, `symbol` + targets)
5. **Performance**: Only runs after chunk concatenation, minimal overhead
6. **Maintains order**: Preserves chronological ordering of data

## 📊 Technical Details

### Key Design Decisions

1. **Automatic execution**: Always run deduplication after chunk concatenation
2. **Smart column detection**: Use most granular unique key available:
   - `ts_event` + `seqnum` (most granular)
   - `ts_event` + `ts_recv` (fallback)
   - `ts_event` + `symbol` (fallback)
   - `ts_event` only (fallback)
   - Full row deduplication (ultimate fallback)
3. **Order preservation**: Use `maintain_order=True` for chronological consistency
4. **Verbose reporting**: Show deduplication statistics when verbose=True

### Integration Strategy

- **Minimal footprint**: Only ~20 lines of new code
- **Zero breaking changes**: All existing code works unchanged
- **Performance optimized**: Only runs when chunks are concatenated
- **Proven logic**: Reuses tested deduplication from `build_symbol_datasets_from_dbn.py`

## ✅ Success Criteria

1. **Functional**: `process_symbols_with_triple_methods.py` handles duplicates correctly
2. **Performance**: No degradation in processing speed
3. **Compatibility**: All existing scripts work unchanged
4. **Logging**: Clear visibility into deduplication actions
5. **Data integrity**: Target-only output format preserved
6. **Order preservation**: Chronological data ordering maintained

## 🚨 Risk Mitigation

- **Proven logic**: Deduplication code copied from working implementation
- **Conservative approach**: Always remove duplicates (safer than potentially leaving them)
- **Isolated changes**: All new code contained in one private method
- **Extensive logging**: Clear visibility into what's happening
- **Backward compatibility**: Existing method signatures unchanged

## Implementation Steps

1. **Add `_remove_duplicates()` method** to `ModularDatasetBuilder`
2. **Integrate deduplication call** in `build_targets_from_parquet_chunked()`
3. **Test with existing scripts** to ensure no breaking changes
4. **Verify duplicate removal** works as expected
5. **Update tests** to cover new functionality

This plan provides a surgical integration that solves the duplicate handling issue while maintaining all existing behavior and performance characteristics, with automatic execution whenever deduplication is needed.