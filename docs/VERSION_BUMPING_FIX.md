# Version Bumping Fix Documentation

## Problem Summary

The semantic-release configuration was not properly updating the `__version__` variable in `represent/__init__.py` during automated version bumping. While `pyproject.toml` versions were being updated correctly, the `__init__.py` file remained stuck at older versions.

## Root Cause Analysis

1. **Incorrect version_variables format**: The original configuration included `pyproject.toml:version` which is redundant since pyproject.toml is handled automatically by semantic-release.

2. **Wrong __version__ declaration format**: Using typed assignment `__version__: str = "1.7.0"` instead of simple assignment `__version__ = "1.7.0"` prevented semantic-release's regex matching.

3. **Configuration mismatch**: The `version_variables` array format didn't match the documented semantic-release requirements.

## Applied Fixes

### 1. Updated pyproject.toml Configuration

**Before:**
```toml
[tool.semantic_release]
version_variables = ["pyproject.toml:version", "represent/__init__.py:__version__"]
```

**After:**
```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
version_variables = ["represent/__init__.py:__version__"]
```

**Rationale:** 
- `version_toml` is the correct way to handle TOML files like pyproject.toml
- `version_variables` is for non-TOML files like Python files
- This ensures both locations are properly updated during version bumping

### 2. Fixed __version__ Declaration Format

**Before:**
```python
__version__: str = "1.7.0"
```

**After:**
```python
__version__ = "1.7.1"
```

**Rationale:** Semantic-release uses regex patterns to find and replace version variables. The simple assignment format is required for proper matching.

### 3. Synchronized All Version Locations

All version locations are now synchronized at `1.7.1`:
- `pyproject.toml`: 1.7.1
- `[tool.commitizen]`: 1.7.1
- `represent/__init__.py`: 1.7.1

## Testing the Fix

### Verification Script

Run the included verification script to check version synchronization:

```bash
uv run python scripts/test_version_sync.py
```

Expected output:
```
🔍 Testing Version Synchronization
========================================
📄 pyproject.toml version: 1.7.1
⚙️  commitizen version: 1.7.1
🐍 __init__.py version: 1.7.1

✅ All versions are synchronized!

🔧 Checking semantic-release configuration...
📝 Version TOML: ['pyproject.toml:project.version']
📝 Version variables: ['represent/__init__.py:__version__']
✅ Semantic-release is configured to update both locations!
   📄 pyproject.toml via version_toml
   🐍 __init__.py via version_variables

🎉 Version bumping system is properly configured!
```

### Package Import Test

Verify the package imports correctly with the new version:

```bash
uv run python -c "import represent; print(f'Version: {represent.__version__}')"
```

Expected output:
```
Version: 1.7.1
```

## How Semantic-Release Works Now

1. **Automatic Detection**: Semantic-release analyzes commit messages for conventional commit patterns (`feat:`, `fix:`, `BREAKING CHANGE:`)

2. **Version Calculation**: Determines the next version based on the semantic versioning rules

3. **File Updates**: 
   - **pyproject.toml**: Updated via `version_toml` configuration
   - **represent/__init__.py**: Updated via `version_variables` configuration
   - **[tool.commitizen]**: Updated automatically as part of pyproject.toml

4. **Git Operations**: Creates commit, tags, and updates changelog

## Future Version Bumping Workflow

1. **Make Changes**: Implement features or fixes
2. **Commit with Conventional Format**:
   ```bash
   git commit -m "feat: add new functionality"
   git commit -m "fix: resolve bug in calculation" 
   git commit -m "fix!: breaking change in API"  # for breaking changes
   ```
3. **Run Semantic Release** (when ready):
   ```bash
   semantic-release version
   ```
4. **Verify Updates**: All version locations should be updated automatically

## Configuration Best Practices

### ✅ Do:
- Use simple assignment for `__version__` in Python files
- Use `version_toml` for TOML files like pyproject.toml
- Use `version_variables` for non-TOML files like Python files
- Follow conventional commit message format

### ❌ Don't:
- Include TOML files in `version_variables` (use `version_toml` instead)
- Use typed assignments for version variables (`__version__: str = ...`)
- Manually edit version numbers (let semantic-release handle it)
- Mix manual and automated version bumping

## Troubleshooting

### Version Sync Issues

If versions get out of sync, run the verification script to identify the problem:

```bash
uv run python scripts/test_version_sync.py
```

### Manual Sync (Emergency)

If manual intervention is needed:

1. Check current versions:
   ```bash
   grep 'version = ' pyproject.toml
   grep '__version__' represent/__init__.py
   ```

2. Update manually if needed:
   ```bash
   # Update __init__.py to match pyproject.toml
   sed -i 's/__version__ = ".*"/__version__ = "X.Y.Z"/' represent/__init__.py
   ```

3. Verify sync:
   ```bash
   uv run python scripts/test_version_sync.py
   ```

## Configuration Files Changed

- **pyproject.toml**: Updated `[tool.semantic_release]` configuration
- **represent/__init__.py**: Changed `__version__` declaration format
- **scripts/test_version_sync.py**: Updated regex patterns and expectations

## Validation

- ✅ All linting checks pass (ruff, pyright)
- ✅ Package imports correctly
- ✅ Version synchronization verified
- ✅ Semantic-release configuration validated
- ✅ Verification script runs successfully

The version bumping system is now properly configured and should work automatically for future releases.