# Diagnostics Directory

This directory contains diagnostic and testing scripts used during the development and debugging of the `represent` package. These scripts were instrumental in identifying and fixing critical issues in the target generation methods.

## Key Issues Resolved

### 1. Triple Barrier Method Fixes
- **Fixed directional logic**: Corrected label assignment to properly indicate long (+1) vs short (-1) signals
- **Fixed returns calculation**: Implemented proper long/short position returns calculation
- **Fixed barrier calculation**: Changed from percentage-based to absolute barriers

### 2. Triple Exceedance Method Redesign
- **Corrected approach**: Changed from first-hit to fixed-duration methodology
- **Dual-sided classification**: Separate binary assessments for long and short exceedance
- **Independent thresholds**: Each direction evaluated separately

### 3. General Optimization Issues
- **Transaction cost errors**: Corrected 0.7 pip round-trip cost calculations
- **Parameter conversion**: Fixed integer parameter handling
- **Label format compatibility**: Resolved mapping issues between different labeling systems

## Script Categories

### Comprehensive Analysis
- `comprehensive_method_debugging.py`: Final working visualization system (✅ **ACTIVE**)
- `comprehensive_assessment.py`: Multi-method performance analysis
- `full_timeseries_debugging.py`: Time series visualization for debugging

### Method-Specific Debugging
- `diagnose_triple_methods.py`: Triple Barrier/Exceedance issue identification
- `diagnose_binary_ctl.py`: Binary CTL method debugging  
- `diagnose_ternary_ctl.py`: Ternary CTL method debugging

### Validation Scripts
- `test_fixed_triple_barrier_directions.py`: Validates corrected Triple Barrier logic
- `test_corrected_triple_exceedance.py`: Validates corrected Triple Exceedance logic
- `test_triple_barrier_fixes.py`: Manual verification of barrier logic
- `final_system_test.py`: Complete system validation

### Economic Analysis
- `deep_dive_triple_economics.py`: Triple method profitability analysis
- `correct_transaction_cost_analysis.py`: Transaction cost verification
- `pnl_consistency_verification.py`: Returns calculation verification

### Parameter Optimization
- `parameter_sensitivity_fix.py`: Parameter bounds optimization
- `test_economically_viable_bounds.py`: Boundary condition testing
- `sampling_variability_analysis.py`: Optimization stability analysis

## Current Status

✅ **All major issues have been resolved**:
- Triple Barrier directional logic is correct
- Triple Exceedance implements proper dual-sided assessment  
- All methods show realistic performance metrics
- Visualizations accurately reflect method behavior

## Important Note

Most of these diagnostic scripts are now **historical** and were used to identify and fix issues. The main working visualization system is:
- `comprehensive_method_debugging.py` - This is the **ACTIVE** script that generates the corrected plots

For ongoing development, use the main optimization and visualization tools in the parent directory rather than these diagnostic scripts.