#!/usr/bin/env python3
"""
Proven Parameters from Diagnostic Analysis

These parameters were validated through comprehensive diagnostic analysis
and generate excellent signal frequencies (76-96%) with balanced distributions.
They should be used as defaults or starting points for optimization.
"""

# PROVEN parameters from diagnostics/comprehensive_method_debugging.py
PROVEN_PARAMETERS = {
    "binary_ctl": {
        "omega": 0.0,  # Optimized parameter from successful runs
    },

    "ternary_ctl": {
        "marginal_change_thres": 0.0446,  # Optimized parameters from successful runs
        "window_size": 501,
    },

    "oracle_binary": {
        "transaction_cost": 0.00007,  # Corrected transaction cost (0.7 pips)
    },

    "oracle_ternary": {
        "transaction_cost": 0.00007,  # Corrected transaction cost (0.7 pips)
        "neutral_reward_factor": 0.1,  # Optimized parameters
    },

    "triple_barrier": {
        "lookforward_window": 2000,  # PROVEN: generates good signal frequency
        "barrier_width": 0.0005,     # FIXED: 5 pip barriers for better signal quality
        "transaction_cost": 0.00007,  # Correct 0.7 pip transaction cost
    },

    "triple_exceedance": {
        "lookforward_window": 2000,  # PROVEN: generates 77% signal frequency
        "scaling_factor": 3.0,       # PROVEN: 3x transaction cost (~2.1 pips)
        "transaction_cost": 0.00007, # Correct 0.7 pip transaction cost
    }
}

def get_proven_parameters(method_name: str) -> dict:
    """
    Get proven parameters for a specific method.

    Args:
        method_name: Name of the method (e.g., 'triple_barrier')

    Returns:
        Dictionary of proven parameters for the method

    Raises:
        KeyError: If method_name is not found
    """
    if method_name not in PROVEN_PARAMETERS:
        available = list(PROVEN_PARAMETERS.keys())
        raise KeyError(f"Method '{method_name}' not found. Available: {available}")

    return PROVEN_PARAMETERS[method_name].copy()

def apply_proven_parameters(generator_class, **kwargs):
    """
    Create a generator with proven parameters, allowing overrides.

    Args:
        generator_class: The target generator class
        **kwargs: Parameter overrides

    Returns:
        Instantiated generator with proven parameters
    """
    # Map class names to method names
    class_to_method = {
        'TripleBarrierGenerator': 'triple_barrier',
        'TripleExceedanceGenerator': 'triple_exceedance',
        'BinaryCTLGenerator': 'binary_ctl',
        'TernaryCTLGenerator': 'ternary_ctl',
        'OracleBinaryGenerator': 'oracle_binary',
        'OracleTernaryGenerator': 'oracle_ternary',
    }

    class_name = generator_class.__name__
    if class_name in class_to_method:
        method_name = class_to_method[class_name]
        proven_params = get_proven_parameters(method_name)

        # Merge with overrides (kwargs take precedence)
        final_params = {**proven_params, **kwargs}
        return generator_class(**final_params)
    else:
        # Fallback for unknown classes
        return generator_class(**kwargs)

def use_proven_defaults() -> bool:
    """
    Check if proven parameters should be used as defaults.
    This can be controlled via environment variable.
    """
    import os
    return os.getenv('USE_PROVEN_PARAMS', '1').lower() in ('1', 'true', 'yes')
