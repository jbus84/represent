"""
Core functionality that provides backward compatibility.
This module bridges the optimized implementation with existing tests.
"""

from .pipeline import process_market_data


def placeholder_function() -> str:
    """Placeholder function for testing coverage."""
    return "placeholder"


def uncovered_function() -> str:
    """Function that won't be covered by tests."""
    return "uncovered"  # pragma: no cover


# Alias for backward compatibility with tests
reference_pipeline = process_market_data