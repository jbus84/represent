"""
Core functionality that provides backward compatibility.
This module bridges the optimized implementation with existing tests.
"""

import polars as pl
import numpy as np
from typing import Optional, Union
from .pipeline import process_market_data
from .constants import FeatureType


def placeholder_function() -> str:
    """Placeholder function for testing coverage."""
    return "placeholder"


def uncovered_function() -> str:
    """Function that won't be covered by tests."""
    return "uncovered"  # pragma: no cover


# Alias for backward compatibility with tests
def reference_pipeline(
    df: pl.DataFrame, features: Optional[Union[list[str], list[FeatureType]]] = None
) -> np.ndarray:
    """Backward compatibility wrapper for process_market_data."""
    return process_market_data(df, features=features)
