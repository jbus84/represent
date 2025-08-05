"""
Core functionality - simplified interface.
"""

import polars as pl
import numpy as np
from typing import Optional, Union
from .pipeline import process_market_data
from .constants import FeatureType


# Alias for tests
def reference_pipeline(
    df: pl.DataFrame, features: Optional[Union[list[str], list[FeatureType]]] = None
) -> np.ndarray:
    """Process market data - alias for tests."""
    return process_market_data(df, features=features)
