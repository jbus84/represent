"""
Base Target Generator Interface

This module defines the core interface that all target generators must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import polars as pl


class TargetGenerator(ABC):
    """
    Abstract base class for all target generators.

    This interface ensures consistent behavior across all target generation
    approaches, whether classification, regression, or custom logic.
    """

    @abstractmethod
    def generate_targets(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """
        Generate target arrays from market data.

        Args:
            df: Market data DataFrame with required columns

        Returns:
            Dict mapping target names to numpy arrays of the same length as df

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        pass

    @abstractmethod
    def get_target_info(self) -> dict[str, Any]:
        """
        Return metadata about the targets this generator creates.

        Returns:
            Dict containing:
            - target_names: List of target column names
            - target_type: 'classification' or 'regression'
            - description: Human-readable description
            - parameters: Generator-specific parameters
        """
        pass

    @property
    @abstractmethod
    def target_type(self) -> str:
        """
        Return the type of targets generated.

        Returns:
            Either 'classification' or 'regression'
        """
        pass

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        Return list of required DataFrame columns.

        Returns:
            List of column names that must be present in the input DataFrame
        """
        pass

    def validate_input(self, df: pl.DataFrame) -> None:
        """
        Validate that input DataFrame has required columns.

        Args:
            df: Input DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns for {self.__class__.__name__}: {sorted(missing_columns)}"
            )

    def __repr__(self) -> str:
        """Return string representation of the generator."""
        info = self.get_target_info()
        return f"{self.__class__.__name__}(type={self.target_type}, targets={info.get('target_names', [])})"
