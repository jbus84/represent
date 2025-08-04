"""
Test the core module for coverage validation.
"""

import pytest
from represent.core import placeholder_function, reference_pipeline
from represent.constants import SAMPLES
from tests.unit.fixtures.sample_data import generate_realistic_market_data


@pytest.mark.unit
def test_placeholder_function():
    """Test the placeholder function."""
    result = placeholder_function()
    assert result == "placeholder"


@pytest.mark.unit
def test_reference_pipeline_alias():
    """Test that reference_pipeline alias works correctly."""
    data = generate_realistic_market_data(n_samples=SAMPLES, seed=42)
    result = reference_pipeline(data)

    # Should match expected output format
    assert result.shape == (402, 500)
    assert result.dtype.kind == "f"  # Float type
