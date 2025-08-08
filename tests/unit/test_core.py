"""
Test the core module for coverage validation.
"""

import pytest
from represent.core import reference_pipeline
from tests.unit.fixtures.sample_data import generate_realistic_market_data




@pytest.mark.unit
def test_reference_pipeline_alias():
    """Test that reference_pipeline alias works correctly."""
    # Use a reasonable test size based on expected dataset size
    expected_samples = 50000  # Standard expected input size for pipeline
    data = generate_realistic_market_data(n_samples=expected_samples, seed=42)
    result = reference_pipeline(data)

    # Should match expected output format
    assert result.shape == (402, 500)
    assert result.dtype.kind == "f"  # Float type
