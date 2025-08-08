"""
Test the process_market_data function directly (core.py was redundant and removed).
"""

import pytest
from represent import process_market_data
from tests.unit.fixtures.sample_data import generate_realistic_market_data


@pytest.mark.unit
def test_process_market_data_direct():
    """Test that process_market_data works correctly without redundant alias."""
    # Use a reasonable test size based on expected dataset size
    expected_samples = 50000  # Standard expected input size for pipeline
    data = generate_realistic_market_data(n_samples=expected_samples, seed=42)
    result = process_market_data(data)

    # Should match expected output format
    assert result.shape == (402, 500)
    assert result.dtype.kind == "f"  # Float type
