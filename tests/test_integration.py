"""
Integration tests to verify the complete pipeline works as expected.
"""
import pytest
import numpy as np
import polars as pl

from .reference_implementation import reference_pipeline, PRICE_LEVELS, TIME_BINS
from .fixtures.sample_data import generate_realistic_market_data


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate realistic test data
        data = generate_realistic_market_data(n_samples=50000, seed=42)
        
        # Run the pipeline
        result = reference_pipeline(data)
        
        # Validate output
        assert result.shape == (PRICE_LEVELS, TIME_BINS)
        assert np.all(np.isfinite(result))
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
        
        # Check that we have meaningful variation
        assert np.std(result) > 0.01
        assert not np.all(result == 0)
    
    def test_pipeline_reproducibility(self):
        """Test that pipeline produces reproducible results."""
        # Generate same data twice
        data1 = generate_realistic_market_data(n_samples=50000, seed=123)
        data2 = generate_realistic_market_data(n_samples=50000, seed=123)
        
        # Run pipeline
        result1 = reference_pipeline(data1)
        result2 = reference_pipeline(data2)
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
    
    def test_pipeline_with_different_market_conditions(self):
        """Test pipeline under different market conditions."""
        # Test with different base prices and spreads
        conditions = [
            {"base_price": 0.6600, "spread": 0.0002},  # Normal AUDUSD
            {"base_price": 1.2000, "spread": 0.0003},  # Higher price, wider spread
            {"base_price": 0.9000, "spread": 0.0001},  # Mid price, tight spread
        ]
        
        results = []
        for condition in conditions:
            data = generate_realistic_market_data(
                n_samples=50000,
                base_price=condition["base_price"],
                spread=condition["spread"],
                seed=42
            )
            result = reference_pipeline(data)
            results.append(result)
            
            # Each result should be valid
            assert result.shape == (PRICE_LEVELS, TIME_BINS)
            assert np.all(np.isfinite(result))
            assert np.all(result >= -1.0)
            assert np.all(result <= 1.0)
        
        # Results should be different for different conditions
        assert not np.array_equal(results[0], results[1])
        assert not np.array_equal(results[1], results[2])
    
    def test_pipeline_output_characteristics(self):
        """Test that pipeline output has expected characteristics."""
        data = generate_realistic_market_data(n_samples=50000, seed=42)
        result = reference_pipeline(data)
        
        # Should have both positive and negative values
        assert np.any(result > 0), "Should have positive values (ask dominance)"
        assert np.any(result < 0), "Should have negative values (bid dominance)"
        
        # Should have some zero values (no volume difference)
        assert np.any(result == 0), "Should have some zero values"
        
        # Values should be distributed across the range
        assert np.max(result) > 0.1, "Should have some strong positive values"
        assert np.min(result) < -0.1, "Should have some strong negative values"
        
        # Check that normalization worked correctly
        assert abs(np.max(np.abs(result)) - 1.0) < 1e-10, "Should be normalized to [-1, 1]"


class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_minimum_data_requirements(self):
        """Test behavior with minimum required data."""
        # Test with exactly the minimum samples
        data = generate_realistic_market_data(n_samples=50000, seed=42)
        result = reference_pipeline(data)
        
        assert result.shape == (PRICE_LEVELS, TIME_BINS)
        assert np.all(np.isfinite(result))
    
    def test_handles_price_gaps(self):
        """Test that pipeline handles price gaps gracefully."""
        # Create data with price gaps (using edge case data)
        from .fixtures.sample_data import create_edge_case_data
        
        edge_data = create_edge_case_data()
        # Pad to required size
        if len(edge_data) < 50000:
            repeats = (50000 // len(edge_data)) + 1
            edge_data = pl.concat([edge_data] * repeats).head(50000)
        
        result = reference_pipeline(edge_data)
        
        # Should still produce valid output
        assert result.shape == (PRICE_LEVELS, TIME_BINS)
        assert np.all(np.isfinite(result))
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


class TestCompatibilityMatrix:
    """Test compatibility with different data formats and conditions."""
    
    @pytest.mark.parametrize("seed", [42, 123, 456, 789])
    def test_different_random_seeds(self, seed):
        """Test pipeline with different random seeds."""
        data = generate_realistic_market_data(n_samples=50000, seed=seed)
        result = reference_pipeline(data)
        
        assert result.shape == (PRICE_LEVELS, TIME_BINS)
        assert np.all(np.isfinite(result))
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
    
    @pytest.mark.parametrize("base_price", [0.5000, 0.7500, 1.0000, 1.5000])
    def test_different_price_levels(self, base_price):
        """Test pipeline with different base price levels."""
        data = generate_realistic_market_data(
            n_samples=50000, 
            base_price=base_price, 
            seed=42
        )
        result = reference_pipeline(data)
        
        assert result.shape == (PRICE_LEVELS, TIME_BINS)
        assert np.all(np.isfinite(result))
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
    
    @pytest.mark.parametrize("spread", [0.0001, 0.0005, 0.0010])
    def test_different_spreads(self, spread):
        """Test pipeline with different bid-ask spreads."""
        data = generate_realistic_market_data(
            n_samples=50000,
            base_price=0.6600,
            spread=spread,
            seed=42
        )
        result = reference_pipeline(data)
        
        assert result.shape == (PRICE_LEVELS, TIME_BINS)
        assert np.all(np.isfinite(result))
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)