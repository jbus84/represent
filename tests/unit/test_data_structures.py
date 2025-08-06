"""
Tests for represent.data_structures module.
"""

import numpy as np
from represent.data_structures import PriceLookupTable, VolumeGrid, OutputBuffer


class TestPriceLookupTable:
    """Test PriceLookupTable data structure."""
    
    def test_price_lookup_table_creation(self):
        """Test PriceLookupTable can be created."""
        mid_price = 1.25000
        table = PriceLookupTable(mid_price=mid_price)
        assert hasattr(table, '_mid_price_int')
        assert table._mid_price_int == int(mid_price)
        
    def test_price_lookup_table_operations(self):
        """Test basic price lookup operations."""
        mid_price = 1.25000
        table = PriceLookupTable(mid_price=mid_price)
        
        # Test that the table has lookup arrays
        assert hasattr(table, '_bid_indices')
        assert hasattr(table, '_price_range')
        assert table._price_range == 200  # Default range


class TestVolumeGrid:
    """Test VolumeGrid data structure."""
    
    def test_volume_grid_creation(self):
        """Test VolumeGrid can be created."""
        grid = VolumeGrid()
        assert hasattr(grid, '_grid')
        assert grid._grid.shape == (402, 500)  # PRICE_LEVELS, TIME_BINS
        
    def test_volume_grid_operations(self):
        """Test volume grid operations."""
        grid = VolumeGrid()
        
        # Test grid clear operation
        grid.clear()
        assert np.all(grid._grid == 0.0)
        
        # Test grid properties
        assert grid._grid.shape == (402, 500)
        

class TestOutputBuffer:
    """Test OutputBuffer data structure."""
    
    def test_output_buffer_creation(self):
        """Test OutputBuffer can be created."""
        buffer = OutputBuffer()
        assert hasattr(buffer, '_buffer')
        assert buffer._buffer.shape == (402, 500)
        
    def test_output_buffer_operations(self):
        """Test output buffer operations."""
        buffer = OutputBuffer()
        
        # Test buffer has temp arrays
        assert hasattr(buffer, '_temp_combined')
        assert hasattr(buffer, '_temp_abs')
        
        # Test prepare_output method exists
        assert hasattr(buffer, 'prepare_output')
        
        # Create dummy grids for testing
        ask_grid = np.random.random((402, 500))
        bid_grid = np.random.random((402, 500))
        
        # Test output preparation
        result = buffer.prepare_output(ask_grid, bid_grid)
        assert result.shape == (402, 500)

    def test_correct_signed_normalization(self):
        """
        CRITICAL TEST: Ensure normalization follows notebook approach with signed output.
        This test prevents regression to incorrect unsigned [0,1] normalization.
        """
        buffer = OutputBuffer()
        
        # Test Case 1: Ask dominance (ask > bid) should produce positive values
        ask_grid = np.ones((402, 500)) * 100  # High ask volume
        bid_grid = np.ones((402, 500)) * 50   # Lower bid volume
        
        result = buffer.prepare_output(ask_grid, bid_grid)
        
        # Should produce positive values (ask dominance)
        assert np.all(result >= 0), "Ask dominance should produce non-negative values"
        assert np.max(result) == 1.0, "Maximum should be normalized to 1.0"
        
        # Test Case 2: Bid dominance (bid > ask) should produce negative values
        ask_grid = np.ones((402, 500)) * 50   # Lower ask volume  
        bid_grid = np.ones((402, 500)) * 100  # High bid volume
        
        result = buffer.prepare_output(ask_grid, bid_grid)
        
        # Should produce negative values (bid dominance)
        assert np.all(result <= 0), "Bid dominance should produce non-positive values"
        assert np.min(result) == -1.0, "Minimum should be normalized to -1.0"
        
        # Test Case 3: Mixed scenario with both positive and negative values
        ask_grid = np.random.random((402, 500)) * 100
        bid_grid = np.random.random((402, 500)) * 100
        
        result = buffer.prepare_output(ask_grid, bid_grid)
        
        # Should have signed range
        assert result.min() >= -1.0, "Values should not go below -1.0"
        assert result.max() <= 1.0, "Values should not go above 1.0"
        assert (result.min() < 0 and result.max() > 0), "Should have both negative and positive values"
        
        # Test Case 4: Zero difference should produce zero
        ask_grid = np.ones((402, 500)) * 50
        bid_grid = np.ones((402, 500)) * 50  # Same values
        
        result = buffer.prepare_output(ask_grid, bid_grid)
        
        # All values should be zero when ask == bid
        assert np.allclose(result, 0.0), "Equal ask/bid should produce zero values"

    def test_normalization_preserves_sign_information(self):
        """
        CRITICAL TEST: Ensure sign information is preserved after normalization.
        This prevents the bug where absolute values lose directional information.
        """
        buffer = OutputBuffer()
        
        # Create a specific pattern: left half ask-dominant, right half bid-dominant
        ask_grid = np.ones((402, 500))
        bid_grid = np.ones((402, 500))
        
        # Left half: ask > bid (should be positive)
        ask_grid[:, :250] = 100
        bid_grid[:, :250] = 50
        
        # Right half: bid > ask (should be negative)  
        ask_grid[:, 250:] = 50
        bid_grid[:, 250:] = 100
        
        result = buffer.prepare_output(ask_grid, bid_grid)
        
        # Left half should be positive (ask dominance)
        left_half = result[:, :250]
        assert np.all(left_half >= 0), "Ask-dominant region should have non-negative values"
        
        # Right half should be negative (bid dominance)
        right_half = result[:, 250:]
        assert np.all(right_half <= 0), "Bid-dominant region should have non-positive values"
        
        # Values should be symmetric
        assert np.allclose(left_half, -right_half), "Symmetric inputs should produce symmetric outputs"

    def test_normalization_range_bounds(self):
        """
        CRITICAL TEST: Ensure output is always in [-1, 1] range.
        This test prevents normalization from producing values outside expected bounds.
        """
        buffer = OutputBuffer()
        
        # Test with extreme values
        test_cases = [
            (np.ones((402, 500)) * 1000, np.zeros((402, 500))),  # Max ask, no bid
            (np.zeros((402, 500)), np.ones((402, 500)) * 1000),  # No ask, max bid  
            (np.random.random((402, 500)) * 1e6, np.random.random((402, 500)) * 1e6),  # Very large values
            (np.ones((402, 500)) * 0.001, np.ones((402, 500)) * 0.002),  # Very small values
        ]
        
        for ask_grid, bid_grid in test_cases:
            result = buffer.prepare_output(ask_grid, bid_grid)
            
            assert result.min() >= -1.0, f"Minimum value {result.min()} should be >= -1.0"
            assert result.max() <= 1.0, f"Maximum value {result.max()} should be <= 1.0"
            assert not np.any(np.isnan(result)), "Result should not contain NaN values"
            assert not np.any(np.isinf(result)), "Result should not contain infinite values"

    def test_compute_normalized_difference_alias(self):
        """Test that compute_normalized_difference works identically to prepare_output."""
        buffer = OutputBuffer()
        
        ask_grid = np.random.random((402, 500)) * 100
        bid_grid = np.random.random((402, 500)) * 100
        
        result1 = buffer.prepare_output(ask_grid, bid_grid)
        result2 = buffer.compute_normalized_difference(ask_grid, bid_grid)
        
        assert np.array_equal(result1, result2), "compute_normalized_difference should be identical to prepare_output"