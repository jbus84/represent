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