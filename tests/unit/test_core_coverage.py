"""
Tests to improve code coverage for core modules.
"""
import numpy as np
import polars as pl
import pytest
from represent.core import placeholder_function, reference_pipeline
from represent.data_structures import PriceLookupTable, VolumeGrid, OutputBuffer


class TestCoreCoverage:
    """Test core module functions to improve coverage."""
    
    def test_placeholder_function(self):
        """Test placeholder function."""
        result = placeholder_function()
        assert result == "placeholder"
    
    def test_reference_pipeline_with_features(self):
        """Test reference pipeline with different feature combinations."""
        # Create test data
        data = {
            'ask_px_00': np.random.uniform(100.0, 101.0, 50000),
            'bid_px_00': np.random.uniform(99.0, 100.0, 50000),
            'ask_sz_00': np.random.uniform(10, 100, 50000),
            'bid_sz_00': np.random.uniform(10, 100, 50000),
            'ask_ct_00': np.random.randint(1, 10, 50000),
            'bid_ct_00': np.random.randint(1, 10, 50000),
        }
        
        # Add all required columns
        for i in range(1, 10):
            data[f'ask_px_{str(i).zfill(2)}'] = data['ask_px_00'] + np.random.uniform(0, 0.1, 50000)
            data[f'bid_px_{str(i).zfill(2)}'] = data['bid_px_00'] - np.random.uniform(0, 0.1, 50000)
            data[f'ask_sz_{str(i).zfill(2)}'] = np.random.uniform(5, 50, 50000)
            data[f'bid_sz_{str(i).zfill(2)}'] = np.random.uniform(5, 50, 50000)
            data[f'ask_ct_{str(i).zfill(2)}'] = np.random.randint(1, 5, 50000)
            data[f'bid_ct_{str(i).zfill(2)}'] = np.random.randint(1, 5, 50000)
        
        df = pl.DataFrame(data)
        
        # Test with None features (default)
        result = reference_pipeline(df, features=None)
        assert result.shape == (402, 500)
        
        # Test with single feature list
        result = reference_pipeline(df, features=['volume'])
        assert result.shape == (402, 500)
        
        # Test with multiple features
        result = reference_pipeline(df, features=['volume', 'variance'])
        assert result.shape == (2, 402, 500)


class TestDataStructuresCoverage:
    """Test data structures to improve coverage."""
    
    @pytest.mark.skip("RingBuffer removed in new parquet-based architecture")
    def test_ring_buffer_operations(self):
        """Test ring buffer operations comprehensively."""
        # This test is disabled as RingBuffer is no longer used in the new architecture
        pass
    
    def test_price_lookup_table_operations(self):
        """Test price lookup table operations."""
        mid_price = 100.0 * 100000  # In micro-pips
        lookup = PriceLookupTable(mid_price)
        
        # Test valid price lookups
        valid_price = int(mid_price)
        result = lookup.price_to_index(valid_price)
        assert result >= 0
        
        # Test out-of-bounds price lookups
        out_of_bounds = int(mid_price) + 10000
        result = lookup.price_to_index(out_of_bounds)
        assert result == -1
        
        # Test vectorized lookup
        prices = np.array([int(mid_price), int(mid_price) + 1, int(mid_price) - 1])
        results = lookup.vectorized_lookup(prices)
        assert len(results) == 3
        assert all(r >= 0 for r in results)
        
        # Test vectorized with out-of-bounds
        prices = np.array([int(mid_price) + 10000, int(mid_price) - 10000])
        results = lookup.vectorized_lookup(prices)
        assert all(r == -1 for r in results)
    
    def test_volume_grid_operations(self):
        """Test volume grid operations."""
        grid = VolumeGrid()
        
        # Test grid initialization
        assert grid.data.shape == (402, 500)
        
        # Test clear operation
        grid.clear()
        assert np.all(grid.data == 0)
        
        # Test setting volumes
        y_coords = np.array([10, 20, 30])
        x_coords = np.array([5, 15, 25])
        volumes = np.array([100.0, 200.0, 300.0])
        
        grid.set_volumes(y_coords, x_coords, volumes)
        assert grid.data[10, 5] == 100.0
        assert grid.data[20, 15] == 200.0
        assert grid.data[30, 25] == 300.0
        
        # Test out-of-bounds coordinates (should be filtered)
        y_coords_bad = np.array([500, -1, 10])
        x_coords_bad = np.array([600, -1, 10])
        volumes_bad = np.array([1000.0, 2000.0, 3000.0])
        
        grid.set_volumes(y_coords_bad, x_coords_bad, volumes_bad)
        assert grid.data[10, 10] == 3000.0  # Only valid coordinate should be set
        
        # Test cumulative volume calculations
        cumulative_normal = grid.get_cumulative_volume(reverse=False)
        cumulative_reverse = grid.get_cumulative_volume(reverse=True)
        
        assert cumulative_normal.shape == (402, 500)
        assert cumulative_reverse.shape == (402, 500)
    
    def test_output_buffer_operations(self):
        """Test output buffer operations."""
        buffer = OutputBuffer()
        
        # Test buffer initialization
        assert buffer.data.shape == (402, 500)
        
        # Create test data
        ask_volume = np.random.random((402, 500)) * 100
        bid_volume = np.random.random((402, 500)) * 100
        
        # Test normalized difference computation
        result = buffer.compute_normalized_difference(ask_volume, bid_volume)
        assert result.shape == (402, 500)
        assert np.all(np.abs(result) <= 1.0)  # Should be normalized
        
        # Test edge case with all zeros
        zero_ask = np.zeros((402, 500))
        zero_bid = np.zeros((402, 500))
        result = buffer.compute_normalized_difference(zero_ask, zero_bid)
        assert np.all(result == 0.0)
        
        # Test copy operation
        copy_result = buffer.get_copy()
        assert copy_result.shape == (402, 500)
        assert not np.shares_memory(copy_result, buffer.data)