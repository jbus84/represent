"""
Test the reference implementation against expected behavior from the notebook.
"""
import pytest
import numpy as np
import polars as pl

from .reference_implementation import (
    reference_pipeline,
    validate_output_shape,
    validate_output_range,
    convert_prices_to_micro_pips,
    create_time_bins,
    calculate_mid_price,
    create_price_bins,
    PRICE_LEVELS,
    TIME_BINS,
    MICRO_PIP_SIZE,
)
from .fixtures.sample_data import (
    generate_realistic_market_data,
    create_simple_test_data,
    create_edge_case_data,
)


class TestReferenceImplementation:
    """Test the reference implementation functions."""
    
    def test_convert_prices_to_micro_pips(self):
        """Test price conversion to micro-pip format."""
        simple_data = create_simple_test_data()
        converted = convert_prices_to_micro_pips(simple_data)
        
        # Check that prices are now integers
        assert converted["ask_px_00"].dtype == pl.Int64
        assert converted["bid_px_00"].dtype == pl.Int64
        
        # Check conversion math (0.6600 * 100000 = 66000)
        expected_ask = int(0.6600 / MICRO_PIP_SIZE)
        assert converted["ask_px_00"][0] == expected_ask
    
    def test_create_time_bins(self):
        """Test time bin creation."""
        simple_data = create_simple_test_data()
        with_bins = create_time_bins(simple_data, samples=1000)
        
        assert "tick_bin" in with_bins.columns
        assert with_bins["tick_bin"].max() == 9  # 1000 / 100 - 1
        assert with_bins["tick_bin"].min() == 0
    
    def test_calculate_mid_price(self):
        """Test mid price calculation."""
        simple_data = create_simple_test_data()
        converted = convert_prices_to_micro_pips(simple_data)
        mid_price = calculate_mid_price(converted)
        
        # Should be approximately (66000 + 65998) / 2 = 65999
        assert abs(mid_price - 65999) < 1
    
    def test_create_price_bins(self):
        """Test price bin creation."""
        mid_price = 65999.0
        price_bins, price_to_index = create_price_bins(mid_price)
        
        assert len(price_bins) == PRICE_LEVELS
        assert len(price_to_index) == PRICE_LEVELS
        
        # Check that mid price is roughly in the middle
        mid_index = price_to_index.get(int(mid_price))
        assert mid_index is not None
        assert 180 <= mid_index <= 220  # Should be around index 201
    
    def test_reference_pipeline_shape(self):
        """Test that reference pipeline produces correct output shape."""
        realistic_data = generate_realistic_market_data(n_samples=50000)
        result = reference_pipeline(realistic_data)
        
        assert validate_output_shape(result)
        assert result.shape == (PRICE_LEVELS, TIME_BINS)
    
    def test_reference_pipeline_range(self):
        """Test that reference pipeline produces values in correct range."""
        realistic_data = generate_realistic_market_data(n_samples=50000)
        result = reference_pipeline(realistic_data)
        
        assert validate_output_range(result)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
    
    def test_reference_pipeline_deterministic(self):
        """Test that reference pipeline is deterministic."""
        data1 = generate_realistic_market_data(seed=42)
        data2 = generate_realistic_market_data(seed=42)
        
        result1 = reference_pipeline(data1)
        result2 = reference_pipeline(data2)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_reference_pipeline_different_seeds(self):
        """Test that different seeds produce different results."""
        data1 = generate_realistic_market_data(seed=42)
        data2 = generate_realistic_market_data(seed=123)
        
        result1 = reference_pipeline(data1)
        result2 = reference_pipeline(data2)
        
        # Results should be different
        assert not np.array_equal(result1, result2)
    
    def test_edge_cases(self):
        """Test reference implementation with edge case data."""
        edge_data = create_edge_case_data()
        # Pad to minimum required samples
        if len(edge_data) < 50000:
            repeats = (50000 // len(edge_data)) + 1
            edge_data = pl.concat([edge_data] * repeats).head(50000)
        
        result = reference_pipeline(edge_data)
        
        # Should still produce valid output despite edge cases
        assert validate_output_shape(result)
        assert validate_output_range(result)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestPerformance:
    """Performance tests for reference implementation."""
    
    @pytest.mark.performance
    def test_reference_pipeline_performance(self):
        """Test reference pipeline performance against targets."""
        import time
        
        realistic_data = generate_realistic_market_data(n_samples=50000)
        
        # Warm-up run
        reference_pipeline(realistic_data)
        
        # Timed run
        start_time = time.perf_counter()
        result = reference_pipeline(realistic_data)
        end_time = time.perf_counter()
        
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        # This is the reference implementation, so we set a generous baseline
        # The optimized implementation should beat this significantly
        assert processing_time < 5000  # 5 seconds for reference
        assert validate_output_shape(result)
        
        print(f"Reference pipeline took {processing_time:.2f}ms")
    
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage of reference implementation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        realistic_data = generate_realistic_market_data(n_samples=50000)
        result = reference_pipeline(realistic_data)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Should use less than 1GB for reference implementation
        assert memory_used < 1024  # MB
        assert validate_output_shape(result)
        
        print(f"Reference implementation used {memory_used:.2f}MB")


class TestCompatibility:
    """Test compatibility with notebook expectations."""
    
    def test_column_names_match_notebook(self):
        """Test that our column names match the notebook exactly."""
        from .reference_implementation import (
            ASK_PRICE_COLUMNS, BID_PRICE_COLUMNS,
            BID_VOL_COLUMNS,
            ASK_COUNT_COLUMNS
        )
        
        # Verify column naming pattern from notebook
        assert ASK_PRICE_COLUMNS[0] == "ask_px_00"
        assert ASK_PRICE_COLUMNS[9] == "ask_px_09"
        assert BID_PRICE_COLUMNS[0] == "bid_px_00"
        assert BID_VOL_COLUMNS[5] == "bid_sz_05"
        assert ASK_COUNT_COLUMNS[7] == "ask_ct_07"
    
    def test_constants_match_notebook(self):
        """Test that constants match notebook values."""
        assert MICRO_PIP_SIZE == 0.00001
        assert PRICE_LEVELS == 402
        assert TIME_BINS == 500
    
    def test_output_similar_to_notebook_example(self):
        """Test that output characteristics are similar to notebook."""
        # Use similar parameters to notebook
        realistic_data = generate_realistic_market_data(
            n_samples=50000,
            base_price=0.6600,  # Similar to AUDUSD
            seed=42
        )
        
        result = reference_pipeline(realistic_data)
        
        # Basic sanity checks based on notebook output
        assert result.shape == (402, 500)
        assert not np.all(result == 0)  # Should have non-zero values
        assert np.any(result > 0)  # Should have positive values (ask dominance)
        assert np.any(result < 0)  # Should have negative values (bid dominance)
        
        # Check that there's reasonable variation
        assert np.std(result) > 0.01  # Some variation in the data