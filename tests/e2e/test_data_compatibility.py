"""
End-to-end tests for data format compatibility and validation.
Tests various data formats and edge cases with real data.
"""
import pytest
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path

from represent import process_market_data, create_processor
from represent.constants import OUTPUT_SHAPE


class TestDataFormatCompatibility:
    """Test compatibility with different data formats and sources."""
    
    @pytest.mark.e2e
    def test_databento_format_compatibility(self, sample_real_data):
        """Test compatibility with databento format data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")
            
        # Verify databento-specific columns if present
        databento_columns = ['ts_event', 'ts_recv', 'rtype', 'publisher_id', 'symbol']
        available_columns = sample_real_data.columns
        
        databento_cols_present = [col for col in databento_columns if col in available_columns]
        
        if len(databento_cols_present) > 0:
            print(f"Databento columns found: {databento_cols_present}")
            
            # Process the data
            result = process_market_data(sample_real_data)
            
            assert result.shape == OUTPUT_SHAPE
            assert np.all(np.isfinite(result))
        else:
            pytest.skip("No databento-specific columns found")
    
    @pytest.mark.e2e
    def test_column_name_variations(self, sample_real_data):
        """Test handling of different column naming conventions."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")
            
        # Test with the data as-is
        result = process_market_data(sample_real_data)
        assert result.shape == OUTPUT_SHAPE
        
        # If we have the expected columns, test renaming scenarios
        expected_ask_cols = [f'ask_px_{str(i).zfill(2)}' for i in range(10)]
        available_cols = sample_real_data.columns
        
        if all(col in available_cols for col in expected_ask_cols[:3]):  # At least first 3 levels
            print("Standard column naming detected")
        else:
            print(f"Non-standard column naming: {available_cols}")
    
    @pytest.mark.e2e
    def test_missing_levels_handling(self, sample_real_data):
        """Test handling when some price levels are missing."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")
            
        # Count available price levels
        ask_levels = len([col for col in sample_real_data.columns if col.startswith('ask_px_')])
        bid_levels = len([col for col in sample_real_data.columns if col.startswith('bid_px_')])
        
        print(f"Available levels - Ask: {ask_levels}, Bid: {bid_levels}")
        
        # Should handle any number of levels gracefully
        result = process_market_data(sample_real_data)
        assert result.shape == OUTPUT_SHAPE
        assert np.all(np.isfinite(result))
    
    @pytest.mark.e2e
    def test_data_type_handling(self, sample_real_data):
        """Test handling of different data types in real data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")
            
        # Check data types
        schema = sample_real_data.schema
        print(f"Data schema: {schema}")
        
        # Process regardless of input data types
        result = process_market_data(sample_real_data)
        
        # Output should always be float32
        assert result.dtype == np.float32
        assert result.shape == OUTPUT_SHAPE


class TestRealDataQuality:
    """Test data quality aspects with real market data."""
    
    @pytest.mark.e2e
    def test_null_value_handling(self, sample_real_data):
        """Test handling of null/NaN values in real data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")
            
        # Check for null values
        null_counts = sample_real_data.null_count()
        total_nulls = sum(null_counts.row(0))
        
        if total_nulls > 0:
            print(f"Found {total_nulls} null values in real data")
            
            # Should handle nulls gracefully
            result = process_market_data(sample_real_data)
            assert result.shape == OUTPUT_SHAPE
            assert np.all(np.isfinite(result))
        else:
            print("No null values found in real data")
            result = process_market_data(sample_real_data)
            assert result.shape == OUTPUT_SHAPE
    
    @pytest.mark.e2e
    def test_price_reasonableness(self, sample_real_data):
        """Test that real data prices are reasonable."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")
            
        # Check price columns for reasonableness
        price_cols = [col for col in sample_real_data.columns if '_px_' in col]
        
        for col in price_cols[:4]:  # Check first few levels
            if col in sample_real_data.columns:
                prices = sample_real_data[col].drop_nulls()
                if len(prices) > 0:
                    price_array = prices.to_numpy()
                    
                    # Basic sanity checks
                    assert np.all(price_array > 0), f"Non-positive prices in {col}"
                    assert np.all(price_array < 1000), f"Unreasonably high prices in {col}"
                    
                    # Check for reasonable variation
                    price_std = np.std(price_array)
                    price_mean = np.mean(price_array)
                    cv = price_std / price_mean if price_mean > 0 else 0
                    
                    # Coefficient of variation should be reasonable for market data
                    assert cv < 1.0, f"Excessive price variation in {col}: CV={cv:.3f}"
        
        # Process the data
        result = process_market_data(sample_real_data)
        assert result.shape == OUTPUT_SHAPE
    
    @pytest.mark.e2e
    def test_volume_reasonableness(self, sample_real_data):
        """Test that real data volumes are reasonable."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")
            
        # Check volume columns
        volume_cols = [col for col in sample_real_data.columns if '_sz_' in col]
        
        for col in volume_cols[:4]:  # Check first few levels
            if col in sample_real_data.columns:
                volumes = sample_real_data[col].drop_nulls()
                if len(volumes) > 0:
                    volume_array = volumes.to_numpy()
                    
                    # Basic sanity checks
                    assert np.all(volume_array >= 0), f"Negative volumes in {col}"
                    
                    # Check for reasonable maximum values
                    max_volume = np.max(volume_array)
                    assert max_volume < 1e12, f"Unreasonably high volume in {col}: {max_volume}"
        
        # Process the data
        result = process_market_data(sample_real_data)
        assert result.shape == OUTPUT_SHAPE
    
    @pytest.mark.e2e
    def test_timestamp_consistency(self, sample_real_data):
        """Test timestamp consistency in real data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")
            
        timestamp_cols = [col for col in sample_real_data.columns if 'ts_' in col]
        
        for col in timestamp_cols:
            if col in sample_real_data.columns:
                timestamps = sample_real_data[col].drop_nulls()
                if len(timestamps) > 1:
                    ts_array = timestamps.to_numpy()
                    
                    # Check for reasonable timestamp values
                    if ts_array.dtype.kind == 'M':  # datetime64 type
                        # For datetime64, check if timestamps are not null and reasonable
                        assert not np.all(pd.isna(ts_array)), f"All timestamps are null in {col}"
                        # Convert to nanoseconds for comparison
                        ts_ns = ts_array.astype('datetime64[ns]').astype(np.int64)
                        assert np.all(ts_ns > 0), f"Invalid timestamps in {col}"
                    else:
                        # For numeric timestamps
                        assert not np.all(ts_array == 0), f"All timestamps are zero in {col}"
                        assert np.all(ts_array >= 0), f"Negative timestamps in {col}"
                    
                    # Check for reasonable ordering (most should be ascending)
                    diffs = np.diff(ts_array)
                    ascending_ratio = np.sum(diffs >= 0) / len(diffs)
                    
                    # Allow some out-of-order but most should be ascending
                    # Be more lenient for delta timestamps which may have different ordering
                    min_ratio = 0.4 if 'delta' in col.lower() else 0.8
                    assert ascending_ratio > min_ratio, f"Too many out-of-order timestamps in {col}: {ascending_ratio:.1%}"
        
        # Process the data
        result = process_market_data(sample_real_data)
        assert result.shape == OUTPUT_SHAPE


class TestRealDataIntegration:
    """Integration tests with real data and different processing scenarios."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_full_pipeline_with_real_data(self, sample_real_data):
        """Test the complete pipeline with real data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")
            
        # Test with the standard 50K sample size (required by pipeline)
        # Test both direct function and processor
        result1 = process_market_data(sample_real_data)
        
        processor = create_processor()
        result2 = processor.process(sample_real_data)
        
        # Results should be identical
        assert result1.shape == result2.shape == OUTPUT_SHAPE
        assert np.allclose(result1, result2, rtol=1e-6), "API methods give different results"
        
        print(f"Successfully processed {len(sample_real_data)} real records")
    
    @pytest.mark.e2e
    def test_processor_reuse_with_real_data(self, sample_real_data):
        """Test processor reuse with real data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")
            
        processor = create_processor()
        
        # Process the same data multiple times
        results = []
        for i in range(3):
            result = processor.process(sample_real_data)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], rtol=1e-10), f"Result {i} differs from first"
        
        assert results[0].shape == OUTPUT_SHAPE
    
    @pytest.mark.e2e
    def test_concurrent_processing_simulation(self, sample_real_data):
        """Simulate concurrent processing scenarios with real data."""
        if sample_real_data is None:
            pytest.skip("Real market data not available")
            
        # Create multiple processors (simulating concurrent usage)
        processors = [create_processor() for _ in range(3)]
        
        # Process with different processors
        results = []
        for processor in processors:
            result = processor.process(sample_real_data)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], rtol=1e-10), f"Processor {i} gives different result"
        
        assert results[0].shape == OUTPUT_SHAPE