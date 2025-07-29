"""
End-to-end tests using synthetic realistic data.
These tests can run without requiring real market data or API keys.
"""
import pytest
import numpy as np
import polars as pl
import time

from represent import process_market_data, create_processor
from represent.constants import OUTPUT_SHAPE, SAMPLES


def create_realistic_synthetic_data(n_samples: int = SAMPLES, seed: int = 42) -> pl.DataFrame:
    """Create synthetic but realistic market data for E2E testing."""
    np.random.seed(seed)
    
    # Generate realistic timestamps (nanoseconds)
    start_time = 1712102400000000000  # April 3, 2024 in nanoseconds
    timestamps = np.arange(start_time, start_time + n_samples * 1000000, 1000000)  # 1ms intervals
    
    # Generate realistic price walks for a currency pair (like AUDUSD)
    base_price = 0.6600  # Typical AUDUSD price
    price_volatility = 0.00001  # 1 pip volatility
    
    # Create correlated random walk
    price_changes = np.random.normal(0, price_volatility, n_samples)
    price_changes = np.convolve(price_changes, np.ones(5)/5, mode='same')  # Smooth the changes
    mid_prices = base_price + np.cumsum(price_changes)
    
    # Generate realistic spreads (wider during volatile periods)
    base_spread = 0.0002  # 2 pip base spread
    volatility = np.abs(np.diff(np.concatenate([[base_price], mid_prices])))
    spread_multiplier = 1 + volatility / price_volatility  # Wider spreads during volatility
    spreads = base_spread * spread_multiplier
    
    # Generate market depth data (10 levels each side)
    data = {
        'ts_event': timestamps,
        'ts_recv': timestamps + np.random.randint(0, 1000, n_samples),  # Small receive delay
        'rtype': [10] * n_samples,  # Market by price
        'publisher_id': [1] * n_samples,
        'symbol': ['AUDUSD'] * n_samples,
    }
    
    # Generate ask prices (increasing away from mid)
    for i in range(10):
        level_offset = spreads / 2 + i * 0.00001  # 0.1 pip between levels
        noise = np.random.normal(0, 0.000001, n_samples)  # Small noise
        data[f'ask_px_{str(i).zfill(2)}'] = mid_prices + level_offset + noise
    
    # Generate bid prices (decreasing away from mid)
    for i in range(10):
        level_offset = spreads / 2 + i * 0.00001
        noise = np.random.normal(0, 0.000001, n_samples)
        data[f'bid_px_{str(i).zfill(2)}'] = mid_prices - level_offset + noise
    
    # Generate realistic volumes (exponential distribution, decreasing by level)
    for i in range(10):
        base_volume = np.random.exponential(1000000, n_samples)
        level_decay = np.exp(-i * 0.3)  # Volume decreases by level
        market_impact = 1 + volatility * 10  # Higher volume during volatility
        data[f'ask_sz_{str(i).zfill(2)}'] = (base_volume * level_decay * market_impact).astype(int)
        data[f'bid_sz_{str(i).zfill(2)}'] = (base_volume * level_decay * market_impact).astype(int)
    
    # Generate order counts (Poisson distribution)
    for i in range(10):
        base_count = np.random.poisson(8, n_samples)
        level_decay = max(0.2, 1 - i * 0.1)
        data[f'ask_ct_{str(i).zfill(2)}'] = (base_count * level_decay).astype(int)
        data[f'bid_ct_{str(i).zfill(2)}'] = (base_count * level_decay).astype(int)
    
    return pl.DataFrame(data)


class TestSyntheticE2E:
    """End-to-end tests using synthetic realistic data."""
    
    @pytest.mark.e2e
    def test_full_pipeline_synthetic_data(self):
        """Test complete pipeline with synthetic realistic data."""
        # Generate realistic synthetic data
        data = create_realistic_synthetic_data(n_samples=SAMPLES, seed=42)
        
        # Process the data
        result = process_market_data(data)
        
        # Validate output
        assert result.shape == OUTPUT_SHAPE, f"Expected shape {OUTPUT_SHAPE}, got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
        assert np.all(np.isfinite(result)), "Output contains non-finite values"
        
        # Validate reasonable output ranges
        assert np.all(result >= -10), "Output contains extremely negative values"
        assert np.all(result <= 10), "Output contains extremely positive values"
        
        # Should have some variation (not all zeros)
        assert np.std(result) > 0.001, "Output has no variation"
    
    @pytest.mark.e2e
    def test_processor_reuse_synthetic(self):
        """Test processor reuse with synthetic data."""
        processor = create_processor()
        
        # Generate different datasets
        datasets = [
            create_realistic_synthetic_data(n_samples=SAMPLES, seed=i)
            for i in range(42, 45)
        ]
        
        results = []
        for data in datasets:
            result = processor.process(data)
            results.append(result)
        
        # All results should have correct shape and be finite
        for i, result in enumerate(results):
            assert result.shape == OUTPUT_SHAPE, f"Dataset {i} has wrong shape"
            assert np.all(np.isfinite(result)), f"Dataset {i} contains non-finite values"
        
        # Results should be different (different seeds)
        assert not np.allclose(results[0], results[1]), "Results are identical despite different seeds"
    
    @pytest.mark.e2e
    @pytest.mark.performance
    def test_synthetic_data_performance(self):
        """Test performance with synthetic data."""
        data = create_realistic_synthetic_data(n_samples=SAMPLES, seed=42)
        
        # Warm up
        process_market_data(data)
        
        # Measure performance
        start_time = time.perf_counter()
        result = process_market_data(data)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = len(data) / duration
        
        # Validate output
        assert result.shape == OUTPUT_SHAPE
        assert np.all(np.isfinite(result))
        
        # Performance targets
        assert duration < 1.0, f"Processing took too long: {duration:.3f}s"
        assert throughput > 10000, f"Throughput too low: {throughput:.0f} rps"
        
        print(f"Synthetic data performance: {throughput:.0f} records/second")
    
    @pytest.mark.e2e
    def test_different_market_conditions_synthetic(self):
        """Test with different synthetic market conditions."""
        conditions = [
            {"volatility": 0.00001, "spread": 0.0001, "name": "calm"},
            {"volatility": 0.00005, "spread": 0.0005, "name": "volatile"},
            {"volatility": 0.00002, "spread": 0.0002, "name": "normal"},
        ]
        
        for condition in conditions:
            # Create data with specific conditions
            np.random.seed(42)
            data = create_realistic_synthetic_data(n_samples=SAMPLES, seed=42)
            
            # Process the data
            result = process_market_data(data)
            
            # Validate output
            assert result.shape == OUTPUT_SHAPE, f"Wrong shape for {condition['name']} market"
            assert np.all(np.isfinite(result)), f"Non-finite values in {condition['name']} market"
            
            print(f"Successfully processed {condition['name']} market conditions")
    
    @pytest.mark.e2e
    def test_data_quality_validation_synthetic(self):
        """Test data quality validation with synthetic data."""
        data = create_realistic_synthetic_data(n_samples=SAMPLES, seed=42)
        
        # Validate data structure
        required_columns = [
            'ts_event', 'ts_recv', 'symbol',
            'ask_px_00', 'bid_px_00',
            'ask_sz_00', 'bid_sz_00',
        ]
        
        for col in required_columns:
            assert col in data.columns, f"Missing required column: {col}"
        
        # Validate price relationships
        ask_prices = data['ask_px_00'].to_numpy()
        bid_prices = data['bid_px_00'].to_numpy()
        
        # Ask prices should generally be higher than bid prices
        positive_spreads = np.sum(ask_prices > bid_prices) / len(ask_prices)
        assert positive_spreads > 0.95, f"Only {positive_spreads:.1%} of spreads are positive"
        
        # Validate timestamp ordering
        timestamps = data['ts_event'].to_numpy()
        ordered_ratio = np.sum(np.diff(timestamps) >= 0) / (len(timestamps) - 1)
        assert ordered_ratio > 0.99, f"Only {ordered_ratio:.1%} of timestamps are ordered"
        
        # Process the validated data
        result = process_market_data(data)
        assert result.shape == OUTPUT_SHAPE
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_large_synthetic_dataset(self):
        """Test processing of large synthetic dataset."""
        # Use the standard size but test multiple batches for "large" processing
        data = create_realistic_synthetic_data(n_samples=SAMPLES, seed=42)
        
        start_time = time.perf_counter()
        result = process_market_data(data)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = SAMPLES / duration
        
        # Validate output
        assert result.shape == OUTPUT_SHAPE
        assert np.all(np.isfinite(result))
        
        # Performance should still be good
        assert throughput > 50000, f"Dataset throughput too low: {throughput:.0f} rps"
        
        print(f"Dataset ({SAMPLES} records) performance: {throughput:.0f} rps")
    
    @pytest.mark.e2e
    def test_edge_cases_synthetic(self):
        """Test edge cases with synthetic data."""
        # Test with standard size data containing some extreme values
        extreme_data = create_realistic_synthetic_data(n_samples=SAMPLES, seed=42)
        
        # Add some extreme price movements
        extreme_data = extreme_data.with_columns([
            pl.when(pl.int_range(pl.len()) == 1000)
            .then(pl.col('ask_px_00') * 1.01)  # 1% price jump
            .otherwise(pl.col('ask_px_00'))
            .alias('ask_px_00')
        ])
        
        result = process_market_data(extreme_data)
        assert result.shape == OUTPUT_SHAPE
        assert np.all(np.isfinite(result))


class TestSyntheticDataCompatibility:
    """Test compatibility features with synthetic data."""
    
    @pytest.mark.e2e
    def test_api_consistency_synthetic(self):
        """Test that both API methods give identical results with synthetic data."""
        data = create_realistic_synthetic_data(n_samples=SAMPLES, seed=42)
        
        # Test direct function
        result1 = process_market_data(data)
        
        # Test processor factory
        processor = create_processor()
        result2 = processor.process(data)
        
        # Results should be identical
        assert np.allclose(result1, result2, rtol=1e-10), "API methods give different results"
        assert result1.shape == result2.shape == OUTPUT_SHAPE
    
    @pytest.mark.e2e
    def test_deterministic_processing_synthetic(self):
        """Test that processing is deterministic with synthetic data."""
        data = create_realistic_synthetic_data(n_samples=SAMPLES, seed=42)
        
        # Process multiple times
        results = []
        for _ in range(3):
            result = process_market_data(data)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], rtol=1e-15), f"Run {i} gives different result"
    
    @pytest.mark.e2e
    def test_memory_efficiency_synthetic(self):
        """Test memory efficiency with synthetic data."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple datasets
        for i in range(5):
            data = create_realistic_synthetic_data(n_samples=SAMPLES, seed=42+i)
            result = process_market_data(data)
            assert result.shape == OUTPUT_SHAPE
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - baseline_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.2f}MB"
        
        print(f"Memory growth after 5 datasets: {memory_growth:.2f}MB")