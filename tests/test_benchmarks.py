"""
Benchmark tests to establish performance baselines and verify against targets.
"""
import pytest
import time
import numpy as np
import psutil
import os
from typing import Dict, Any

from .reference_implementation import reference_pipeline
from .fixtures.sample_data import generate_realistic_market_data


class PerformanceMetrics:
    """Helper class to collect and validate performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
    
    def record_timing(self, operation: str, duration_ms: float):
        """Record timing for an operation."""
        self.metrics[f"{operation}_duration_ms"] = duration_ms
    
    def record_memory(self, operation: str, memory_mb: float):
        """Record memory usage for an operation."""
        self.metrics[f"{operation}_memory_mb"] = memory_mb
    
    def record_throughput(self, operation: str, records_per_second: float):
        """Record throughput for an operation."""
        self.metrics[f"{operation}_throughput_rps"] = records_per_second
    
    def validate_against_targets(self, targets: Dict[str, float]) -> Dict[str, bool]:
        """Validate metrics against performance targets."""
        results = {}
        for metric, target in targets.items():
            actual = self.metrics.get(metric)
            if actual is not None:
                results[metric] = actual <= target
            else:
                results[metric] = False
        return results
    
    def print_summary(self):
        """Print performance metrics summary."""
        print("\n=== Performance Metrics Summary ===")
        for metric, value in self.metrics.items():
            if "duration_ms" in metric:
                print(f"{metric}: {value:.2f}ms")
            elif "memory_mb" in metric:
                print(f"{metric}: {value:.2f}MB")
            elif "throughput_rps" in metric:
                print(f"{metric}: {value:.0f} records/second")
            else:
                print(f"{metric}: {value}")


@pytest.fixture
def performance_metrics():
    """Fixture to provide performance metrics collector."""
    return PerformanceMetrics()


@pytest.fixture
def sample_data_small():
    """Small dataset for quick performance tests."""
    return generate_realistic_market_data(n_samples=5000, seed=42)


@pytest.fixture
def sample_data_full():
    """Full-size dataset for comprehensive performance tests."""
    return generate_realistic_market_data(n_samples=50000, seed=42)


class TestLatencyBenchmarks:
    """Test latency against strict performance targets."""
    
    @pytest.mark.performance
    def test_single_array_generation_latency(self, sample_data_full, performance_metrics):
        """Test single normed_abs_combined array generation latency."""
        # Warm-up
        reference_pipeline(sample_data_full)
        
        # Measure single generation
        start_time = time.perf_counter()
        result = reference_pipeline(sample_data_full)
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        performance_metrics.record_timing("array_generation", duration_ms)
        
        # Validate against target (reference implementation baseline)
        targets = {"array_generation_duration_ms": 1000.0}  # 1 second for reference
        validation = performance_metrics.validate_against_targets(targets)
        
        performance_metrics.print_summary()
        
        assert result.shape == (402, 500)
        assert validation["array_generation_duration_ms"], f"Array generation took {duration_ms:.2f}ms, target was {targets['array_generation_duration_ms']}ms"
    
    @pytest.mark.performance
    def test_batch_processing_latency(self, performance_metrics):
        """Test batch processing latency for 500-record batches."""
        batch_size = 500
        num_batches = 5
        
        durations = []
        for i in range(num_batches):
            # Generate small batch
            batch_data = generate_realistic_market_data(n_samples=batch_size, seed=42+i)
            
            # Pad to minimum size required by pipeline
            if len(batch_data) < 50000:
                batch_data = batch_data.head(50000)  # Truncate to exact size
            
            start_time = time.perf_counter()
            _ = reference_pipeline(batch_data)
            end_time = time.perf_counter()
            
            duration_ms = (end_time - start_time) * 1000
            durations.append(duration_ms)
        
        avg_duration = np.mean(durations)
        max_duration = np.max(durations)
        
        performance_metrics.record_timing("batch_processing_avg", avg_duration)
        performance_metrics.record_timing("batch_processing_max", max_duration)
        
        # Reference implementation targets (generous)
        targets = {
            "batch_processing_avg_duration_ms": 1000.0,  # 1 second average
            "batch_processing_max_duration_ms": 2000.0,  # 2 second max
        }
        validation = performance_metrics.validate_against_targets(targets)
        
        performance_metrics.print_summary()
        
        assert all(validation.values()), f"Batch processing failed targets: {validation}"


class TestThroughputBenchmarks:
    """Test throughput against performance targets."""
    
    @pytest.mark.performance
    def test_sustained_throughput(self, performance_metrics):
        """Test sustained throughput over multiple operations."""
        num_operations = 10
        records_per_operation = 5000
        
        total_records = 0
        start_time = time.perf_counter()
        
        for i in range(num_operations):
            data = generate_realistic_market_data(n_samples=records_per_operation, seed=42+i)
            # Pad to required size
            if len(data) < 50000:
                data = data.head(50000)
            
            _ = reference_pipeline(data)
            total_records += records_per_operation
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        throughput_rps = total_records / total_duration
        performance_metrics.record_throughput("sustained", throughput_rps)
        
        # Reference implementation target (modest)
        targets = {"sustained_throughput_rps": 1000.0}  # 1K records/second
        validation = performance_metrics.validate_against_targets(targets)
        
        performance_metrics.print_summary()
        
        assert validation["sustained_throughput_rps"], f"Sustained throughput was {throughput_rps:.0f} rps, target was {targets['sustained_throughput_rps']} rps"
    
    @pytest.mark.performance
    def test_peak_throughput(self, sample_data_full, performance_metrics):
        """Test peak throughput for short bursts."""
        num_bursts = 3
        durations = []
        
        for i in range(num_bursts):
            start_time = time.perf_counter()
            _ = reference_pipeline(sample_data_full)
            end_time = time.perf_counter()
            
            durations.append(end_time - start_time)
        
        best_duration = min(durations)
        peak_throughput = len(sample_data_full) / best_duration
        
        performance_metrics.record_throughput("peak", peak_throughput)
        
        targets = {"peak_throughput_rps": 5000.0}  # 5K records/second peak
        validation = performance_metrics.validate_against_targets(targets)
        
        performance_metrics.print_summary()
        
        assert validation["peak_throughput_rps"], f"Peak throughput was {peak_throughput:.0f} rps, target was {targets['peak_throughput_rps']} rps"


class TestMemoryBenchmarks:
    """Test memory usage against constraints."""
    
    @pytest.mark.performance
    def test_memory_usage_single_operation(self, sample_data_full, performance_metrics):
        """Test memory usage for single operation."""
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform operation
        result = reference_pipeline(sample_data_full)
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - baseline_memory
        
        performance_metrics.record_memory("single_operation", memory_used)
        
        # Reference implementation target (generous)
        targets = {"single_operation_memory_mb": 2048.0}  # 2GB
        validation = performance_metrics.validate_against_targets(targets)
        
        performance_metrics.print_summary()
        
        assert result.shape == (402, 500)
        assert validation["single_operation_memory_mb"], f"Memory usage was {memory_used:.2f}MB, target was {targets['single_operation_memory_mb']}MB"
    
    @pytest.mark.performance
    def test_memory_stability_over_time(self, performance_metrics):
        """Test that memory usage is stable over multiple operations."""
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_readings = []
        num_operations = 5
        
        for i in range(num_operations):
            data = generate_realistic_market_data(n_samples=10000, seed=42+i)
            # Pad to required size
            if len(data) < 50000:
                data = data.head(50000)
            
            _ = reference_pipeline(data)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = current_memory - baseline_memory
            memory_readings.append(memory_used)
        
        # Check for memory leaks (memory should not continuously grow)
        memory_growth = memory_readings[-1] - memory_readings[0]
        max_memory = max(memory_readings)
        
        performance_metrics.record_memory("stability_growth", memory_growth)
        performance_metrics.record_memory("stability_max", max_memory)
        
        targets = {
            "stability_growth_memory_mb": 100.0,  # Less than 100MB growth
            "stability_max_memory_mb": 3072.0,    # Less than 3GB max
        }
        validation = performance_metrics.validate_against_targets(targets)
        
        performance_metrics.print_summary()
        
        assert validation["stability_growth_memory_mb"], f"Memory grew by {memory_growth:.2f}MB over {num_operations} operations"
        assert validation["stability_max_memory_mb"], f"Max memory usage was {max_memory:.2f}MB"


class TestScalabilityBenchmarks:
    """Test scalability characteristics."""
    
    @pytest.mark.performance
    def test_scaling_with_data_size(self, performance_metrics):
        """Test how performance scales with input data size."""
        sizes = [1000, 5000, 10000, 25000, 50000]
        durations = []
        
        for size in sizes:
            data = generate_realistic_market_data(n_samples=size, seed=42)
            
            # Pad smaller datasets to minimum required size
            if len(data) < 50000:
                padded_data = data.head(50000)
            else:
                padded_data = data
            
            start_time = time.perf_counter()
            _ = reference_pipeline(padded_data)
            end_time = time.perf_counter()
            
            duration = (end_time - start_time) * 1000  # ms
            durations.append(duration)
        
        # Check that scaling is reasonable (not exponential)
        # For reference implementation, we expect roughly linear scaling
        ratio_5k_to_1k = durations[1] / durations[0] if durations[0] > 0 else float('inf')
        ratio_50k_to_5k = durations[4] / durations[1] if durations[1] > 0 else float('inf')
        
        performance_metrics.record_timing("scaling_1k", durations[0])
        performance_metrics.record_timing("scaling_50k", durations[4])
        performance_metrics.metrics["scaling_ratio_5k_1k"] = ratio_5k_to_1k
        performance_metrics.metrics["scaling_ratio_50k_5k"] = ratio_50k_to_5k
        
        performance_metrics.print_summary()
        
        # Scaling should be reasonable (not worse than quadratic)
        assert ratio_5k_to_1k < 25, f"5K to 1K scaling ratio too high: {ratio_5k_to_1k}"
        assert ratio_50k_to_5k < 25, f"50K to 5K scaling ratio too high: {ratio_50k_to_5k}"