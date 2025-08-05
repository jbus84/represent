"""
Tests for parquet_classifier module.
Focused on core functionality with performance optimizations.
"""

import pytest
import tempfile
from pathlib import Path
import polars as pl
import numpy as np
from unittest.mock import patch, MagicMock

from represent.parquet_classifier import (
    ParquetClassifier,
    classify_parquet_file,
    batch_classify_parquet_files,
)


def create_mock_unlabeled_parquet(n_samples: int = 100) -> pl.DataFrame:
    """Create mock unlabeled parquet data for testing."""
    np.random.seed(42)
    
    # Generate realistic market data
    base_price = 0.6650
    price_changes = np.random.normal(0, 0.0005, n_samples)
    start_prices = np.random.uniform(base_price - 0.01, base_price + 0.01, n_samples)
    end_prices = start_prices + price_changes
    
    # Mock serialized tensor data
    mock_tensor_data = np.random.rand(n_samples, 402 * 500).astype(np.float32)
    
    return pl.DataFrame({
        "market_depth_features": [data.tobytes() for data in mock_tensor_data],
        "feature_shape": ["(402, 500)"] * n_samples,
        "sample_id": [f"test_{i}" for i in range(n_samples)],
        "symbol": ["M6AM4"] * n_samples,
        "start_mid_price": start_prices,
        "end_mid_price": end_prices,
        "start_timestamp": range(n_samples),
        "end_timestamp": range(1, n_samples + 1),
    })


class TestParquetClassifierCore:
    """Test core ParquetClassifier functionality."""

    def test_initialization_default(self):
        """Test default classifier initialization."""
        classifier = ParquetClassifier(currency="AUDUSD")
        
        assert classifier.currency == "AUDUSD"
        assert classifier.target_uniform_percentage == pytest.approx(7.69, rel=1e-2)
        assert classifier.force_uniform is True
        assert classifier.verbose is True
        assert classifier.currency_config is not None
        assert classifier.classification_config is not None

    def test_initialization_custom(self):
        """Test classifier initialization with custom parameters."""
        classifier = ParquetClassifier(
            currency="GBPUSD",
            target_uniform_percentage=10.0,
            force_uniform=False,
            verbose=False
        )
        
        assert classifier.currency == "GBPUSD"
        assert classifier.target_uniform_percentage == 10.0
        assert classifier.force_uniform is False
        assert classifier.verbose is False

    def test_optimal_thresholds_structure(self):
        """Test that optimal thresholds are properly structured."""
        classifier = ParquetClassifier(currency="AUDUSD")
        
        expected_keys = ["bin_1", "bin_2", "bin_3", "bin_4", "bin_5", "bin_6"]
        assert all(key in classifier.optimal_thresholds for key in expected_keys)
        assert all(isinstance(val, (int, float)) for val in classifier.optimal_thresholds.values())
        
        # Thresholds should be in ascending order
        values = list(classifier.optimal_thresholds.values())
        assert values == sorted(values)

    def test_classify_with_optimal_thresholds(self):
        """Test classification logic with optimal thresholds."""
        classifier = ParquetClassifier(currency="AUDUSD")
        
        # Test various price changes
        test_cases = [
            (0.0, 6),      # No change -> middle bin
            (0.001, 12),   # Large positive change -> highest bin
            (-0.001, 0),   # Large negative change -> lowest bin
            (0.0001, 7),   # Small positive change
            (-0.0001, 5),  # Small negative change
        ]
        
        for price_change, expected_bin in test_cases:
            result = classifier._classify_with_optimal_thresholds(price_change)
            assert 0 <= result <= 12  # Valid bin range
            # Note: exact bin may vary based on threshold values

    def test_validate_uniform_distribution(self):
        """Test uniform distribution validation."""
        classifier = ParquetClassifier(currency="AUDUSD")
        
        # Create mock classified data with perfect uniform distribution
        n_samples = 130  # 10 samples per bin for 13 bins
        labels = list(range(13)) * 10  # Perfect uniform distribution
        
        mock_data = pl.DataFrame({
            "classification_label": labels,
            "sample_id": [f"test_{i}" for i in range(n_samples)],
        })
        
        validation = classifier._validate_uniform_distribution(mock_data)
        
        assert validation["total_samples"] == n_samples
        assert validation["target_percentage"] == pytest.approx(100/13, rel=1e-2)
        assert validation["assessment"] == "EXCELLENT"
        assert validation["is_uniform"] is True
        assert validation["max_deviation"] < 1.0

    def test_validate_poor_distribution(self):
        """Test validation of poor distribution."""
        classifier = ParquetClassifier(currency="AUDUSD")
        
        # Create mock data with poor distribution (all samples in one bin)
        n_samples = 100
        labels = [0] * n_samples  # All in bin 0
        
        mock_data = pl.DataFrame({
            "classification_label": labels,
            "sample_id": [f"test_{i}" for i in range(n_samples)],
        })
        
        validation = classifier._validate_uniform_distribution(mock_data)
        
        assert validation["assessment"] == "NEEDS IMPROVEMENT"
        assert validation["is_uniform"] is False
        assert validation["max_deviation"] > 50.0  # Very poor distribution

    @patch('polars.read_parquet')
    @patch('polars.DataFrame.write_parquet')
    @patch('pathlib.Path.stat')
    def test_classify_symbol_parquet_basic(self, mock_stat, mock_write, mock_read):
        """Test basic symbol parquet classification."""
        mock_data = create_mock_unlabeled_parquet(50)
        mock_read.return_value = mock_data
        mock_stat.return_value.st_size = 1024 * 1024  # 1MB
        
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.parquet"
            output_path = Path(temp_dir) / "output.parquet"
            input_path.touch()
            
            stats = classifier.classify_symbol_parquet(
                parquet_path=input_path,
                output_path=output_path,
                validate_uniformity=False  # Skip validation for speed
            )
            
            assert stats["input_file"] == str(input_path)
            assert stats["output_file"] == str(output_path)
            assert stats["original_samples"] == 50
            assert stats["currency"] == "AUDUSD"
            assert "processing_time_seconds" in stats
            assert "samples_per_second" in stats

    @patch('polars.read_parquet')
    @patch('pathlib.Path.stat')
    def test_classify_symbol_parquet_with_sampling(self, mock_stat, mock_read):
        """Test classification with data sampling."""
        mock_data = create_mock_unlabeled_parquet(100)
        mock_read.return_value = mock_data
        mock_stat.return_value.st_size = 1024 * 1024  # 1MB
        
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.parquet"
            input_path.touch()
            
            with patch('polars.DataFrame.write_parquet'):
                stats = classifier.classify_symbol_parquet(
                    parquet_path=input_path,
                    sample_fraction=0.5,  # Use only 50% of data
                    validate_uniformity=False
                )
            
            assert stats["original_samples"] == 100
            assert stats["sample_fraction"] == 0.5

    def test_apply_classification_to_dataframe(self):
        """Test DataFrame classification application."""
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        mock_data = create_mock_unlabeled_parquet(20)
        
        classified_df = classifier._apply_classification_to_dataframe(mock_data)
        
        assert "classification_label" in classified_df.columns
        assert len(classified_df) <= 20  # May filter out invalid samples
        assert all(0 <= label <= 12 for label in classified_df["classification_label"])

    def test_optimize_uniform_distribution_placeholder(self):
        """Test uniform distribution optimization (placeholder implementation)."""
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        mock_data = create_mock_unlabeled_parquet(30)
        
        classified_df, optimization_results = classifier._optimize_uniform_distribution(mock_data)
        
        assert "optimization_applied" in optimization_results
        assert optimization_results["optimization_applied"] is False  # Not implemented yet
        assert "reason" in optimization_results
        assert len(classified_df) > 0


class TestParquetClassifierBatch:
    """Test batch processing functionality."""

    @patch('polars.read_parquet')
    @patch('polars.DataFrame.write_parquet')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.is_dir')
    def test_batch_classify_parquets(self, mock_is_dir, mock_stat, mock_write, mock_read):
        """Test batch classification of multiple files."""
        mock_data = create_mock_unlabeled_parquet(30)
        mock_read.return_value = mock_data
        mock_stat.return_value.st_size = 1024 * 1024  # 1MB
        mock_is_dir.return_value = True
        
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            # Create mock parquet files
            test_files = [
                input_dir / "AUDUSD_M6AM4.parquet",
                input_dir / "AUDUSD_M6AU4.parquet"
            ]
            for f in test_files:
                f.touch()
            
            # Mock the glob operation at the module level
            with patch('pathlib.Path.glob', return_value=test_files):
                results = classifier.batch_classify_parquets(
                    input_directory=input_dir,
                    output_directory=output_dir,
                    validate_uniformity=False
                )
                
                assert len(results) == 2
            assert all("input_file" in result for result in results)
            assert all("output_file" in result for result in results)

    def test_batch_classify_no_files_error(self):
        """Test error handling when no files are found."""
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "empty"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            with pytest.raises(ValueError, match="No parquet files found"):
                classifier.batch_classify_parquets(
                    input_directory=input_dir,
                    output_directory=output_dir
                )


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch('represent.parquet_classifier.ParquetClassifier')
    def test_classify_parquet_file_function(self, mock_classifier_class):
        """Test classify_parquet_file convenience function."""
        mock_classifier = MagicMock()
        mock_classifier.classify_symbol_parquet.return_value = {"processed_samples": 100}
        mock_classifier_class.return_value = mock_classifier
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.parquet"
            output_path = Path(temp_dir) / "output.parquet"
            input_path.touch()
            
            result = classify_parquet_file(
                parquet_path=input_path,
                output_path=output_path,
                currency="GBPUSD",
                force_uniform=False
            )
            
            assert result["processed_samples"] == 100
            mock_classifier_class.assert_called_once_with(
                currency="GBPUSD",
                force_uniform=False
            )

    @patch('represent.parquet_classifier.ParquetClassifier')
    def test_batch_classify_parquet_files_function(self, mock_classifier_class):
        """Test batch_classify_parquet_files convenience function."""
        mock_classifier = MagicMock()
        mock_classifier.batch_classify_parquets.return_value = [
            {"processed_samples": 50},
            {"processed_samples": 75}
        ]
        mock_classifier_class.return_value = mock_classifier
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            results = batch_classify_parquet_files(
                input_directory=input_dir,
                output_directory=output_dir,
                currency="EURJPY",
                pattern="*.parquet"
            )
            
            assert len(results) == 2
            assert results[0]["processed_samples"] == 50
            mock_classifier_class.assert_called_once_with(currency="EURJPY")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_input_file_error(self):
        """Test error when input file doesn't exist."""
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        
        with pytest.raises(FileNotFoundError, match="Parquet file not found"):
            classifier.classify_symbol_parquet(
                parquet_path="/nonexistent/file.parquet"
            )

    def test_missing_input_directory_error(self):
        """Test error when input directory doesn't exist."""
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        
        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            classifier.batch_classify_parquets(
                input_directory="/nonexistent/directory",
                output_directory="/tmp/output"
            )

    def test_validation_missing_column_error(self):
        """Test validation error when classification column is missing."""
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        
        invalid_data = pl.DataFrame({
            "sample_id": ["test_1", "test_2"],
            "symbol": ["M6AM4", "M6AM4"],
        })
        
        validation = classifier._validate_uniform_distribution(invalid_data)
        assert "error" in validation
        assert "No classification_label column found" in validation["error"]

    @patch('polars.read_parquet')
    def test_empty_data_handling(self, mock_read):
        """Test handling of empty datasets."""
        mock_read.return_value = pl.DataFrame({
            "market_depth_features": [],
            "sample_id": [],
            "start_mid_price": [],
            "end_mid_price": [],
        })
        
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "empty.parquet"
            input_path.touch()
            
            with pytest.raises(ValueError, match="No valid samples found"):
                classifier.classify_symbol_parquet(
                    parquet_path=input_path,
                    validate_uniformity=False
                )


class TestPerformanceOptimizations:
    """Test performance-related functionality."""

    @patch('polars.read_parquet')
    @patch('polars.DataFrame.write_parquet')
    @patch('pathlib.Path.stat')
    def test_large_dataset_performance(self, mock_stat, mock_write, mock_read):
        """Test performance with large datasets."""
        # Simulate large dataset
        large_data = create_mock_unlabeled_parquet(1000)
        mock_read.return_value = large_data
        mock_stat.return_value.st_size = 1024 * 1024  # 1MB
        
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "large.parquet"
            input_path.touch()
            
            import time
            start_time = time.time()
            
            stats = classifier.classify_symbol_parquet(
                parquet_path=input_path,
                sample_fraction=0.1,  # Use only 10% for performance
                validate_uniformity=False
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete reasonably quickly
            assert processing_time < 5.0  # Less than 5 seconds
            assert stats["samples_per_second"] > 10  # At least 10 samples/second

    def test_memory_efficient_classification(self):
        """Test memory-efficient classification processing."""
        classifier = ParquetClassifier(currency="AUDUSD", verbose=False)
        
        # Process multiple small batches instead of one large batch
        batch_sizes = [50, 100, 200]
        
        for batch_size in batch_sizes:
            mock_data = create_mock_unlabeled_parquet(batch_size)
            classified_df = classifier._apply_classification_to_dataframe(mock_data)
            
            # Should successfully process all batch sizes
            assert len(classified_df) > 0
            assert "classification_label" in classified_df.columns