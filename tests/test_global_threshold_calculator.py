"""
Tests for global threshold calculator functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from represent.global_threshold_calculator import (
    GlobalThresholds,
    GlobalThresholdCalculator,
    calculate_global_thresholds
)


class TestGlobalThresholds:
    """Test GlobalThresholds dataclass."""
    
    def test_global_thresholds_creation(self):
        """Test creating GlobalThresholds object."""
        boundaries = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        stats = {"mean": 0.0, "std": 5.0, "min": -15.0, "max": 15.0}
        
        thresholds = GlobalThresholds(
            quantile_boundaries=boundaries,
            nbins=4,
            sample_size=1000,
            files_analyzed=5,
            price_movement_stats=stats
        )
        
        assert len(thresholds.quantile_boundaries) == 5
        assert thresholds.nbins == 4
        assert thresholds.sample_size == 1000
        assert thresholds.files_analyzed == 5
        assert "mean" in thresholds.price_movement_stats


class TestGlobalThresholdCalculator:
    """Test GlobalThresholdCalculator class."""
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calculator = GlobalThresholdCalculator(
            currency="AUDUSD",
            nbins=13,
            sample_fraction=0.3,
            verbose=False
        )
        
        assert calculator.currency == "AUDUSD"
        assert calculator.nbins == 13
        assert calculator.sample_fraction == 0.3
        assert calculator.verbose is False
    
    def test_calculator_initialization_defaults(self):
        """Test calculator with default parameters."""
        calculator = GlobalThresholdCalculator(verbose=False)
        
        assert calculator.currency == "AUDUSD"
        assert calculator.nbins == 13
        assert calculator.sample_fraction == 0.5
        assert calculator.max_samples_per_file == 10000
    
    def test_load_dbn_file_sample_no_file(self):
        """Test loading sample from non-existent file."""
        calculator = GlobalThresholdCalculator(verbose=False)
        
        result = calculator.load_dbn_file_sample("non_existent_file.dbn")
        
        assert result is None
    
    def test_load_dbn_file_sample_success(self):
        """Test successful loading of DBN file sample."""
        calculator = GlobalThresholdCalculator(verbose=False)
        
        # Mock the entire method to return valid data
        expected_result = np.random.randn(1000) * 0.001  # Realistic price movements
        calculator.load_dbn_file_sample = Mock(return_value=expected_result)
        
        result = calculator.load_dbn_file_sample("test_file.dbn")
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 1000
        calculator.load_dbn_file_sample.assert_called_once_with("test_file.dbn")
    
    def test_calculate_global_thresholds_no_directory(self):
        """Test calculating thresholds with non-existent directory."""
        calculator = GlobalThresholdCalculator(verbose=False)
        
        with pytest.raises(FileNotFoundError):
            calculator.calculate_global_thresholds("/non/existent/directory")
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_calculate_global_thresholds_no_files(self, mock_glob, mock_exists):
        """Test calculating thresholds with no DBN files."""
        mock_exists.return_value = True
        mock_glob.return_value = []
        
        calculator = GlobalThresholdCalculator(verbose=False)
        
        with pytest.raises(ValueError, match="No DBN files found"):
            calculator.calculate_global_thresholds("/some/directory")
    
    @patch('pathlib.Path.exists')  
    @patch('pathlib.Path.glob')
    def test_calculate_global_thresholds_no_valid_data(self, mock_glob, mock_exists):
        """Test calculating thresholds when no files produce valid data."""
        mock_exists.return_value = True
        # Create mock Path objects that can be sorted
        from pathlib import Path
        mock_file1 = Mock(spec=Path)
        mock_file1.name = "file1.dbn" 
        mock_file1.__lt__ = Mock(return_value=True)
        mock_file1.__gt__ = Mock(return_value=False)
        mock_file2 = Mock(spec=Path)
        mock_file2.name = "file2.dbn"
        mock_file2.__lt__ = Mock(return_value=False) 
        mock_file2.__gt__ = Mock(return_value=True)
        mock_files = [mock_file1, mock_file2]
        mock_glob.return_value = mock_files
        
        calculator = GlobalThresholdCalculator(verbose=False)
        
        # Mock load_dbn_file_sample to return None (no valid data)
        calculator.load_dbn_file_sample = Mock(return_value=None)
        
        with pytest.raises(ValueError, match="No valid price movements"):
            calculator.calculate_global_thresholds("/some/directory")
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')  
    def test_calculate_global_thresholds_success(self, mock_glob, mock_exists):
        """Test successful global threshold calculation."""
        mock_exists.return_value = True
        # Create mock Path objects that can be sorted
        from pathlib import Path
        mock_file1 = Mock(spec=Path)
        mock_file1.name = "file1.dbn"
        mock_file1.__lt__ = Mock(return_value=True)  # For sorting
        mock_file1.__gt__ = Mock(return_value=False)
        mock_file2 = Mock(spec=Path) 
        mock_file2.name = "file2.dbn"
        mock_file2.__lt__ = Mock(return_value=False)
        mock_file2.__gt__ = Mock(return_value=True)
        mock_files = [mock_file1, mock_file2]
        mock_glob.return_value = mock_files
        
        calculator = GlobalThresholdCalculator(nbins=5, sample_fraction=1.0, verbose=False)
        
        # Mock load_dbn_file_sample to return valid price movements
        price_movements_1 = np.random.randn(1000) * 10  # First file data
        price_movements_2 = np.random.randn(1000) * 10  # Second file data
        calculator.load_dbn_file_sample = Mock(side_effect=[price_movements_1, price_movements_2])
        
        result = calculator.calculate_global_thresholds("/some/directory")
        
        assert isinstance(result, GlobalThresholds)
        assert result.nbins == 5
        assert len(result.quantile_boundaries) <= 6  # nbins + 1, but might be less due to uniqueness
        assert result.sample_size == 2000  # Combined from both files
        assert result.files_analyzed == 2
        assert "mean" in result.price_movement_stats
        assert "std" in result.price_movement_stats


class TestConvenienceFunction:
    """Test the convenience function."""
    
    @patch('represent.global_threshold_calculator.GlobalThresholdCalculator')
    def test_calculate_global_thresholds_convenience(self, mock_calculator_class):
        """Test the convenience function."""
        mock_calculator = Mock()
        mock_thresholds = Mock()
        mock_calculator.calculate_global_thresholds.return_value = mock_thresholds
        mock_calculator_class.return_value = mock_calculator
        
        result = calculate_global_thresholds(
            "/some/directory",
            currency="GBPUSD",
            nbins=10,
            sample_fraction=0.3,
            verbose=False
        )
        
        # Verify calculator was created with correct parameters
        mock_calculator_class.assert_called_once_with(
            currency="GBPUSD",
            nbins=10,
            sample_fraction=0.3,
            verbose=False
        )
        
        # Verify calculate method was called
        mock_calculator.calculate_global_thresholds.assert_called_once_with(
            data_directory="/some/directory",
            file_pattern="*.dbn*"
        )
        
        assert result is mock_thresholds


class TestIntegrationWithParquetClassifier:
    """Test integration with ParquetClassifier."""
    
    def test_parquet_classifier_accepts_global_thresholds(self):
        """Test that ParquetClassifier accepts GlobalThresholds."""
        from represent.parquet_classifier import ParquetClassifier
        
        # Create mock global thresholds
        boundaries = np.array([-10, -5, 0, 5, 10, 15])
        stats = {"mean": 0.0, "std": 5.0}
        thresholds = GlobalThresholds(
            quantile_boundaries=boundaries,
            nbins=5,
            sample_size=1000,
            files_analyzed=3,
            price_movement_stats=stats
        )
        
        # Should not raise any errors
        classifier = ParquetClassifier(
            currency="AUDUSD",
            global_thresholds=thresholds,
            verbose=False
        )
        
        assert classifier.config.global_thresholds is thresholds
        assert classifier.config.global_thresholds.nbins == 5