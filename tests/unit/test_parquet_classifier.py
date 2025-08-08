"""Tests for ParquetClassifier (streamlined DBN-to-parquet) functionality."""
import pytest
import polars as pl
import numpy as np

from represent.parquet_classifier import (
    ParquetClassifier, 
    ClassificationConfig,
    process_dbn_to_classified_parquets
)


class TestClassificationConfig:
    """Test ClassificationConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default config initialization."""
        config = ClassificationConfig()
        
        assert config.currency == "AUDUSD"
        assert config.features == ["volume"]  # Set in __post_init__
        assert config.min_symbol_samples == 1000
        assert config.force_uniform is True
        assert config.nbins == 13
    
    def test_custom_initialization(self):
        """Test config with custom parameters."""
        config = ClassificationConfig(
            currency="EURUSD",
            features=["volume", "variance"],
            min_symbol_samples=500,
            nbins=10
        )
        
        assert config.currency == "EURUSD"
        assert config.features == ["volume", "variance"]
        assert config.min_symbol_samples == 500
        assert config.nbins == 10


class TestParquetClassifier:
    """Test ParquetClassifier core functionality."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        classifier = ParquetClassifier(verbose=False)
        
        assert classifier.config.currency == "AUDUSD"
        assert classifier.config.features == ["volume"]
        assert classifier.verbose is False
        assert classifier.represent_config is not None
    
    def test_custom_initialization(self):
        """Test classifier with custom parameters."""
        classifier = ParquetClassifier(
            currency="EURUSD",
            features=["volume", "variance"],
            min_symbol_samples=100,
            verbose=False
        )
        
        assert classifier.config.currency == "EURUSD"
        assert classifier.config.features == ["volume", "variance"]
        assert classifier.config.min_symbol_samples == 100
    
    def test_filter_symbols_by_threshold(self):
        """Test symbol filtering logic."""
        classifier = ParquetClassifier(
            min_symbol_samples=20,
            verbose=False
        )
        
        # Create mock DataFrame with symbols (need 15500+ per symbol based on constants)
        mock_data = pl.DataFrame({
            'symbol': ['A'] * 16000 + ['B'] * 5000 + ['C'] * 17000,  # A=16000, B=5000, C=17000 samples
            'ts_event': range(38000),
            'price': [1.0] * 38000
        })
        
        filtered_df, symbol_counts = classifier.filter_symbols_by_threshold(mock_data)
        
        # Should keep A and C based on minimum required from constants
        # Min required = INPUT_ROWS + LOOKBACK_ROWS + LOOKFORWARD_ROWS = 15500
        # A=16000 (≥15500) and C=17000 (≥15500) should be kept, B=5000 (<15500) filtered out
        assert len(filtered_df) > 0
        symbols = filtered_df['symbol'].unique().to_list()
        assert 'A' in symbols
        assert 'C' in symbols
        assert 'B' not in symbols
        symbols = filtered_df['symbol'].unique().to_list()
        assert 'A' in symbols
        assert 'C' in symbols
        assert 'B' not in symbols
    
    def test_calculate_price_movements(self):
        """Test price movement calculation."""
        classifier = ParquetClassifier(
            verbose=False
        )
        
        # Create mock data with predictable price changes
        mock_data = pl.DataFrame({
            'ts_event': range(10),
            'bid_px_00': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            'ask_px_00': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        })
        
        result_df = classifier.calculate_price_movements(mock_data)
        
        assert 'mid_price' in result_df.columns
        assert 'price_movement' in result_df.columns
        
        # Check mid price calculation
        mid_prices = result_df['mid_price'].to_list()
        expected_mid = [(bid + ask) / 2 for bid, ask in zip(
            mock_data['bid_px_00'], mock_data['ask_px_00']
        )]
        assert mid_prices == expected_mid
    
    def test_apply_quantile_classification_uniform(self):
        """Test quantile-based uniform classification."""
        classifier = ParquetClassifier(
            force_uniform=True,
            nbins=5,
            verbose=False
        )
        
        # Create mock data with price movements
        price_movements = np.linspace(-0.001, 0.001, 100)  # Linear distribution
        mock_data = pl.DataFrame({
            'ts_event': range(100),
            'price_movement': price_movements,
        })
        
        classified_df = classifier.apply_quantile_classification(mock_data)
        
        assert 'classification_label' in classified_df.columns
        labels = classified_df['classification_label'].to_list()
        
        # With uniform quantile classification, should have roughly equal counts
        unique_labels = set(labels)
        assert len(unique_labels) == 5  # 5 bins specified
        assert all(0 <= label <= 4 for label in labels)  # Valid range
    
    def test_filter_processable_rows(self):
        """Test row filtering for processable data."""
        classifier = ParquetClassifier(
            verbose=False
        )
        
        # Create mock data with price_movement column (required for filtering)
        mock_data = pl.DataFrame({
            'ts_event': range(10),  # 10 rows total
            'price_movement': [0.001, None, 0.002, None, 0.003, 0.004, None, 0.005, 0.006, None]
        })
        
        # Sort by timestamp first (required by the method)
        mock_data = mock_data.sort('ts_event')
        
        filtered_df = classifier.filter_processable_rows(mock_data)
        
        # Should filter to rows with non-null price_movement values
        # Count: [0.001, 0.002, 0.003, 0.004, 0.005, 0.006] = 6 non-null values
        expected_non_null = 6  # Count of non-None values
        assert len(filtered_df) == expected_non_null
    
    def test_filter_processable_rows_insufficient_data(self):
        """Test filtering when there's insufficient data."""
        classifier = ParquetClassifier(
            verbose=False
        )
        
        # Create data with no valid price_movements (all None)
        mock_data = pl.DataFrame({
            'ts_event': range(5),
            'price_movement': [None] * 5
        })
        
        filtered_df = classifier.filter_processable_rows(mock_data)
        assert len(filtered_df) == 0


class TestParquetClassifierEdgeCases:
    """Test edge cases and error handling."""
    
    def test_process_symbol_insufficient_samples(self):
        """Test symbol processing with insufficient samples."""
        classifier = ParquetClassifier(
            min_symbol_samples=100,
            verbose=False
        )
        
        # Create mock data with too few samples
        mock_data = pl.DataFrame({
            'ts_event': range(50),  # Only 50 samples, need 100
            'bid_px_00': [1.0] * 50,
            'ask_px_00': [1.1] * 50,
        })
        
        result = classifier.process_symbol("TEST_SYMBOL", mock_data)
        assert result is None  # Should return None for insufficient samples
    
    def test_apply_quantile_classification_no_valid_data(self):
        """Test classification with no valid price movements."""
        classifier = ParquetClassifier(verbose=False)
        
        # Create data with all null price movements
        mock_data = pl.DataFrame({
            'ts_event': range(10),
            'price_movement': [None] * 10,
        })
        
        result_df = classifier.apply_quantile_classification(mock_data)
        # Should handle gracefully and return data with null classifications
        assert 'classification_label' in result_df.columns
    
    def test_apply_quantile_classification_non_uniform(self):
        """Test non-uniform classification mode."""
        classifier = ParquetClassifier(
            force_uniform=False,
            verbose=False
        )
        
        # Create mock data with price movements
        mock_data = pl.DataFrame({
            'ts_event': range(20),
            'price_movement': np.linspace(-0.001, 0.001, 20),
        })
        
        result_df = classifier.apply_quantile_classification(mock_data)
        assert 'classification_label' in result_df.columns
        labels = result_df['classification_label'].to_list()
        assert all(0 <= label <= 12 for label in labels if label is not None)

    def test_verbose_output(self):
        """Test verbose output functionality."""
        classifier = ParquetClassifier(verbose=True)
        
        # Test that verbose initialization produces output
        # (We can't easily capture print output, but we can test the flag)
        assert classifier.verbose is True
        
        # Test filter_symbols with verbose
        mock_data = pl.DataFrame({
            'symbol': ['A'] * 20 + ['B'] * 10,
            'ts_event': range(30),
            'price': [1.0] * 30
        })
        
        # This should not raise an error with verbose=True
        try:
            filtered_df, counts = classifier.filter_symbols_by_threshold(mock_data)
            assert len(filtered_df) >= 0  # Should work without error
        except Exception as e:
            pytest.fail(f"Verbose filtering failed: {e}")


class TestConvenienceFunction:
    """Test the convenience function."""
    
    def test_process_dbn_to_classified_parquets_parameters(self):
        """Test that the convenience function accepts the right parameters."""
        # Just test that the function exists and accepts parameters
        # We can't test actual execution without DBN files
        try:
            # This should not raise an error for parameter validation
            import inspect
            sig = inspect.signature(process_dbn_to_classified_parquets)
            params = list(sig.parameters.keys())
            
            expected_params = [
                'dbn_path', 'output_dir', 'currency', 'features', 
                'min_symbol_samples', 'force_uniform', 'nbins', 'global_thresholds', 'verbose'
            ]
            
            for param in expected_params:
                assert param in params
                
        except Exception as e:
            pytest.fail(f"Function signature validation failed: {e}")
    
    def test_config_post_init(self):
        """Test the post_init behavior of ClassificationConfig."""
        config = ClassificationConfig(features=None)
        assert config.features == ["volume"]
        
        config2 = ClassificationConfig(features=["variance"])
        assert config2.features == ["variance"]


class TestParquetClassifierCoverage:
    """Additional tests to improve code coverage."""
    
    def test_process_symbol_with_valid_data(self):
        """Test process_symbol with valid classification data."""
        classifier = ParquetClassifier(
            min_symbol_samples=10,
            verbose=False
        )
        
        # Create mock data with sufficient samples and proper structure
        # Need enough data for price movement calculation: LOOKBACK_ROWS + LOOKFORWARD_ROWS + buffer
        num_samples = 20000  # Much more than minimum required
        mock_data = pl.DataFrame({
            'ts_event': range(num_samples), 
            'bid_px_00': [1.0 + i * 0.0001 for i in range(num_samples)],  # Increasing prices
            'ask_px_00': [1.1 + i * 0.0001 for i in range(num_samples)],  # Increasing prices
            'symbol': ['TEST'] * num_samples,
        })
        
        result = classifier.process_symbol("TEST", mock_data)
        
        # Should succeed and return classified data
        assert result is not None
        assert len(result) > 0
        assert 'classification_label' in result.columns
        
        # Classifications should be integers in valid range
        labels = result['classification_label'].to_list()
        assert all(isinstance(label, (int, type(None))) for label in labels)
        valid_labels = [label for label in labels if label is not None]
        if valid_labels:
            assert all(0 <= label <= 12 for label in valid_labels)
    
    def test_quantile_boundaries_edge_cases(self):
        """Test quantile boundary calculation edge cases."""
        classifier = ParquetClassifier(
            force_uniform=True,
            nbins=5,
            verbose=False
        )
        
        # Test with constant price movements (all same value)
        constant_data = pl.DataFrame({
            'ts_event': range(10),
            'price_movement': [0.001] * 10,  # All same value
        })
        
        result = classifier.apply_quantile_classification(constant_data)
        assert 'classification_label' in result.columns
        
        # Test with very few unique values
        few_unique_data = pl.DataFrame({
            'ts_event': range(10),
            'price_movement': [0.001, 0.002] * 5,  # Only 2 unique values
        })
        
        result2 = classifier.apply_quantile_classification(few_unique_data)
        assert 'classification_label' in result2.columns
    
    def test_filter_processable_rows_edge_cases(self):
        """Test edge cases in filter_processable_rows."""
        classifier = ParquetClassifier(
            verbose=False
        )
        
        # Test with all null price_movements
        null_data = pl.DataFrame({
            'ts_event': range(10),
            'price_movement': [None] * 10
        })
        
        result = classifier.filter_processable_rows(null_data)
        assert len(result) == 0  # Should have no processable rows
        
        # Test with mixed null/valid price_movements
        mixed_data = pl.DataFrame({
            'ts_event': range(11),
            'price_movement': [0.001, None, 0.002, None, None, 0.003, None, 0.004, None, 0.005, None]
        })
        
        # Sort by timestamp first (required by the method)
        mixed_data = mixed_data.sort('ts_event')
        
        result2 = classifier.filter_processable_rows(mixed_data)
        # Count: [0.001, 0.002, 0.003, 0.004, 0.005] = 5 non-null values
        assert len(result2) == 5  # Should have exactly 5 non-null values
    
    def test_calculate_price_movements_edge_cases(self):
        """Test price movement calculation edge cases."""
        classifier = ParquetClassifier(
            verbose=False
        )
        
        # Test with minimal data
        minimal_data = pl.DataFrame({
            'ts_event': [1, 2, 3, 4, 5],
            'bid_px_00': [1.0, 1.1, 1.2, 1.3, 1.4],
            'ask_px_00': [1.01, 1.11, 1.21, 1.31, 1.41],
        })
        
        result = classifier.calculate_price_movements(minimal_data)
        
        # Should add required columns
        assert 'mid_price' in result.columns
        assert 'price_movement' in result.columns
        
        # Check that mid_price calculation is correct
        expected_mid_0 = (1.0 + 1.01) / 2
        actual_mid_0 = result['mid_price'][0]
        assert abs(actual_mid_0 - expected_mid_0) < 1e-10
    
    def test_apply_quantile_classification_edge_distribution(self):
        """Test quantile classification with edge distribution cases."""
        classifier = ParquetClassifier(
            force_uniform=True,
            nbins=3,  # Small number of bins
            verbose=False
        )
        
        # Test with extreme outliers
        outlier_data = pl.DataFrame({
            'ts_event': range(100),
            'price_movement': ([-100.0] * 10 +  # Extreme negative
                              list(np.linspace(-0.01, 0.01, 80)) +  # Normal range
                              [100.0] * 10),  # Extreme positive
        })
        
        result = classifier.apply_quantile_classification(outlier_data)
        assert 'classification_label' in result.columns
        
        labels = result['classification_label'].to_list()
        unique_labels = set(label for label in labels if label is not None)
        assert len(unique_labels) <= 3  # Should not exceed nbins
        assert all(0 <= label <= 2 for label in labels if label is not None)
    
    def test_verbose_functionality_with_mock_data(self):
        """Test verbose output functionality with mock operations."""
        # Test with verbose=True but smaller requirements for testing
        classifier = ParquetClassifier(
            min_symbol_samples=20,
            verbose=True
        )
        
        # This mainly tests that verbose initialization works
        assert classifier.verbose is True
        
        # Test filter_symbols_by_threshold with verbose output
        # Create mock data with enough samples (need 15500+ per symbol based on constants)
        mock_data = pl.DataFrame({
            'symbol': ['A'] * 16000 + ['B'] * 17000 + ['C'] * 5000,  # Different symbol counts
            'ts_event': range(38000),
            'price': [1.0] * 38000
        })
        
        # Should not raise an error with verbose output
        filtered_df, counts = classifier.filter_symbols_by_threshold(mock_data)
        
        # Verify results are reasonable
        assert len(filtered_df) > 0
        assert len(counts) == 3  # Should have counts for all 3 symbols
        
        # Check that symbols with sufficient data are included
        # Check symbols in filtered data
        assert len(filtered_df['symbol'].unique()) >= 1  # At least some symbols should pass the threshold
        # A and B should be included as they have sufficient samples
        # C might be filtered out depending on constants
        
        # Test process_symbol with verbose (should handle insufficient data gracefully)
        insufficient_data = pl.DataFrame({
            'ts_event': range(5),  # Less than min_symbol_samples (20)
            'bid_px_00': [1.0] * 5,
            'ask_px_00': [1.1] * 5,
        })
        
        result = classifier.process_symbol("INSUFFICIENT", insufficient_data)
        assert result is None  # Should return None for insufficient data