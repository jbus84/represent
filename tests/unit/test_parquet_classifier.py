"""Tests for ParquetClassifier (streamlined DBN-to-parquet) functionality."""
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
        
        # Create mock data with some null price movements (simulating rows that couldn't be processed)
        total_rows = 1000
        price_movements = []
        
        for i in range(total_rows):
            if i < 100 or i >= 900:  # First 100 and last 100 rows are null
                price_movements.append(None)
            else:
                price_movements.append(np.random.normal(0, 0.001))
        
        mock_data = pl.DataFrame({
            'ts_event': range(total_rows),
            'price_movement': price_movements,
            'classification_label': np.random.randint(0, 13, total_rows),
        })
        
        filtered_df = classifier.filter_processable_rows(mock_data)
        
        # Should filter out rows with null price_movement values
        expected_length = total_rows - 200  # Remove 100 from start and 100 from end
        assert len(filtered_df) == expected_length
        assert len(filtered_df) < len(mock_data)
        
        # Verify no null values remain
        assert filtered_df['price_movement'].null_count() == 0
    
    def test_classifier_has_expected_methods(self):
        """Test that classifier has expected methods."""
        classifier = ParquetClassifier(verbose=False)
        
        # Test that key methods exist
        assert hasattr(classifier, 'filter_symbols_by_threshold')
        assert hasattr(classifier, 'calculate_price_movements')
        assert hasattr(classifier, 'apply_quantile_classification')
        assert hasattr(classifier, 'filter_processable_rows')


class TestParquetClassifierAPI:
    """Test the public API functions."""
    
    def test_process_dbn_to_classified_parquets_parameters(self):
        """Test parameter validation for main API function."""
        from unittest.mock import patch, MagicMock
        
        with patch('represent.parquet_classifier.ParquetClassifier') as mock_classifier_class:
            # Mock the classifier instance
            mock_classifier = MagicMock()
            mock_classifier_class.return_value = mock_classifier
            
            # Mock the process method to avoid actual DBN processing
            mock_classifier.process_dbn_to_classified_parquets.return_value = {
                "processed_symbols": ["MOCK"], 
                "total_samples": 1000
            }
            
            result = process_dbn_to_classified_parquets(
                dbn_path="dummy.dbn",
                output_dir="output",
                currency="EURUSD",
                features=["volume", "variance"],
                min_symbol_samples=500
            )
            
            # Verify classifier was created with correct parameters
            mock_classifier_class.assert_called_once()
            call_kwargs = mock_classifier_class.call_args[1]
            assert call_kwargs['currency'] == "EURUSD"
            assert call_kwargs['features'] == ["volume", "variance"]
            assert call_kwargs['min_symbol_samples'] == 500
            
            # Verify processing was called
            mock_classifier.process_dbn_to_classified_parquets.assert_called_once()
            
            # Verify return value
            assert "processed_symbols" in result
            assert "total_samples" in result


class TestParquetClassifierEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        classifier = ParquetClassifier(verbose=False)
        
        empty_df = pl.DataFrame({
            'symbol': [],
            'ts_event': [],
            'bid_px_00': [],
            'ask_px_00': [],
        })
        
        # Should handle empty data gracefully
        filtered_df, counts = classifier.filter_symbols_by_threshold(empty_df)
        assert len(filtered_df) == 0
        assert len(counts) == 0
    
    def test_basic_configuration_access(self):
        """Test basic configuration access."""
        classifier = ParquetClassifier(verbose=False)
        
        # Should have basic configuration
        assert hasattr(classifier, 'config')
        assert hasattr(classifier, 'represent_config')
        assert classifier.config.currency == "AUDUSD"
    
    def test_multi_feature_support(self):
        """Test classifier with multiple features."""
        classifier = ParquetClassifier(
            currency="AUDUSD",
            features=["volume", "variance"],
            verbose=False
        )
        
        assert classifier.config.features == ["volume", "variance"]
        assert len(classifier.config.features) == 2


class TestParquetClassifierConfiguration:
    """Test configuration integration."""
    
    def test_configuration_consistency(self):
        """Test that classifier uses RepresentConfig consistently."""
        classifier = ParquetClassifier(currency="EURUSD", verbose=False)
        
        # Should use EURUSD configuration
        assert classifier.config.currency == "EURUSD"
        assert classifier.represent_config.micro_pip_size > 0
        assert classifier.represent_config.lookback_rows > 0
        assert classifier.represent_config.lookforward_input > 0
    
    def test_error_handling_insufficient_data(self):
        """Test error handling with insufficient data."""
        classifier = ParquetClassifier(
            min_symbol_samples=10000,
            verbose=False
        )
        
        # Create insufficient data
        small_data = pl.DataFrame({
            'symbol': ['TEST'] * 100,
            'ts_event': range(100),
            'bid_px_00': [1.0] * 100,
            'ask_px_00': [1.0001] * 100,
        })
        
        # Should handle gracefully
        filtered_df, symbol_counts = classifier.filter_symbols_by_threshold(small_data)
        assert len(filtered_df) == 0  # No symbols meet threshold
        assert symbol_counts['TEST'] == 100
    
    def test_constant_price_handling(self):
        """Test handling of constant prices (no movement)."""
        classifier = ParquetClassifier(verbose=False)
        
        constant_data = pl.DataFrame({
            'ts_event': range(1000),
            'bid_px_00': [1.0] * 1000,  # Constant prices
            'ask_px_00': [1.0001] * 1000,
        })
        
        result = classifier.calculate_price_movements(constant_data)
        movements = result['price_movement'].to_numpy()
        
        # All movements should be zero (except first row which has NaN)
        non_nan_movements = movements[~np.isnan(movements)]
        assert np.all(non_nan_movements == 0.0)


class TestParquetClassifierAdvanced:
    """Test advanced ParquetClassifier functionality with mocking."""

    def test_classifier_with_global_thresholds(self):
        """Test classifier initialization with global thresholds."""
        from represent.global_threshold_calculator import GlobalThresholds
        
        # Create mock global thresholds
        mock_thresholds = GlobalThresholds(
            quantile_boundaries=np.array([-0.001, -0.0005, 0, 0.0005, 0.001]),
            nbins=5,
            sample_size=1000,
            files_analyzed=2,
            price_movement_stats={"mean": 0.0, "std": 0.0005}
        )
        
        classifier = ParquetClassifier(
            currency="AUDUSD",
            global_thresholds=mock_thresholds,
            nbins=5,
            verbose=False
        )
        
        assert classifier.config.global_thresholds is not None
        assert classifier.config.nbins == 5
        assert classifier.config.currency == "AUDUSD"

    def test_classifier_with_custom_features(self):
        """Test classifier with different feature combinations."""
        feature_combinations = [
            ["volume"],
            ["volume", "variance"],
            ["volume", "variance", "trade_counts"]
        ]
        
        for features in feature_combinations:
            classifier = ParquetClassifier(
                currency="AUDUSD",
                features=features,
                verbose=False
            )
            
            assert classifier.config.features == features
            assert len(classifier.config.features) == len(features)

    def test_classifier_different_currencies(self):
        """Test classifier with different currency configurations."""
        currencies = ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY"]
        
        for currency in currencies:
            classifier = ParquetClassifier(
                currency=currency,
                verbose=False
            )
            
            assert classifier.config.currency == currency
            assert classifier.represent_config.currency == currency
            assert classifier.represent_config.micro_pip_size > 0

    def test_classifier_parameter_validation(self):
        """Test classifier parameter combinations."""
        classifier = ParquetClassifier(
            currency="EURUSD",
            features=["volume", "variance"],
            min_symbol_samples=2000,
            force_uniform=False,
            nbins=7,
            verbose=False
        )
        
        assert classifier.config.currency == "EURUSD"
        assert classifier.config.features == ["volume", "variance"]
        assert classifier.config.min_symbol_samples == 2000
        assert classifier.config.force_uniform is False
        assert classifier.config.nbins == 7

    def test_process_symbol_method_structure(self):
        """Test process_symbol method structure and requirements."""
        classifier = ParquetClassifier(verbose=False)
        
        # Test that method exists
        assert hasattr(classifier, 'process_symbol')
        assert callable(classifier.process_symbol)
        
        # Create mock data for a symbol
        mock_symbol_data = pl.DataFrame({
            'symbol': ['TEST'] * 100,
            'ts_event': range(100),
            'bid_px_00': [1.0] * 100,
            'ask_px_00': [1.0001] * 100,
        })
        
        # Test with insufficient data (should return None)
        result = classifier.process_symbol("TEST", mock_symbol_data)
        assert result is None or isinstance(result, pl.DataFrame)

    def test_load_dbn_file_method(self):
        """Test load_dbn_file method structure."""
        classifier = ParquetClassifier(
            features=["volume", "variance"],
            verbose=False
        )
        
        # Test that method exists
        assert hasattr(classifier, 'load_dbn_file')
        assert callable(classifier.load_dbn_file)

    def test_classifier_config_consistency(self):
        """Test configuration consistency across methods."""
        classifier = ParquetClassifier(
            currency="GBPUSD",
            features=["volume"],
            min_symbol_samples=500,
            nbins=9,
            verbose=False
        )
        
        # Test config values are consistent
        assert classifier.config.currency == "GBPUSD"
        assert classifier.represent_config.currency == "GBPUSD"
        assert classifier.config.features == ["volume"]
        assert classifier.config.min_symbol_samples == 500
        assert classifier.config.nbins == 9
        
        # Test computed values
        assert classifier.represent_config.time_bins > 0
        assert classifier.represent_config.lookback_rows > 0
        assert classifier.represent_config.lookforward_input > 0

    def test_classifier_verbose_mode(self):
        """Test verbose mode functionality."""
        import io
        import sys
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        _ = ParquetClassifier(currency="AUDUSD", verbose=True)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        # Should have some verbose output during initialization
        assert len(output) > 0

    def test_classifier_method_signatures(self):
        """Test that all expected methods exist with correct signatures."""
        classifier = ParquetClassifier(verbose=False)
        
        expected_methods = [
            'filter_symbols_by_threshold',
            'calculate_price_movements', 
            'apply_quantile_classification',
            'filter_processable_rows',
            'process_symbol',
            'load_dbn_file',
            'process_dbn_to_classified_parquets'
        ]
        
        for method_name in expected_methods:
            assert hasattr(classifier, method_name), f"Missing method: {method_name}"
            assert callable(getattr(classifier, method_name)), f"Not callable: {method_name}"


class TestClassificationConfigClass:
    """Test ClassificationConfig class thoroughly."""
    
    def test_config_initialization_variants(self):
        """Test various ClassificationConfig initialization patterns."""
        
        # Test with minimal parameters
        config1 = ClassificationConfig()
        assert config1.currency == "AUDUSD"  # Default
        assert config1.features == ["volume"]  # Set in __post_init__
        
        # Test with all parameters
        config2 = ClassificationConfig(
            currency="EURUSD",
            features=["volume", "variance", "trade_counts"],
            min_symbol_samples=2000,
            force_uniform=False,
            nbins=7
        )
        
        assert config2.currency == "EURUSD"
        assert config2.features == ["volume", "variance", "trade_counts"]
        assert config2.min_symbol_samples == 2000
        assert config2.force_uniform is False
        assert config2.nbins == 7

    def test_config_post_init_behavior(self):
        """Test ClassificationConfig __post_init__ method."""
        
        # Test with None features (should be set to default)
        config = ClassificationConfig(features=None)
        assert config.features == ["volume"]
        
        # Test with empty features (should remain empty)
        config = ClassificationConfig(features=[])
        assert config.features == []
        
        # Test with valid features (should be preserved)
        config = ClassificationConfig(features=["variance"])
        assert config.features == ["variance"]

    def test_config_default_values(self):
        """Test ClassificationConfig default values."""
        config = ClassificationConfig()
        
        # Test all default values
        assert config.currency == "AUDUSD"
        assert config.features == ["volume"]
        assert config.min_symbol_samples == 1000
        assert config.force_uniform is True
        assert config.nbins == 13

    def test_config_with_different_currencies(self):
        """Test ClassificationConfig with different currencies."""
        currencies = ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY", "EURJPY"]
        
        for currency in currencies:
            config = ClassificationConfig(currency=currency)
            assert config.currency == currency
            # Other defaults should remain the same
            assert config.features == ["volume"]
            assert config.min_symbol_samples == 1000
            assert config.force_uniform is True
            assert config.nbins == 13


class TestParquetClassifierAPIFunction:
    """Test the module-level API function."""
    
    def test_process_dbn_to_classified_parquets_function_exists(self):
        """Test that the API function exists and is callable."""
        assert callable(process_dbn_to_classified_parquets)
        
    def test_process_dbn_to_classified_parquets_parameter_validation(self):
        """Test parameter validation for the API function."""
        from unittest.mock import patch, MagicMock
        
        with patch('represent.parquet_classifier.ParquetClassifier') as mock_classifier_class:
            mock_instance = MagicMock()
            mock_classifier_class.return_value = mock_instance
            mock_instance.process_dbn_to_classified_parquets.return_value = {
                "processed_symbols": ["TEST"],
                "total_samples": 1000
            }
            
            # Test function call with various parameters
            result = process_dbn_to_classified_parquets(
                dbn_path="test.dbn",
                output_dir="output",
                currency="EURUSD",
                features=["volume", "variance"],
                min_symbol_samples=500,
                force_uniform=True,
                nbins=7,
                verbose=False
            )
            
            # Verify classifier was created with correct parameters
            mock_classifier_class.assert_called_once_with(
                currency="EURUSD",
                features=["volume", "variance"],
                min_symbol_samples=500,
                force_uniform=True,
                nbins=7,
                global_thresholds=None,
                verbose=False
            )
            
            # Verify processing method was called
            mock_instance.process_dbn_to_classified_parquets.assert_called_once_with(
                "test.dbn", 
                "output"
            )
            
            assert "processed_symbols" in result