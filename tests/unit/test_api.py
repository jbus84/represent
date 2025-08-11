"""
Enhanced tests for API module coverage.
Focused on core functionality with performance optimizations.
"""

# Import API components directly to avoid torch issues
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from represent.api import RepresentAPI
    from represent.config import create_represent_config

    API_AVAILABLE = True
except ImportError as e:
    print(f"API import failed: {e}")
    API_AVAILABLE = False


def create_mock_classified_parquet(n_samples: int = 50) -> pl.DataFrame:
    """Create mock classified parquet data for testing."""
    np.random.seed(42)

    mock_tensor_data = np.random.rand(n_samples, 402 * 250).astype(np.float32)  # Use correct time_bins

    return pl.DataFrame({
        "market_depth_features": [data.tobytes() for data in mock_tensor_data],
        "classification_label": np.random.randint(0, 13, n_samples),
        "feature_shape": ["(402, 250)"] * n_samples,  # Use correct time_bins
        "sample_id": [f"test_{i}" for i in range(n_samples)],
        "symbol": ["M6AM4"] * n_samples,
    })


@pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
class TestRepresentAPI:
    """Test RepresentAPI functionality."""

    def test_api_initialization(self):
        """Test API initialization."""
        api = RepresentAPI()
        assert api is not None
        assert hasattr(api, '_available_currencies')

    def test_api_available_currencies(self):
        """Test API available currencies."""
        api = RepresentAPI()
        assert isinstance(api._available_currencies, list)
        assert len(api._available_currencies) > 0
        assert "AUDUSD" in api._available_currencies

    def test_api_dataloader_removed(self):
        """Test that dataloader methods have been removed."""
        api = RepresentAPI()

        # Test that dataloader methods no longer exist
        assert not hasattr(api, 'create_dataloader')
        assert not hasattr(api, 'load_dataset')
        assert not hasattr(api, 'create_ml_dataloader')


    def test_api_config_creation(self):
        """Test direct config creation (no longer via API method)."""
        # Test direct config creation since get_currency_config is removed
        config = create_represent_config("AUDUSD")
        assert config is not None
        assert config.currency == "AUDUSD"
        assert config.nbins > 0
        assert config.time_bins > 0

    def test_api_list_available_currencies(self):
        """Test API list_available_currencies method."""
        api = RepresentAPI()

        # Test that method exists and is callable
        assert hasattr(api, 'list_available_currencies')
        assert callable(api.list_available_currencies)

        # Test the method returns currencies
        currencies = api.list_available_currencies()
        assert isinstance(currencies, list)
        assert len(currencies) > 0
        expected_currencies = ["AUDUSD", "GBPUSD", "EURJPY", "EURUSD", "USDJPY"]
        for currency in expected_currencies:
            assert currency in currencies

    def test_api_get_package_info(self):
        """Test API get_package_info method."""
        api = RepresentAPI()

        # Test that method exists and is callable
        assert hasattr(api, 'get_package_info')
        assert callable(api.get_package_info)

        # Test the package info structure
        info = api.get_package_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "architecture" in info
        assert "available_currencies" in info
        assert "supported_features" in info
        assert "tensor_shape" in info
        assert "pipeline_approaches" in info
        assert "dynamic_features" in info

        # Test specific values
        assert isinstance(info["available_currencies"], list)
        assert isinstance(info["supported_features"], list)
        assert "volume" in info["supported_features"]
        assert isinstance(info["pipeline_approaches"], dict)
        assert "classic_3_stage" in info["pipeline_approaches"]
        assert "streamlined_2_stage" in info["pipeline_approaches"]


class TestAPIFunctions:
    """Test standalone API functions."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
    def test_dataloader_functionality_removed(self):
        """Test that dataloader functionality has been properly removed."""
        # Test that create_training_dataloader is no longer available
        import represent.api as api_module
        with pytest.raises(AttributeError):
            _ = api_module.create_training_dataloader

    @pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
    def test_api_module_imports(self):
        """Test that API module imports are working."""
        from represent.api import RepresentAPI, api, load_training_dataset

        # Test singleton API instance
        assert api is not None
        assert isinstance(api, RepresentAPI)

        # Test that load_training_dataset raises NotImplementedError
        assert callable(load_training_dataset)
        with pytest.raises(NotImplementedError):
            load_training_dataset(None)

    def test_mock_data_creation(self):
        """Test our mock data creation utility."""
        mock_data = create_mock_classified_parquet(25)

        assert len(mock_data) == 25
        assert "market_depth_features" in mock_data.columns
        assert "classification_label" in mock_data.columns
        assert "feature_shape" in mock_data.columns
        assert "sample_id" in mock_data.columns
        assert "symbol" in mock_data.columns

        # Check data types
        labels = mock_data["classification_label"].to_list()
        assert all(isinstance(label, int | np.integer) for label in labels)
        assert all(0 <= label <= 12 for label in labels)

        shapes = mock_data["feature_shape"].to_list()
        assert all(shape == "(402, 250)" for shape in shapes)  # Updated to correct time_bins

    def test_mock_data_serialization(self):
        """Test that mock data can be properly serialized/deserialized."""
        mock_data = create_mock_classified_parquet(10)

        # Test that tensor data is properly serialized as bytes
        tensor_data = mock_data["market_depth_features"][0]
        assert isinstance(tensor_data, bytes)

        # Test deserialization
        import numpy as np
        deserialized = np.frombuffer(tensor_data, dtype=np.float32)
        assert len(deserialized) == 402 * 250  # Should match expected size with correct time_bins

        # Test reshaping
        reshaped = deserialized.reshape((402, 250))
        assert reshaped.shape == (402, 250)


@pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
class TestAPIIntegration:
    """Integration tests for API functionality."""

    def test_api_configuration_access(self):
        """Test accessing configuration through API."""
        # Test that we can create configurations through the API
        config = create_represent_config("AUDUSD")
        assert config.time_bins == 250  # Expected for AUDUSD
        assert config.nbins == 13
        assert config.micro_pip_size == 0.00001

    def test_api_memory_efficiency(self):
        """Test that API doesn't consume excessive memory."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create multiple API instances
        apis = [RepresentAPI() for _ in range(10)]

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory for multiple instances
        assert memory_increase < 100  # Less than 100MB increase
        assert len(apis) == 10

        # All should be RepresentAPI instances
        for api in apis:
            assert isinstance(api, RepresentAPI)
            assert hasattr(api, '_available_currencies')

    def test_api_concurrent_usage(self):
        """Test API usage from multiple contexts."""
        import queue
        import threading

        results = queue.Queue()

        def worker(worker_id):
            try:
                api = RepresentAPI()
                currencies = api._available_currencies
                results.put((worker_id, "SUCCESS", len(currencies)))
            except Exception as e:
                results.put((worker_id, "ERROR", str(e)))

        # Create multiple threads using API
        threads = []
        for i in range(4):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Collect results
        thread_results = []
        while not results.empty():
            thread_results.append(results.get())

        assert len(thread_results) == 4
        for _worker_id, status, result in thread_results:
            assert status == "SUCCESS"
            assert isinstance(result, int)
            assert result > 0  # Should have some currencies available


class TestSymbolSplitMergeAPI:
    """Test Symbol-Split-Merge API functionality."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
    def test_build_comprehensive_symbol_datasets_method_exists(self):
        """Test that build_comprehensive_symbol_datasets method exists."""
        api = RepresentAPI()
        assert hasattr(api, 'build_comprehensive_symbol_datasets')
        assert callable(api.build_comprehensive_symbol_datasets)

    @pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
    def test_build_datasets_from_directory_method_exists(self):
        """Test that build_datasets_from_directory method exists."""
        api = RepresentAPI()
        assert hasattr(api, 'build_datasets_from_directory')
        assert callable(api.build_datasets_from_directory)

    @pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
    def test_create_dataset_builder_method_exists(self):
        """Test that create_dataset_builder method exists."""
        api = RepresentAPI()
        assert hasattr(api, 'create_dataset_builder')
        assert callable(api.create_dataset_builder)


class TestAPIUtilities:
    """Test API utility functions and helpers."""

    def test_config_consistency(self):
        """Test configuration consistency across multiple calls."""
        config1 = create_represent_config("AUDUSD")
        config2 = create_represent_config("AUDUSD")

        # Should have same values
        assert config1.time_bins == config2.time_bins
        assert config1.nbins == config2.nbins
        assert config1.micro_pip_size == config2.micro_pip_size

    def test_multiple_currency_configs(self):
        """Test creating configurations for different currencies."""
        currencies = ["AUDUSD", "EURUSD", "GBPUSD"]
        configs = {}

        for currency in currencies:
            configs[currency] = create_represent_config(currency)

        # All should be valid configs
        for _currency, config in configs.items():
            assert config is not None
            assert config.time_bins > 0
            assert config.nbins > 0
            assert config.micro_pip_size > 0

    def test_api_package_consistency(self):
        """Test that API integrates properly with the package."""
        # Test that we can import all expected API components
        from represent.api import RepresentAPI, api, load_training_dataset

        # Verify they are callable functions
        assert callable(load_training_dataset)

        # Verify that the deprecated function raises NotImplementedError
        with pytest.raises(NotImplementedError):
            load_training_dataset(None)

        # Test singleton
        assert isinstance(api, RepresentAPI)


@pytest.mark.skipif(not API_AVAILABLE, reason="API imports not available")
class TestSymbolSplitMergeAPIMethods:
    """Test Symbol-Split-Merge API methods with mocking."""

    def test_create_dataset_builder_method(self):
        """Test API create_dataset_builder method."""
        api = RepresentAPI()

        # Test that method exists and is callable
        assert hasattr(api, 'create_dataset_builder')
        assert callable(api.create_dataset_builder)

        # Test dataset builder creation with default parameters
        config = create_represent_config("AUDUSD")
        builder = api.create_dataset_builder(config, verbose=False)
        assert builder is not None
        assert hasattr(builder, 'represent_config')
        assert builder.represent_config.currency == "AUDUSD"
        assert builder.represent_config.features == ["volume"]

        # Test dataset builder creation with custom parameters
        from represent.dataset_builder import DatasetBuildConfig
        custom_config = create_represent_config("EURUSD", features=["volume", "variance"], nbins=7)
        dataset_config = DatasetBuildConfig(currency="EURUSD", features=["volume", "variance"], nbins=7)
        builder = api.create_dataset_builder(
            custom_config,
            dataset_config=dataset_config,
            verbose=False
        )
        assert builder.represent_config.currency == "EURUSD"
        assert builder.represent_config.features == ["volume", "variance"]
        assert builder.represent_config.nbins == 7

    def test_api_method_signatures(self):
        """Test that all expected API methods exist."""
        api = RepresentAPI()

        # Test existence of current symbol-split-merge API methods
        expected_methods = [
            'build_comprehensive_symbol_datasets', 'build_datasets_from_directory',
            'create_dataset_builder',
            'list_available_currencies', 'generate_classification_config',
            'calculate_global_thresholds', 'get_package_info'
        ]

        for method_name in expected_methods:
            assert hasattr(api, method_name), f"API missing method: {method_name}"
            assert callable(getattr(api, method_name)), f"Method not callable: {method_name}"

    def test_api_with_different_currencies(self):
        """Test API methods with different currencies."""
        api = RepresentAPI()

        currencies = ["AUDUSD", "EURUSD", "GBPUSD"]

        for currency in currencies:
            # Test config creation for each currency (direct creation)
            config = create_represent_config(currency)
            assert config.currency == currency
            assert config.nbins > 0

            # Test dataset builder creation for each currency
            builder = api.create_dataset_builder(config, verbose=False)
            assert builder.represent_config.currency == currency

    def test_api_parameter_validation(self):
        """Test API parameter handling."""
        api = RepresentAPI()

        # Test with various parameter combinations
        from represent.dataset_builder import DatasetBuildConfig
        config = create_represent_config(
            currency="AUDUSD",
            features=["volume", "variance"],
            nbins=13
        )
        dataset_config = DatasetBuildConfig(
            currency="AUDUSD",
            features=["volume", "variance"],
            min_symbol_samples=500,
            force_uniform=True,
            nbins=13
        )
        builder = api.create_dataset_builder(
            config,
            dataset_config=dataset_config,
            verbose=False
        )

        assert builder.represent_config.currency == "AUDUSD"
        assert builder.represent_config.features == ["volume", "variance"]
        # min_symbol_samples will be auto-updated to meet requirements, so just check it's >= 500
        assert builder.dataset_config.min_symbol_samples >= 500
        assert builder.dataset_config.force_uniform is True
        assert builder.dataset_config.nbins == 13

    def test_api_convenience_functions(self):
        """Test the module-level convenience functions."""
        from represent.api import load_training_dataset

        # Test that convenience function exists and is callable
        assert callable(load_training_dataset)

        # Test that it raises NotImplementedError with helpful message
        with pytest.raises(NotImplementedError) as exc_info:
            load_training_dataset(None)

        # Verify the error message mentions the symbol-split-merge demo
        assert "symbol_split_merge_demo.py" in str(exc_info.value)
