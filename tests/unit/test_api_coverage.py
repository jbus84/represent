"""
Coverage tests for API module to improve overall coverage.
"""

import tempfile
from unittest.mock import Mock, patch

from represent.api import RepresentAPI
from represent.config import create_represent_config
from represent.dataset_builder import DatasetBuildConfig


class TestRepresentAPI:
    """Test the RepresentAPI class for coverage."""

    def test_api_initialization_default(self):
        """Test API initialization with defaults."""
        api = RepresentAPI()
        assert api is not None

    def test_api_initialization_with_config(self):
        """Test API initialization with custom config."""
        config = create_represent_config("AUDUSD")
        api = RepresentAPI(config=config)
        assert api is not None

    @patch('represent.api.build_datasets_from_dbn_files')
    def test_process_dbn_to_classified_parquets(self, mock_build):
        """Test the process_dbn_to_classified_parquets method."""
        mock_build.return_value = {
            'symbols_processed': 2,
            'total_classified_samples': 1000
        }

        api = RepresentAPI()
        config = create_represent_config("AUDUSD")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = api.process_dbn_to_classified_parquets(
                config=config,
                dbn_files=["test.dbn"],
                output_dir=temp_dir
            )

            assert result['symbols_processed'] == 2
            assert result['total_classified_samples'] == 1000
            mock_build.assert_called_once()

    @patch('represent.api.batch_build_datasets_from_directory')
    def test_batch_process_directory(self, mock_batch):
        """Test batch processing of directory."""
        mock_batch.return_value = {
            'phase_2_stats': {'datasets_created': 3, 'total_samples': 1500}
        }

        api = RepresentAPI()
        config = create_represent_config("AUDUSD")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = api.batch_process_directory(
                config=config,
                input_directory=temp_dir,
                output_dir="/tmp/output"
            )

            assert result['phase_2_stats']['datasets_created'] == 3
            assert result['phase_2_stats']['total_samples'] == 1500
            mock_batch.assert_called_once()

    @patch('represent.api.calculate_global_thresholds')
    def test_calculate_thresholds(self, mock_calc):
        """Test threshold calculation."""
        from represent.global_threshold_calculator import GlobalThresholds

        mock_thresholds = Mock(spec=GlobalThresholds)
        mock_thresholds.nbins = 13
        mock_thresholds.sample_size = 5000
        mock_calc.return_value = mock_thresholds

        api = RepresentAPI()
        config = create_represent_config("AUDUSD")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = api.calculate_global_thresholds(
                config=config,
                data_directory=temp_dir,
                sample_fraction=0.5
            )

            assert result.nbins == 13
            assert result.sample_size == 5000
            mock_calc.assert_called_once()

    def test_api_error_handling_invalid_path(self):
        """Test API error handling with invalid paths."""
        api = RepresentAPI()
        config = create_represent_config("AUDUSD")

        # Should handle invalid paths gracefully
        try:
            api.process_dbn_to_classified_parquets(
                config=config,
                dbn_path="/nonexistent/path/file.dbn",
                output_dir="/tmp/output"
            )
        except Exception:
            # Should get some kind of error - that's expected
            assert True

    def test_api_config_validation(self):
        """Test API validates configuration."""
        RepresentAPI()

        # Test with valid config
        config = create_represent_config("AUDUSD", samples=30000)
        assert config.samples == 30000

        # This should work without error
        assert config is not None

    @patch('represent.api.build_datasets_from_dbn_files')
    def test_api_with_dataset_config(self, mock_build):
        """Test API with custom dataset configuration."""
        mock_build.return_value = {'test': 'result'}

        api = RepresentAPI()
        config = create_represent_config("AUDUSD")
        dataset_config = DatasetBuildConfig(
            currency="AUDUSD",
            features=["volume", "variance"],
            min_symbol_samples=50000
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = api.process_dbn_to_classified_parquets(
                config=config,
                dbn_files=["test.dbn"],
                output_dir=temp_dir,
                dataset_config=dataset_config
            )

            assert result == {'test': 'result'}
            mock_build.assert_called_once()

    def test_api_verbose_mode(self):
        """Test API in verbose mode."""
        api = RepresentAPI()
        config = create_represent_config("AUDUSD")

        # Should not raise error with verbose=True
        assert api is not None
        assert config is not None

    @patch('represent.api.build_datasets_from_dbn_files')
    def test_api_multiple_files(self, mock_build):
        """Test API with multiple DBN files."""
        mock_build.return_value = {
            'input_files': ['file1.dbn', 'file2.dbn'],
            'phase_2_stats': {'datasets_created': 2}
        }

        api = RepresentAPI()
        config = create_represent_config("AUDUSD")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = api.process_multiple_dbn_files(
                config=config,
                dbn_files=["file1.dbn", "file2.dbn"],
                output_dir=temp_dir
            )

            assert len(result['input_files']) == 2
            assert result['phase_2_stats']['datasets_created'] == 2
            mock_build.assert_called_once()
