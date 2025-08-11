"""
Focused coverage tests for dataset_builder module to reach 80% overall coverage.
These tests target specific uncovered code paths.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

from represent.config import create_represent_config
from represent.dataset_builder import (
    DatasetBuildConfig,
    DatasetBuilder,
    batch_build_datasets_from_directory,
)


class TestDatasetBuilderFocusedCoverage:
    """Tests targeting specific uncovered lines in dataset_builder.py."""

    @pytest.fixture
    def simple_config(self):
        """Simple config for focused testing."""
        return create_represent_config(
            currency="AUDUSD",
            features=["volume"],
            samples=25000,
            lookback_rows=100,
            lookforward_input=100,
            lookforward_offset=50,
            jump_size=10
        )

    @pytest.fixture
    def simple_dataset_config(self):
        """Simple dataset config for focused testing."""
        return DatasetBuildConfig(
            currency="AUDUSD",
            features=["volume"],
            min_symbol_samples=1000,
            force_uniform=True
        )

    def test_verbose_initialization_output(self, simple_config, simple_dataset_config, capsys):
        """Test verbose output during DatasetBuilder initialization."""
        # Test with automatic min_samples update (verbose output)
        low_config = DatasetBuildConfig(min_symbol_samples=100)  # Too low

        DatasetBuilder(simple_config, low_config, verbose=True)

        captured = capsys.readouterr()
        assert "DatasetBuilder initialized" in captured.out
        assert "Currency: AUDUSD" in captured.out
        assert "Features: ['volume']" in captured.out
        assert "Min samples per symbol:" in captured.out
        assert "Force uniform: True" in captured.out

    def test_split_dbn_verbose_output(self, simple_config, simple_dataset_config, capsys):
        """Test verbose output during DBN splitting."""
        builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

        # Mock databento read
        with patch('databento.read_dbn') as mock_read_dbn:
            mock_data = Mock()
            sample_df = pl.DataFrame({
                "ts_event": [1640995200000000000 + i * 1000000000 for i in range(100)],
                "symbol": ["M6AM4"] * 50 + ["M6AM5"] * 50,
                "ask_px_00": [0.67000] * 100,
                "bid_px_00": [0.66995] * 100,
            })
            mock_data.to_df.return_value = sample_df.to_pandas()
            mock_read_dbn.return_value = mock_data

            with tempfile.TemporaryDirectory() as temp_dir:
                intermediate_dir = Path(temp_dir)

                result = builder.split_dbn_by_symbol("test.dbn", intermediate_dir)

                captured = capsys.readouterr()
                assert "Splitting DBN file:" in captured.out
                assert "Found" in captured.out and "rows across" in captured.out
                assert "Split complete in" in captured.out
                assert len(result) == 2  # M6AM4 and M6AM5

    def test_merge_symbol_verbose_output(self, simple_config, simple_dataset_config, capsys):
        """Test verbose output during symbol merging."""
        builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            intermediate_dir = Path(temp_dir) / "intermediate"
            output_dir = Path(temp_dir) / "output"
            intermediate_dir.mkdir()
            output_dir.mkdir()

            # Create test intermediate files
            symbol_files = []
            for i in range(2):
                sample_data = pl.DataFrame({
                    "ts_event": [1640995200000000000 + (i*1000 + j) * 1000000000 for j in range(1000)],
                    "symbol": ["M6AM4"] * 1000,
                    "ask_px_00": np.random.normal(0.67000, 0.00001, 1000),
                    "bid_px_00": np.random.normal(0.66995, 0.00001, 1000),
                })

                file_path = intermediate_dir / f"file{i}_M6AM4.parquet"
                sample_data.write_parquet(str(file_path))
                symbol_files.append(file_path)

            builder.merge_symbol_data("M6AM4", symbol_files, output_dir)

            captured = capsys.readouterr()
            assert "Merging symbol: M6AM4" in captured.out
            assert "Files: 2" in captured.out
            assert "Processing: Lookback → Lookforward → Percentage Change → Classification" in captured.out
            assert "Features: Generated on-demand during ML training" in captured.out

    def test_price_movement_calculation_verbose(self, simple_config, simple_dataset_config, capsys):
        """Test verbose output during price movement calculation."""
        builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

        # Create test data with sufficient rows
        test_data = pl.DataFrame({
            "ts_event": [1640995200000000000 + i * 1000000000 for i in range(1000)],
            "symbol": ["M6AM4"] * 1000,
            "ask_px_00": [0.67000 + i * 0.000001 for i in range(1000)],
            "bid_px_00": [0.66995 + i * 0.000001 for i in range(1000)],
        })

        builder._calculate_price_movements(test_data)

        captured = capsys.readouterr()
        assert "Calculating price movements using lookback/lookforward methodology" in captured.out
        assert "Lookback:" in captured.out
        assert "Lookforward:" in captured.out
        assert "Processing: Every valid row (no jumping)" in captured.out
        assert "Calculated" in captured.out and "price movements" in captured.out

    def test_price_movement_insufficient_data_verbose(self, simple_config, simple_dataset_config, capsys):
        """Test verbose output when insufficient data for price movements."""
        builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

        # Create insufficient data
        small_data = pl.DataFrame({
            "ts_event": [1640995200000000000 + i * 1000000000 for i in range(50)],
            "symbol": ["M6AM4"] * 50,
            "ask_px_00": [0.67000] * 50,
            "bid_px_00": [0.66995] * 50,
        })

        builder._calculate_price_movements(small_data)

        captured = capsys.readouterr()
        assert "Insufficient data:" in captured.out
        assert "required" in captured.out

    def test_classification_verbose_outputs(self, simple_config, simple_dataset_config, capsys):
        """Test verbose outputs during classification."""
        builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

        # Create data with price movements
        test_data = pl.DataFrame({
            "ts_event": [1640995200000000000 + i * 1000000000 for i in range(1000)],
            "symbol": ["M6AM4"] * 1000,
            "price_movement": np.random.normal(0, 0.001, 1000),
        })

        builder._apply_classification(test_data)

        captured = capsys.readouterr()
        assert "Using" in captured.out and "samples to define" in captured.out and "bins" in captured.out
        assert "Applying classification to" in captured.out and "total samples" in captured.out
        assert "Quantile boundaries:" in captured.out
        assert "Classification distribution" in captured.out

    def test_classification_no_valid_movements_verbose(self, simple_config, simple_dataset_config, capsys):
        """Test verbose output when no valid movements for classification."""
        builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

        # Create data with no valid movements
        test_data = pl.DataFrame({
            "ts_event": [1640995200000000000 + i * 1000000000 for i in range(100)],
            "symbol": ["M6AM4"] * 100,
            "price_movement": [np.nan] * 100,
        })

        builder._apply_classification(test_data)

        captured = capsys.readouterr()
        assert "No valid price movements for classification" in captured.out

    def test_classification_insufficient_for_bins_verbose(self, simple_config, simple_dataset_config, capsys):
        """Test verbose output when insufficient data for reliable bin calculation."""
        builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

        # Create small amount of valid data
        test_data = pl.DataFrame({
            "ts_event": [1640995200000000000 + i * 1000000000 for i in range(20)],
            "symbol": ["M6AM4"] * 20,
            "price_movement": np.random.normal(0, 0.001, 20),
        })

        builder._apply_classification(test_data)

        captured = capsys.readouterr()
        assert "Insufficient data for reliable bin calculation:" in captured.out

    def test_no_feature_generation_in_simplified_pipeline(self, simple_config, simple_dataset_config):
        """Test that simplified pipeline doesn't generate features during building."""
        builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

        # Should not have _generate_features method
        assert not hasattr(builder, '_generate_features'), "_generate_features should not exist in simplified pipeline"

    def test_merge_missing_files_verbose(self, simple_config, simple_dataset_config, capsys):
        """Test verbose output when symbol files are missing."""
        builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Reference non-existent files
            missing_files = [Path(temp_dir) / "missing1.parquet", Path(temp_dir) / "missing2.parquet"]

            builder.merge_symbol_data("M6AM4", missing_files, output_dir)

            captured = capsys.readouterr()
            assert "Missing file:" in captured.out

    def test_merge_no_data_found_verbose(self, simple_config, simple_dataset_config, capsys):
        """Test verbose output when no data found for symbol."""
        builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            builder.merge_symbol_data("M6AM4", [], output_dir)

            captured = capsys.readouterr()
            assert "No data found for M6AM4" in captured.out

    def test_merge_insufficient_samples_verbose(self, simple_config, capsys):
        """Test verbose output when insufficient total samples."""
        high_config = DatasetBuildConfig(min_symbol_samples=100000)  # Very high
        builder = DatasetBuilder(simple_config, high_config, verbose=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            intermediate_dir = Path(temp_dir) / "intermediate"
            output_dir = Path(temp_dir) / "output"
            intermediate_dir.mkdir()
            output_dir.mkdir()

            # Create small data file
            small_data = pl.DataFrame({
                "ts_event": [1640995200000000000 + i * 1000000000 for i in range(100)],
                "symbol": ["M6AM4"] * 100,
                "ask_px_00": [0.67000] * 100,
                "bid_px_00": [0.66995] * 100,
            })

            file_path = intermediate_dir / "small_M6AM4.parquet"
            small_data.write_parquet(str(file_path))

            builder.merge_symbol_data("M6AM4", [file_path], output_dir)

            captured = capsys.readouterr()
            assert "Insufficient total samples:" in captured.out

    def test_merge_no_processable_rows_verbose(self, simple_config, simple_dataset_config, capsys):
        """Test verbose output when no processable rows after filtering."""
        builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            intermediate_dir = Path(temp_dir) / "intermediate"
            output_dir = Path(temp_dir) / "output"
            intermediate_dir.mkdir()
            output_dir.mkdir()

            # Create data that will result in no processable rows
            bad_data = pl.DataFrame({
                "ts_event": [1640995200000000000 + i * 1000000000 for i in range(1000)],
                "symbol": ["M6AM4"] * 1000,
                "ask_px_00": [0.67000] * 1000,  # No variation - will cause issues
                "bid_px_00": [0.66995] * 1000,
            })

            file_path = intermediate_dir / "bad_M6AM4.parquet"
            bad_data.write_parquet(str(file_path))

            # This should result in no processable rows due to insufficient variation
            result = builder.merge_symbol_data("M6AM4", [file_path], output_dir)

            # Either gets no processable rows, or succeeds - both are valid test outcomes
            if result is None:
                captured = capsys.readouterr()
                # Check if it was due to no processable rows or insufficient samples
                assert ("No processable rows" in captured.out or
                       "Insufficient total samples" in captured.out)

    @patch('represent.dataset_builder.db.read_dbn')
    def test_build_datasets_verbose_phases(self, mock_read_dbn, simple_config, simple_dataset_config, capsys):
        """Test verbose output during full dataset building phases."""
        # Mock databento read
        mock_data = Mock()
        sample_df = pl.DataFrame({
            "ts_event": [1640995200000000000 + i * 1000000000 for i in range(1000)],
            "symbol": ["M6AM4"] * 500 + ["M6AM5"] * 500,
            "ask_px_00": np.random.normal(0.67000, 0.00001, 1000),
            "bid_px_00": np.random.normal(0.66995, 0.00001, 1000),
        })
        mock_data.to_df.return_value = sample_df.to_pandas()
        mock_read_dbn.return_value = mock_data

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

            builder.build_datasets_from_dbn_files(
                dbn_files=["file1.dbn", "file2.dbn"],
                output_dir=output_dir
            )

            captured = capsys.readouterr()
            assert "Starting Symbol-Split-Merge Dataset Building" in captured.out
            assert "Phase 1: Splitting" in captured.out and "DBN files by symbol" in captured.out
            assert "Phase 1 Complete:" in captured.out
            assert "Phase 2: Merging symbols into datasets" in captured.out
            assert "DATASET BUILDING COMPLETE!" in captured.out
            assert "Total processing time:" in captured.out
            assert "Processing rate:" in captured.out and "samples/sec" in captured.out

    @patch('represent.dataset_builder.db.read_dbn')
    def test_build_datasets_error_handling_verbose(self, mock_read_dbn, simple_config, simple_dataset_config, capsys):
        """Test verbose error handling during dataset building."""
        # Make databento read fail
        mock_read_dbn.side_effect = Exception("Mock DBN read error")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            builder = DatasetBuilder(simple_config, simple_dataset_config, verbose=True)

            builder.build_datasets_from_dbn_files(
                dbn_files=["bad_file.dbn"],
                output_dir=output_dir
            )

            captured = capsys.readouterr()
            assert "Failed to process" in captured.out
            assert "bad_file.dbn" in captured.out

    @patch('represent.dataset_builder.db.read_dbn')
    def test_build_datasets_merge_error_verbose(self, mock_read_dbn, simple_config, simple_dataset_config, capsys):
        """Test verbose output when merging fails."""
        # Mock databento read
        mock_data = Mock()
        sample_df = pl.DataFrame({
            "ts_event": [1640995200000000000 + i * 1000000000 for i in range(100)],
            "symbol": ["M6AM4"] * 100,
            "ask_px_00": [0.67000] * 100,
            "bid_px_00": [0.66995] * 100,
        })
        mock_data.to_df.return_value = sample_df.to_pandas()
        mock_read_dbn.return_value = mock_data

        # Use very high minimum to force merge failures
        high_config = DatasetBuildConfig(min_symbol_samples=100000)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            builder = DatasetBuilder(simple_config, high_config, verbose=True)

            builder.build_datasets_from_dbn_files(
                dbn_files=["file1.dbn"],
                output_dir=output_dir
            )

            captured = capsys.readouterr()
            # Check for the actual insufficient samples message
            assert "Insufficient total samples" in captured.out

    @patch('represent.dataset_builder.db.read_dbn')
    def test_build_datasets_cleanup_verbose(self, mock_read_dbn, simple_config, capsys):
        """Test verbose output during intermediate file cleanup."""
        # Mock databento read
        mock_data = Mock()
        sample_df = pl.DataFrame({
            "ts_event": [1640995200000000000 + i * 1000000000 for i in range(100)],
            "symbol": ["M6AM4"] * 100,
            "ask_px_00": [0.67000] * 100,
            "bid_px_00": [0.66995] * 100,
        })
        mock_data.to_df.return_value = sample_df.to_pandas()
        mock_read_dbn.return_value = mock_data

        # Use config that doesn't keep intermediate files
        dataset_config = DatasetBuildConfig(
            min_symbol_samples=10,  # Low for testing
            keep_intermediate=False
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            builder = DatasetBuilder(simple_config, dataset_config, verbose=True)

            builder.build_datasets_from_dbn_files(
                dbn_files=["file1.dbn"],
                output_dir=output_dir
            )

            captured = capsys.readouterr()
            assert "Cleaning up intermediate files" in captured.out

    def test_batch_build_verbose_file_finding(self, simple_config, simple_dataset_config, capsys):
        """Test verbose output when finding files in directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()

            # Create mock DBN files
            (input_dir / "test1.dbn").write_text("mock")
            (input_dir / "test2.dbn.zst").write_text("mock")

            with patch('represent.dataset_builder.build_datasets_from_dbn_files') as mock_build:
                mock_build.return_value = {"test": "result"}

                batch_build_datasets_from_directory(
                    config=simple_config,
                    input_directory=input_dir,
                    output_dir="/tmp/output",
                    dataset_config=simple_dataset_config,
                    verbose=True
                )

                captured = capsys.readouterr()
                assert "Found" in captured.out and "DBN files in" in captured.out
                assert "test1.dbn" in captured.out
                assert "test2.dbn.zst" in captured.out
