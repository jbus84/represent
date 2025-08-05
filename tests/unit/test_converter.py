"""
Tests for the DBN to Parquet converter functionality.
"""

import pytest
from pathlib import Path
import tempfile

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from represent.converter import DBNToParquetConverter, convert_dbn_file, batch_convert_dbn_files
from represent.config import ClassificationConfig


class TestDBNToParquetConverter:
    """Test DBN to Parquet converter initialization and configuration."""

    def test_converter_initialization_with_currency(self):
        """Test converter initializes correctly with currency."""
        converter = DBNToParquetConverter(currency="AUDUSD")

        assert converter.currency == "AUDUSD"
        assert converter.features == ["volume"]  # Default
        assert converter.classification_config is not None
        assert converter.processor is not None
        assert converter.batch_size == 2000

    def test_converter_initialization_with_features(self):
        """Test converter with multiple features."""
        converter = DBNToParquetConverter(currency="GBPUSD", features=["volume", "variance"])

        assert converter.currency == "GBPUSD"
        assert converter.features == ["volume", "variance"]

    def test_converter_initialization_with_custom_config(self):
        """Test converter with custom classification config."""
        custom_config = ClassificationConfig(
            nbins=5,
            lookforward_input=1000,
            lookforward_offset=0,
            ticks_per_bin=100,
            micro_pip_size=0.00001,
        )

        converter = DBNToParquetConverter(classification_config=custom_config, features=["volume"])

        assert converter.currency == "CUSTOM"
        assert converter.classification_config.nbins == 5
        assert converter.classification_config.lookforward_input == 1000

    def test_converter_default_configuration(self):
        """Test converter with default configuration."""
        converter = DBNToParquetConverter()

        assert converter.currency == "AUDUSD"
        assert converter.features == ["volume"]
        assert converter.classification_config is not None

    def test_converter_batch_size_configuration(self):
        """Test converter with custom batch size."""
        converter = DBNToParquetConverter(currency="EURJPY", batch_size=5000)

        assert converter.batch_size == 5000

    def test_classification_method(self):
        """Test classification logic."""
        converter = DBNToParquetConverter(currency="AUDUSD")

        # Test various percentage changes (new method expects relative changes)
        true_pip_size = converter.classification_config.true_pip_size
        percentage_changes = [0.5 * true_pip_size, 1.5 * true_pip_size, 3.0 * true_pip_size, 
                            5.0 * true_pip_size, 10.0 * true_pip_size]
        labels = [converter._classify_price_movement_percentage(change) for change in percentage_changes]

        # Should return valid labels
        for label in labels:
            assert 0 <= label < converter.classification_config.nbins

        # Check that we get reasonable classification results
        assert isinstance(labels[0], int)
        assert isinstance(labels[-1], int)

    def test_classification_fallback_logic(self):
        """Test classification edge cases and expected behavior."""
        # Test with supported nbins but different lookforward_input and ticks_per_bin combinations
        custom_config = ClassificationConfig(
            nbins=7,  # Supported number of bins
            lookforward_input=5000,
            lookforward_offset=0,
            ticks_per_bin=100,
            micro_pip_size=0.00001,
        )

        converter = DBNToParquetConverter(classification_config=custom_config)

        # Test classification with various percentage changes
        true_pip_size = custom_config.true_pip_size
        small_movement = converter._classify_price_movement_percentage(0.5 * true_pip_size)
        medium_movement = converter._classify_price_movement_percentage(2.0 * true_pip_size)
        large_movement = converter._classify_price_movement_percentage(5.0 * true_pip_size)
        negative_movement = converter._classify_price_movement_percentage(-2.0 * true_pip_size)

        # Check that we get reasonable results within valid range
        assert 0 <= small_movement < custom_config.nbins
        assert 0 <= medium_movement < custom_config.nbins
        assert 0 <= large_movement < custom_config.nbins
        assert 0 <= negative_movement < custom_config.nbins
        
        # Check that different movements produce different classifications
        assert small_movement != large_movement or medium_movement != large_movement

    def test_add_metadata_columns(self):
        """Test metadata column addition."""
        import polars as pl
        from datetime import datetime

        converter = DBNToParquetConverter(currency="AUDUSD")

        # Create sample dataframe with proper datetime timestamps
        df = pl.DataFrame(
            {
                "ts_event": pl.Series(
                    [
                        datetime(2023, 12, 1, 10, 0, 0),
                        datetime(2023, 12, 1, 10, 0, 1),
                        datetime(2023, 12, 1, 10, 0, 2),
                    ],
                    dtype=pl.Datetime,
                ),
                "symbol": ["M6AM4", "M6AM4", "M6AM4"],
            }
        )

        source_path = Path("/path/to/test-20231201.dbn")
        result_df = converter._add_metadata_columns(df, source_path)

        # Check metadata columns were added
        assert "source_file" in result_df.columns
        assert "file_date" in result_df.columns
        assert "timestamp" in result_df.columns
        assert "date" in result_df.columns
        assert "hour" in result_df.columns
        assert "row_id" in result_df.columns

        # Check values
        assert result_df["source_file"][0] == "test-20231201.dbn"
        assert result_df["file_date"][0] == "20231201"
        assert result_df["hour"][0] == 10  # Check hour extraction works


class TestConverterFunctions:
    """Test converter convenience functions."""

    def test_convert_dbn_file_with_missing_file(self):
        """Test convert_dbn_file with missing input file."""
        with pytest.raises(FileNotFoundError):
            convert_dbn_file(
                dbn_path="/nonexistent/file.dbn",
                output_path="/tmp/output.parquet",
                currency="AUDUSD",
            )

    def test_batch_convert_dbn_files_missing_directory(self):
        """Test batch conversion with missing input directory."""
        with pytest.raises(FileNotFoundError):
            batch_convert_dbn_files(
                input_directory="/nonexistent/directory",
                output_directory="/tmp/output",
                currency="AUDUSD",
            )

    def test_batch_convert_dbn_files_no_files(self):
        """Test batch conversion with no matching files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="No DBN files found"):
                batch_convert_dbn_files(
                    input_directory=temp_dir,
                    output_directory=temp_dir,
                    currency="AUDUSD",
                    pattern="*.dbn",
                )


class TestConverterErrorHandling:
    """Test converter error handling and edge cases."""

    def test_process_chunk_with_insufficient_data(self):
        """Test chunk processing with insufficient data."""
        import polars as pl

        converter = DBNToParquetConverter(currency="AUDUSD", batch_size=2000)

        # Create a small chunk (less than batch size)
        small_chunk = pl.DataFrame(
            {
                "ts_event": list(range(1000)),  # Only 1000 rows
                "bid_px_00": [125000] * 1000,
                "ask_px_00": [125010] * 1000,
                "bid_sz_00": [100] * 1000,
                "ask_sz_00": [100] * 1000,
            }
        )

        result = converter._process_chunk_with_labels(small_chunk, 0)

        # Should return None for insufficient data
        assert result is None

    def test_process_chunk_with_no_valid_samples(self):
        """Test chunk processing with no valid sample positions."""
        import polars as pl

        converter = DBNToParquetConverter(currency="AUDUSD", batch_size=1000)

        # Create a chunk that's too small for the lookforward window
        small_chunk = pl.DataFrame(
            {
                "ts_event": list(range(2000)),  # Enough for batch but not lookforward
                "bid_px_00": [125000] * 2000,
                "ask_px_00": [125010] * 2000,
                "bid_sz_00": [100] * 2000,
                "ask_sz_00": [100] * 2000,
            }
        )

        result = converter._process_chunk_with_labels(small_chunk, 0)

        # Should return None when no valid samples can be generated
        assert result is None
