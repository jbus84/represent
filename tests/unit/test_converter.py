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

        # Test various movement sizes
        movements = [0.5, 1.5, 3.0, 5.0, 10.0]
        labels = [converter._classify_price_movement(mov) for mov in movements]

        # Should return valid labels
        for label in labels:
            assert 0 <= label < converter.classification_config.nbins

        # Larger movements should generally get higher labels (or fallback logic)
        assert isinstance(labels[0], int)
        assert isinstance(labels[-1], int)

    def test_classification_fallback_logic(self):
        """Test classification fallback when no thresholds available."""
        # Create converter with valid structure but missing specific thresholds
        # to trigger fallback logic
        custom_config = ClassificationConfig(
            nbins=3,
            lookforward_input=999,  # Use different lookforward to trigger fallback
            lookforward_offset=0,
            ticks_per_bin=100,
            micro_pip_size=0.00001,
            bin_thresholds={
                "3": {"100": {"1000": {"bin_1": 1.0}}}
            },  # Valid format but wrong lookforward
        )

        converter = DBNToParquetConverter(classification_config=custom_config)

        # Test fallback classification (should use fallback logic since lookforward=999 not in thresholds)
        small_movement = converter._classify_price_movement(0.5)
        medium_movement = converter._classify_price_movement(2.0)
        large_movement = converter._classify_price_movement(5.0)

        assert small_movement == 0  # Small movement
        assert medium_movement == 1  # Medium movement
        assert large_movement == 2  # Large movement

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
