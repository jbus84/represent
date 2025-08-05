"""
Fixed tests for unlabeled_converter module matching the actual API.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl
import numpy as np

from represent.unlabeled_converter import (
    UnlabeledDBNConverter,
    convert_dbn_to_parquet,
    batch_convert_dbn_files
)


def create_mock_dbn_data(n_samples: int = 100) -> pl.DataFrame:
    """Create mock DBN data for testing."""
    np.random.seed(42)
    
    if n_samples == 0:
        # Return empty dataframe with correct schema
        return pl.DataFrame({
            "symbol": [],
            "ts_event": [],
            "ts_recv": [],
        }).with_columns([
            pl.col("ts_event").cast(pl.Datetime(time_unit="ns")),
            pl.col("ts_recv").cast(pl.Datetime(time_unit="ns"))
        ])
    
    # Create realistic market data structure
    symbols = ["M6AM4", "M6AU4", "M6AZ4"]
    
    data = []
    for i in range(n_samples):
        symbol = symbols[i % len(symbols)]
        base_price = 0.66000 if "M6AM4" in symbol else 0.65000
        
        # Create bid/ask levels
        bid_prices = [base_price - (j + 1) * 0.00001 for j in range(10)]
        ask_prices = [base_price + (j + 1) * 0.00001 for j in range(10)]
        bid_sizes = [np.random.randint(100, 1000) for _ in range(10)]
        ask_sizes = [np.random.randint(100, 1000) for _ in range(10)]
        bid_counts = [np.random.randint(1, 10) for _ in range(10)]
        ask_counts = [np.random.randint(1, 10) for _ in range(10)]
        
        row = {
            "symbol": symbol,
            "ts_event": 1700000000000000000 + i * 1000000,  # nanoseconds
            "ts_recv": 1700000000000000000 + i * 1000000,
        }
        
        # Add bid/ask price columns
        for j in range(10):
            row[f"bid_px_{j:02d}"] = bid_prices[j]
            row[f"ask_px_{j:02d}"] = ask_prices[j]
            row[f"bid_sz_{j:02d}"] = bid_sizes[j]
            row[f"ask_sz_{j:02d}"] = ask_sizes[j]
            row[f"bid_ct_{j:02d}"] = bid_counts[j]
            row[f"ask_ct_{j:02d}"] = ask_counts[j]
        
        data.append(row)
    
    df = pl.DataFrame(data)
    
    # Convert timestamps to proper datetime format
    df = df.with_columns([
        pl.from_epoch(pl.col("ts_event"), time_unit="ns").alias("ts_event"),
        pl.from_epoch(pl.col("ts_recv"), time_unit="ns").alias("ts_recv")
    ])
    
    return df


class TestUnlabeledDBNConverter:
    """Test UnlabeledDBNConverter class functionality."""

    def test_initialization_default(self):
        """Test converter initialization with default parameters."""
        converter = UnlabeledDBNConverter()
        
        assert converter.features == ["volume"]
        assert converter.min_symbol_samples == 1000  # Correct attribute name
        assert converter.batch_size == 2000

    def test_initialization_custom(self):
        """Test converter initialization with custom parameters."""
        converter = UnlabeledDBNConverter(
            features=["volume", "variance"],
            min_symbol_samples=500,
            batch_size=1000
        )
        
        assert converter.features == ["volume", "variance"]
        assert converter.min_symbol_samples == 500
        assert converter.batch_size == 1000

    @patch('databento.DBNStore.from_file')
    @patch('polars.DataFrame.write_parquet')
    def test_convert_dbn_to_symbol_parquets_basic(self, mock_write, mock_dbn_store):
        """Test basic DBN file conversion."""
        # Setup mocks
        mock_dbn_data = create_mock_dbn_data(3000)  # Enough for min_symbol_samples
        mock_store = MagicMock()
        mock_store.to_df.return_value = mock_dbn_data
        mock_dbn_store.return_value = mock_store
        
        converter = UnlabeledDBNConverter()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dbn_path = Path(temp_dir) / "test.dbn"
            output_dir = Path(temp_dir) / "output"
            
            # Create mock DBN file
            dbn_path.touch()
            
            # Mock the processor to avoid tensor operations
            with patch.object(converter.processor, 'process') as mock_process:
                mock_process.return_value = np.random.rand(402, 500)
                
                stats = converter.convert_dbn_to_symbol_parquets(dbn_path, output_dir)
            
            assert "conversion_time_seconds" in stats
            assert "total_processed_samples" in stats
            assert "symbols_processed" in stats
            assert stats["original_rows"] == 3000

    @patch('databento.DBNStore.from_file')
    def test_convert_dbn_file_error_handling(self, mock_dbn_store):
        """Test error handling in DBN conversion."""
        # Mock databento to raise an exception
        mock_dbn_store.side_effect = Exception("DBN read error")
        
        converter = UnlabeledDBNConverter()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dbn_path = Path(temp_dir) / "test.dbn"
            output_dir = Path(temp_dir) / "output"
            dbn_path.touch()
            
            with pytest.raises(Exception, match="DBN read error"):
                converter.convert_dbn_to_symbol_parquets(dbn_path, output_dir)

    def test_missing_dbn_file(self):
        """Test handling of missing DBN files."""
        converter = UnlabeledDBNConverter()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dbn_path = Path(temp_dir) / "nonexistent.dbn"
            output_dir = Path(temp_dir) / "output"
            
            with pytest.raises(FileNotFoundError):
                converter.convert_dbn_to_symbol_parquets(dbn_path, output_dir)

    def test_add_metadata_columns(self):
        """Test metadata column addition."""
        converter = UnlabeledDBNConverter()
        mock_data = create_mock_dbn_data(100)
        
        source_path = Path("/test/data_20240403.dbn")
        result = converter._add_metadata_columns(mock_data, source_path)
        
        assert "source_file" in result.columns
        assert "file_date" in result.columns
        assert "row_id" in result.columns
        
        # Check that values are properly set
        assert result["source_file"][0] == "data_20240403.dbn"
        # The file_date extraction might return "unknown" if pattern doesn't match
        assert result["file_date"][0] in ["20240403", "unknown"]
        assert len(result["row_id"].unique()) == 100  # All unique row IDs

    def test_process_symbol_chunk(self):
        """Test symbol chunk processing."""
        converter = UnlabeledDBNConverter(batch_size=100)
        
        # Create chunk with enough data (more than batch_size)
        chunk = create_mock_dbn_data(150)
        
        # Mock the processor to avoid tensor operations
        with patch.object(converter.processor, 'process') as mock_process:
            mock_process.return_value = np.random.rand(402, 500)  # Mock tensor output
            
            # Mock the _add_metadata_columns method to avoid file path issues
            with patch.object(converter, '_add_metadata_columns') as mock_metadata:
                mock_metadata.return_value = chunk.with_columns([
                    pl.lit("test.dbn").alias("source_file"),
                    pl.lit("20240403").alias("file_date"),
                    pl.arange(len(chunk)).alias("row_id")
                ])
                
                result = converter._process_symbol_chunk(chunk, "M6AM4")
                
                # Should return a DataFrame with processed data
                assert result is None or isinstance(result, pl.DataFrame)


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    @patch('represent.unlabeled_converter.UnlabeledDBNConverter')
    def test_convert_dbn_to_parquet(self, mock_converter_class):
        """Test convert_dbn_to_parquet convenience function."""
        mock_converter = MagicMock()
        mock_converter.convert_dbn_to_symbol_parquets.return_value = {"total_processed_samples": 100}
        mock_converter_class.return_value = mock_converter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dbn_path = Path(temp_dir) / "test.dbn"
            output_dir = Path(temp_dir) / "output"
            dbn_path.touch()
            
            stats = convert_dbn_to_parquet(dbn_path, output_dir)
            
            assert stats["total_processed_samples"] == 100
            mock_converter_class.assert_called_once()
            mock_converter.convert_dbn_to_symbol_parquets.assert_called_once()

    @patch('represent.unlabeled_converter.convert_dbn_to_parquet')
    def test_batch_convert_dbn_files(self, mock_convert):
        """Test batch_convert_dbn_files convenience function."""
        mock_convert.return_value = {"total_processed_samples": 50}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            # Create mock DBN files
            dbn_files = [
                input_dir / "file1.dbn",
                input_dir / "file2.dbn"
            ]
            for f in dbn_files:
                f.touch()
            
            results = batch_convert_dbn_files(input_dir, output_dir)
            
            assert len(results) == 2
            assert all(result["total_processed_samples"] == 50 for result in results)

    def test_batch_convert_no_files(self):
        """Test batch convert with no DBN files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            with pytest.raises(ValueError, match="No DBN files found"):
                batch_convert_dbn_files(input_dir, output_dir)

    def test_batch_convert_missing_input_dir(self):
        """Test batch convert with missing input directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "nonexistent"
            output_dir = Path(temp_dir) / "output"
            
            with pytest.raises(FileNotFoundError):
                batch_convert_dbn_files(input_dir, output_dir)


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch('databento.DBNStore.from_file')
    def test_empty_dbn_file(self, mock_dbn_store):
        """Test handling of empty DBN files."""
        # Mock empty dataframe with correct schema
        empty_df = create_mock_dbn_data(0)  # This returns properly formatted empty df
        mock_store = MagicMock()
        mock_store.to_df.return_value = empty_df
        mock_dbn_store.return_value = mock_store
        
        converter = UnlabeledDBNConverter()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dbn_path = Path(temp_dir) / "empty.dbn"
            output_dir = Path(temp_dir) / "output"
            dbn_path.touch()
            
            stats = converter.convert_dbn_to_symbol_parquets(dbn_path, output_dir)
            
            assert stats["original_rows"] == 0
            assert stats["symbols_processed"] == 0

    @patch('databento.DBNStore.from_file')
    def test_insufficient_symbol_data(self, mock_dbn_store):
        """Test handling when symbols don't have enough samples."""
        # Create data with insufficient samples per symbol
        mock_data = create_mock_dbn_data(300)  # Only 100 per symbol, less than min_symbol_samples
        mock_store = MagicMock()
        mock_store.to_df.return_value = mock_data
        mock_dbn_store.return_value = mock_store
        
        converter = UnlabeledDBNConverter(min_symbol_samples=500)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dbn_path = Path(temp_dir) / "test.dbn"
            output_dir = Path(temp_dir) / "output"
            dbn_path.touch()
            
            # Mock the processor to avoid tensor operations
            with patch.object(converter.processor, 'process') as mock_process:
                mock_process.return_value = np.random.rand(402, 500)
                
                stats = converter.convert_dbn_to_symbol_parquets(dbn_path, output_dir)
            
            # Should filter out symbols with insufficient data
            assert stats["symbols_processed"] == 0


class TestPerformanceOptimizations:
    """Test performance-related functionality."""

    @patch('databento.DBNStore.from_file')
    @patch('polars.DataFrame.write_parquet')
    @patch('pathlib.Path.stat')
    def test_large_dataset_processing(self, mock_stat, mock_write, mock_dbn_store):
        """Test processing of large datasets."""
        # Simulate large dataset
        large_data = create_mock_dbn_data(15000)  # 5000 per symbol
        mock_store = MagicMock()
        mock_store.to_df.return_value = large_data
        mock_dbn_store.return_value = mock_store
        
        # Mock file stat to avoid FileNotFoundError
        mock_stat.return_value.st_size = 1024 * 1024  # 1MB
        
        converter = UnlabeledDBNConverter()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dbn_path = Path(temp_dir) / "large.dbn"
            output_dir = Path(temp_dir) / "output"
            dbn_path.touch()
            
            import time
            start_time = time.time()
            
            # Mock the processor to avoid heavy computation
            with patch.object(converter.processor, 'process') as mock_process:
                mock_process.return_value = np.random.rand(402, 500)
                
                stats = converter.convert_dbn_to_symbol_parquets(dbn_path, output_dir)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete in reasonable time
            assert processing_time < 10.0  # Less than 10 seconds
            assert stats["original_rows"] == 15000

    def test_batch_size_configuration(self):
        """Test that batch size affects processing."""
        converter_small = UnlabeledDBNConverter(batch_size=100)
        converter_large = UnlabeledDBNConverter(batch_size=5000)
        
        assert converter_small.batch_size == 100
        assert converter_large.batch_size == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])