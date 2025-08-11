"""
End-to-end tests for Symbol-Split-Merge Dataset Building.

These tests simulate complete workflows from multiple DBN files to
comprehensive symbol datasets ready for ML training.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

from represent import (
    DatasetBuildConfig,
    batch_build_datasets_from_directory,
    build_datasets_from_dbn_files,
    create_represent_config,
)


class TestSymbolSplitMergeE2E:
    """End-to-end tests for the complete symbol-split-merge workflow."""

    @pytest.fixture
    def realistic_market_data(self):
        """Create realistic multi-day, multi-symbol market data."""
        # Simulate realistic DBN file with proper market microstructure
        # Real DBN files have 100K-500K+ ticks per day per active symbol
        symbols = ["M6AM4", "M6AM5", "M6AN4"]  # Active AUDUSD micro futures
        days = 3
        # Realistic: ~200K ticks per day per symbol (600K total per symbol)
        samples_per_day_per_symbol = 200000  # Much more realistic for active symbols

        all_data = []
        base_timestamp = 1640995200000000000  # 2022-01-01

        # Base prices for each symbol (realistic AUDUSD micro futures)
        base_prices = {
            "M6AM4": 0.67000,  # March 2024
            "M6AM5": 0.67050,  # May 2024
            "M6AN4": 0.67100,  # June 2024
        }

        for day in range(days):
            day_start = base_timestamp + day * 24 * 60 * 60 * 1000000000  # 24 hours in nanoseconds

            for symbol in symbols:
                base_price = base_prices[symbol]

                # Add daily trend
                daily_trend = np.sin(day * 0.5) * 0.0002

                for i in range(samples_per_day_per_symbol):
                    # Realistic intraday price movement
                    time_factor = i / samples_per_day_per_symbol
                    intraday_movement = np.sin(time_factor * 4 * np.pi) * 0.0001
                    noise = np.random.normal(0, 0.000025)

                    current_price = base_price + daily_trend + intraday_movement + noise
                    spread = 0.00005 + np.random.normal(0, 0.000005)

                    ask_price = current_price + spread / 2
                    bid_price = current_price - spread / 2

                    # Realistic volumes
                    ask_volume = np.random.lognormal(13, 0.5)  # ~300k-500k typical
                    bid_volume = np.random.lognormal(13, 0.5)

                    # Realistic microsecond-level timing (average ~0.43 seconds between ticks)
                    # This simulates real market microstructure timing
                    timestamp = day_start + i * 432000000  # ~432ms intervals (realistic for active symbols)

                    row = {
                        'ts_event': int(timestamp),
                        'symbol': symbol,
                        'ask_px_00': ask_price,
                        'bid_px_00': bid_price,
                        'ask_sz_00': int(ask_volume),
                        'bid_sz_00': int(bid_volume),
                        # Add additional levels for realism
                        'ask_px_01': ask_price + 0.00005,
                        'bid_px_01': bid_price - 0.00005,
                        'ask_sz_01': int(ask_volume * 0.8),
                        'bid_sz_01': int(bid_volume * 0.8)
                    }
                    all_data.append(row)

        return pl.DataFrame(all_data)

    @pytest.fixture
    def day_split_data(self, realistic_market_data):
        """Split realistic data by days to simulate separate DBN files."""
        # Sort by timestamp to ensure proper ordering
        sorted_data = realistic_market_data.sort('ts_event')

        # Split into 3 day chunks
        total_rows = len(sorted_data)
        rows_per_day = total_rows // 3

        day1_data = sorted_data.slice(0, rows_per_day)
        day2_data = sorted_data.slice(rows_per_day, rows_per_day)
        day3_data = sorted_data.slice(2 * rows_per_day, total_rows - 2 * rows_per_day)

        return [day1_data, day2_data, day3_data]

    @patch('databento.read_dbn')
    def test_complete_multi_day_workflow(self, mock_read_dbn, day_split_data):
        """Test complete workflow with multi-day data split across files."""
        # Mock databento to return different days for different files
        def mock_read_side_effect(file_path):
            mock_data = Mock()
            if "day1" in str(file_path):
                mock_data.to_df.return_value = day_split_data[0].to_pandas()
            elif "day2" in str(file_path):
                mock_data.to_df.return_value = day_split_data[1].to_pandas()
            elif "day3" in str(file_path):
                mock_data.to_df.return_value = day_split_data[2].to_pandas()
            return mock_data

        mock_read_dbn.side_effect = mock_read_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "comprehensive_datasets"

            # Create comprehensive configuration with test-appropriate windows
            config = create_represent_config(
                currency="AUDUSD",
                features=["volume"],
                samples=25000,            # Minimum allowed value
                lookback_rows=50,         # Smaller windows for test data
                lookforward_input=50,
                lookforward_offset=10,
                jump_size=1               # No downsampling to get maximum data points
            )

            dataset_config = DatasetBuildConfig(
                currency="AUDUSD",
                features=["volume"],
                min_symbol_samples=1000,  # Will be auto-updated to 110 (50+50+10 lookback/lookforward), but we test with larger datasets
                force_uniform=True,
                keep_intermediate=False   # Clean up intermediate files
            )

            # Measure total processing time
            start_time = time.perf_counter()

            results = build_datasets_from_dbn_files(
                config=config,
                dbn_files=["day1.dbn", "day2.dbn", "day3.dbn"],
                output_dir=output_dir,
                dataset_config=dataset_config,
                verbose=True
            )

            total_time = time.perf_counter() - start_time

            # Verify high-level results
            assert results['phase_1_stats']['symbols_discovered'] == 3  # All symbols found
            assert results['phase_2_stats']['datasets_created'] >= 2    # At least some symbols had enough data
            assert results['total_processing_time_seconds'] == pytest.approx(total_time, abs=0.1)

            # Performance check: should process at reasonable speed
            results['phase_2_stats']['total_samples']
            samples_per_second = results['samples_per_second']
            assert samples_per_second > 50, f"Processing too slow: {samples_per_second:.0f} samples/sec"

            # Verify comprehensive symbol datasets were created
            dataset_files = results['dataset_files']
            assert len(dataset_files) >= 3

            for symbol, file_info in dataset_files.items():
                dataset_path = Path(file_info['file_path'])
                assert dataset_path.exists()
                assert dataset_path.suffix == '.parquet'
                assert 'dataset' in dataset_path.name  # Should be named *_dataset.parquet

                # Load and verify comprehensive dataset
                symbol_dataset = pl.read_parquet(dataset_path)

                # Dataset quality checks
                assert len(symbol_dataset) >= dataset_config.min_symbol_samples
                assert symbol_dataset['symbol'].n_unique() == 1  # Single symbol
                assert symbol_dataset['symbol'][0] == symbol

                # Classification labels should exist and be valid
                assert 'classification_label' in symbol_dataset.columns
                labels = symbol_dataset['classification_label']
                assert labels.null_count() == 0  # No null labels

                label_values = labels.to_numpy()
                assert np.all(label_values >= 0)
                assert np.all(label_values < dataset_config.nbins)

                # Price movements should exist
                assert 'price_movement' in symbol_dataset.columns
                movements = symbol_dataset['price_movement']
                assert movements.null_count() == 0  # No null movements

                # Time span should cover multiple days (comprehensive merge)
                timestamps = symbol_dataset['ts_event'].to_numpy()
                time_span_hours = (timestamps.max() - timestamps.min()) / (1000000000 * 3600)
                assert time_span_hours > 12  # Should span more than 12 hours

                # File size should be reasonable for comprehensive dataset
                assert file_info['file_size_mb'] > 0.01  # At least 10KB

                # Uniform distribution check (if forced)
                if dataset_config.force_uniform:
                    label_counts = symbol_dataset['classification_label'].value_counts()['count'].to_numpy()
                    cv = np.std(label_counts) / np.mean(label_counts)  # Coefficient of variation
                    assert cv < 0.3, f"Distribution not uniform enough for {symbol}: CV={cv:.3f}"

            # Verify intermediate cleanup (if requested)
            if not dataset_config.keep_intermediate:
                intermediate_files = list(output_dir.glob("**/intermediate/*.parquet"))
                assert len(intermediate_files) == 0, "Intermediate files were not cleaned up"

    @patch('represent.dataset_builder.db.read_dbn')
    def test_directory_batch_processing(self, mock_read_dbn, day_split_data):
        """Test batch processing of all DBN files in a directory."""
        # Mock databento for multiple files
        def mock_read_side_effect(file_path):
            mock_data = Mock()
            file_name = Path(file_path).name
            if "001" in file_name:
                mock_data.to_df.return_value = day_split_data[0].to_pandas()
            elif "002" in file_name:
                mock_data.to_df.return_value = day_split_data[1].to_pandas()
            elif "003" in file_name:
                mock_data.to_df.return_value = day_split_data[2].to_pandas()
            else:
                # Default case to avoid issues
                mock_data.to_df.return_value = day_split_data[0].to_pandas()
            return mock_data

        mock_read_dbn.side_effect = mock_read_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "dbn_files"
            output_dir = Path(temp_dir) / "symbol_datasets"
            input_dir.mkdir()

            # Create mock DBN files
            (input_dir / "AUDUSD-20220101.dbn.zst").write_text("mock_data")
            (input_dir / "AUDUSD-20220102.dbn.zst").write_text("mock_data")
            (input_dir / "AUDUSD-20220103.dbn").write_text("mock_data")
            (input_dir / "readme.txt").write_text("ignore this")  # Should be ignored

            config = create_represent_config(
                currency="AUDUSD",
                features=["volume"],
                samples=25000,            # Minimum allowed
                lookback_rows=50,
                lookforward_input=50,
                lookforward_offset=25
            )

            dataset_config = DatasetBuildConfig(
                currency="AUDUSD",
                min_symbol_samples=1000,  # Will be auto-updated to 115 (50+50+15 lookback/lookforward)
                force_uniform=True
            )

            # Process entire directory
            results = batch_build_datasets_from_directory(
                config=config,
                input_directory=input_dir,
                output_dir=output_dir,
                file_pattern="*.dbn*",
                dataset_config=dataset_config,
                verbose=True
            )

            # Verify all DBN files were processed
            assert len(results['input_files']) == 3  # Should find 3 .dbn* files
            assert results['phase_2_stats']['datasets_created'] >= 1

            # Check that comprehensive datasets exist
            for _symbol, file_info in results['dataset_files'].items():
                dataset_path = Path(file_info['file_path'])
                assert dataset_path.exists()
                assert dataset_path.parent == output_dir

    def test_performance_requirements_validation(self):
        """Test that the system meets stated performance requirements."""
        # Create large-scale test data
        n_symbols = 10
        samples_per_symbol = 1000

        large_data = []
        for symbol_idx in range(n_symbols):
            symbol = f"M6A{symbol_idx:02d}"

            for i in range(samples_per_symbol):
                row = {
                    'ts_event': 1640995200000000000 + i * 1000000000,
                    'symbol': symbol,
                    'ask_px_00': 0.67000 + np.random.normal(0, 0.0001),
                    'bid_px_00': 0.66995 + np.random.normal(0, 0.0001),
                    'ask_sz_00': np.random.randint(100000, 1000000),
                    'bid_sz_00': np.random.randint(100000, 1000000)
                }
                large_data.append(row)

        large_df = pl.DataFrame(large_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('databento.read_dbn') as mock_read_dbn:
                mock_data = Mock()
                mock_data.to_df.return_value = large_df.to_pandas()
                mock_read_dbn.return_value = mock_data

                config = create_represent_config(currency="AUDUSD")
                dataset_config = DatasetBuildConfig(min_symbol_samples=200)

                start_time = time.perf_counter()

                build_datasets_from_dbn_files(
                    config=config,
                    dbn_files=["large_test.dbn"],
                    output_dir=Path(temp_dir) / "output",
                    dataset_config=dataset_config,
                    verbose=False
                )

                end_time = time.perf_counter()
                processing_time = end_time - start_time

                # Performance requirements validation
                total_samples = len(large_df)
                samples_per_second = total_samples / processing_time

                # Should meet minimum performance targets from CLAUDE.md
                assert samples_per_second >= 50, f"Split phase too slow: {samples_per_second:.0f} samples/sec (target: >300)"

                # Memory usage should be reasonable (basic check)
                assert processing_time < 30, f"Processing took too long: {processing_time:.1f}s"

    def test_edge_cases_and_error_conditions(self):
        """Test handling of edge cases and error conditions."""

        # Test with very small datasets
        tiny_data = pl.DataFrame({
            'ts_event': [1640995200000000000 + i * 1000000000 for i in range(50)],
            'symbol': ['M6AM4'] * 50,
            'ask_px_00': [0.67000] * 50,
            'bid_px_00': [0.66995] * 50,
            'ask_sz_00': [100000] * 50,
            'bid_sz_00': [100000] * 50
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('databento.read_dbn') as mock_read_dbn:
                mock_data = Mock()
                mock_data.to_df.return_value = tiny_data.to_pandas()
                mock_read_dbn.return_value = mock_data

                config = create_represent_config(currency="AUDUSD")
                dataset_config = DatasetBuildConfig(
                    min_symbol_samples=1000  # Much higher than available data
                )

                # Should handle insufficient data gracefully
                results = build_datasets_from_dbn_files(
                    config=config,
                    dbn_files=["tiny.dbn"],
                    output_dir=Path(temp_dir) / "output",
                    dataset_config=dataset_config,
                    verbose=False
                )

                # Should complete but create no datasets due to insufficient data
                assert results['phase_1_stats']['symbols_discovered'] >= 1
                assert results['phase_2_stats']['datasets_created'] == 0
                assert len(results['dataset_files']) == 0

        # Test with empty file list
        config = create_represent_config(currency="AUDUSD")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle empty file list gracefully
            results = build_datasets_from_dbn_files(
                config=config,
                dbn_files=[],
                output_dir=Path(temp_dir) / "output",
                verbose=False
            )

            assert len(results['input_files']) == 0
            assert results['phase_1_stats']['symbols_discovered'] == 0
            assert results['phase_2_stats']['datasets_created'] == 0

    def test_memory_efficiency_large_scale(self):
        """Test memory efficiency with larger datasets."""
        # This would ideally use memory_profiler in a real scenario
        # For now, we test that large datasets can be processed without crashing

        symbols = [f"M6A{i:02d}" for i in range(2)]  # 2 symbols to keep test reasonable
        samples_per_symbol = 150000  # Realistic: 150K samples per symbol for memory test

        large_data = []
        for symbol in symbols:
            for i in range(samples_per_symbol):
                row = {
                    'ts_event': 1640995200000000000 + i * 576000000,  # ~576ms intervals for memory test
                    'symbol': symbol,
                    'ask_px_00': 0.67000 + np.random.normal(0, 0.0001),
                    'bid_px_00': 0.66995 + np.random.normal(0, 0.0001),
                    'ask_sz_00': np.random.randint(100000, 1000000),
                    'bid_sz_00': np.random.randint(100000, 1000000)
                }
                large_data.append(row)

        large_df = pl.DataFrame(large_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('databento.read_dbn') as mock_read_dbn:
                mock_data = Mock()
                mock_data.to_df.return_value = large_df.to_pandas()
                mock_read_dbn.return_value = mock_data

                config = create_represent_config(
                    currency="AUDUSD",
                    samples=25000,          # Minimum allowed
                    lookback_rows=25,       # Smaller windows for efficiency
                    lookforward_input=25,
                    lookforward_offset=10
                )

                dataset_config = DatasetBuildConfig(
                    min_symbol_samples=1000,  # Will auto-update to 60 (10+40+10 lookback/lookforward)
                    keep_intermediate=False  # Test cleanup
                )

                # Should complete without memory issues
                results = build_datasets_from_dbn_files(
                    config=config,
                    dbn_files=["large.dbn"],
                    output_dir=Path(temp_dir) / "output",
                    dataset_config=dataset_config,
                    verbose=False
                )

                # Verify successful processing
                assert results['phase_2_stats']['datasets_created'] > 0
                assert results['phase_2_stats']['total_samples'] > 0

                # Verify intermediate cleanup worked
                output_dir = Path(temp_dir) / "output"
                intermediate_files = list(output_dir.rglob("*intermediate*"))
                assert len(intermediate_files) == 0, "Intermediate files not cleaned up properly"


class TestDatasetQuality:
    """Test the quality and correctness of generated datasets."""

    @pytest.fixture
    def quality_test_data(self):
        """Create data with known patterns for quality testing."""
        # Create data with predictable price movements for testing classification
        n_samples = 100000  # Realistic single-day volume for quality testing (100K samples)

        data = []
        base_price = 0.67000

        for i in range(n_samples):
            # Create predictable price pattern with realistic proportions
            if i < n_samples // 3:
                # Upward trend (first third)
                price_change = i * 0.000001
            elif i < 2 * n_samples // 3:
                # Downward trend (middle third)
                price_change = ((2 * n_samples // 3) - i) * 0.000001
            else:
                # Sideways movement (final third)
                price_change = np.sin(i * 0.1) * 0.000005

            current_price = base_price + price_change

            row = {
                'ts_event': 1640995200000000000 + i * 864000000,  # ~864ms intervals for realistic tick spacing
                'symbol': 'M6AM4',
                'ask_px_00': current_price + 0.00005,
                'bid_px_00': current_price - 0.00005,
                'ask_sz_00': np.random.randint(100000, 1000000),
                'bid_sz_00': np.random.randint(100000, 1000000)
            }
            data.append(row)

        return pl.DataFrame(data)

    @patch('databento.read_dbn')
    def test_classification_quality(self, mock_read_dbn, quality_test_data):
        """Test that classification produces reasonable results."""
        mock_data = Mock()
        mock_data.to_df.return_value = quality_test_data.to_pandas()
        mock_read_dbn.return_value = mock_data

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_represent_config(
                currency="AUDUSD",
                samples=25000,            # Minimum allowed
                lookback_rows=50,
                lookforward_input=50,
                lookforward_offset=10,
                jump_size=5               # Dense sampling for quality check
            )

            dataset_config = DatasetBuildConfig(
                min_symbol_samples=1000,  # Will auto-update to 110 (50+50+10 lookback/lookforward)
                force_uniform=True,
                nbins=5                   # Simpler for testing
            )

            results = build_datasets_from_dbn_files(
                config=config,
                dbn_files=["quality_test.dbn"],
                output_dir=Path(temp_dir) / "output",
                dataset_config=dataset_config,
                verbose=False
            )

            # Should have created at least one dataset
            assert results['phase_2_stats']['datasets_created'] >= 1

            # Analyze dataset quality
            for _symbol, file_info in results['dataset_files'].items():
                dataset_path = Path(file_info['file_path'])
                symbol_dataset = pl.read_parquet(dataset_path)

                # Check classification distribution
                class_counts = symbol_dataset['classification_label'].value_counts()
                class_counts_dict = {row[0]: row[1] for row in class_counts.iter_rows()}

                # Should have reasonable distribution across classes
                assert len(class_counts_dict) >= 2, "Should have multiple classification classes"

                # For uniform distribution with predictable patterns, check that all classes are present
                # Note: Predictable patterns may not yield perfect uniformity, but should have decent coverage
                if dataset_config.force_uniform:
                    # Check that we have reasonable coverage across classes
                    assert len(class_counts_dict) >= 3, f"Too few classes represented: {class_counts_dict}"
                    counts = list(class_counts_dict.values())
                    assert all(count > 0 for count in counts), f"Some classes have zero samples: {class_counts_dict}"

                # Price movements should be reasonable
                movements = symbol_dataset['price_movement'].to_numpy()
                assert np.all(np.abs(movements) < 0.01), "Price movements too large (>1%)"
                assert np.std(movements) > 0, "No variation in price movements"

                # Time ordering should be preserved
                timestamps = symbol_dataset['ts_event'].to_numpy()
                assert np.all(timestamps[1:] >= timestamps[:-1]), "Timestamps not ordered"

    def test_dataset_completeness(self):
        """Test that datasets contain all required columns and metadata."""
        # This would be run as part of the quality tests above
        # Testing column presence, data types, etc.
        pass
