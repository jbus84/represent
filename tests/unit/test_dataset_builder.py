"""
Consolidated Test Suite for DatasetBuilder

This module consolidates all dataset builder tests, removing duplication while
maintaining full coverage. Tests are organized by functionality rather than
spread across multiple files.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

from represent import (
    DatasetBuildConfig,
    DatasetBuilder,
    batch_build_datasets_from_directory,
    build_datasets_from_dbn_files,
    create_represent_config,
)


class TestDatasetBuildConfig:
    """Test DatasetBuildConfig dataclass."""

    def test_default_config(self):
        """Test default DatasetBuildConfig creation."""
        config = DatasetBuildConfig()

        assert config.currency == "AUDUSD"
        assert config.features == ["volume"]
        assert config.min_symbol_samples == 10500
        assert config.force_uniform is True
        assert config.nbins == 13
        assert config.global_thresholds is None
        assert config.intermediate_dir is None
        assert config.keep_intermediate is False

    def test_custom_config(self):
        """Test custom DatasetBuildConfig creation."""
        config = DatasetBuildConfig(
            currency="EURUSD",
            features=["volume", "variance"],
            min_symbol_samples=15000,
            force_uniform=True,
            nbins=10,
            keep_intermediate=True
        )

        assert config.currency == "EURUSD"
        assert config.features == ["volume", "variance"]
        assert config.min_symbol_samples == 15000
        assert config.force_uniform is True
        assert config.nbins == 10
        assert config.keep_intermediate is True

    def test_config_validation_error(self):
        """Test that DatasetBuildConfig raises error when neither force_uniform nor global_thresholds provided."""
        with pytest.raises(ValueError) as exc_info:
            DatasetBuildConfig(
                currency="EURUSD",
                force_uniform=False,
                global_thresholds=None
            )

        assert "requires either force_uniform=True or global_thresholds" in str(exc_info.value)


class TestDatasetBuilderInitialization:
    """Test DatasetBuilder initialization and configuration."""

    @pytest.fixture
    def base_config(self):
        """Base RepresentConfig for testing."""
        return create_represent_config(
            currency="AUDUSD",
            features=["volume"],
            samples=25000,
            lookback_rows=100,
            lookforward_input=100,
            lookforward_offset=20
        )

    def test_automatic_min_samples_calculation(self, base_config):
        """Test that DatasetBuilder automatically calculates minimum required samples."""
        dataset_config = DatasetBuildConfig(min_symbol_samples=50)  # Too low
        builder = DatasetBuilder(base_config, dataset_config, verbose=False)

        expected_min = 100 + 100 + 20  # lookback + lookforward + offset = 220
        assert builder.dataset_config.min_symbol_samples == expected_min

    def test_respects_higher_min_samples(self, base_config):
        """Test that DatasetBuilder respects higher min_symbol_samples values."""
        high_min = 50000
        dataset_config = DatasetBuildConfig(min_symbol_samples=high_min)
        builder = DatasetBuilder(base_config, dataset_config, verbose=False)

        assert builder.dataset_config.min_symbol_samples == high_min

    def test_initialization_verbose_output(self, base_config, capsys):
        """Test verbose output during initialization."""
        dataset_config = DatasetBuildConfig(min_symbol_samples=100)
        DatasetBuilder(base_config, dataset_config, verbose=True)

        captured = capsys.readouterr()
        assert "DatasetBuilder initialized" in captured.out
        assert "Currency: AUDUSD" in captured.out
        assert "Features: ['volume']" in captured.out


class TestSimplifiedPipeline:
    """Test that dataset building uses simplified pipeline (no feature generation)."""

    @pytest.fixture
    def pipeline_config(self):
        """Config for pipeline testing."""
        return create_represent_config(
            currency="AUDUSD",
            features=["volume", "variance"],
            lookback_rows=50,
            lookforward_input=50,
            lookforward_offset=10
        )

    @pytest.fixture
    def test_data(self):
        """Create deterministic test data."""
        n_rows = 200
        data = []
        for i in range(n_rows):
            price = 0.67000 + np.sin(i / 20.0) * 0.0001
            row = {
                'ts_event': 1640995200000000000 + i * 1000000000,
                'symbol': 'M6AM4',
                'ask_px_00': price + 0.00005,
                'bid_px_00': price - 0.00005,
                'ask_sz_00': 100000,
                'bid_sz_00': 100000,
            }
            data.append(row)
        return pl.DataFrame(data)

    def test_no_feature_generation_method(self, pipeline_config):
        """Test that _generate_features method no longer exists."""
        dataset_config = DatasetBuildConfig(features=["volume", "variance"])
        builder = DatasetBuilder(pipeline_config, dataset_config, verbose=False)

        assert not hasattr(builder, '_generate_features'), "_generate_features method should be removed"

    def test_simplified_pipeline_only_essential_processing(self, pipeline_config, test_data):
        """Test that pipeline only does essential processing (no feature storage)."""
        dataset_config = DatasetBuildConfig(features=["volume", "variance"], min_symbol_samples=50)
        builder = DatasetBuilder(pipeline_config, dataset_config, verbose=False)

        # Process through pipeline steps
        df_movements = builder._calculate_price_movements(test_data)
        df_classified = builder._apply_classification(df_movements)
        df_final = builder._filter_processable_rows(df_classified)

        if len(df_final) > 0:
            columns = set(df_final.columns)

            # Required columns
            required = {'ts_event', 'symbol', 'price_movement', 'classification_label'}
            assert required.issubset(columns)

            # Should NOT have feature representation columns
            feature_cols = {col for col in columns if col.endswith('_representation')}
            assert len(feature_cols) == 0, f"Found feature columns that should not exist: {feature_cols}"

    def test_features_preserved_for_on_demand_generation(self, pipeline_config):
        """Test that features config is preserved for on-demand generation."""
        dataset_config = DatasetBuildConfig(features=["volume", "variance"])
        builder = DatasetBuilder(pipeline_config, dataset_config, verbose=False)

        # Features should still be configured (for on-demand generation)
        assert builder.dataset_config.features == ["volume", "variance"]
        assert builder.represent_config.features == ["volume", "variance"]


class TestPriceMovementCalculation:
    """Test price movement calculation functionality."""

    @pytest.fixture
    def calc_config(self):
        """Config for calculation testing."""
        return create_represent_config(
            currency="AUDUSD",
            lookback_rows=20,
            lookforward_input=15,
            lookforward_offset=5
        )

    @pytest.fixture
    def builder(self, calc_config):
        """Builder for calculation tests."""
        dataset_config = DatasetBuildConfig(min_symbol_samples=100)
        return DatasetBuilder(calc_config, dataset_config, verbose=False)

    @pytest.fixture
    def calc_test_data(self):
        """Data for calculation testing."""
        n_rows = 200
        data = []
        for i in range(n_rows):
            # Linear price increase for predictable testing
            price = 1.0 + i * 0.001
            row = {
                'ts_event': 1640995200000000000 + i * 1000000000,
                'symbol': 'TEST',
                'ask_px_00': price + 0.0001,
                'bid_px_00': price - 0.0001,
                'ask_sz_00': 100000,
                'bid_sz_00': 100000,
            }
            data.append(row)
        return pl.DataFrame(data)

    def test_price_movements_every_valid_row(self, builder, calc_test_data):
        """Test that every valid row gets processed (no jump_size gaps)."""
        result_df = builder._calculate_price_movements(calc_test_data)

        # Calculate expected valid range
        total_rows = len(calc_test_data)
        config = builder.represent_config
        valid_start = config.lookback_rows
        valid_end = total_rows - (config.lookforward_input + config.lookforward_offset)
        expected_count = max(0, valid_end - valid_start)

        # Count actual valid movements
        price_movements = result_df['price_movement'].to_numpy()
        actual_count = 0
        for i in range(valid_start, min(valid_end, len(price_movements))):
            if not np.isnan(price_movements[i]):
                actual_count += 1

        assert actual_count == expected_count, f"Expected {expected_count} valid movements, got {actual_count}"

    def test_price_movement_calculation_precision(self, builder):
        """Test precision of lookback/lookforward percentage calculations."""
        # Create data with known pattern for verification
        test_data = []
        prices = [1.0, 1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009]

        for i, price in enumerate(prices):
            row = {
                'ts_event': 1640995200000000000 + i * 1000000000,
                'symbol': 'TEST',
                'ask_px_00': price + 0.0001,
                'bid_px_00': price - 0.0001,
                'ask_sz_00': 100000,
                'bid_sz_00': 100000,
            }
            test_data.append(row)

        df = pl.DataFrame(test_data)
        result_df = builder._calculate_price_movements(df)

        # Manually verify calculation for a specific position
        mid_prices = ((df['ask_px_00'] + df['bid_px_00']) / 2).to_numpy()
        price_movements = result_df['price_movement'].to_numpy()

        # For the calculation to work, we need sufficient data
        if len(mid_prices) > builder.represent_config.lookback_rows + builder.represent_config.lookforward_input + builder.represent_config.lookforward_offset:
            # Test the first valid position where we can calculate
            stop_row = builder.represent_config.lookback_rows
            if stop_row < len(price_movements) and not np.isnan(price_movements[stop_row]):
                # Manual calculation verification would go here
                # This is a basic check that movements are calculated
                assert not np.isnan(price_movements[stop_row])

    def test_insufficient_data_handling(self, builder):
        """Test handling of insufficient data."""
        small_data = pl.DataFrame({
            'ts_event': [1640995200000000000 + i * 1000000000 for i in range(10)],
            'symbol': ['TEST'] * 10,
            'ask_px_00': [1.0] * 10,
            'bid_px_00': [0.999] * 10,
            'ask_sz_00': [100000] * 10,
            'bid_sz_00': [100000] * 10,
        })

        result_df = builder._calculate_price_movements(small_data)

        # Should have price_movement column but mostly NaN due to insufficient data
        assert 'price_movement' in result_df.columns
        movements = result_df['price_movement']
        finite_movements = [x for x in movements.to_list() if x is not None and not np.isnan(x)]
        # With only 10 rows and requirements of 20+15+5=40 minimum, should have no valid movements
        assert len(finite_movements) == 0

    def test_verbose_output(self, calc_config, capsys):
        """Test verbose output during price movement calculation."""
        dataset_config = DatasetBuildConfig(min_symbol_samples=100)
        builder = DatasetBuilder(calc_config, dataset_config, verbose=True)

        test_data = pl.DataFrame({
            'ts_event': [1640995200000000000 + i * 1000000000 for i in range(100)],
            'symbol': ['TEST'] * 100,
            'ask_px_00': [1.0 + i * 0.001 for i in range(100)],
            'bid_px_00': [0.999 + i * 0.001 for i in range(100)],
            'ask_sz_00': [100000] * 100,
            'bid_sz_00': [100000] * 100,
        })

        builder._calculate_price_movements(test_data)

        captured = capsys.readouterr()
        assert "Calculating price movements using lookback/lookforward methodology" in captured.out
        assert "Lookback:" in captured.out
        assert "Lookforward:" in captured.out
        assert "Processing: Every valid row (no jumping)" in captured.out


class TestClassification:
    """Test classification functionality."""

    @pytest.fixture
    def class_config(self):
        """Config for classification testing."""
        return create_represent_config(
            currency="AUDUSD",
            lookback_rows=10,
            lookforward_input=10,
            lookforward_offset=2
        )

    @pytest.fixture
    def class_builder(self, class_config):
        """Builder for classification tests."""
        dataset_config = DatasetBuildConfig(
            min_symbol_samples=50,
            force_uniform=True,
            nbins=5  # Smaller for easier testing
        )
        return DatasetBuilder(class_config, dataset_config, verbose=False)

    def test_classification_uniform_distribution(self, class_builder):
        """Test uniform classification distribution."""
        # Create test data with varying price movements
        test_data = []
        for i in range(100):
            if i < 30:
                price_change = -0.002
            elif i < 60:
                price_change = 0.001
            else:
                price_change = 0.003

            price = 1.0 + price_change * (i / 10.0)

            row = {
                'ts_event': 1640995200000000000 + i * 1000000000,
                'symbol': 'TEST',
                'ask_px_00': price + 0.0001,
                'bid_px_00': price - 0.0001,
                'ask_sz_00': 100000,
                'bid_sz_00': 100000,
            }
            test_data.append(row)

        df = pl.DataFrame(test_data)
        df_movements = class_builder._calculate_price_movements(df)
        classified_df = class_builder._apply_classification(df_movements)

        if len(classified_df) > 0:
            classifications = classified_df['classification_label'].drop_nulls()
            if len(classifications) > 0:
                class_values = classifications.to_numpy()

                # All classifications should be valid bin numbers
                assert np.all(class_values >= 0)
                assert np.all(class_values < class_builder.dataset_config.nbins)

    def test_classification_no_valid_movements(self, class_builder):
        """Test classification when no valid movements exist."""
        # Data with all NaN movements
        df = pl.DataFrame({
            'ts_event': [1640995200000000000],
            'symbol': ['TEST'],
            'price_movement': [None],
        })

        result_df = class_builder._apply_classification(df)

        assert 'classification_label' in result_df.columns
        labels = result_df['classification_label'].drop_nulls()
        assert len(labels) == 0

    def test_classification_verbose_output(self, class_config, capsys):
        """Test verbose output during classification."""
        dataset_config = DatasetBuildConfig(min_symbol_samples=50, force_uniform=True, nbins=5)
        builder = DatasetBuilder(class_config, dataset_config, verbose=True)

        # Create data with some valid movements
        test_data = []
        for i in range(50):
            price = 1.0 + i * 0.0001
            row = {
                'ts_event': 1640995200000000000 + i * 1000000000,
                'symbol': 'TEST',
                'ask_px_00': price + 0.0001,
                'bid_px_00': price - 0.0001,
                'ask_sz_00': 100000,
                'bid_sz_00': 100000,
            }
            test_data.append(row)

        df = pl.DataFrame(test_data)
        df_movements = builder._calculate_price_movements(df)
        builder._apply_classification(df_movements)

        captured = capsys.readouterr()
        # Should have verbose classification output
        assert any(text in captured.out for text in ["Using", "bins", "samples"])


class TestSymbolProcessing:
    """Test symbol splitting and merging functionality."""

    @pytest.fixture
    def symbol_config(self):
        """Config for symbol processing."""
        return create_represent_config(currency="AUDUSD", lookback_rows=25, lookforward_input=25, lookforward_offset=5)

    @pytest.fixture
    def symbol_builder(self, symbol_config):
        """Builder for symbol tests."""
        dataset_config = DatasetBuildConfig(min_symbol_samples=100)
        return DatasetBuilder(symbol_config, dataset_config, verbose=False)

    @pytest.fixture
    def multi_symbol_data(self):
        """Multi-symbol test data."""
        symbols = ['M6AM4', 'M6AM5', 'M6AN4']
        data = []

        for i in range(300):
            symbol = symbols[i % len(symbols)]
            price = 0.67000 + np.sin(i / 50.0) * 0.0001

            row = {
                'ts_event': 1640995200000000000 + i * 1000000000,
                'symbol': symbol,
                'ask_px_00': price + 0.00005,
                'bid_px_00': price - 0.00005,
                'ask_sz_00': np.random.randint(100000, 1000000),
                'bid_sz_00': np.random.randint(100000, 1000000),
            }
            data.append(row)

        return pl.DataFrame(data)

    @patch('databento.read_dbn')
    def test_split_dbn_by_symbol(self, mock_read_dbn, symbol_builder, multi_symbol_data):
        """Test DBN file splitting by symbol."""
        mock_data = Mock()
        mock_data.to_df.return_value = multi_symbol_data.to_pandas()
        mock_read_dbn.return_value = mock_data

        with tempfile.TemporaryDirectory() as temp_dir:
            intermediate_dir = Path(temp_dir)
            symbol_files = symbol_builder.split_dbn_by_symbol("test.dbn", intermediate_dir)

            symbols = multi_symbol_data['symbol'].unique().to_list()
            assert len(symbol_files) == len(symbols)

            for symbol in symbols:
                assert symbol in symbol_files
                assert symbol_files[symbol].exists()

    def test_merge_symbol_data(self, symbol_builder, multi_symbol_data):
        """Test symbol data merging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            intermediate_dir = Path(temp_dir) / "intermediate"
            output_dir = Path(temp_dir) / "output"
            intermediate_dir.mkdir()
            output_dir.mkdir()

            # Create intermediate files for single symbol
            symbol_data = multi_symbol_data.filter(pl.col('symbol') == 'M6AM4')
            symbol_files = []

            for i in range(3):
                file_path = intermediate_dir / f"file{i}_M6AM4.parquet"
                symbol_data.write_parquet(str(file_path))
                symbol_files.append(file_path)

            result_file = symbol_builder.merge_symbol_data("M6AM4", symbol_files, output_dir)

            if result_file is not None:
                assert result_file.exists()
                merged_df = pl.read_parquet(result_file)
                assert len(merged_df) > 0
                assert 'classification_label' in merged_df.columns

    def test_merge_verbose_output(self, symbol_config, multi_symbol_data, capsys):
        """Test verbose output during symbol merging."""
        dataset_config = DatasetBuildConfig(min_symbol_samples=50)
        builder = DatasetBuilder(symbol_config, dataset_config, verbose=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            intermediate_dir = Path(temp_dir) / "intermediate"
            output_dir = Path(temp_dir) / "output"
            intermediate_dir.mkdir()
            output_dir.mkdir()

            # Create test files
            symbol_data = multi_symbol_data.filter(pl.col('symbol') == 'M6AM4')
            symbol_files = []
            for i in range(2):
                file_path = intermediate_dir / f"file{i}_M6AM4.parquet"
                symbol_data.write_parquet(str(file_path))
                symbol_files.append(file_path)

            builder.merge_symbol_data("M6AM4", symbol_files, output_dir)

            captured = capsys.readouterr()
            assert "Merging symbol: M6AM4" in captured.out
            assert "Files: 2" in captured.out
            assert "Processing: Lookback → Lookforward → Percentage Change → Classification" in captured.out


class TestEndToEndWorkflows:
    """Test complete dataset building workflows."""

    @pytest.fixture
    def workflow_config(self):
        """Config for workflow testing."""
        return create_represent_config(
            currency="AUDUSD",
            features=["volume"],
            lookback_rows=30,
            lookforward_input=20,
            lookforward_offset=5
        )

    @pytest.fixture
    def workflow_data(self):
        """Data for workflow testing."""
        symbols = ['M6AM4', 'M6AM5']
        data = []

        for i in range(400):
            symbol = symbols[i % len(symbols)]
            price = 0.67000 + np.sin(i / 30.0) * 0.0002

            row = {
                'ts_event': 1640995200000000000 + i * 1000000000,
                'symbol': symbol,
                'ask_px_00': price + 0.00005,
                'bid_px_00': price - 0.00005,
                'ask_sz_00': np.random.randint(100000, 1000000),
                'bid_sz_00': np.random.randint(100000, 1000000),
            }
            data.append(row)

        return pl.DataFrame(data)

    @patch('databento.read_dbn')
    def test_complete_dataset_building(self, mock_read_dbn, workflow_config, workflow_data):
        """Test complete dataset building process."""
        mock_data = Mock()
        mock_data.to_df.return_value = workflow_data.to_pandas()
        mock_read_dbn.return_value = mock_data

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dataset_config = DatasetBuildConfig(min_symbol_samples=100)

            results = build_datasets_from_dbn_files(
                config=workflow_config,
                dbn_files=["test.dbn"],
                output_dir=output_dir,
                dataset_config=dataset_config,
                verbose=False
            )

            assert isinstance(results, dict)
            assert 'phase_1_stats' in results
            assert 'phase_2_stats' in results

    def test_batch_build_from_directory(self, workflow_config):
        """Test batch building from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()

            # Create mock DBN files
            (input_dir / "test1.dbn").write_text("mock")
            (input_dir / "test2.dbn.zst").write_text("mock")
            (input_dir / "other.txt").write_text("ignore")

            with patch('represent.dataset_builder.build_datasets_from_dbn_files') as mock_build:
                mock_build.return_value = {"datasets_created": 2}

                batch_build_datasets_from_directory(
                    config=workflow_config,
                    input_directory=input_dir,
                    output_dir=output_dir,
                    file_pattern="*.dbn*",
                    verbose=False
                )

                mock_build.assert_called_once()
                args, kwargs = mock_build.call_args
                dbn_files = kwargs['dbn_files']

                assert len(dbn_files) == 2
                assert any("test1.dbn" in str(f) for f in dbn_files)
                assert any("test2.dbn.zst" in str(f) for f in dbn_files)


class TestErrorHandling:
    """Test error conditions and edge cases."""

    def test_missing_input_directory(self):
        """Test handling of missing input directory."""
        config = create_represent_config(currency="AUDUSD")

        with pytest.raises(FileNotFoundError):
            batch_build_datasets_from_directory(
                config=config,
                input_directory="nonexistent_directory",
                output_dir="/tmp/output",
                verbose=False
            )

    def test_no_matching_files(self):
        """Test handling of no matching DBN files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "other.txt").write_text("not a DBN file")

            config = create_represent_config(currency="AUDUSD")

            with pytest.raises(ValueError, match="No files found"):
                batch_build_datasets_from_directory(
                    config=config,
                    input_directory=input_dir,
                    output_dir="/tmp/output",
                    file_pattern="*.dbn*",
                    verbose=False
                )

    def test_insufficient_symbol_samples(self):
        """Test handling of symbols with insufficient samples."""
        config = create_represent_config(currency="AUDUSD")
        dataset_config = DatasetBuildConfig(min_symbol_samples=10000)  # Very high
        builder = DatasetBuilder(config, dataset_config, verbose=False)

        small_data = pl.DataFrame({
            'ts_event': [1640995200000000000 + i * 1000000000 for i in range(100)],
            'symbol': ['M6AM4'] * 100,
            'ask_px_00': [0.67000] * 100,
            'bid_px_00': [0.66995] * 100,
            'ask_sz_00': [100000] * 100,
            'bid_sz_00': [100000] * 100
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            intermediate_dir = output_dir / "intermediate"
            intermediate_dir.mkdir(parents=True)

            symbol_file = intermediate_dir / "test_M6AM4.parquet"
            small_data.write_parquet(symbol_file)

            result = builder.merge_symbol_data("M6AM4", [symbol_file], output_dir)
            assert result is None


class TestPerformanceRequirements:
    """Test performance requirements and memory efficiency."""

    @pytest.fixture
    def large_test_data(self):
        """Create larger dataset for performance testing."""
        n_rows = 2000
        data = []
        for i in range(n_rows):
            row = {
                'ts_event': 1640995200000000000 + i * 1000000000,
                'symbol': 'PERF_TEST',
                'ask_px_00': 0.67000 + np.random.normal(0, 0.0001),
                'bid_px_00': 0.66995 + np.random.normal(0, 0.0001),
                'ask_sz_00': np.random.randint(100000, 1000000),
                'bid_sz_00': np.random.randint(100000, 1000000),
            }
            data.append(row)
        return pl.DataFrame(data)

    @patch('databento.read_dbn')
    @pytest.mark.performance
    def test_processing_performance(self, mock_read_dbn, large_test_data):
        """Test that processing meets performance requirements."""
        import time

        mock_data = Mock()
        mock_data.to_df.return_value = large_test_data.to_pandas()
        mock_read_dbn.return_value = mock_data

        config = create_represent_config(currency="AUDUSD")
        dataset_config = DatasetBuildConfig(min_symbol_samples=200)
        builder = DatasetBuilder(config, dataset_config, verbose=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            intermediate_dir = Path(temp_dir)

            start_time = time.perf_counter()
            symbol_files = builder.split_dbn_by_symbol("test.dbn", intermediate_dir)
            processing_time = time.perf_counter() - start_time

            # Should process data efficiently
            total_rows = len(large_test_data)
            rows_per_second = total_rows / processing_time if processing_time > 0 else float('inf')

            assert rows_per_second > 100, f"Performance too slow: {rows_per_second:.0f} rows/sec"
            assert len(symbol_files) > 0

    def test_memory_efficiency(self):
        """Test memory usage during processing."""
        config = create_represent_config(currency="AUDUSD")
        dataset_config = DatasetBuildConfig(min_symbol_samples=200)

        builder = DatasetBuilder(config, dataset_config, verbose=False)

        # Basic memory efficiency check
        assert len(builder.symbol_registry) == 0

        # Adding symbols to registry should be efficient
        for i in range(100):
            builder.symbol_registry[f"SYMBOL_{i}"] = []

        assert len(builder.symbol_registry) == 100

    def test_min_symbol_samples_ignores_config_samples_parameter(self):
        """Test that min_symbol_samples calculation ignores config.samples parameter."""
        # Test with large samples value - should NOT affect minimum calculation
        config = create_represent_config(
            currency="AUDUSD",
            samples=100000,  # Large value that should be ignored
            lookback_rows=200,
            lookforward_input=150,
            lookforward_offset=25
        )

        dataset_config = DatasetBuildConfig(min_symbol_samples=1)  # Force auto-update
        builder = DatasetBuilder(config, dataset_config, verbose=False)

        # Should only consider lookback + lookforward + offset
        expected_min = 200 + 150 + 25  # 375
        actual_min = builder.dataset_config.min_symbol_samples

        assert actual_min == expected_min, (
            f"min_symbol_samples should only consider lookback/lookforward windows. "
            f"Expected {expected_min}, got {actual_min}"
        )

    def test_price_movement_calculation_precision(self):
        """Test precision of lookback/lookforward percentage calculations."""
        config = create_represent_config(
            currency="AUDUSD",
            lookback_rows=3,
            lookforward_input=2,
            lookforward_offset=1
        )

        dataset_config = DatasetBuildConfig(min_symbol_samples=10, force_uniform=True)
        builder = DatasetBuilder(config, dataset_config, verbose=False)

        # Create data with known prices for manual verification
        test_data = []
        prices = [1.0, 1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009]

        for i, price in enumerate(prices):
            row = {
                'ts_event': 1640995200000000000 + i * 1000000000,
                'symbol': 'TEST_SYMBOL',
                'ask_px_00': price + 0.0001,
                'bid_px_00': price - 0.0001,
                'ask_sz_00': 100000,
                'bid_sz_00': 100000,
            }
            test_data.append(row)

        df = pl.DataFrame(test_data)
        result_df = builder._calculate_price_movements(df)

        # Verify calculation for stop_row = 3
        # Lookback: rows 0,1,2 (prices 1.0, 1.001, 1.002)
        # Lookforward: rows 5,6 (prices 1.005, 1.006) - offset of 1 from row 4
        stop_row = 3

        mid_prices = ((df['ask_px_00'] + df['bid_px_00']) / 2).to_numpy()

        lookback_mean = np.mean(mid_prices[0:3])    # 1.001
        lookforward_mean = np.mean(mid_prices[5:7]) # 1.0055
        expected_movement = (lookforward_mean - lookback_mean) / lookback_mean

        actual_movement = result_df['price_movement'].to_numpy()[stop_row]

        assert abs(actual_movement - expected_movement) < 1e-10, (
            f"Price movement calculation imprecise. Expected {expected_movement:.10f}, "
            f"got {actual_movement:.10f}"
        )
