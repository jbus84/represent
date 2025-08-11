"""
Tests for GlobalThresholdCalculator functionality.
Focused on testing the actual API and functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from represent.config import create_represent_config
from represent.global_threshold_calculator import (
    GlobalThresholdCalculator,
    GlobalThresholds,
    calculate_global_thresholds,
)


class TestGlobalThresholds:
    """Test GlobalThresholds dataclass."""

    def test_global_thresholds_creation(self):
        """Test GlobalThresholds dataclass creation."""
        quantiles = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        stats = {"mean": 0.1, "std": 5.2, "min": -50.0, "max": 45.0}

        thresholds = GlobalThresholds(
            quantile_boundaries=quantiles,
            nbins=5,
            sample_size=10000,
            files_analyzed=5,
            price_movement_stats=stats
        )

        assert thresholds.nbins == 5
        assert thresholds.sample_size == 10000
        assert thresholds.files_analyzed == 5
        assert np.array_equal(thresholds.quantile_boundaries, quantiles)
        assert thresholds.price_movement_stats == stats

    def test_global_thresholds_validation(self):
        """Test that GlobalThresholds contains expected data types."""
        quantiles = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        stats = {"mean": 0.1, "std": 5.2}

        thresholds = GlobalThresholds(
            quantile_boundaries=quantiles,
            nbins=5,
            sample_size=10000,
            files_analyzed=3,
            price_movement_stats=stats
        )

        assert isinstance(thresholds.quantile_boundaries, np.ndarray)
        assert isinstance(thresholds.nbins, int)
        assert isinstance(thresholds.sample_size, int)
        assert isinstance(thresholds.files_analyzed, int)
        assert isinstance(thresholds.price_movement_stats, dict)


class TestGlobalThresholdCalculator:
    """Test GlobalThresholdCalculator functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = create_represent_config("AUDUSD")

    def test_calculator_initialization(self):
        """Test calculator initializes with correct configuration."""
        calc = GlobalThresholdCalculator(
            config=self.config,
            sample_fraction=0.3,
            verbose=False
        )

        assert calc.currency == "AUDUSD"
        assert calc.sample_fraction == 0.3
        assert calc.max_samples_per_file == self.config.max_samples_per_file
        assert calc.verbose is False

    def test_calculator_uses_config_nbins(self):
        """Test calculator uses RepresentConfig nbins when not specified."""
        calc = GlobalThresholdCalculator(config=self.config)

        # Should use config nbins (13 for AUDUSD)
        assert calc.config.nbins == 13

    def test_calculator_different_currencies(self):
        """Test calculator with different currencies."""
        currencies = ["AUDUSD", "EURUSD", "GBPUSD"]

        for currency in currencies:
            currency_config = create_represent_config(currency)
            calc = GlobalThresholdCalculator(config=currency_config, verbose=False)
            assert calc.currency == currency
            assert hasattr(calc, 'config')
            assert calc.config.nbins > 0

    def test_calculator_parameter_validation(self):
        """Test calculator parameter validation."""
        # Test with valid parameters
        calc = GlobalThresholdCalculator(
            config=self.config,
            sample_fraction=0.5,
            verbose=True
        )

        assert calc.sample_fraction == 0.5
        assert calc.max_samples_per_file == self.config.max_samples_per_file
        assert calc.verbose is True

    def test_calculator_config_integration(self):
        """Test that calculator integrates with RepresentConfig."""
        calc = GlobalThresholdCalculator(config=self.config, verbose=False)

        # Should have valid config
        assert hasattr(calc, 'config')
        assert calc.config.micro_pip_size > 0
        assert calc.config.nbins > 0


class TestGlobalThresholdCalculatorAPI:
    """Test the public API function."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = create_represent_config("AUDUSD")

    def test_calculate_global_thresholds_function_signature(self):
        """Test the function signature and parameters."""
        # Test that function exists and is callable
        assert callable(calculate_global_thresholds)

        # Test with mock to avoid actual file processing
        with patch.object(GlobalThresholdCalculator, 'calculate_global_thresholds') as mock_calc:
            mock_thresholds = GlobalThresholds(
                quantile_boundaries=np.array([-5, 0, 5]),
                nbins=3,
                sample_size=1000,
                files_analyzed=2,
                price_movement_stats={"mean": 0.0, "std": 3.0}
            )
            mock_calc.return_value = mock_thresholds

            calculate_global_thresholds(
                config=self.config,
                data_directory="dummy_path",
                sample_fraction=0.3
            )

            # Should call the calculator method
            mock_calc.assert_called_once()

    def test_calculate_global_thresholds_parameters(self):
        """Test parameter passing to GlobalThresholdCalculator."""
        with patch.object(GlobalThresholdCalculator, '__init__', return_value=None) as mock_init:
            with patch.object(GlobalThresholdCalculator, 'calculate_global_thresholds') as mock_calc:
                mock_calc.return_value = MagicMock()

                calculate_global_thresholds(
                    config=self.config,
                    data_directory="test_dir",
                    sample_fraction=0.2,
                    verbose=False
                )

                # Verify correct parameters passed to constructor
                mock_init.assert_called_once_with(
                    config=self.config,
                    sample_fraction=0.2,
                    verbose=False
                )


class TestGlobalThresholdCalculatorErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = create_represent_config("AUDUSD")

    def test_unsupported_currency(self):
        """Test behavior with unsupported currency that still matches format."""
        # Use a valid format currency that's not in our predefined list
        nzdusd_config = create_represent_config("NZDUSD")
        calc = GlobalThresholdCalculator(config=nzdusd_config, verbose=False)
        assert calc.currency == "NZDUSD"
        # Config should still be created with defaults
        assert hasattr(calc, 'config')
        assert calc.config.nbins > 0

    def test_edge_case_parameters(self):
        """Test behavior with edge case parameters."""
        # Test with extreme sample fraction
        calc = GlobalThresholdCalculator(config=self.config, sample_fraction=1.5, verbose=False)
        assert calc.sample_fraction == 1.5  # Should accept it

        # Test with very small sample fraction
        calc = GlobalThresholdCalculator(config=self.config, sample_fraction=0.01, verbose=False)
        assert calc.sample_fraction == 0.01

        # Test with zero max samples
        # Test that we're using config value
        calc = GlobalThresholdCalculator(config=self.config, verbose=False)
        assert calc.max_samples_per_file == self.config.max_samples_per_file

    def test_calculator_uses_config_nbins(self):
        """Test calculator uses config nbins (no override)."""
        calc = GlobalThresholdCalculator(config=self.config, verbose=False)
        assert calc.nbins == self.config.nbins

        # Test with different currency config
        gbpusd_config = create_represent_config("GBPUSD")
        calc2 = GlobalThresholdCalculator(config=gbpusd_config, verbose=False)
        assert calc2.nbins == gbpusd_config.nbins


class TestGlobalThresholdCalculatorConfiguration:
    """Test configuration integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = create_represent_config("AUDUSD")

    def test_configuration_consistency(self):
        """Test that calculator maintains configuration consistency."""
        calc = GlobalThresholdCalculator(config=self.config, verbose=False)
        config = create_represent_config("AUDUSD")

        # Should use same micro pip size
        assert calc.config.micro_pip_size == config.micro_pip_size

    def test_different_currency_configs(self):
        """Test calculator with different currency configurations."""
        currencies = ["AUDUSD", "EURUSD", "GBPUSD"]
        calculators = {}

        for currency in currencies:
            currency_config = create_represent_config(currency)
            calculators[currency] = GlobalThresholdCalculator(config=currency_config, verbose=False)

        # All should have valid configurations
        for currency, calc in calculators.items():
            assert calc.currency == currency
            assert hasattr(calc, 'config')
            assert calc.config.micro_pip_size > 0
            assert calc.config.nbins > 0

    def test_nbins_from_config(self):
        """Test that nbins always comes from config (no parameter override)."""
        # Test that nbins always matches config
        calc = GlobalThresholdCalculator(config=self.config, verbose=False)
        assert calc.nbins == self.config.nbins

        # Test with different currency configs
        for currency in ["EURUSD", "GBPUSD", "USDJPY"]:
            currency_config = create_represent_config(currency)
            calc = GlobalThresholdCalculator(config=currency_config, verbose=False)
            assert calc.nbins == currency_config.nbins

    def test_calculator_memory_efficiency(self):
        """Test that calculator doesn't consume excessive memory."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create multiple calculators
        calcs = [GlobalThresholdCalculator(config=self.config, verbose=False) for _ in range(10)]

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory
        assert memory_increase < 50  # Less than 50MB increase
        assert len(calcs) == 10

        # All should have same configuration for same currency
        first_config = calcs[0].config
        for calc in calcs[1:]:
            assert calc.config.nbins == first_config.nbins
            assert calc.config.micro_pip_size == first_config.micro_pip_size


class TestGlobalThresholdCalculatorFileProcessing:
    """Test file processing methods with simpler approaches."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = create_represent_config("AUDUSD")

    def test_load_dbn_file_sample_method_exists(self):
        """Test that load_dbn_file_sample method exists."""
        calc = GlobalThresholdCalculator(config=self.config, verbose=False)

        # Test that method exists and is callable
        assert hasattr(calc, 'load_dbn_file_sample')
        assert callable(calc.load_dbn_file_sample)

    def test_calculate_global_thresholds_method_exists(self):
        """Test that calculate_global_thresholds method exists."""
        calc = GlobalThresholdCalculator(config=self.config, verbose=False)

        # Test that method exists and is callable
        assert hasattr(calc, 'calculate_global_thresholds')
        assert callable(calc.calculate_global_thresholds)

    def test_calculator_config_integration(self):
        """Test that calculator integrates properly with config."""
        calc = GlobalThresholdCalculator(config=self.config, verbose=False)

        # Test config attributes are accessible
        assert calc.config.lookback_rows > 0
        assert calc.config.lookforward_input > 0
        assert calc.config.lookforward_offset >= 0
        assert calc.config.jump_size > 0


class TestGlobalThresholdCalculatorVerboseOutput:
    """Test verbose output and edge cases."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = create_represent_config("AUDUSD")

    def test_verbose_initialization(self):
        """Test verbose output during initialization."""
        import io
        import sys

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        _ = GlobalThresholdCalculator(config=self.config, verbose=True)

        # Restore stdout
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        assert "GlobalThresholdCalculator initialized" in output
        assert "Currency: AUDUSD" in output
        assert "Bins:" in output

    def test_unique_quantile_boundaries_handling(self):
        """Test handling of non-unique quantile boundaries."""
        calc = GlobalThresholdCalculator(config=self.config, verbose=False)

        # Create data with very little variance (will create duplicate quantiles)
        mock_movements = np.array([0.0] * 500 + [0.0001] * 500)  # Mostly zeros

        with patch.object(calc, 'load_dbn_file_sample', return_value=mock_movements), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.glob') as mock_glob:

            mock_files = [Path("file1.dbn")]
            mock_glob.return_value = mock_files

            result = calc.calculate_global_thresholds("dummy_dir")

            # Should handle duplicate quantiles and create valid boundaries
            assert isinstance(result, GlobalThresholds)
            assert len(result.quantile_boundaries) >= 2  # At least min and max

    def test_sample_fraction_behavior(self):
        """Test different sample fraction values."""
        # Test with different sample fractions
        for fraction in [0.1, 0.5, 1.0]:
            calc = GlobalThresholdCalculator(
                config=self.config,
                sample_fraction=fraction,
                verbose=False
            )
            assert calc.sample_fraction == fraction

            # Test that the fraction is used in file selection logic
            mock_movements = np.random.normal(0, 0.001, 1000)

            with patch.object(calc, 'load_dbn_file_sample', return_value=mock_movements), \
                 patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.glob') as mock_glob:

                # Mock 10 files
                mock_files = [Path(f"file{i}.dbn") for i in range(10)]
                mock_glob.return_value = mock_files

                result = calc.calculate_global_thresholds("dummy_dir")

                # Verify that sample size reflects the fraction
                expected_files = max(1, int(10 * fraction))
                assert result.files_analyzed <= expected_files
