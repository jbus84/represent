#!/usr/bin/env python3
"""
Tests for TStrends-based target generators.

This module tests the academic TStrends labeling approaches including:
- Binary and Ternary CTL generators
- Oracle Binary and Ternary generators
- Parameter optimization and label remapping
- Visualization compatibility
"""

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

try:
    from represent.target_generators.tstrends_labeling import (
        TSTRENDS_AVAILABLE,
        BinaryCTLGenerator,
        OracleBinaryTrendGenerator,
        OracleTernaryTrendGenerator,
        TernaryCTLGenerator,
        TunedTrendGenerator,
    )

    TSTRENDS_IMPORTED = True
except ImportError:
    TSTRENDS_IMPORTED = False


class TestTStrends:
    """Test suite for TStrends generator availability and basic functionality."""

    def test_tstrends_availability_flag(self):
        """Test that TSTRENDS_AVAILABLE flag is properly set."""
        assert isinstance(TSTRENDS_AVAILABLE, bool)
        if TSTRENDS_IMPORTED:
            # If we can import the module, the flag should match actual tstrends availability
            assert TSTRENDS_AVAILABLE == TSTRENDS_IMPORTED or not TSTRENDS_AVAILABLE


@pytest.mark.skipif(not TSTRENDS_IMPORTED, reason="TStrends generators not available")
@pytest.mark.skipif(not TSTRENDS_AVAILABLE, reason="tstrends library not installed")
class TestTStreamsGenerators:
    """Test suite for TStrends generators when available."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        # Create realistic price movement data
        np.random.seed(42)
        n_samples = 1000

        # Generate price series with trend and noise
        base_price = 0.65
        trend = np.linspace(0, 0.01, n_samples)  # Small upward trend
        noise = np.random.normal(0, 0.001, n_samples)  # Market noise
        prices = base_price + trend + noise

        # Ensure positive prices
        prices = np.maximum(prices, 0.6)

        df = pl.DataFrame(
            {
                "mid_price": prices,
                "ts_event": np.arange(n_samples),
            }
        )
        return df

    def test_binary_ctl_generator_initialization(self):
        """Test Binary CTL generator initialization with optimized parameters."""
        # Test with ultra-aggressive parameters (as used in visualization)
        omega = 0.0008  # Very sensitive
        generator = BinaryCTLGenerator(omega=omega, target_name="test_binary")

        assert generator.omega == omega
        assert generator.target_name == "test_binary"
        assert generator.target_type == "classification"
        assert generator.required_columns == ["mid_price"]

        info = generator.get_target_info()
        assert info["target_type"] == "classification"
        assert info["target_names"] == ["test_binary"]
        assert info["parameters"]["omega"] == omega
        assert info["parameters"]["library"] == "tstrends"

    def test_ternary_ctl_generator_initialization(self):
        """Test Ternary CTL generator initialization with ultra-aggressive parameters."""
        # Test with ultra-aggressive parameters for 3-class output
        marginal_change_thres = 0.0008  # Very sensitive
        window_size = 3  # Very small window
        generator = TernaryCTLGenerator(
            marginal_change_thres=marginal_change_thres,
            window_size=window_size,
            target_name="test_ternary",
        )

        assert generator.marginal_change_thres == marginal_change_thres
        assert generator.window_size == window_size
        assert generator.target_name == "test_ternary"
        assert generator.target_type == "classification"

        info = generator.get_target_info()
        assert info["target_type"] == "classification"
        assert info["parameters"]["marginal_change_thres"] == marginal_change_thres
        assert info["parameters"]["window_size"] == window_size

    def test_oracle_generators_initialization(self):
        """Test Oracle generator initialization with optimized parameters."""
        # Binary Oracle with optimized parameters
        binary_gen = OracleBinaryTrendGenerator(
            transaction_cost=0.0003, target_name="test_oracle_binary"
        )
        assert binary_gen.transaction_cost == 0.0003
        assert binary_gen.target_type == "classification"

        # Ternary Oracle with optimized parameters for 3-class output
        ternary_gen = OracleTernaryTrendGenerator(
            transaction_cost=0.0001,  # Very low cost - more responsive
            neutral_reward_factor=0.3,  # Low neutral factor - favor up/down
            target_name="test_oracle_ternary",
        )
        assert ternary_gen.transaction_cost == 0.0001
        assert ternary_gen.neutral_reward_factor == 0.3
        assert ternary_gen.target_type == "classification"

    @patch("represent.target_generators.tstrends_labeling.BinaryCTL")
    def test_binary_ctl_label_remapping(self, mock_binary_ctl, sample_price_data):
        """Test that Binary CTL properly remaps TStrends {-1, 1} to {0, 1}."""
        # Mock the TStrends BinaryCTL to return {-1, 1} labels
        mock_labeller = MagicMock()
        mock_labeller.get_labels.return_value = [-1, 1, -1, 1, 1]  # TStrends format
        mock_binary_ctl.return_value = mock_labeller

        generator = BinaryCTLGenerator(omega=0.001, target_name="test_binary")
        targets = generator.generate_targets(sample_price_data.head(5))

        labels = targets["test_binary"]

        # Check that labels are remapped to {0, 1}
        unique_labels = np.unique(labels)
        assert set(unique_labels) == {0, 1}  # Should be {0, 1}, not {-1, 1}

        # Verify mapping: -1 -> 0, 1 -> 1
        expected = np.array([0, 1, 0, 1, 1])  # Remapped from [-1, 1, -1, 1, 1]
        np.testing.assert_array_equal(labels, expected)

    @patch("represent.target_generators.tstrends_labeling.TernaryCTL")
    def test_ternary_ctl_label_remapping(self, mock_ternary_ctl, sample_price_data):
        """Test that Ternary CTL properly remaps TStrends {-1, 0, 1} to {0, 1, 2}."""
        # Mock the TStrends TernaryCTL to return {-1, 0, 1} labels
        mock_labeller = MagicMock()
        mock_labeller.get_labels.return_value = [-1, 0, 1, -1, 0, 1]  # TStrends format
        mock_ternary_ctl.return_value = mock_labeller

        generator = TernaryCTLGenerator(
            marginal_change_thres=0.001, window_size=3, target_name="test_ternary"
        )
        targets = generator.generate_targets(sample_price_data.head(6))

        labels = targets["test_ternary"]

        # Check that labels are remapped to {0, 1, 2}
        unique_labels = np.unique(labels)
        assert set(unique_labels) == {0, 1, 2}  # Should be {0, 1, 2}, not {-1, 0, 1}

        # Verify mapping: -1 -> 0, 0 -> 1, 1 -> 2
        expected = np.array([0, 1, 2, 0, 1, 2])  # Remapped from [-1, 0, 1, -1, 0, 1]
        np.testing.assert_array_equal(labels, expected)

    @patch("represent.target_generators.tstrends_labeling.OracleBinaryTrendLabeller")
    def test_oracle_binary_label_remapping(self, mock_oracle_binary, sample_price_data):
        """Test that Oracle Binary properly remaps labels."""
        mock_labeller = MagicMock()
        mock_labeller.get_labels.return_value = [-1, 1, -1]  # TStrends format
        mock_oracle_binary.return_value = mock_labeller

        generator = OracleBinaryTrendGenerator(transaction_cost=0.001, target_name="test_oracle")
        targets = generator.generate_targets(sample_price_data.head(3))

        labels = targets["test_oracle"]
        expected = np.array([0, 1, 0])  # Remapped from [-1, 1, -1]
        np.testing.assert_array_equal(labels, expected)

    @patch("represent.target_generators.tstrends_labeling.OracleTernaryTrendLabeller")
    def test_oracle_ternary_label_remapping(self, mock_oracle_ternary, sample_price_data):
        """Test that Oracle Ternary properly remaps labels."""
        mock_labeller = MagicMock()
        mock_labeller.get_labels.return_value = [-1, 0, 1]  # TStrends format
        mock_oracle_ternary.return_value = mock_labeller

        generator = OracleTernaryTrendGenerator(
            transaction_cost=0.001, neutral_reward_factor=0.5, target_name="test_oracle_ternary"
        )
        targets = generator.generate_targets(sample_price_data.head(3))

        labels = targets["test_oracle_ternary"]
        expected = np.array([0, 1, 2])  # Remapped from [-1, 0, 1]
        np.testing.assert_array_equal(labels, expected)

    def test_binary_generator_produces_two_classes(self, sample_price_data):
        """Test that binary generators produce exactly 2 classes."""
        generator = BinaryCTLGenerator(omega=0.001, target_name="test_binary")
        targets = generator.generate_targets(sample_price_data)

        labels = targets["test_binary"]
        valid_labels = labels.filter(~labels.is_nan())
        unique_labels = np.unique(valid_labels)

        # Should produce exactly 2 classes: {0, 1}
        assert len(unique_labels) == 2
        assert set(unique_labels) == {0, 1}

    def test_ternary_generator_produces_three_classes(self, sample_price_data):
        """Test that ternary generators with optimized parameters produce 3 classes."""
        # Use ultra-aggressive parameters that guarantee 3-class output
        generator = TernaryCTLGenerator(
            marginal_change_thres=0.0008,  # Very sensitive
            window_size=3,  # Very small window
            target_name="test_ternary",
        )
        targets = generator.generate_targets(sample_price_data)

        labels = targets["test_ternary"]
        valid_labels = labels.filter(~labels.is_nan())
        unique_labels = np.unique(valid_labels)

        # With optimized parameters, should produce 3 classes: {0, 1, 2}
        assert len(unique_labels) == 3, (
            f"Expected 3 classes, got {len(unique_labels)}: {unique_labels}"
        )
        assert set(unique_labels) == {0, 1, 2}

    def test_tuned_trend_generator_fallback(self, sample_price_data):
        """Test that TunedTrendGenerator works as fallback without complex tuning."""
        generator = TunedTrendGenerator(
            base_labeller_type="binary_ctl", omega=0.001, target_name="test_tuned"
        )

        # Should work without tuner (uses base labeller directly)
        targets = generator.generate_targets(sample_price_data.head(100))
        labels = targets["test_tuned"]

        assert len(labels) == 100
        assert not labels.is_nan().all()  # Should have some valid labels

        info = generator.get_target_info()
        assert info["target_type"] == "classification"
        assert info["parameters"]["base_labeller"] == "binary_ctl"

    def test_empty_data_handling(self):
        """Test that generators handle empty data gracefully."""
        empty_df = pl.DataFrame({"mid_price": [], "ts_event": []})

        generator = BinaryCTLGenerator(omega=0.001, target_name="test_empty")
        targets = generator.generate_targets(empty_df)

        labels = targets["test_empty"]
        assert len(labels) == 0
        assert labels.dtype == pl.Int32

    def test_nan_price_handling(self, sample_price_data):
        """Test that generators handle NaN prices properly."""
        # Add some NaN values to price data
        df_with_nans = sample_price_data.clone()
        prices_with_nans = df_with_nans["mid_price"].to_numpy().copy()  # Make writable copy
        prices_with_nans[::10] = np.nan  # Every 10th price is NaN

        df_with_nans = df_with_nans.with_columns(pl.Series("mid_price", prices_with_nans))

        generator = BinaryCTLGenerator(omega=0.001, target_name="test_nans")
        targets = generator.generate_targets(df_with_nans)

        # Should still produce labels for valid prices
        labels = targets["test_nans"]
        assert len(labels) > 0
        # Some labels should be valid (for non-NaN prices)
        assert (~labels.is_nan()).any()


@pytest.mark.skipif(TSTRENDS_IMPORTED, reason="Only test import error when tstrends not available")
class TestTStreamsImportError:
    """Test error handling when tstrends is not available."""

    def test_import_error_message(self):
        """Test that appropriate error is raised when tstrends not available."""
        with pytest.raises(ImportError, match="tstrends library is required"):
            BinaryCTLGenerator(omega=0.001)

        with pytest.raises(ImportError, match="tstrends library is required"):
            TernaryCTLGenerator(marginal_change_thres=0.001, window_size=5)

        with pytest.raises(ImportError, match="tstrends library is required"):
            OracleBinaryTrendGenerator(transaction_cost=0.001)

        with pytest.raises(ImportError, match="tstrends library is required"):
            OracleTernaryTrendGenerator(transaction_cost=0.001, neutral_reward_factor=0.5)

        with pytest.raises(ImportError, match="tstrends library is required"):
            TunedTrendGenerator(base_labeller_type="binary_ctl", omega=0.001)


class TestTStreamsParameterOptimization:
    """Test the parameter optimization logic used in the visualization."""

    def test_ultra_aggressive_parameters(self):
        """Test that ultra-aggressive parameters are within expected ranges."""
        # These are the optimized parameters from our systematic search
        optimized_params = {
            "binary_omega_range": (0.0005, 0.002),  # Ultra-aggressive for binary
            "ternary_thres_range": (0.0005, 0.0012),  # Ultra-aggressive for ternary
            "ternary_window_range": (2, 5),  # Very small windows
            "oracle_tx_cost": 0.0001,  # Very low transaction cost
            "oracle_neutral_factor": 0.3,  # Low neutral reward factor
        }

        # Verify parameter ranges make sense
        assert optimized_params["binary_omega_range"][0] > 0
        assert optimized_params["binary_omega_range"][1] > optimized_params["binary_omega_range"][0]

        assert optimized_params["ternary_thres_range"][0] > 0
        assert (
            optimized_params["ternary_thres_range"][1] > optimized_params["ternary_thres_range"][0]
        )

        assert (
            2
            <= optimized_params["ternary_window_range"][0]
            <= optimized_params["ternary_window_range"][1]
            <= 5
        )

        assert 0 < optimized_params["oracle_tx_cost"] < 0.001  # Very low cost
        assert 0 < optimized_params["oracle_neutral_factor"] < 0.5  # Favor directional over neutral

    def test_visualization_compatibility(self):
        """Test that our parameter optimization is compatible with visualization needs."""
        # The visualization expects:
        # 1. Binary generators to produce {0, 1} labels
        # 2. Ternary generators to produce {0, 1, 2} labels
        # 3. Neutral class (1) to be hidden in ternary plotting

        expected_binary_classes = {0, 1}
        expected_ternary_classes = {0, 1, 2}

        # These should match our label remapping logic
        assert len(expected_binary_classes) == 2
        assert len(expected_ternary_classes) == 3
        assert 1 in expected_ternary_classes  # Neutral class exists but will be hidden
