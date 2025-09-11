#!/usr/bin/env python3
"""
Tests for Cumulative Returns Generator

This module tests the CumulativeReturnsGenerator regression target generator.
"""

import numpy as np
import polars as pl
import pytest

from represent.target_generators.regression import CumulativeReturnsGenerator


class TestCumulativeReturnsGenerator:
    """Test suite for CumulativeReturnsGenerator."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        n_samples = 5000

        # Create price series with realistic movements
        base_price = 1.0
        returns = np.random.normal(0, 0.001, n_samples - 1)  # 0.1% volatility
        log_prices = np.concatenate([[np.log(base_price)], np.cumsum(returns)])
        prices = np.exp(log_prices)

        df = pl.DataFrame(
            {
                "mid_price": prices,
                "ts_event": np.arange(n_samples),
            }
        )
        return df

    def test_initialization(self):
        """Test generator initialization with default parameters."""
        generator = CumulativeReturnsGenerator()

        assert generator.lookforward_samples == 3000
        assert generator.target_name == "cumulative_returns_bps"
        assert generator.target_type == "regression"
        assert generator.required_columns == ["mid_price"]

    def test_initialization_custom_parameters(self):
        """Test generator initialization with custom parameters."""
        generator = CumulativeReturnsGenerator(
            lookforward_samples=1500, target_name="custom_cumret"
        )

        assert generator.lookforward_samples == 1500
        assert generator.target_name == "custom_cumret"

    def test_generate_targets_basic(self, sample_price_data):
        """Test basic target generation."""
        generator = CumulativeReturnsGenerator(lookforward_samples=100)
        targets = generator.generate_targets(sample_price_data)

        assert "cumulative_returns_bps" in targets.columns
        cumulative_returns = targets["cumulative_returns_bps"]

        # Should have same length as input data
        assert len(cumulative_returns) == len(sample_price_data)

        # Should be Polars Series
        assert isinstance(cumulative_returns, pl.Series)

        # Should have valid values for most positions
        valid_values = cumulative_returns.drop_nulls()
        assert len(valid_values) > 0
        assert len(valid_values) <= len(sample_price_data)  # Can't have more valid values than input data

    def test_cumulative_calculation_logic(self):
        """Test the cumulative returns calculation logic."""
        # Create simple test data where we can verify calculations manually
        prices = np.array([1.0, 1.01, 1.02, 1.01, 1.03])  # 5 prices, 4 returns
        df = pl.DataFrame(
            {
                "mid_price": prices,
                "ts_event": np.arange(len(prices)),
            }
        )

        generator = CumulativeReturnsGenerator(lookforward_samples=3)
        targets = generator.generate_targets(df)
        cumulative_returns = targets["cumulative_returns_bps"]

        # For position 0: should sum returns from positions 0, 1, 2
        # log(1.01/1.0) + log(1.02/1.01) + log(1.01/1.02) = log(1.01) + log(1.02/1.01) + log(1.01/1.02)
        # = log(1.01 * (1.02/1.01) * (1.01/1.02)) = log(1.01) ≈ 0.00995 in log returns

        expected_log_sum = np.log(1.01) + np.log(1.02 / 1.01) + np.log(1.01 / 1.02)
        expected_bps = expected_log_sum * 10000

        # Position 0 should have a valid cumulative return
        assert not pl.Series([cumulative_returns[0]]).is_nan()[0]
        # Allow for small floating point differences
        assert abs(cumulative_returns[0] - expected_bps) < 0.1

    def test_insufficient_lookforward_data(self):
        """Test handling when insufficient lookforward data is available."""
        # Create data with only 50 samples
        prices = np.random.uniform(0.9, 1.1, 50)
        df = pl.DataFrame(
            {
                "mid_price": prices,
                "ts_event": np.arange(len(prices)),
            }
        )

        # Request 100 samples lookforward (more than available)
        generator = CumulativeReturnsGenerator(lookforward_samples=100)
        targets = generator.generate_targets(df)
        cumulative_returns = targets["cumulative_returns_bps"]

        # Should all be NaN since we don't have enough lookforward data
        assert cumulative_returns.is_nan().all()

    def test_nan_price_handling(self, sample_price_data):
        """Test handling of NaN prices in input data."""
        # Add some NaN values at the end so they don't affect early positions
        df_with_nans = sample_price_data.clone()
        prices_with_nans = df_with_nans["mid_price"].to_numpy().copy()
        prices_with_nans[4500:4510] = np.nan  # Add NaN block near end

        df_with_nans = df_with_nans.with_columns(pl.Series("mid_price", prices_with_nans))

        generator = CumulativeReturnsGenerator(lookforward_samples=200)
        targets = generator.generate_targets(df_with_nans)
        cumulative_returns = targets["cumulative_returns_bps"]

        # Should handle NaN gracefully - positions affected by NaN should be NaN
        assert len(cumulative_returns) == len(df_with_nans)

        # Early positions (well before NaN and with enough lookforward) should have valid values
        early_positions = cumulative_returns[:100]  # First 100 positions
        assert (~early_positions.is_nan()).any(), "Should have some valid early positions"

        # Positions that would include NaN in their lookforward window should be NaN
        affected_positions = cumulative_returns[
            4300:4400
        ]  # Positions whose lookforward hits the NaN
        assert affected_positions.is_nan().all(), "Positions affected by NaN should be NaN"

    def test_edge_case_single_price(self):
        """Test edge case with single price point."""
        df = pl.DataFrame(
            {
                "mid_price": [1.0],
                "ts_event": [0],
            }
        )

        generator = CumulativeReturnsGenerator(lookforward_samples=3)
        targets = generator.generate_targets(df)
        cumulative_returns = targets["cumulative_returns_bps"]

        # Should return NaN for single price (no returns possible)
        assert len(cumulative_returns) == 1
        assert cumulative_returns.is_nan()[0]

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pl.DataFrame(
            {
                "mid_price": [],
                "ts_event": [],
            }
        )

        generator = CumulativeReturnsGenerator()
        targets = generator.generate_targets(df)
        cumulative_returns = targets["cumulative_returns_bps"]

        assert len(cumulative_returns) == 0
        assert cumulative_returns.dtype == pl.Float64

    def test_get_target_info(self):
        """Test target info metadata."""
        generator = CumulativeReturnsGenerator(lookforward_samples=1500, target_name="test_cumret")

        info = generator.get_target_info()

        assert info["target_names"] == ["test_cumret"]
        assert info["target_type"] == "regression"
        assert info["description"] == "Cumulative returns over next 1500 samples"
        assert info["parameters"]["lookforward_samples"] == 1500

    def test_different_lookforward_samples(self, sample_price_data):
        """Test with different lookforward sample sizes."""
        for lookforward in [50, 500, 1000, 2000]:
            generator = CumulativeReturnsGenerator(lookforward_samples=lookforward)
            targets = generator.generate_targets(sample_price_data)
            cumulative_returns = targets["cumulative_returns_bps"]

            # Should have valid targets for positions with sufficient lookforward data
            valid_positions = len(sample_price_data) - lookforward - 1
            if valid_positions > 0:
                valid_returns = cumulative_returns[:valid_positions]
                assert (~valid_returns.is_nan()).any(), (
                    f"No valid returns for lookforward={lookforward}"
                )

    def test_returns_magnitude(self, sample_price_data):
        """Test that returns magnitude is reasonable."""
        generator = CumulativeReturnsGenerator(lookforward_samples=1000)
        targets = generator.generate_targets(sample_price_data)
        cumulative_returns = targets["cumulative_returns_bps"]

        valid_returns = cumulative_returns.filter(~cumulative_returns.is_nan())

        if len(valid_returns) > 0:
            # For typical FX data with 0.1% daily vol, cumulative returns over 1000 samples
            # should typically be within reasonable bounds
            assert (valid_returns.abs() < 10000).all(), (
                "Cumulative returns seem unreasonably large"
            )

            # Should have some variability (not all the same)
            assert valid_returns.std() > 0.01, "Cumulative returns lack expected variability"

    def test_custom_target_name(self, sample_price_data):
        """Test using custom target name."""
        custom_name = "my_cumulative_returns"
        generator = CumulativeReturnsGenerator(lookforward_samples=100, target_name=custom_name)

        targets = generator.generate_targets(sample_price_data)

        assert custom_name in targets
        assert "cumulative_returns_bps" not in targets
        assert len(targets[custom_name]) == len(sample_price_data)
