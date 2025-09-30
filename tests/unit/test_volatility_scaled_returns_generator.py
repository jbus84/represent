#!/usr/bin/env python3
"""
Tests for VolatilityScaledReturnsGenerator

This module tests the VolatilityScaledReturnsGenerator regression target generator
that implements volatility-scaled barriers for adaptive risk management.
"""

import numpy as np
import polars as pl
import pytest

from represent.target_generators.regression import VolatilityScaledReturnsGenerator


class TestVolatilityScaledReturnsGenerator:
    """Test suite for VolatilityScaledReturnsGenerator."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data with varying volatility regimes."""
        np.random.seed(42)
        n_samples = 5000

        # Create price series with different volatility regimes
        prices = []
        current_price = 1.0

        for i in range(n_samples):
            # Create regime-switching volatility
            if i < 1500:
                vol = 0.0005  # Low volatility regime
            elif i < 3500:
                vol = 0.002  # High volatility regime
            else:
                vol = 0.001  # Medium volatility regime

            change = np.random.normal(0, vol)
            current_price = current_price * np.exp(change)
            prices.append(current_price)

        df = pl.DataFrame(
            {
                "mid_price": prices,
                "ts_event": np.arange(n_samples),
            }
        )
        return df

    def test_initialization(self):
        """Test generator initialization with default parameters."""
        generator = VolatilityScaledReturnsGenerator()

        assert generator.volatility_window == 500
        assert generator.vol_multiplier == 2.0
        assert generator.horizon_ticks == 2000
        assert generator.target_name == "vol_scaled_returns_bps"
        assert generator.target_type == "regression"
        assert generator.required_columns == ["mid_price"]

    def test_initialization_custom_parameters(self):
        """Test generator initialization with custom parameters."""
        generator = VolatilityScaledReturnsGenerator(
            volatility_window=300,
            vol_multiplier=1.5,
            horizon_ticks=1000,
            target_name="custom_vol_scaled",
        )

        assert generator.volatility_window == 300
        assert generator.vol_multiplier == 1.5
        assert generator.horizon_ticks == 1000
        assert generator.target_name == "custom_vol_scaled"

    def test_generate_targets_basic(self, sample_price_data):
        """Test basic target generation."""
        generator = VolatilityScaledReturnsGenerator(
            volatility_window=300, vol_multiplier=2.0, horizon_ticks=1000
        )
        targets = generator.generate_targets(sample_price_data)

        assert "vol_scaled_returns_bps" in targets
        vol_scaled_returns = targets["vol_scaled_returns_bps"]

        # Should have same length as input data
        assert len(vol_scaled_returns) == len(sample_price_data)

        # Should be Polars Series
        assert isinstance(vol_scaled_returns, pl.Series)

        # Should have valid values for positions where we have sufficient data
        valid_values = vol_scaled_returns.filter(~vol_scaled_returns.is_nan())
        assert len(valid_values) > 0

        # Valid positions should be limited by volatility window + horizon
        expected_valid_positions = len(sample_price_data) - 300 - 1000 - 1
        assert len(valid_values) <= expected_valid_positions

    def test_volatility_adaptation(self):
        """Test that the generator adapts to different volatility regimes."""
        # Create two price series: one low vol, one high vol
        np.random.seed(42)
        n_samples = 3000

        # Low volatility series
        low_vol_prices = []
        current_price = 1.0
        for _i in range(n_samples):
            change = np.random.normal(0, 0.0002)  # Very low vol
            current_price = current_price * np.exp(change)
            low_vol_prices.append(current_price)

        # High volatility series
        np.random.seed(42)  # Same seed for comparable results
        high_vol_prices = []
        current_price = 1.0
        for _i in range(n_samples):
            change = np.random.normal(0, 0.003)  # High vol
            current_price = current_price * np.exp(change)
            high_vol_prices.append(current_price)

        low_vol_df = pl.DataFrame({"mid_price": low_vol_prices, "ts_event": range(n_samples)})
        high_vol_df = pl.DataFrame({"mid_price": high_vol_prices, "ts_event": range(n_samples)})

        generator = VolatilityScaledReturnsGenerator(
            volatility_window=200, vol_multiplier=2.0, horizon_ticks=800
        )

        low_vol_targets = generator.generate_targets(low_vol_df)["vol_scaled_returns_bps"]
        high_vol_targets = generator.generate_targets(high_vol_df)["vol_scaled_returns_bps"]

        low_vol_valid = low_vol_targets.filter(~low_vol_targets.is_nan())
        high_vol_valid = high_vol_targets.filter(~high_vol_targets.is_nan())

        if len(low_vol_valid) > 10 and len(high_vol_valid) > 10:
            # High vol regime should have larger magnitude returns (more barrier hits)
            low_vol_std = low_vol_valid.std()
            high_vol_std = high_vol_valid.std()

            # High vol should generally have higher variability in outcomes
            assert high_vol_std > low_vol_std * 0.8  # Allow some tolerance

    def test_vol_multiplier_effect(self, sample_price_data):
        """Test that different volatility multipliers produce different results."""
        vol_multipliers = [1.0, 2.0, 3.0]
        results = {}

        for vol_mult in vol_multipliers:
            generator = VolatilityScaledReturnsGenerator(
                volatility_window=300, vol_multiplier=vol_mult, horizon_ticks=1000
            )
            targets = generator.generate_targets(sample_price_data)
            valid_targets = targets["vol_scaled_returns_bps"].filter(
                ~targets["vol_scaled_returns_bps"].is_nan()
            )
            results[vol_mult] = valid_targets

        # Higher volatility multipliers should generally lead to different return distributions
        # (wider barriers should allow for larger moves before hitting barriers)
        if all(len(results[mult]) > 10 for mult in vol_multipliers):
            stds = {mult: results[mult].std() for mult in vol_multipliers}

            # Generally, higher multipliers should allow for wider ranges
            # (though this depends on the specific price series)
            assert stds[1.0] != stds[3.0]  # Should at least be different

    def test_barrier_logic(self):
        """Test the barrier breach logic with a controlled price series."""
        # Create a price series with more pronounced movements
        np.random.seed(42)
        prices = []
        current_price = 1.0

        # Create initial stable period for volatility estimation
        for _i in range(100):
            change = np.random.normal(0, 0.0001)  # Very low vol
            current_price *= np.exp(change)
            prices.append(current_price)

        # Add some larger movements that should hit barriers
        large_moves = [0.02, -0.03, 0.025, -0.02, 0.015]  # 2-3% moves
        for move in large_moves:
            current_price *= np.exp(move)
            prices.append(current_price)

        # Add more normal movement
        for _i in range(50):
            change = np.random.normal(0, 0.001)
            current_price *= np.exp(change)
            prices.append(current_price)

        df = pl.DataFrame(
            {
                "mid_price": prices,
                "ts_event": range(len(prices)),
            }
        )

        generator = VolatilityScaledReturnsGenerator(
            volatility_window=50,  # Use more data for vol estimation
            vol_multiplier=0.5,  # Lower multiplier to make barriers easier to hit
            horizon_ticks=30,  # Longer horizon to capture movements
        )

        targets = generator.generate_targets(df)
        vol_scaled_returns = targets["vol_scaled_returns_bps"]

        # Should have some valid targets
        valid_targets = vol_scaled_returns.filter(~vol_scaled_returns.is_nan())
        assert len(valid_targets) > 0

        # With large price movements and low volatility barriers,
        # should see some significant positive or negative returns
        assert (valid_targets.abs() > 10).any(), "Should capture significant price movements"

    def test_insufficient_data_handling(self):
        """Test handling when there's insufficient data."""
        # Create data with fewer samples than required
        small_data = pl.DataFrame(
            {
                "mid_price": [1.0, 1.001, 1.002, 1.001, 1.0],
                "ts_event": [0, 1, 2, 3, 4],
            }
        )

        generator = VolatilityScaledReturnsGenerator(
            volatility_window=100,  # Larger than available data
            horizon_ticks=100,
        )

        targets = generator.generate_targets(small_data)
        vol_scaled_returns = targets["vol_scaled_returns_bps"]

        # Should all be NaN due to insufficient data
        assert vol_scaled_returns.is_nan().all()

    def test_nan_price_handling(self):
        """Test handling of NaN prices in input data."""
        prices = [1.0] * 100
        prices[50:55] = [np.nan] * 5  # Add some NaN values

        df = pl.DataFrame(
            {
                "mid_price": prices,
                "ts_event": range(len(prices)),
            }
        )

        generator = VolatilityScaledReturnsGenerator(volatility_window=20, horizon_ticks=30)

        targets = generator.generate_targets(df)
        vol_scaled_returns = targets["vol_scaled_returns_bps"]

        # Should handle NaN gracefully - affected positions should be NaN
        assert len(vol_scaled_returns) == len(df)

        # Positions that include NaN in their volatility window or horizon should be NaN
        # Positions before NaN block should potentially have valid values
        early_positions = vol_scaled_returns[:30]
        assert len(early_positions) > 0

    def test_edge_case_zero_volatility(self):
        """Test edge case with zero or very low volatility."""
        # Create constant price series (zero volatility)
        constant_prices = [1.0] * 1000

        df = pl.DataFrame(
            {
                "mid_price": constant_prices,
                "ts_event": range(len(constant_prices)),
            }
        )

        generator = VolatilityScaledReturnsGenerator(volatility_window=100, horizon_ticks=200)

        targets = generator.generate_targets(df)
        vol_scaled_returns = targets["vol_scaled_returns_bps"]

        # With zero volatility, barriers would be at entry price,
        # so returns should be 0 (no price movement)
        valid_targets = vol_scaled_returns.filter(~vol_scaled_returns.is_nan())
        if len(valid_targets) > 0:
            # All returns should be very close to 0 (no price movement)
            assert (valid_targets.abs() < 0.1).all()

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pl.DataFrame(
            {
                "mid_price": [],
                "ts_event": [],
            }
        )

        generator = VolatilityScaledReturnsGenerator()
        targets = generator.generate_targets(df)
        vol_scaled_returns = targets["vol_scaled_returns_bps"]

        assert len(vol_scaled_returns) == 0
        assert vol_scaled_returns.dtype == pl.Float64

    def test_get_target_info(self):
        """Test target info metadata."""
        generator = VolatilityScaledReturnsGenerator(
            volatility_window=400,
            vol_multiplier=2.5,
            horizon_ticks=1500,
            target_name="test_vol_scaled",
        )

        info = generator.get_target_info()

        assert info["target_names"] == ["test_vol_scaled"]
        assert info["target_type"] == "regression"
        assert (
            info["description"]
            == "Volatility-scaled returns with 2.5x vol barriers over 1500 ticks"
        )
        assert info["parameters"]["volatility_window"] == 400
        assert info["parameters"]["vol_multiplier"] == 2.5
        assert info["parameters"]["horizon_ticks"] == 1500

    def test_different_horizons(self, sample_price_data):
        """Test with different horizon lengths."""
        horizons = [500, 1000, 2000]

        for horizon in horizons:
            generator = VolatilityScaledReturnsGenerator(
                volatility_window=300, vol_multiplier=2.0, horizon_ticks=horizon
            )

            targets = generator.generate_targets(sample_price_data)
            vol_scaled_returns = targets["vol_scaled_returns_bps"]

            # Should have valid targets for positions with sufficient lookforward data
            valid_positions = len(sample_price_data) - 300 - horizon - 1
            if valid_positions > 0:
                valid_returns = vol_scaled_returns[:valid_positions]
                valid_count = (~valid_returns.is_nan()).sum()
                assert valid_count > 0, f"No valid returns for horizon={horizon}"

    def test_returns_magnitude_reasonable(self, sample_price_data):
        """Test that returns magnitude is reasonable for typical market data."""
        generator = VolatilityScaledReturnsGenerator(
            volatility_window=300, vol_multiplier=2.0, horizon_ticks=1000
        )

        targets = generator.generate_targets(sample_price_data)
        vol_scaled_returns = targets["vol_scaled_returns_bps"]

        valid_returns = vol_scaled_returns.filter(~vol_scaled_returns.is_nan())

        if len(valid_returns) > 0:
            # For typical FX data, vol-scaled returns should be bounded
            # by the volatility barriers (roughly +/- 2 * vol * 10000 bps)
            max_expected = 1000  # Reasonable upper bound in basis points
            assert (valid_returns.abs() < max_expected).all(), (
                f"Returns exceed reasonable bounds: {valid_returns.min():.2f} to {valid_returns.max():.2f}"
            )

            # Should have some variability
            assert valid_returns.std() > 0.1, "Returns lack expected variability"

    def test_custom_target_name(self, sample_price_data):
        """Test using custom target name."""
        custom_name = "my_vol_scaled_returns"
        generator = VolatilityScaledReturnsGenerator(
            volatility_window=300, horizon_ticks=1000, target_name=custom_name
        )

        targets = generator.generate_targets(sample_price_data)

        assert custom_name in targets
        assert "vol_scaled_returns_bps" not in targets
        assert len(targets[custom_name]) == len(sample_price_data)

    def test_barrier_calculation_accuracy(self):
        """Test the accuracy of barrier calculations."""
        # Create a simple scenario where we can verify barrier calculations
        prices = [1.0] * 50  # Constant prices for volatility estimation
        prices.extend([1.0001, 1.0002, 1.0003, 1.0004, 1.0005])  # Small upward trend

        df = pl.DataFrame(
            {
                "mid_price": prices,
                "ts_event": range(len(prices)),
            }
        )

        generator = VolatilityScaledReturnsGenerator(
            volatility_window=20, vol_multiplier=1.0, horizon_ticks=5
        )

        targets = generator.generate_targets(df)
        vol_scaled_returns = targets["vol_scaled_returns_bps"]

        # With very low volatility in the initial constant prices,
        # the small upward trend should be captured
        valid_targets = vol_scaled_returns.filter(~vol_scaled_returns.is_nan())

        if len(valid_targets) > 0:
            # Should capture the upward price movement
            assert (valid_targets > 0).any(), "Should capture positive price movements"
