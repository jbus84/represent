"""
Comprehensive unit tests for regression target generators.
Tests DirectionalMFEGenerator and RemainingValueTunerGenerator.
"""

import numpy as np
import polars as pl

from represent.target_generators.regression import (
    DirectionalMFEGenerator,
    LogReturnHorizonsGenerator,
    LookbackLookforwardReturnsGenerator,
    RemainingValueTunerGenerator,
)


class TestDirectionalMFEGenerator:
    """Test suite for DirectionalMFEGenerator."""

    def test_basic_mfe_calculation(self):
        """Test MFE calculation with a simple upward trend."""
        # Create simple upward trend: 1.0000 -> 1.0100 over 100 ticks
        prices = np.linspace(1.0000, 1.0100, 200)
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = DirectionalMFEGenerator(
            lookforward_horizon=50,
            lookback_window=10,
            expected_fee_pips=0.0,
            target_names=("mfe_buy", "mfe_sell"),
        )

        targets = generator.generate_targets(df)
        mfe_buy = targets["mfe_buy"]
        mfe_sell = targets["mfe_sell"]

        # Test valid range
        valid_indices = ~mfe_buy.is_nan()
        assert valid_indices.any(), "Should have some valid MFE values"

        # For upward trend, buy MFE should be positive, sell MFE should be 0 or negative
        valid_buy_mfe = mfe_buy.filter(valid_indices)
        valid_sell_mfe = mfe_sell.filter(valid_indices)

        assert (valid_buy_mfe >= 0).all(), "Buy MFE should be non-negative in uptrend"
        assert (valid_sell_mfe <= 0).all(), "Sell MFE should be non-positive in uptrend"

    def test_mfe_with_controlled_scenario(self):
        """Test MFE with a controlled price scenario."""
        # Controlled scenario: stable -> up 50 bps -> stable
        prices = []
        current_price = 1.0000

        # Stable period (100 ticks)
        for _ in range(100):
            prices.append(current_price)

        # Up movement (50 ticks, 1 bp each = 50 bps total)
        for _ in range(50):
            current_price += 0.0001
            prices.append(current_price)

        # Stable period (100 ticks)
        for _ in range(100):
            prices.append(current_price)

        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = DirectionalMFEGenerator(
            lookforward_horizon=80,  # Cover the movement
            lookback_window=10,
            expected_fee_pips=0.0,
            target_names=("mfe_buy", "mfe_sell"),
        )

        targets = generator.generate_targets(df)
        mfe_buy = targets["mfe_buy"]
        mfe_sell = targets["mfe_sell"]

        # Test at position 50 (in first stable period)
        # Note: With lookforward_horizon=80, from position 50 we look to position 130
        # The movement starts at position 100, so we only see 31 bps of the 50 bps move
        test_idx = 50
        if not mfe_buy.is_nan()[test_idx]:
            # Buy MFE should be close to 31 bps (partial movement within horizon)
            assert abs(mfe_buy[test_idx] - 31.0) < 5.0, f"Expected ~31 bps, got {mfe_buy[test_idx]}"
            # Sell MFE should be 0 (price never goes down from entry)
            assert abs(mfe_sell[test_idx] - 0.0) < 1.0, f"Expected ~0 bps, got {mfe_sell[test_idx]}"

    def test_mfe_with_fees(self):
        """Test MFE calculation includes fees correctly."""
        # Simple upward trend
        prices = np.linspace(1.0000, 1.0100, 200)
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        # Test with fees
        generator_with_fees = DirectionalMFEGenerator(
            lookforward_horizon=50,
            lookback_window=10,
            expected_fee_pips=1.0,  # 1 pip = 10 bps
            target_names=("mfe_buy", "mfe_sell"),
        )

        # Test without fees
        generator_no_fees = DirectionalMFEGenerator(
            lookforward_horizon=50,
            lookback_window=10,
            expected_fee_pips=0.0,
            target_names=("mfe_buy", "mfe_sell"),
        )

        targets_with_fees = generator_with_fees.generate_targets(df)
        targets_no_fees = generator_no_fees.generate_targets(df)

        mfe_buy_fees = targets_with_fees["mfe_buy"]
        mfe_buy_no_fees = targets_no_fees["mfe_buy"]

        # Find valid comparison point
        valid_mask = ~mfe_buy_fees.is_nan() & ~mfe_buy_no_fees.is_nan()
        if valid_mask.any():
            # With fees should be 10 bps lower than without fees
            fee_difference = mfe_buy_no_fees.filter(valid_mask) - mfe_buy_fees.filter(valid_mask)
            assert np.allclose(fee_difference, 10.0, atol=0.1), (
                "Fee adjustment not applied correctly"
            )

    def test_mfe_boundary_conditions(self):
        """Test MFE with edge cases and boundary conditions."""
        # Flat prices (no movement)
        prices = np.full(100, 1.0000)
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = DirectionalMFEGenerator(
            lookforward_horizon=20,
            lookback_window=5,
            expected_fee_pips=0.0,
            target_names=("mfe_buy", "mfe_sell"),
        )

        targets = generator.generate_targets(df)
        mfe_buy = targets["mfe_buy"]
        mfe_sell = targets["mfe_sell"]

        # With flat prices, MFE should be 0 (or close to 0)
        valid_mask = ~mfe_buy.is_nan()
        if valid_mask.any():
            assert np.allclose(mfe_buy.filter(valid_mask), 0.0, atol=0.1), "Flat prices should give ~0 MFE"
            assert np.allclose(mfe_sell.filter(valid_mask), 0.0, atol=0.1), (
                "Flat prices should give ~0 MFE"
            )


class TestLogReturnHorizonsGenerator:
    """Test suite for LogReturnHorizonsGenerator."""

    def test_basic_log_return_calculation(self):
        """Test log return calculation with simple price data."""
        # Create simple exponential growth: price increases by 1% every 1000 ticks
        initial_price = 1.0000
        n_samples = 6000
        prices = []

        for i in range(n_samples):
            # Exponential growth: 0.01% per tick for smooth log returns
            growth_factor = 1.0001 ** i
            prices.append(initial_price * growth_factor)

        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = LogReturnHorizonsGenerator(
            horizons=[1000, 2000, 3000],
            lookback_window=500,
            target_prefix="test_log_return"
        )

        targets = generator.generate_targets(df)

        # Should have all expected target names
        expected_names = ["test_log_return_1000t", "test_log_return_2000t", "test_log_return_3000t"]
        for name in expected_names:
            assert name in targets, f"Missing target {name}"

        # Test that longer horizons generally have larger log returns (due to exponential growth)
        log_1000 = targets["test_log_return_1000t"]
        log_2000 = targets["test_log_return_2000t"]
        log_3000 = targets["test_log_return_3000t"]

        # Find a test index where all horizons have valid values
        valid_mask = (~log_1000.is_nan() & ~log_2000.is_nan() & ~log_3000.is_nan())
        if valid_mask.any():
            valid_indices = np.where(valid_mask.to_numpy())[0]
            test_idx = int(valid_indices[len(valid_indices) // 2])  # Middle valid index

            # Due to exponential growth, longer horizons should have larger returns
            assert log_3000[test_idx] > log_2000[test_idx], "3000 tick horizon should have larger return than 2000"
            assert log_2000[test_idx] > log_1000[test_idx], "2000 tick horizon should have larger return than 1000"

    def test_log_return_with_flat_prices(self):
        """Test log return calculation with flat prices."""
        # Constant prices
        prices = np.full(2000, 1.2345)
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = LogReturnHorizonsGenerator(
            horizons=[500, 1000, 1500],
            lookback_window=100,
            target_prefix="flat_test"
        )

        targets = generator.generate_targets(df)

        # All log returns should be very close to zero
        for name in ["flat_test_500t", "flat_test_1000t", "flat_test_1500t"]:
            log_returns = targets[name]
            valid_values = log_returns.filter(~log_returns.is_nan())

            if len(valid_values) > 0:
                assert np.allclose(valid_values, 0.0, atol=0.1), f"Flat prices should give ~0 log returns for {name}"

    def test_log_return_with_controlled_movement(self):
        """Test log return with controlled price movement."""
        # Create price series that increases by exactly 1% (100 bps) every 1000 ticks
        prices = []
        base_price = 1.0000

        for i in range(3000):
            # Increase by 1% every 1000 ticks
            multiplier = (1.01) ** (i // 1000)
            prices.append(base_price * multiplier)

        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = LogReturnHorizonsGenerator(
            horizons=[1000],  # Single horizon for precise testing
            lookback_window=100,
            target_prefix="controlled"
        )

        targets = generator.generate_targets(df)
        log_returns = targets["controlled_1000t"]

        # Test at position where we expect exactly 1% growth over 1000 ticks
        # Position 500 (current) -> Position 1500 (future) spans the transition from 1.00 to 1.01
        test_idx = 500

        if not log_returns.is_nan()[test_idx]:
            # Log return of 1% is approximately 99.5 bps (ln(1.01) * 10000)
            expected_return = np.log(1.01) * 10000
            actual_return = log_returns[test_idx]

            assert abs(actual_return - expected_return) < 5.0, (
                f"Expected ~{expected_return:.1f} bps, got {actual_return:.1f} bps"
            )

    def test_log_return_default_parameters(self):
        """Test log return generator with default parameters."""
        # Simple price data
        prices = np.linspace(1.0000, 1.0050, 8000)  # 50 bps increase over 8000 ticks
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        # Use default parameters
        generator = LogReturnHorizonsGenerator()

        targets = generator.generate_targets(df)

        # Should have default horizons
        expected_defaults = ["log_return_1000t", "log_return_2000t", "log_return_3000t",
                           "log_return_4000t", "log_return_5000t"]

        for name in expected_defaults:
            assert name in targets, f"Missing default target {name}"

        # Should have some valid values
        for name in expected_defaults:
            log_returns = targets[name]
            valid_count = (~log_returns.is_nan()).sum()
            assert valid_count > 0, f"Should have some valid values for {name}"

    def test_log_return_custom_horizons(self):
        """Test log return generator with custom horizons."""
        prices = np.linspace(1.0000, 1.0100, 4000)  # 100 bps increase
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        # Custom configuration
        generator = LogReturnHorizonsGenerator(
            horizons=[250, 750, 1250],
            lookback_window=50,
            target_prefix="custom_horizon"
        )

        targets = generator.generate_targets(df)

        # Should have custom target names
        expected_custom = ["custom_horizon_250t", "custom_horizon_750t", "custom_horizon_1250t"]

        for name in expected_custom:
            assert name in targets, f"Missing custom target {name}"

    def test_log_return_edge_cases(self):
        """Test log return generator with edge cases."""
        # Very small dataset
        prices = [1.0000, 1.0001, 1.0002, 1.0003, 1.0004]
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = LogReturnHorizonsGenerator(
            horizons=[1, 2],  # Very short horizons for small dataset
            lookback_window=1,
            target_prefix="edge"
        )

        targets = generator.generate_targets(df)

        # Should handle small dataset gracefully
        assert "edge_1t" in targets
        assert "edge_2t" in targets

        # Most values might be NaN due to boundary conditions, but shouldn't crash
        assert len(targets["edge_1t"]) == len(prices)
        assert len(targets["edge_2t"]) == len(prices)

    def test_log_return_target_info(self):
        """Test target info metadata."""
        generator = LogReturnHorizonsGenerator(
            horizons=[1000, 2000, 3000],
            lookback_window=500,
            target_prefix="info_test"
        )

        info = generator.get_target_info()

        # Check required fields
        assert "target_names" in info
        assert "target_type" in info
        assert "description" in info
        assert "parameters" in info

        # Check values
        expected_names = ["info_test_1000t", "info_test_2000t", "info_test_3000t"]
        assert info["target_names"] == expected_names
        assert info["target_type"] == "regression"
        assert "1000-3000" in info["description"]  # Should mention horizon range
        assert info["parameters"]["horizons"] == [1000, 2000, 3000]
        assert info["parameters"]["lookback_window"] == 500


class TestLookbackLookforwardReturnsGenerator:
    """Test suite for LookbackLookforwardReturnsGenerator."""

    def test_returns_positive_in_uptrend(self):
        """Percentage returns should be positive in a linear uptrend."""
        prices = np.linspace(1.0000, 1.0200, 12000)
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = LookbackLookforwardReturnsGenerator(
            window_lengths=[400],
            target_prefix="up",
        )

        targets = generator.generate_targets(df)
        base_returns = targets["up_return_400t"]
        valid_values = base_returns.filter(~base_returns.is_nan()).to_numpy()

        assert valid_values.size > 0
        assert (valid_values > 0).all(), "Returns should be positive in an uptrend"

    def test_returns_negative_in_downtrend(self):
        """Percentage returns should be negative in a linear downtrend."""
        prices = np.linspace(1.0200, 1.0000, 12000)
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = LookbackLookforwardReturnsGenerator(
            window_lengths=[400],
            target_prefix="down",
        )

        targets = generator.generate_targets(df)
        base_returns = targets["down_return_400t"]
        valid_values = base_returns.filter(~base_returns.is_nan()).to_numpy()

        assert valid_values.size > 0
        assert (valid_values < 0).all(), "Returns should be negative in a downtrend"

    def test_manual_return_matches_expected(self):
        """Generated returns should match manual calculations."""
        prices = np.linspace(1.0000, 1.0500, 20000)
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        window_len = 200
        scale_factor = 0.4
        generator = LookbackLookforwardReturnsGenerator(
            window_lengths=[window_len],
            target_prefix="manual",
            scale_factor=scale_factor,
        )

        targets = generator.generate_targets(df)
        idx = 8000  # Well inside the valid range

        base_value = targets["manual_return_200t"][idx]
        scaled_value = targets["manual_scaled_return_200t"][idx]
        current_value = targets["manual_current_return_200t"][idx]

        mid_prices = df["mid_price"].to_numpy()

        lookback_start = idx - window_len + 1
        lookforward_start = idx + 1
        lookforward_end = lookforward_start + window_len
        scaled_start = int(idx - window_len * scale_factor + 1)

        lookback_mean = mid_prices[lookback_start:idx + 1].mean()
        lookforward_mean = mid_prices[lookforward_start:lookforward_end].mean()
        scaled_mean = mid_prices[scaled_start:idx + 1].mean()
        current_price = mid_prices[idx]

        expected_base = (lookforward_mean - lookback_mean) / lookback_mean
        expected_scaled = (lookforward_mean - scaled_mean) / scaled_mean
        expected_current = (lookforward_mean - current_price) / current_price

        assert np.isclose(base_value, expected_base, atol=1e-12)
        assert np.isclose(scaled_value, expected_scaled, atol=1e-12)
        assert np.isclose(current_value, expected_current, atol=1e-12)

    def test_default_parameters(self):
        """Default configuration should expose all expected columns."""
        prices = np.linspace(1.0000, 1.0100, 30000)
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = LookbackLookforwardReturnsGenerator()
        assert generator.window_lengths == [100, 250, 500, 1000]

        targets = generator.generate_targets(df)
        for window_len in generator.window_lengths:
            assert f"lookback_lookforward_return_{window_len}t" in targets
            assert f"lookback_lookforward_scaled_return_{window_len}t" in targets
            assert f"lookback_lookforward_current_return_{window_len}t" in targets

    def test_flat_prices(self):
        """Flat prices should yield near-zero percentage returns."""
        prices = np.full(8000, 1.2345)
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = LookbackLookforwardReturnsGenerator(
            window_lengths=[300],
            target_prefix="flat",
        )

        targets = generator.generate_targets(df)
        base_returns = targets["flat_return_300t"].filter(~targets["flat_return_300t"].is_nan()).to_numpy()
        scaled_returns = targets["flat_scaled_return_300t"].filter(~targets["flat_scaled_return_300t"].is_nan()).to_numpy()
        current_returns = targets["flat_current_return_300t"].filter(~targets["flat_current_return_300t"].is_nan()).to_numpy()

        for values in (base_returns, scaled_returns, current_returns):
            if values.size:
                assert np.allclose(values, 0.0, atol=1e-9)

    def test_edge_case_small_window(self):
        """Generator should handle very small windows without errors."""
        prices = [1.0000, 1.0001, 1.0002, 1.0003, 1.0004]
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = LookbackLookforwardReturnsGenerator(
            window_lengths=[1],
            target_prefix="edge",
        )

        targets = generator.generate_targets(df)
        assert "edge_return_1t" in targets
        assert "edge_scaled_return_1t" in targets
        assert "edge_current_return_1t" in targets
        assert len(targets["edge_return_1t"]) == len(prices)

    def test_nan_handling(self):
        """Windows that touch NaNs or non-positive prices should remain NaN."""
        prices = np.linspace(1.0000, 1.0200, 5000)
        prices[900:915] = np.nan
        prices[2500] = 0.0
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = LookbackLookforwardReturnsGenerator(
            window_lengths=[200],
            target_prefix="nan_case",
        )

        targets = generator.generate_targets(df)
        returns = targets["nan_case_return_200t"].to_numpy()

        assert np.any(~np.isnan(returns[:600]))
        assert np.all(np.isnan(returns[850:1050]))
        assert np.any(~np.isnan(returns[1400:1800]))

    def test_target_info(self):
        """Metadata should reflect configuration and target names."""
        generator = LookbackLookforwardReturnsGenerator(
            window_lengths=[200, 800],
            target_prefix="info",
        )

        info = generator.get_target_info()
        assert info["target_type"] == "regression"
        assert info["parameters"]["window_lengths"] == [200, 800]
        assert len(info["target_names"]) == 6  # 2 windows * 3 targets each

    def test_factory_creation(self):
        """Factory should instantiate the correct generator."""
        from represent.target_generators.factory import TargetGeneratorFactory

        generator = TargetGeneratorFactory.create(
            "lookback_lookforward_returns",
            window_lengths=[150, 300],
            target_prefix="factory_test",
        )

        assert isinstance(generator, LookbackLookforwardReturnsGenerator)
        assert generator.window_lengths == [150, 300]
        assert generator.target_prefix == "factory_test"


class TestRemainingValueTunerGenerator:
    """Test suite for RemainingValueTunerGenerator."""

    def test_basic_remaining_value_calculation(self):
        """Test remaining value calculation with a simple trend."""
        # Create gradual uptrend
        np.random.seed(42)  # For reproducible random noise
        prices = []
        current_price = 1.0000

        # Stable start
        for _ in range(500):
            prices.append(current_price + np.random.normal(0, 0.00001))

        # Gradual uptrend (1000 ticks, +100 bps)
        trend_increment = 0.0001  # 1 bp per tick
        for _ in range(1000):
            current_price += trend_increment + np.random.normal(0, 0.00001)
            prices.append(current_price)

        # Stable end
        for _ in range(500):
            prices.append(current_price + np.random.normal(0, 0.00001))

        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = RemainingValueTunerGenerator(
            lookback_rows=100,
            lookforward_input=800,
            lookforward_offset=50,
            trend_threshold_bps=20.0,
            neutral_factor=0.5,
            enforce_monotonicity=True,
            target_name="remaining_value",
        )

        targets = generator.generate_targets(df)
        remaining_values = targets["remaining_value"]

        # Should have some valid values
        valid_mask = ~remaining_values.is_nan()
        assert valid_mask.any(), "Should have some valid remaining value calculations"

        # Test monotonicity in trend region (positions 600-1200)
        trend_start = 600
        trend_end = 1200
        trend_values = remaining_values[trend_start:trend_end]
        valid_trend_values = trend_values.filter(~trend_values.is_nan())

        if len(valid_trend_values) > 50:
            # Check that values generally decrease (allowing some noise)
            third = len(valid_trend_values) // 3
            early_values = valid_trend_values.head(third)
            late_values = valid_trend_values.tail(third)

            assert early_values.mean() > late_values.mean(), (
                "Remaining values should decrease through the trend"
            )

    def test_remaining_value_with_no_trend(self):
        """Test remaining value with sideways/no-trend market."""
        # Create sideways market
        np.random.seed(42)
        prices = []
        base_price = 1.0000

        for _ in range(2000):
            # Small random fluctuations around base price
            prices.append(base_price + np.random.normal(0, 0.0001))

        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        generator = RemainingValueTunerGenerator(
            lookback_rows=200,
            lookforward_input=500,
            lookforward_offset=50,
            trend_threshold_bps=30.0,  # Higher threshold
            neutral_factor=0.5,
            enforce_monotonicity=True,
            target_name="remaining_value",
        )

        targets = generator.generate_targets(df)
        remaining_values = targets["remaining_value"]

        # Most values should be NaN or close to neutral in sideways market
        valid_mask = ~remaining_values.is_nan()
        valid_values = remaining_values.filter(valid_mask)

        if len(valid_values) > 0:
            # Values should be relatively small (no strong trend)
            assert (valid_values.abs() < 100.0).all(), (
                "Sideways market should have small remaining values"
            )

    def test_remaining_value_monotonicity_enforcement(self):
        """Test monotonicity enforcement option."""
        # Create clear trend scenario
        prices = []
        current_price = 1.0000

        # Stable period
        for _ in range(200):
            prices.append(current_price)

        # Strong uptrend
        for _ in range(500):
            current_price += 0.0002  # 2 bps per tick
            prices.append(current_price)

        # Stable period
        for _ in range(200):
            prices.append(current_price)

        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        # Test with monotonicity
        generator_mono = RemainingValueTunerGenerator(
            lookback_rows=50,
            lookforward_input=300,
            lookforward_offset=20,
            trend_threshold_bps=20.0,
            neutral_factor=0.5,
            enforce_monotonicity=True,
            target_name="remaining_value",
        )

        # Test without monotonicity
        generator_no_mono = RemainingValueTunerGenerator(
            lookback_rows=50,
            lookforward_input=300,
            lookforward_offset=20,
            trend_threshold_bps=20.0,
            neutral_factor=0.5,
            enforce_monotonicity=False,
            target_name="remaining_value",
        )

        targets_mono = generator_mono.generate_targets(df)
        targets_no_mono = generator_no_mono.generate_targets(df)

        values_mono = targets_mono["remaining_value"]
        values_no_mono = targets_no_mono["remaining_value"]

        # Both should have valid values
        assert (~values_mono.is_nan()).any(), "Monotonic version should have valid values"
        assert (~values_no_mono.is_nan()).any(), "Non-monotonic version should have valid values"

        # Results can differ (we're not asserting they're the same, just that both work)

    def test_remaining_value_different_configurations(self):
        """Test remaining value with different configurations."""
        # Create test data
        prices = np.linspace(1.0000, 1.0100, 1000)  # Simple uptrend
        df = pl.DataFrame({"mid_price": prices, "ts_event": range(len(prices))})

        # Test different trend thresholds
        generator_sensitive = RemainingValueTunerGenerator(
            lookback_rows=50,
            lookforward_input=200,
            lookforward_offset=20,
            trend_threshold_bps=5.0,  # Very sensitive
            target_name="test_sensitive",
        )

        generator_conservative = RemainingValueTunerGenerator(
            lookback_rows=50,
            lookforward_input=200,
            lookforward_offset=20,
            trend_threshold_bps=50.0,  # Conservative
            target_name="test_conservative",
        )

        targets_sensitive = generator_sensitive.generate_targets(df)
        targets_conservative = generator_conservative.generate_targets(df)

        # Both should generate some values (though potentially different amounts)
        sensitive_valid = ~targets_sensitive["test_sensitive"].is_nan()
        conservative_valid = ~targets_conservative["test_conservative"].is_nan()

        # At least one should have valid values with clear uptrend data
        assert sensitive_valid.any() or conservative_valid.any(), (
            "At least one configuration should detect the uptrend"
        )


class TestRegressionGeneratorIntegration:
    """Integration tests for regression generators."""

    def test_generators_with_factory(self):
        """Test that generators can be created via factory."""
        from represent.target_generators.factory import TargetGeneratorFactory

        # Test DirectionalMFE creation
        mfe_generator = TargetGeneratorFactory.create(
            "directional_mfe", lookforward_horizon=1000, lookback_window=100, expected_fee_pips=0.5
        )
        assert isinstance(mfe_generator, DirectionalMFEGenerator)
        assert mfe_generator.lookforward_horizon == 1000
        assert mfe_generator.lookback_window == 100
        assert mfe_generator.expected_fee_pips == 0.5

        # Test RemainingValueTuner creation
        rv_generator = TargetGeneratorFactory.create(
            "remaining_value_tuner",
            lookback_rows=500,
            lookforward_input=2000,
            lookforward_offset=100,
            trend_threshold_bps=25.0,
        )
        assert isinstance(rv_generator, RemainingValueTunerGenerator)
        assert rv_generator.lookback_rows == 500
        assert rv_generator.lookforward_input == 2000
        assert rv_generator.lookforward_offset == 100
        assert rv_generator.trend_threshold_bps == 25.0

        # Test LogReturnHorizons creation
        log_return_generator = TargetGeneratorFactory.create(
            "log_return_horizons",
            horizons=[1000, 2000, 3000],
            lookback_window=500,
            target_prefix="test_log"
        )
        assert isinstance(log_return_generator, LogReturnHorizonsGenerator)
        assert log_return_generator.horizons == [1000, 2000, 3000]
        assert log_return_generator.lookback_window == 500
        assert log_return_generator.target_prefix == "test_log"

    def test_generators_with_real_data_structure(self):
        """Test generators with realistic data structure."""
        # Create realistic market data structure
        np.random.seed(42)
        n_samples = 2000

        # Generate realistic price series with trends and noise
        prices = []
        current_price = 1.2345
        trend = 0.0

        for i in range(n_samples):
            # Occasional trend changes
            if i % 400 == 0:
                trend = np.random.normal(0, 0.000005)

            # Price evolution with trend and noise
            current_price += trend + np.random.normal(0, 0.00002)
            prices.append(current_price)

        df = pl.DataFrame(
            {
                "mid_price": prices,
                "ts_event": range(n_samples),
                "symbol": ["EURUSD"] * n_samples,  # Additional column
                "volume": np.random.randint(100, 1000, n_samples),  # Additional column
            }
        )

        # Test both generators on same data
        mfe_generator = DirectionalMFEGenerator(
            lookforward_horizon=500,
            lookback_window=50,
            expected_fee_pips=0.5,
            target_names=("mfe_buy", "mfe_sell"),
        )

        rv_generator = RemainingValueTunerGenerator(
            lookback_rows=100,
            lookforward_input=400,
            lookforward_offset=50,
            trend_threshold_bps=15.0,
            neutral_factor=0.6,
            target_name="remaining_value",
        )

        log_return_generator = LogReturnHorizonsGenerator(
            horizons=[200, 400, 800],
            lookback_window=50,
            target_prefix="realistic_log"
        )

        # All should process successfully
        mfe_targets = mfe_generator.generate_targets(df)
        rv_targets = rv_generator.generate_targets(df)
        log_targets = log_return_generator.generate_targets(df)

        # Verify output structure
        assert "mfe_buy" in mfe_targets
        assert "mfe_sell" in mfe_targets
        assert "remaining_value" in rv_targets
        assert "realistic_log_200t" in log_targets
        assert "realistic_log_400t" in log_targets
        assert "realistic_log_800t" in log_targets

        # Verify output lengths match input
        assert len(mfe_targets["mfe_buy"]) == n_samples
        assert len(mfe_targets["mfe_sell"]) == n_samples
        assert len(rv_targets["remaining_value"]) == n_samples
        assert len(log_targets["realistic_log_200t"]) == n_samples
        assert len(log_targets["realistic_log_400t"]) == n_samples
        assert len(log_targets["realistic_log_800t"]) == n_samples

        # Should have some valid values
        assert (~mfe_targets["mfe_buy"].is_nan()).any()
        assert (~mfe_targets["mfe_sell"].is_nan()).any()
        assert (~log_targets["realistic_log_200t"].is_nan()).any()
        assert (~log_targets["realistic_log_400t"].is_nan()).any()
        assert (~log_targets["realistic_log_800t"].is_nan()).any()
        # RemainingValue may have fewer valid values depending on trend detection
