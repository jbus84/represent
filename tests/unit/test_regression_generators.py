"""
Comprehensive unit tests for regression target generators.
Tests DirectionalMFEGenerator and RemainingValueTunerGenerator.
"""

import numpy as np
import polars as pl

from represent.target_generators.regression import (
    DirectionalMFEGenerator,
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
        valid_indices = ~np.isnan(mfe_buy)
        assert np.any(valid_indices), "Should have some valid MFE values"

        # For upward trend, buy MFE should be positive, sell MFE should be 0 or negative
        valid_buy_mfe = mfe_buy[valid_indices]
        valid_sell_mfe = mfe_sell[valid_indices]

        assert np.all(valid_buy_mfe >= 0), "Buy MFE should be non-negative in uptrend"
        assert np.all(valid_sell_mfe <= 0), "Sell MFE should be non-positive in uptrend"

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
        if not np.isnan(mfe_buy[test_idx]):
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
        valid_mask = ~np.isnan(mfe_buy_fees) & ~np.isnan(mfe_buy_no_fees)
        if np.any(valid_mask):
            # With fees should be 10 bps lower than without fees
            fee_difference = mfe_buy_no_fees[valid_mask] - mfe_buy_fees[valid_mask]
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
        valid_mask = ~np.isnan(mfe_buy)
        if np.any(valid_mask):
            assert np.allclose(mfe_buy[valid_mask], 0.0, atol=0.1), "Flat prices should give ~0 MFE"
            assert np.allclose(mfe_sell[valid_mask], 0.0, atol=0.1), (
                "Flat prices should give ~0 MFE"
            )


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
        valid_mask = ~np.isnan(remaining_values)
        assert np.any(valid_mask), "Should have some valid remaining value calculations"

        # Test monotonicity in trend region (positions 600-1200)
        trend_start = 600
        trend_end = 1200
        trend_values = remaining_values[trend_start:trend_end]
        valid_trend_values = trend_values[~np.isnan(trend_values)]

        if len(valid_trend_values) > 50:
            # Check that values generally decrease (allowing some noise)
            early_values = valid_trend_values[: len(valid_trend_values) // 3]
            late_values = valid_trend_values[-len(valid_trend_values) // 3 :]

            assert np.mean(early_values) > np.mean(late_values), (
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
        valid_mask = ~np.isnan(remaining_values)
        valid_values = remaining_values[valid_mask]

        if len(valid_values) > 0:
            # Values should be relatively small (no strong trend)
            assert np.all(np.abs(valid_values) < 100.0), (
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
        assert np.any(~np.isnan(values_mono)), "Monotonic version should have valid values"
        assert np.any(~np.isnan(values_no_mono)), "Non-monotonic version should have valid values"

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
        sensitive_valid = ~np.isnan(targets_sensitive["test_sensitive"])
        conservative_valid = ~np.isnan(targets_conservative["test_conservative"])

        # At least one should have valid values with clear uptrend data
        assert np.any(sensitive_valid) or np.any(conservative_valid), (
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

        # Both should process successfully
        mfe_targets = mfe_generator.generate_targets(df)
        rv_targets = rv_generator.generate_targets(df)

        # Verify output structure
        assert "mfe_buy" in mfe_targets
        assert "mfe_sell" in mfe_targets
        assert "remaining_value" in rv_targets

        # Verify output lengths match input
        assert len(mfe_targets["mfe_buy"]) == n_samples
        assert len(mfe_targets["mfe_sell"]) == n_samples
        assert len(rv_targets["remaining_value"]) == n_samples

        # Should have some valid values
        assert np.any(~np.isnan(mfe_targets["mfe_buy"]))
        assert np.any(~np.isnan(mfe_targets["mfe_sell"]))
        # RemainingValue may have fewer valid values depending on trend detection
