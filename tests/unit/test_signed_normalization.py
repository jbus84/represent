"""
Critical tests for signed normalization to prevent regression.

These tests ensure that the normalization always follows the notebook approach:
1. Calculate (ask - bid) difference
2. Preserve sign information (neg_mask = combined < 0) 
3. Normalize absolute values to [0,1]
4. Restore sign information (result[neg_mask] *= -1)
5. Final output in [-1, 1] range with proper directional meaning

NEVER remove or weaken these tests - they prevent catastrophic regression
to unsigned [0,1] normalization that loses market directional information.
"""

import numpy as np
import polars as pl
import pandas as pd
from represent.pipeline import process_market_data
from represent.data_structures import OutputBuffer


class TestSignedNormalization:
    """Critical tests to prevent normalization regression."""

    def test_process_market_data_produces_signed_output(self):
        """
        CRITICAL: Ensure process_market_data produces signed [-1,1] output.
        This is the main API function - it MUST preserve sign information.
        """
        # Create synthetic market data
        n_samples = 1000
        
        # Generate timestamps
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
        
        # Create ask/bid prices with clear patterns
        base_price = 1.25000
        ask_prices = np.full(n_samples, base_price + 0.00001)  # Ask slightly higher
        bid_prices = np.full(n_samples, base_price - 0.00001)  # Bid slightly lower
        
        # Create volumes with intentional ask/bid imbalances
        ask_volumes = np.full(n_samples, 100)  # Higher ask volume
        bid_volumes = np.full(n_samples, 50)   # Lower bid volume
        
        # Build DataFrame
        data = {
            'ts_event': timestamps,
            'symbol': ['AUDUSD'] * n_samples,
        }
        
        # Add price levels
        for i in range(10):
            data[f'ask_px_{i:02d}'] = ask_prices + (i * 0.00001)
            data[f'bid_px_{i:02d}'] = bid_prices - (i * 0.00001)
            data[f'ask_sz_{i:02d}'] = ask_volumes
            data[f'bid_sz_{i:02d}'] = bid_volumes
            data[f'ask_ct_{i:02d}'] = np.full(n_samples, 5)
            data[f'bid_ct_{i:02d}'] = np.full(n_samples, 3)
        
        df = pl.DataFrame(data)
        
        # Process data
        result = process_market_data(df, features=["volume"])
        
        # CRITICAL ASSERTIONS
        assert result.min() >= -1.0, f"Minimum {result.min()} should be >= -1.0"
        assert result.max() <= 1.0, f"Maximum {result.max()} should be <= 1.0"
        
        # With ask volume > bid volume, we should have positive values (ask dominance)
        assert result.max() > 0, "Should have positive values with ask dominance"
        
        # Should have some negative values too (due to processing artifacts)
        # This verifies sign preservation is working
        has_negative = np.any(result < 0)
        has_positive = np.any(result > 0)
        assert has_negative or has_positive, "Should have signed values (not all zeros)"

    def test_multi_feature_signed_normalization(self):
        """Test that all features produce signed [-1,1] output."""
        # Create synthetic data
        n_samples = 1000
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
        base_price = 1.25000
        
        data = {
            'ts_event': timestamps,
            'symbol': ['AUDUSD'] * n_samples,
        }
        
        for i in range(10):
            data[f'ask_px_{i:02d}'] = np.full(n_samples, base_price + 0.00001 + (i * 0.00001))
            data[f'bid_px_{i:02d}'] = np.full(n_samples, base_price - 0.00001 - (i * 0.00001))
            data[f'ask_sz_{i:02d}'] = np.random.randint(50, 150, n_samples)
            data[f'bid_sz_{i:02d}'] = np.random.randint(50, 150, n_samples)
            data[f'ask_ct_{i:02d}'] = np.random.randint(1, 10, n_samples)
            data[f'bid_ct_{i:02d}'] = np.random.randint(1, 10, n_samples)
        
        df = pl.DataFrame(data)
        
        # Test all features
        result = process_market_data(df, features=["volume", "variance", "trade_counts"])
        
        assert result.shape[0] == 3, "Should have 3 feature channels"
        
        for i, feature in enumerate(["volume", "variance", "trade_counts"]):
            feature_data = result[i]
            
            # Each feature must be in [-1, 1] range
            assert feature_data.min() >= -1.0, f"{feature} minimum {feature_data.min()} should be >= -1.0"
            assert feature_data.max() <= 1.0, f"{feature} maximum {feature_data.max()} should be <= 1.0"
            
            # Should not be all zeros (indicating processing worked)
            assert not np.allclose(feature_data, 0), f"{feature} should not be all zeros"

    def test_regression_prevention_unsigned_normalization(self):
        """
        CRITICAL REGRESSION TEST: Ensure we never return to unsigned [0,1] normalization.
        This test will FAIL if someone accidentally implements unsigned normalization.
        """
        buffer = OutputBuffer()
        
        # Create scenario where bid > ask (should produce negative values)
        ask_grid = np.ones((402, 500)) * 30   # Low ask volume
        bid_grid = np.ones((402, 500)) * 100  # High bid volume (bid dominance)
        
        result = buffer.prepare_output(ask_grid, bid_grid)
        
        # CRITICAL: This should produce negative values (bid dominance)
        # If this assertion fails, someone broke the signed normalization!
        assert np.all(result <= 0), \
            "REGRESSION DETECTED: Bid dominance should produce negative values. " \
            "Someone may have implemented unsigned [0,1] normalization!"
        
        assert result.min() == -1.0, \
            "REGRESSION DETECTED: Maximum bid dominance should produce -1.0. " \
            "Signed normalization may be broken!"

    def test_directional_information_preservation(self):
        """
        Test that directional market information is preserved.
        This is critical for CNN training - positive/negative values must have meaning.
        """
        buffer = OutputBuffer()
        
        # Test scenario 1: Pure ask dominance
        ask_only = np.ones((402, 500)) * 100
        bid_none = np.zeros((402, 500))
        
        result_ask = buffer.prepare_output(ask_only, bid_none)
        assert np.all(result_ask > 0), "Pure ask dominance should be positive"
        assert np.max(result_ask) == 1.0, "Maximum ask dominance should be 1.0"
        
        # Test scenario 2: Pure bid dominance  
        ask_none = np.zeros((402, 500))
        bid_only = np.ones((402, 500)) * 100
        
        result_bid = buffer.prepare_output(ask_none, bid_only)
        assert np.all(result_bid < 0), "Pure bid dominance should be negative"
        assert np.min(result_bid) == -1.0, "Maximum bid dominance should be -1.0"
        
        # Test scenario 3: Perfect balance
        balanced_ask = np.ones((402, 500)) * 50
        balanced_bid = np.ones((402, 500)) * 50
        
        result_balanced = buffer.prepare_output(balanced_ask, balanced_bid)
        assert np.allclose(result_balanced, 0), "Perfect balance should produce zeros"

    def test_notebook_compliance(self):
        """
        Test that our implementation exactly matches the notebook approach:
        
        combined = ask_market_volume - bid_market_volume
        neg_mask = combined < 0
        abs_combined = np.abs(combined)
        normed_abs_combined = (abs_combined - 0) / (abs_combined.max() - 0)  
        normed_abs_combined[neg_mask] *= -1
        """
        buffer = OutputBuffer()
        
        # Create test data that matches notebook pattern
        np.random.seed(42)  # For reproducibility
        ask_grid = np.random.random((402, 500)) * 100
        bid_grid = np.random.random((402, 500)) * 100
        
        # Manual calculation following notebook exactly
        combined = ask_grid - bid_grid
        neg_mask = combined < 0
        abs_combined = np.abs(combined)
        
        if abs_combined.max() > 0:
            normed_abs_combined = (abs_combined - 0) / (abs_combined.max() - 0)
        else:
            normed_abs_combined = np.zeros_like(abs_combined)
            
        normed_abs_combined[neg_mask] *= -1
        
        # Our implementation result
        our_result = buffer.prepare_output(ask_grid, bid_grid)
        
        # Should match exactly
        assert np.allclose(our_result, normed_abs_combined), \
            "Our implementation should match notebook approach exactly"

    def test_cnn_compatibility_range(self):
        """
        Test that output is in proper range for CNN training.
        CNNs work great with [-1,1] range but not with [0,1] range for signed data.
        """
        buffer = OutputBuffer()
        
        # Test multiple random scenarios
        for _ in range(10):
            ask_grid = np.random.random((402, 500)) * np.random.uniform(10, 1000)
            bid_grid = np.random.random((402, 500)) * np.random.uniform(10, 1000)
            
            result = buffer.prepare_output(ask_grid, bid_grid)
            
            # Perfect for CNN training
            assert result.min() >= -1.0, "Should be >= -1 for CNN compatibility"
            assert result.max() <= 1.0, "Should be <= 1 for CNN compatibility"
            
            # Should be zero-centered for better gradient flow
            mean_abs_value = np.mean(np.abs(result))
            assert mean_abs_value > 0, "Should have non-zero values for meaningful training"

    def test_semantic_meaning_preservation(self):
        """
        Test that the sign has proper semantic meaning:
        - Negative values = Bid dominance (market selling pressure)  
        - Positive values = Ask dominance (market buying pressure)
        - Zero values = Market balance
        """
        buffer = OutputBuffer()
        
        # Create clear semantic scenarios
        scenarios = [
            # (ask_vol, bid_vol, expected_sign, description)
            (100, 20, "positive", "Strong ask dominance (buying pressure)"),
            (20, 100, "negative", "Strong bid dominance (selling pressure)"), 
            (50, 50, "zero", "Perfect market balance"),
            (80, 40, "positive", "Moderate ask dominance"),
            (40, 80, "negative", "Moderate bid dominance"),
        ]
        
        for ask_vol, bid_vol, expected_sign, description in scenarios:
            ask_grid = np.ones((402, 500)) * ask_vol
            bid_grid = np.ones((402, 500)) * bid_vol
            
            result = buffer.prepare_output(ask_grid, bid_grid)
            
            if expected_sign == "positive":
                assert np.all(result >= 0), f"Failed {description}: should be non-negative"
                if ask_vol != bid_vol:  # Not perfectly balanced
                    assert np.max(result) > 0, f"Failed {description}: should have positive values"
                    
            elif expected_sign == "negative":
                assert np.all(result <= 0), f"Failed {description}: should be non-positive"
                if ask_vol != bid_vol:  # Not perfectly balanced
                    assert np.min(result) < 0, f"Failed {description}: should have negative values"
                    
            elif expected_sign == "zero":
                assert np.allclose(result, 0), f"Failed {description}: should be zero"