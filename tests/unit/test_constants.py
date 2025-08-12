"""
Tests for represent.constants module.
"""

from represent.configs import create_compatible_configs
from represent.constants import (
    DEFAULT_FEATURES,
    FEATURE_INDEX_MAP,
    MAX_FEATURES,
    PRICE_LEVELS,
    FeatureType,
    get_output_shape,
)


class TestConstants:
    """Test constant values."""

    def test_basic_constants(self):
        """Test basic constant values."""
        assert PRICE_LEVELS == 402
        # TIME_BINS moved to RepresentConfig.time_bins
        _, _, processor_config = create_compatible_configs()
        # time_bins = samples // ticks_per_bin = 50000 // 100 = 500
        assert processor_config.time_bins == 500

    def test_config_constants(self):
        """Test constants now available through RepresentConfig."""
        _, _, processor_config = create_compatible_configs(currency="AUDUSD")
        assert processor_config.micro_pip_size == 0.00001
        assert processor_config.ticks_per_bin == 100
        assert processor_config.samples == 50000  # Processing batch size

        # Test currency-specific variations
        _, _, jpy_config = create_compatible_configs(currency="USDJPY")
        assert jpy_config.micro_pip_size == 0.001  # Different for JPY pairs
        # Note: true_pip_size is not part of processor config anymore

    def test_feature_constants(self):
        """Test feature-related constants."""
        assert "volume" in DEFAULT_FEATURES
        assert len(DEFAULT_FEATURES) == 1  # Only volume by default

        # Test all available features through FeatureType
        all_features = [f.value for f in FeatureType]
        assert "volume" in all_features
        assert "variance" in all_features
        assert "trade_counts" in all_features
        assert len(all_features) == MAX_FEATURES

        # Test feature index mapping
        for feature in all_features:
            assert feature in FEATURE_INDEX_MAP
            assert isinstance(FEATURE_INDEX_MAP[feature], int)

    def test_get_output_shape(self):
        """Test output shape calculation."""
        # Single feature - now needs time_bins parameter
        _, _, processor_config = create_compatible_configs()
        shape = get_output_shape(["volume"], time_bins=processor_config.time_bins)
        assert shape == (PRICE_LEVELS, processor_config.time_bins)  # (402, 500)

        # Multiple features
        shape = get_output_shape(["volume", "variance"], time_bins=processor_config.time_bins)
        assert shape == (2, PRICE_LEVELS, processor_config.time_bins)  # (2, 402, 500)

        # All features
        all_features = [f.value for f in FeatureType]
        shape = get_output_shape(all_features, time_bins=processor_config.time_bins)
        assert shape == (MAX_FEATURES, PRICE_LEVELS, processor_config.time_bins)  # (3, 402, 500)

    def test_get_output_shape_edge_cases(self):
        """Test output shape edge cases."""
        # Empty list
        _, _, processor_config = create_compatible_configs()
        shape = get_output_shape([], time_bins=processor_config.time_bins)
        assert shape == (0, PRICE_LEVELS, processor_config.time_bins)  # (0, 402, 500)

        # Single feature vs multiple features
        single_shape = get_output_shape(["volume"], time_bins=processor_config.time_bins)
        multi_shape = get_output_shape(["volume", "variance"], time_bins=processor_config.time_bins)
        assert single_shape == (PRICE_LEVELS, processor_config.time_bins)  # 2D for single feature: (402, 500)
        assert multi_shape == (2, PRICE_LEVELS, processor_config.time_bins)  # 3D for multiple features: (2, 402, 500)
