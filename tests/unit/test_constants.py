"""
Tests for represent.constants module.
"""

from represent.constants import (
    PRICE_LEVELS,
    DEFAULT_FEATURES, FEATURE_INDEX_MAP, MAX_FEATURES, get_output_shape, FeatureType
)
from represent.config import create_represent_config


class TestConstants:
    """Test constant values."""
    
    def test_basic_constants(self):
        """Test basic constant values."""
        assert PRICE_LEVELS == 402
        # TIME_BINS moved to RepresentConfig.time_bins
        config = create_represent_config()
        # time_bins = samples // ticks_per_bin = 25000 // 100 = 250
        assert config.time_bins == 250
        
    def test_config_constants(self):
        """Test constants now available through RepresentConfig."""
        config = create_represent_config("AUDUSD")
        assert config.micro_pip_size == 0.00001
        assert config.ticks_per_bin == 100
        assert config.samples == 25000  # Processing batch size
        
        # Test currency-specific variations
        jpy_config = create_represent_config("USDJPY")
        assert jpy_config.micro_pip_size == 0.001  # Different for JPY pairs
        assert jpy_config.true_pip_size == 0.01
    
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
        config = create_represent_config()
        shape = get_output_shape(["volume"], time_bins=config.time_bins)
        assert shape == (PRICE_LEVELS, config.time_bins)  # (402, 250)
        
        # Multiple features
        shape = get_output_shape(["volume", "variance"], time_bins=config.time_bins)
        assert shape == (2, PRICE_LEVELS, config.time_bins)  # (2, 402, 250)
        
        # All features
        all_features = [f.value for f in FeatureType]
        shape = get_output_shape(all_features, time_bins=config.time_bins)
        assert shape == (MAX_FEATURES, PRICE_LEVELS, config.time_bins)  # (3, 402, 250)
        
    def test_get_output_shape_edge_cases(self):
        """Test output shape edge cases."""
        # Empty list
        config = create_represent_config()
        shape = get_output_shape([], time_bins=config.time_bins)
        assert shape == (0, PRICE_LEVELS, config.time_bins)  # (0, 402, 250)
        
        # Single feature vs multiple features
        single_shape = get_output_shape(["volume"], time_bins=config.time_bins)
        multi_shape = get_output_shape(["volume", "variance"], time_bins=config.time_bins)
        assert single_shape == (PRICE_LEVELS, config.time_bins)  # 2D for single feature: (402, 250)
        assert multi_shape == (2, PRICE_LEVELS, config.time_bins)  # 3D for multiple features: (2, 402, 250)