"""
Tests for represent.constants module.
"""

from represent.constants import (
    PRICE_LEVELS, TIME_BINS, SAMPLES, TICKS_PER_BIN, MICRO_PIP_SIZE,
    DEFAULT_FEATURES, FEATURE_INDEX_MAP, MAX_FEATURES, get_output_shape, FeatureType
)


class TestConstants:
    """Test constant values."""
    
    def test_basic_constants(self):
        """Test basic constant values."""
        assert PRICE_LEVELS == 402
        assert TIME_BINS == 500
        assert SAMPLES == 50000
        assert TICKS_PER_BIN == 100
        assert MICRO_PIP_SIZE == 0.00001
    
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
        # Single feature
        shape = get_output_shape(["volume"])
        assert shape == (PRICE_LEVELS, TIME_BINS)
        
        # Multiple features
        shape = get_output_shape(["volume", "variance"])
        assert shape == (2, PRICE_LEVELS, TIME_BINS)
        
        # All features
        all_features = [f.value for f in FeatureType]
        shape = get_output_shape(all_features)
        assert shape == (MAX_FEATURES, PRICE_LEVELS, TIME_BINS)
        
    def test_get_output_shape_edge_cases(self):
        """Test output shape edge cases."""
        # Empty list
        shape = get_output_shape([])
        assert shape == (0, PRICE_LEVELS, TIME_BINS)
        
        # Single feature vs multiple features
        single_shape = get_output_shape(["volume"])
        multi_shape = get_output_shape(["volume", "variance"])
        assert single_shape == (PRICE_LEVELS, TIME_BINS)  # 2D for single feature
        assert multi_shape == (2, PRICE_LEVELS, TIME_BINS)  # 3D for multiple features