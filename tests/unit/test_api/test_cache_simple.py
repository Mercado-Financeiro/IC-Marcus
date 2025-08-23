"""Simplified unit tests for API cache module without async."""

import pytest
import json
from unittest.mock import Mock, MagicMock
from src.api.cache import CacheManager


class TestCacheManager:
    """Test cases for CacheManager."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock = Mock()
        mock.get = Mock(return_value=None)
        mock.setex = Mock()
        mock.delete = Mock()
        mock.keys = Mock(return_value=[])
        mock.ping = Mock()
        return mock
    
    @pytest.fixture
    def cache_manager(self, mock_redis):
        """Create cache manager with mock Redis."""
        return CacheManager(mock_redis, ttl=300)
    
    def test_generate_key(self, cache_manager):
        """Test cache key generation."""
        features = {"feature1": 1.0, "feature2": 2.0}
        key = cache_manager.generate_key(features)
        
        assert key.startswith("pred:")
        assert len(key) > 5
        
        # Same features should generate same key
        key2 = cache_manager.generate_key(features)
        assert key == key2
        
        # Different features should generate different key
        features3 = {"feature1": 1.0, "feature3": 3.0}
        key3 = cache_manager.generate_key(features3)
        assert key != key3
    
    def test_generate_key_order_independence(self, cache_manager):
        """Test that key generation is order-independent."""
        features1 = {"a": 1.0, "b": 2.0, "c": 3.0}
        features2 = {"c": 3.0, "a": 1.0, "b": 2.0}
        
        key1 = cache_manager.generate_key(features1)
        key2 = cache_manager.generate_key(features2)
        
        assert key1 == key2
    
    def test_is_connected(self, cache_manager, mock_redis):
        """Test checking connection status."""
        assert cache_manager.is_connected() is True
        mock_redis.ping.assert_called_once()
    
    def test_is_connected_no_redis(self):
        """Test connection status when Redis is None."""
        cache_manager = CacheManager(None, ttl=300)
        assert cache_manager.is_connected() is False
    
    def test_is_connected_error(self, cache_manager, mock_redis):
        """Test connection status when ping fails."""
        mock_redis.ping.side_effect = Exception("Connection error")
        assert cache_manager.is_connected() is False