"""Unit tests for API cache module."""

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
    
    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache_manager, mock_redis):
        """Test getting value from cache (hit)."""
        cached_data = {"prediction": 0.8, "version": "1.0"}
        mock_redis.get.return_value = json.dumps(cached_data)
        
        result = await cache_manager.get("test_key")
        
        assert result == cached_data
        mock_redis.get.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache_manager, mock_redis):
        """Test getting value from cache (miss)."""
        mock_redis.get.return_value = None
        
        result = await cache_manager.get("test_key")
        
        assert result is None
        mock_redis.get.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_get_no_redis(self):
        """Test get when Redis is not available."""
        cache_manager = CacheManager(None, ttl=300)
        result = await cache_manager.get("test_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_set_cache(self, cache_manager, mock_redis):
        """Test setting value in cache."""
        data = {"prediction": 0.8, "version": "1.0"}
        
        await cache_manager.set("test_key", data)
        
        mock_redis.setex.assert_called_once_with(
            "test_key",
            300,
            json.dumps(data)
        )
    
    @pytest.mark.asyncio
    async def test_set_no_redis(self):
        """Test set when Redis is not available."""
        cache_manager = CacheManager(None, ttl=300)
        # Should not raise exception
        await cache_manager.set("test_key", {"data": "value"})
    
    @pytest.mark.asyncio
    async def test_delete_cache(self, cache_manager, mock_redis):
        """Test deleting value from cache."""
        await cache_manager.delete("test_key")
        mock_redis.delete.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, cache_manager, mock_redis):
        """Test clearing cache entries."""
        mock_redis.keys.return_value = ["pred:key1", "pred:key2", "pred:key3"]
        
        await cache_manager.clear("pred:*")
        
        mock_redis.keys.assert_called_once_with("pred:*")
        mock_redis.delete.assert_called_once_with(
            "pred:key1", "pred:key2", "pred:key3"
        )
    
    @pytest.mark.asyncio
    async def test_clear_cache_no_keys(self, cache_manager, mock_redis):
        """Test clearing cache when no keys match."""
        mock_redis.keys.return_value = []
        
        await cache_manager.clear("pred:*")
        
        mock_redis.keys.assert_called_once()
        mock_redis.delete.assert_not_called()
    
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
    
    @pytest.mark.asyncio
    async def test_exception_handling_get(self, cache_manager, mock_redis):
        """Test exception handling in get method."""
        mock_redis.get.side_effect = Exception("Redis error")
        
        result = await cache_manager.get("test_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_exception_handling_set(self, cache_manager, mock_redis):
        """Test exception handling in set method."""
        mock_redis.setex.side_effect = Exception("Redis error")
        
        # Should not raise exception
        await cache_manager.set("test_key", {"data": "value"})