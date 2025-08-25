"""Cache management for API."""

import json
import hashlib
from typing import Dict, Optional
import structlog

log = structlog.get_logger()


class CacheManager:
    """Manages caching operations."""
    
    def __init__(self, redis_client, ttl: int = 300):
        """
        Initialize cache manager.
        
        Args:
            redis_client: Redis client instance
            ttl: Time to live in seconds
        """
        self.redis_client = redis_client
        self.ttl = ttl
    
    def generate_key(self, features: Dict[str, float]) -> str:
        """
        Generate cache key from features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Cache key string
        """
        # Sort features for consistency
        sorted_features = sorted(features.items())
        features_str = json.dumps(sorted_features)
        
        # Hash for compact key
        hash_obj = hashlib.md5(features_str.encode())
        return f"pred:{hash_obj.hexdigest()}"
    
    async def get(self, key: str) -> Optional[Dict]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            log.warning(f"Cache get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Dict):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(value)
            )
        except Exception as e:
            log.warning(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        """
        if not self.redis_client:
            return
        
        try:
            self.redis_client.delete(key)
        except Exception as e:
            log.warning(f"Cache delete error: {e}")
    
    async def clear(self, pattern: str = "pred:*"):
        """
        Clear cache entries matching pattern.
        
        Args:
            pattern: Key pattern to match
        """
        if not self.redis_client:
            return
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                log.info(f"Cleared {len(keys)} cache entries")
        except Exception as e:
            log.warning(f"Cache clear error: {e}")
    
    def is_connected(self) -> bool:
        """Check if cache is connected."""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False