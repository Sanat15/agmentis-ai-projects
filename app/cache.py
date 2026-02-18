"""Redis caching layer for query results."""
import redis
import json
import hashlib
from typing import Optional, Any, List, Dict
import logging
import time

from app.config import get_settings

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Simple in-memory cache when Redis is not available."""
    
    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._ttl = ttl
    
    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            del self._cache[key]
        return None
    
    def setex(self, key: str, ttl: int, value: str):
        self._cache[key] = (value, time.time() + ttl)
    
    def delete(self, *keys):
        for key in keys:
            self._cache.pop(key, None)
    
    def keys(self, pattern: str = "*"):
        return list(self._cache.keys())
    
    def ping(self):
        return True


class CacheService:
    """
    Redis cache for search query results.
    
    Features:
    - Query result caching with TTL
    - Automatic key generation from query parameters
    - Cache statistics tracking
    """
    
    _instance: Optional['CacheService'] = None
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        ttl: Optional[int] = None,
        use_in_memory: Optional[bool] = None
    ):
        """
        Initialize Redis connection or in-memory fallback.
        
        Args:
            host: Redis host
            port: Redis port
            ttl: Default TTL in seconds
            use_in_memory: Force in-memory mode
        """
        settings = get_settings()
        
        self.host = host or settings.redis_host
        self.port = port or settings.redis_port
        self.ttl = ttl or settings.redis_ttl
        self._use_in_memory = use_in_memory if use_in_memory is not None else settings.use_in_memory
        
        # Track cache statistics
        self._hits = 0
        self._misses = 0
        
        if self._use_in_memory:
            logger.info("Using in-memory cache (no Redis)")
            self.client = InMemoryCache(self.ttl)
        else:
            logger.info(f"Connecting to Redis at {self.host}:{self.port}")
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5
            )
    
    @classmethod
    def get_instance(cls) -> 'CacheService':
        """Get singleton instance of cache service."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _generate_key(self, query: str, top_k: int, threshold: float = 0.7) -> str:
        """
        Generate cache key from query parameters.
        
        Args:
            query: Search query text
            top_k: Number of results
            threshold: Similarity threshold
            
        Returns:
            Cache key string
        """
        # Create deterministic key from parameters
        key_data = f"{query.lower().strip()}:{top_k}:{threshold}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"search:{key_hash}"
    
    def get(self, query: str, top_k: int, threshold: float = 0.7) -> Optional[List[dict]]:
        """
        Get cached search results.
        
        Args:
            query: Search query
            top_k: Number of results
            threshold: Similarity threshold
            
        Returns:
            Cached results or None if not found
        """
        try:
            key = self._generate_key(query, top_k, threshold)
            cached = self.client.get(key)
            
            if cached:
                self._hits += 1
                logger.debug(f"Cache HIT for query: {query[:50]}...")
                return json.loads(cached)
            else:
                self._misses += 1
                logger.debug(f"Cache MISS for query: {query[:50]}...")
                return None
                
        except redis.RedisError as e:
            logger.error(f"Redis error on get: {e}")
            return None
    
    def set(
        self,
        query: str,
        top_k: int,
        results: List[dict],
        threshold: float = 0.7,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache search results.
        
        Args:
            query: Search query
            top_k: Number of results
            results: Results to cache
            threshold: Similarity threshold
            ttl: TTL in seconds (default from settings)
            
        Returns:
            True if cached successfully
        """
        try:
            key = self._generate_key(query, top_k, threshold)
            ttl = ttl or self.ttl
            
            self.client.setex(
                key,
                ttl,
                json.dumps(results)
            )
            logger.debug(f"Cached results for query: {query[:50]}... (TTL: {ttl}s)")
            return True
            
        except redis.RedisError as e:
            logger.error(f"Redis error on set: {e}")
            return False
    
    def invalidate(self, pattern: str = "search:*") -> int:
        """
        Invalidate cached entries matching pattern.
        
        Args:
            pattern: Redis key pattern to match
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries")
                return deleted
            return 0
            
        except redis.RedisError as e:
            logger.error(f"Redis error on invalidate: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with hit rate and counts
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            'hits': self._hits,
            'misses': self._misses,
            'total_requests': total,
            'hit_rate': round(hit_rate, 3)
        }
    
    def health_check(self) -> bool:
        """Check if cache is healthy."""
        try:
            return self.client.ping()
        except (redis.RedisError, Exception):
            return False
    
    def clear_stats(self):
        """Reset cache statistics."""
        self._hits = 0
        self._misses = 0


# Singleton accessor
def get_cache_service() -> CacheService:
    """Get the cache service singleton."""
    return CacheService.get_instance()
