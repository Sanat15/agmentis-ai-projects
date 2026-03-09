"""Enhanced caching layer for query results and embeddings."""
import json
import hashlib
import numpy as np
from typing import Optional, Any, List, Dict, Tuple
import logging
import time
from dataclasses import dataclass, field
from collections import OrderedDict

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Track cache performance metrics."""
    embedding_hits: int = 0
    embedding_misses: int = 0
    result_hits: int = 0
    result_misses: int = 0
    total_time_saved_ms: float = 0.0
    latencies_with_cache: List[float] = field(default_factory=list)
    latencies_without_cache: List[float] = field(default_factory=list)
    
    def record_embedding_hit(self, time_saved_ms: float = 0):
        self.embedding_hits += 1
        self.total_time_saved_ms += time_saved_ms
    
    def record_embedding_miss(self):
        self.embedding_misses += 1
    
    def record_result_hit(self, time_saved_ms: float = 0):
        self.result_hits += 1
        self.total_time_saved_ms += time_saved_ms
    
    def record_result_miss(self):
        self.result_misses += 1
    
    def record_latency(self, latency_ms: float, cached: bool):
        if cached:
            self.latencies_with_cache.append(latency_ms)
        else:
            self.latencies_without_cache.append(latency_ms)
    
    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        total_embedding = self.embedding_hits + self.embedding_misses
        total_result = self.result_hits + self.result_misses
        
        avg_cached = np.mean(self.latencies_with_cache) if self.latencies_with_cache else 0
        avg_uncached = np.mean(self.latencies_without_cache) if self.latencies_without_cache else 0
        
        latency_improvement = 0
        if avg_uncached > 0 and avg_cached > 0:
            latency_improvement = ((avg_uncached - avg_cached) / avg_uncached) * 100
        
        return {
            'embedding_cache': {
                'hits': self.embedding_hits,
                'misses': self.embedding_misses,
                'total': total_embedding,
                'hit_rate': self.embedding_hits / total_embedding if total_embedding > 0 else 0
            },
            'result_cache': {
                'hits': self.result_hits,
                'misses': self.result_misses,
                'total': total_result,
                'hit_rate': self.result_hits / total_result if total_result > 0 else 0
            },
            'latency': {
                'avg_with_cache_ms': round(avg_cached, 2),
                'avg_without_cache_ms': round(avg_uncached, 2),
                'improvement_percent': round(latency_improvement, 1),
                'total_time_saved_ms': round(self.total_time_saved_ms, 2)
            }
        }
    
    def reset(self):
        """Reset all metrics."""
        self.embedding_hits = 0
        self.embedding_misses = 0
        self.result_hits = 0
        self.result_misses = 0
        self.total_time_saved_ms = 0.0
        self.latencies_with_cache.clear()
        self.latencies_without_cache.clear()


class LRUCache:
    """LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return value
            # Expired, remove
            del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        ttl = ttl or self._ttl
        # Remove oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = (value, time.time() + ttl)
        self._cache.move_to_end(key)
    
    def delete(self, key: str):
        self._cache.pop(key, None)
    
    def clear(self):
        self._cache.clear()
    
    def size(self) -> int:
        return len(self._cache)


class InMemoryCache:
    """Simple in-memory cache when Redis is not available."""
    
    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, tuple] = {}
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


class EmbeddingCache:
    """
    Cache for query embeddings to avoid redundant embedding generation.
    
    This is critical for performance since embedding generation is typically
    the most expensive operation (500-800ms on CPU).
    """
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self._cache = LRUCache(max_size=max_size, ttl=ttl)
        self._embedding_dim: Optional[int] = None
    
    def _generate_key(self, text: str) -> str:
        """Generate cache key from text."""
        normalized = text.lower().strip()
        key_hash = hashlib.md5(normalized.encode()).hexdigest()
        return f"emb:{key_hash}"
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        key = self._generate_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            return np.array(cached, dtype=np.float32)
        return None
    
    def set(self, text: str, embedding: np.ndarray, ttl: Optional[int] = None):
        """Cache embedding for text."""
        key = self._generate_key(text)
        # Store as list for JSON serialization compatibility
        self._cache.set(key, embedding.tolist(), ttl)
        if self._embedding_dim is None:
            self._embedding_dim = len(embedding)
    
    def size(self) -> int:
        return self._cache.size()
    
    def clear(self):
        self._cache.clear()


class CacheService:
    """
    Comprehensive caching service for search operations.
    
    Features:
    - Query result caching with TTL
    - Query embedding caching (eliminates repeated model inference)
    - Automatic key generation from query parameters
    - Detailed cache statistics tracking
    - Measurable latency improvements
    """
    
    _instance: Optional['CacheService'] = None
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        ttl: Optional[int] = None,
        use_in_memory: Optional[bool] = None,
        embedding_cache_size: int = 10000
    ):
        """
        Initialize cache service.
        
        Args:
            host: Redis host (unused if in-memory)
            port: Redis port (unused if in-memory)
            ttl: Default TTL in seconds
            use_in_memory: Force in-memory mode
            embedding_cache_size: Max embeddings to cache
        """
        settings = get_settings()
        
        self.host = host or settings.redis_host
        self.port = port or settings.redis_port
        self.ttl = ttl or settings.redis_ttl
        self._use_in_memory = use_in_memory if use_in_memory is not None else settings.use_in_memory
        
        # Initialize metrics
        self.metrics = CacheMetrics()
        
        # Initialize result cache
        if self._use_in_memory:
            logger.info("Using in-memory cache (no Redis)")
            self.client = InMemoryCache(self.ttl)
        else:
            try:
                import redis
                logger.info(f"Connecting to Redis at {self.host}:{self.port}")
                self.client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
            except ImportError:
                logger.warning("Redis not available, falling back to in-memory cache")
                self.client = InMemoryCache(self.ttl)
        
        # Initialize embedding cache
        self.embedding_cache = EmbeddingCache(max_size=embedding_cache_size, ttl=ttl or 3600)
        
        logger.info(f"Cache service initialized (embedding cache size: {embedding_cache_size})")
    
    @classmethod
    def get_instance(cls) -> 'CacheService':
        """Get singleton instance of cache service."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance."""
        cls._instance = None
    
    def _generate_key(self, query: str, top_k: int, threshold: float = 0.7) -> str:
        """Generate cache key from query parameters."""
        key_data = f"{query.lower().strip()}:{top_k}:{threshold}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"search:{key_hash}"
    
    # =================== Embedding Cache Methods ===================
    
    def get_embedding(self, text: str) -> Tuple[Optional[np.ndarray], bool]:
        """
        Get cached embedding for text.
        
        Returns:
            Tuple of (embedding or None, cache_hit boolean)
        """
        embedding = self.embedding_cache.get(text)
        if embedding is not None:
            self.metrics.record_embedding_hit(time_saved_ms=600)  # Est. embedding time
            logger.debug(f"Embedding cache HIT for: {text[:50]}...")
            return embedding, True
        else:
            self.metrics.record_embedding_miss()
            logger.debug(f"Embedding cache MISS for: {text[:50]}...")
            return None, False
    
    def set_embedding(self, text: str, embedding: np.ndarray, ttl: Optional[int] = None):
        """Cache embedding for text."""
        self.embedding_cache.set(text, embedding, ttl)
        logger.debug(f"Cached embedding for: {text[:50]}...")
    
    # =================== Result Cache Methods ===================
    
    def get(self, query: str, top_k: int, threshold: float = 0.7) -> Optional[List[dict]]:
        """Get cached search results."""
        try:
            key = self._generate_key(query, top_k, threshold)
            cached = self.client.get(key)
            
            if cached:
                self.metrics.record_result_hit(time_saved_ms=700)  # Est. full search time
                logger.debug(f"Result cache HIT for query: {query[:50]}...")
                return json.loads(cached)
            else:
                self.metrics.record_result_miss()
                logger.debug(f"Result cache MISS for query: {query[:50]}...")
                return None
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(
        self,
        query: str,
        top_k: int,
        results: List[dict],
        threshold: float = 0.7,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache search results."""
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
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def invalidate(self, pattern: str = "search:*") -> int:
        """Invalidate cached entries matching pattern."""
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries")
                return len(keys)
            return 0
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return 0
    
    def record_latency(self, latency_ms: float, cached: bool):
        """Record query latency for metrics."""
        self.metrics.record_latency(latency_ms, cached)
    
    def get_stats(self) -> dict:
        """Get comprehensive cache statistics."""
        stats = self.metrics.get_stats()
        stats['embedding_cache_size'] = self.embedding_cache.size()
        return stats
    
    def reset_metrics(self):
        """Reset cache statistics."""
        self.metrics.reset()
    
    def health_check(self) -> bool:
        """Check if cache is healthy."""
        try:
            return self.client.ping()
        except Exception:
            return False
    
    def clear_all(self):
        """Clear all caches."""
        self.invalidate("*")
        self.embedding_cache.clear()
        self.reset_metrics()


# Singleton accessor
def get_cache_service() -> CacheService:
    """Get the cache service singleton."""
    return CacheService.get_instance()
