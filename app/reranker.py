"""
Re-ranking service for improving retrieval quality.

Provides cross-encoder re-ranking to improve result quality at the cost of
added latency. Includes evaluation tools to measure the accuracy vs. latency trade-off.
"""
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import time
from dataclasses import dataclass, field

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RerankerMetrics:
    """Track re-ranking performance metrics."""
    total_reranks: int = 0
    total_time_ms: float = 0.0
    position_improvements: List[int] = field(default_factory=list)
    
    def record(self, time_ms: float, position_change: int = 0):
        self.total_reranks += 1
        self.total_time_ms += time_ms
        self.position_improvements.append(position_change)
    
    def get_stats(self) -> Dict:
        avg_time = self.total_time_ms / self.total_reranks if self.total_reranks > 0 else 0
        avg_improvement = np.mean(self.position_improvements) if self.position_improvements else 0
        return {
            'total_reranks': self.total_reranks,
            'avg_time_ms': round(avg_time, 2),
            'total_time_ms': round(self.total_time_ms, 2),
            'avg_position_improvement': round(avg_improvement, 2),
            'max_improvement': max(self.position_improvements) if self.position_improvements else 0
        }
    
    def reset(self):
        self.total_reranks = 0
        self.total_time_ms = 0.0
        self.position_improvements = []


class RerankerService:
    """
    Cross-encoder re-ranking service.
    
    Uses a cross-encoder model to re-score query-document pairs for improved
    ranking accuracy. The cross-encoder jointly encodes the query with each
    document, providing more accurate similarity scores than bi-encoder embeddings.
    
    Trade-offs:
    - Accuracy: Typically improves Top-1 accuracy by 5-15%
    - Latency: Adds 50-200ms per query depending on result count
    """
    
    _instance: Optional['RerankerService'] = None
    
    # Available models from fastest to most accurate
    MODELS = {
        'fast': 'cross-encoder/ms-marco-MiniLM-L-2-v2',      # ~20ms per query
        'balanced': 'cross-encoder/ms-marco-MiniLM-L-6-v2',  # ~50ms per query
        'accurate': 'cross-encoder/ms-marco-TinyBERT-L-2-v2' # ~30ms per query
    }
    
    def __init__(self, model_name: Optional[str] = None, enabled: bool = True):
        """
        Initialize re-ranker.
        
        Args:
            model_name: Cross-encoder model name or preset ('fast', 'balanced', 'accurate')
            enabled: Whether re-ranking is enabled
        """
        self.enabled = enabled
        self.metrics = RerankerMetrics()
        self.model = None
        self.model_name = None
        
        if not enabled:
            logger.info("Re-ranker disabled")
            return
        
        # Resolve model name
        if model_name in self.MODELS:
            resolved_name = self.MODELS[model_name]
        else:
            resolved_name = model_name or self.MODELS['fast']
        
        self.model_name = resolved_name
        
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
            self.enabled = False
    
    @classmethod
    def get_instance(cls, model_name: Optional[str] = None, enabled: bool = True) -> 'RerankerService':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, enabled=enabled)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance."""
        cls._instance = None
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict], float]:
        """
        Re-rank results using cross-encoder.
        
        Args:
            query: Search query
            results: List of search results with 'text' field
            top_k: Return only top K results after re-ranking
            
        Returns:
            Tuple of (re-ranked results, time in ms)
        """
        if not self.enabled or not self.model or not results:
            return results, 0.0
        
        start_time = time.time()
        
        # Create query-document pairs
        pairs = [(query, r.get('text', '')) for r in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Track original top-1 position
        original_top1_text = results[0].get('text', '') if results else ''
        
        # Sort by cross-encoder score
        scored_results = list(zip(scores, results))
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        reranked = [r for _, r in scored_results]
        
        # Update with new similarity scores
        for i, (score, result) in enumerate(scored_results):
            result['original_score'] = result.get('similarity_score', 0)
            result['rerank_score'] = float(score)
            result['similarity_score'] = float(score)  # Replace with rerank score
        
        # Calculate position change for metrics
        new_top1_text = reranked[0].get('text', '') if reranked else ''
        position_change = 0
        if original_top1_text != new_top1_text:
            for i, r in enumerate(results):
                if r.get('text', '') == new_top1_text:
                    position_change = i  # How many positions it moved up
                    break
        
        elapsed_ms = (time.time() - start_time) * 1000
        self.metrics.record(elapsed_ms, position_change)
        
        if top_k:
            reranked = reranked[:top_k]
        
        logger.debug(f"Re-ranked {len(results)} results in {elapsed_ms:.1f}ms")
        
        return reranked, elapsed_ms
    
    def evaluate_impact(
        self,
        query: str,
        results: List[Dict],
        expected_keywords: List[str]
    ) -> Dict:
        """
        Evaluate the impact of re-ranking on a single query.
        
        Returns comparison of metrics before and after re-ranking.
        """
        def check_relevance(result: Dict) -> bool:
            text = result.get('text', '').lower()
            return any(kw.lower() in text for kw in expected_keywords)
        
        def find_first_relevant_rank(results_list: List[Dict]) -> Optional[int]:
            for i, r in enumerate(results_list):
                if check_relevance(r):
                    return i + 1
            return None
        
        # Before re-ranking
        before_rank = find_first_relevant_rank(results)
        before_top1 = before_rank == 1 if before_rank else False
        before_top3 = before_rank is not None and before_rank <= 3
        
        # Re-rank
        reranked, rerank_time = self.rerank(query, results.copy())
        
        # After re-ranking
        after_rank = find_first_relevant_rank(reranked)
        after_top1 = after_rank == 1 if after_rank else False
        after_top3 = after_rank is not None and after_rank <= 3
        
        return {
            'before': {
                'first_relevant_rank': before_rank,
                'top1_correct': before_top1,
                'top3_correct': before_top3
            },
            'after': {
                'first_relevant_rank': after_rank,
                'top1_correct': after_top1,
                'top3_correct': after_top3
            },
            'improvement': {
                'rank_change': (before_rank - after_rank) if (before_rank and after_rank) else 0,
                'top1_improved': after_top1 and not before_top1,
                'top3_improved': after_top3 and not before_top3
            },
            'latency_ms': rerank_time
        }
    
    def get_stats(self) -> Dict:
        """Get re-ranking statistics."""
        return self.metrics.get_stats()
    
    def reset_metrics(self):
        """Reset metrics."""
        self.metrics.reset()


# Singleton accessor
def get_reranker_service(model_name: Optional[str] = None, enabled: bool = True) -> RerankerService:
    """Get the re-ranker service singleton."""
    return RerankerService.get_instance(model_name=model_name, enabled=enabled)
