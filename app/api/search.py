"""Search endpoint for document queries."""
from fastapi import APIRouter, HTTPException
import time
import logging

from app.models import SearchRequest, SearchResponse, SearchResult
from app.embedding_service import get_embedding_service
from app.vector_store import get_vector_store
from app.cache import get_cache_service
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])

settings = get_settings()


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search documents with natural language query.
    
    - Generates embedding for query
    - Performs vector similarity search
    - Returns ranked results with metadata
    - Caches results for repeated queries
    """
    start_time = time.time()
    
    query = request.query.strip()
    top_k = request.top_k
    threshold = request.score_threshold or settings.similarity_threshold
    
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    # Check cache first
    cache = get_cache_service()
    cached_results = cache.get(query, top_k, threshold)
    
    if cached_results is not None:
        query_time = (time.time() - start_time) * 1000
        logger.info(f"Cache HIT for query: '{query[:50]}...' ({query_time:.1f}ms)")
        
        return SearchResponse(
            query=query,
            results=[SearchResult(**r) for r in cached_results],
            total_results=len(cached_results),
            query_time_ms=round(query_time, 2),
            cached=True
        )
    
    try:
        # Generate query embedding
        embed_service = get_embedding_service()
        query_embedding = embed_service.encode_text(query)
        
        # Search vector store
        vector_store = get_vector_store()
        results = vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
            score_threshold=threshold
        )
        
        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append(SearchResult(
                text=r['text'],
                pdf_name=r['metadata']['pdf_name'],
                page_number=r['metadata']['page'],
                chunk_index=r['metadata']['chunk_index'],
                similarity_score=round(r['score'], 4)
            ))
        
        # Cache results
        cache.set(
            query, 
            top_k, 
            [r.model_dump() for r in formatted_results],
            threshold
        )
        
        query_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Search: '{query[:50]}...' -> {len(formatted_results)} results ({query_time:.1f}ms)"
        )
        
        return SearchResponse(
            query=query,
            results=formatted_results,
            total_results=len(formatted_results),
            query_time_ms=round(query_time, 2),
            cached=False
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )


@router.get("/search/stats")
async def get_search_stats():
    """Get search and cache statistics."""
    try:
        cache = get_cache_service()
        vector_store = get_vector_store()
        
        cache_stats = cache.get_stats()
        collection_info = vector_store.get_collection_info()
        
        return {
            "cache": cache_stats,
            "vector_store": collection_info
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )
