"""
Real Estate Document Intelligence System

FastAPI application for semantic search over real estate documents.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys

from app import __version__
from app.config import get_settings
from app.api import upload, search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info("Starting Real Estate Doc Intelligence API...")
    settings = get_settings()
    logger.info(f"Environment: {'DEBUG' if settings.debug else 'PRODUCTION'}")
    
    # Pre-load embedding model (slow on first load)
    try:
        from app.embedding_service import get_embedding_service
        embed_service = get_embedding_service()
        logger.info(f"Embedding model loaded: {embed_service.model_name}")
    except Exception as e:
        logger.warning(f"Could not pre-load embedding model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
settings = get_settings()

app = FastAPI(
    title="Real Estate Document Intelligence API",
    description="""
    A semantic search system for real estate documents.
    
    ## Features
    - PDF upload and processing
    - Natural language search
    - Source attribution (PDF name, page number)
    - Fast vector similarity search
    - Result caching
    
    ## Tech Stack
    - **Embeddings**: sentence-transformers (all-mpnet-base-v2)
    - **Vector DB**: Qdrant
    - **Cache**: Redis
    - **Framework**: FastAPI
    """,
    version=__version__,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(search.router)


# Health check endpoints
@app.get("/health", tags=["health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__
    }


@app.get("/health/detailed", tags=["health"])
async def detailed_health_check():
    """Detailed health check with service status."""
    services = {}
    
    # Check Qdrant
    try:
        from app.vector_store import get_vector_store
        vs = get_vector_store()
        vs.client.get_collections()
        services["qdrant"] = "healthy"
    except Exception as e:
        services["qdrant"] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        from app.cache import get_cache_service
        cache = get_cache_service()
        if cache.health_check():
            services["redis"] = "healthy"
        else:
            services["redis"] = "unhealthy"
    except Exception as e:
        services["redis"] = f"unhealthy: {str(e)}"
    
    # Overall status
    all_healthy = all(s == "healthy" for s in services.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "version": __version__,
        "services": services
    }


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Real Estate Document Intelligence API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/debug/collection", tags=["debug"])
async def debug_collection():
    """Debug endpoint to check collection status."""
    try:
        from app.vector_store import get_vector_store
        vs = get_vector_store()
        info = vs.client.get_collection(vs.collection_name)
        
        # Get sample points
        points = vs.client.scroll(vs.collection_name, limit=5, with_vectors=True)
        samples = []
        for p in points[0]:
            samples.append({
                "id": str(p.id),
                "has_vector": p.vector is not None and len(p.vector) > 0 if p.vector else False,
                "vector_length": len(p.vector) if p.vector else 0,
                "text_preview": p.payload.get('text', '')[:100] if p.payload else ''
            })
        
        return {
            "collection_name": vs.collection_name,
            "points_count": info.points_count,
            "indexed_vectors": info.indexed_vectors_count,
            "status": str(info.status),
            "sample_points": samples
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/debug/search", tags=["debug"])
async def debug_search(query: str = "test"):
    """Debug search to see raw results."""
    try:
        from app.embedding_service import get_embedding_service
        from app.vector_store import get_vector_store
        import numpy as np
        
        embed_service = get_embedding_service()
        query_embedding = embed_service.encode_text(query)
        
        vs = get_vector_store()
        
        # Raw query
        raw_results = vs.client.query_points(
            collection_name=vs.collection_name,
            query=query_embedding.tolist(),
            limit=5
        )
        
        return {
            "query": query,
            "embedding_shape": list(query_embedding.shape),
            "embedding_sample": query_embedding[:5].tolist(),
            "raw_result_type": str(type(raw_results)),
            "has_points": hasattr(raw_results, 'points'),
            "points_count": len(raw_results.points) if hasattr(raw_results, 'points') else 0,
            "raw_points": [
                {
                    "id": str(p.id),
                    "score": p.score,
                    "payload_keys": list(p.payload.keys()) if p.payload else []
                }
                for p in (raw_results.points if hasattr(raw_results, 'points') else [])
            ]
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
