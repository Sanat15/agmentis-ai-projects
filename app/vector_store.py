"""Qdrant vector database operations."""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from typing import List, Dict, Optional
import numpy as np
import uuid
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton storage (survives class reimports)
_vector_store_instance = None
_qdrant_client = None


class VectorStore:
    """
    Qdrant vector database wrapper for document embeddings.
    
    Features:
    - HNSW indexing for fast approximate search
    - Cosine similarity for semantic matching
    - Payload storage for metadata
    - Filtering capabilities
    """
    
    _instance: Optional['VectorStore'] = None
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        in_memory: bool = False,
        client: Optional[QdrantClient] = None
    ):
        """
        Initialize Qdrant client.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the vector collection
            in_memory: Use in-memory storage (no Docker required)
            client: Existing QdrantClient to reuse
        """
        global _qdrant_client
        settings = get_settings()
        
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self.vector_size = settings.embedding_dim
        
        # Reuse existing client if provided or available globally
        if client:
            self.client = client
        elif _qdrant_client is not None:
            self.client = _qdrant_client
            logger.info("Reusing existing Qdrant client")
        elif in_memory:
            # Use local file storage instead of pure memory (survives reloads)
            import tempfile
            import os
            qdrant_path = os.path.join(os.path.dirname(__file__), "..", "data", "qdrant_db")
            os.makedirs(qdrant_path, exist_ok=True)
            logger.info(f"Using local Qdrant storage at {qdrant_path}")
            self.client = QdrantClient(path=qdrant_path)
            _qdrant_client = self.client
        else:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port)
            _qdrant_client = self.client
        
        # Ensure collection exists
        self._ensure_collection()
    
    @classmethod
    def get_instance(cls) -> 'VectorStore':
        """Get singleton instance of vector store."""
        global _vector_store_instance
        if _vector_store_instance is None:
            settings = get_settings()
            _vector_store_instance = cls(in_memory=settings.use_in_memory)
            cls._instance = _vector_store_instance
        return _vector_store_instance
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.create_collection()
        else:
            logger.info(f"Collection exists: {self.collection_name}")
    
    def create_collection(self, vector_size: Optional[int] = None):
        """
        Create or recreate vector collection.
        
        Args:
            vector_size: Embedding dimension (default from settings)
        """
        size = vector_size or self.vector_size
        
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=size,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created collection '{self.collection_name}' with vector size {size}")
    
    def insert_chunks(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        pdf_id: Optional[str] = None
    ) -> List[str]:
        """
        Insert document chunks with embeddings.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'
            embeddings: Numpy array of embeddings
            pdf_id: Optional PDF identifier
            
        Returns:
            List of inserted point IDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")
        
        points = []
        point_ids = []
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            payload = {
                'text': chunk['text'],
                'pdf_name': chunk['metadata'].get('pdf_name', ''),
                'page': chunk['metadata'].get('page', 0),
                'chunk_index': chunk['metadata'].get('chunk_index', idx),
                'token_count': chunk['metadata'].get('token_count', 0),
            }
            
            if pdf_id:
                payload['pdf_id'] = pdf_id
            
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        # Batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Inserted {len(points)} chunks into {self.collection_name}")
        return point_ids
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
        pdf_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar chunks.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            pdf_filter: Optional filter by PDF name
            
        Returns:
            List of results with text, metadata, and score
        """
        # Build filter if needed
        search_filter = None
        if pdf_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="pdf_name",
                        match=MatchValue(value=pdf_filter)
                    )
                ]
            )
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=search_filter
        ).points
        
        formatted_results = []
        for hit in results:
            formatted_results.append({
                'id': hit.id,
                'text': hit.payload.get('text', ''),
                'metadata': {
                    'pdf_name': hit.payload.get('pdf_name', ''),
                    'page': hit.payload.get('page', 0),
                    'chunk_index': hit.payload.get('chunk_index', 0),
                    'token_count': hit.payload.get('token_count', 0),
                },
                'score': hit.score
            })
        
        return formatted_results
    
    def delete_by_pdf(self, pdf_name: str) -> int:
        """
        Delete all chunks from a specific PDF.
        
        Supports both exact match and partial match (e.g., "Sanat_SOP.pdf" 
        will match "5e13907b-fc03-4372-b938-1e430c54c089_Sanat_SOP.pdf").
        
        Args:
            pdf_name: Name of PDF to delete (full or partial)
            
        Returns:
            Number of points deleted
        """
        # First try exact match
        points = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="pdf_name",
                        match=MatchValue(value=pdf_name)
                    )
                ]
            ),
            limit=10000
        )[0]
        
        # If no exact match, find by partial match (ends with filename)
        if not points:
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000
            )[0]
            points = [p for p in all_points 
                     if p.payload and p.payload.get('pdf_name', '').endswith(pdf_name)]
        
        if points:
            point_ids = [p.id for p in points]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            logger.info(f"Deleted {len(point_ids)} chunks for {pdf_name}")
            return len(point_ids)
        
        return 0
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            'name': self.collection_name,
            'vectors_count': info.vectors_count,
            'points_count': info.points_count,
            'status': info.status.value
        }
    
    def clear_collection(self):
        """Delete all points in collection."""
        self.create_collection()
        logger.info(f"Cleared collection: {self.collection_name}")


# Singleton accessor
def get_vector_store() -> VectorStore:
    """Get the vector store singleton."""
    return VectorStore.get_instance()
