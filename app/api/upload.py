"""Upload endpoint for PDF processing."""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
import time
import uuid
import os
import logging

from app.config import get_settings
from app.models import UploadResponse
from app.pdf_processor import PDFProcessor
from app.embedding_service import get_embedding_service
from app.vector_store import get_vector_store
from app.cache import get_cache_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

settings = get_settings()


def ensure_upload_dir():
    """Ensure upload directory exists."""
    upload_path = Path(settings.upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.
    
    - Extracts text from PDF
    - Chunks text into semantic units
    - Generates embeddings for each chunk
    - Stores in vector database
    
    Returns processing statistics and file ID.
    """
    start_time = time.time()
    
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_dir = ensure_upload_dir()
    file_path = upload_dir / f"{file_id}_{file.filename}"
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved PDF: {file.filename} ({len(content) / 1024:.1f} KB)")
        
        # Process PDF
        processor = PDFProcessor()
        pages = processor.extract_text_from_pdf(str(file_path))
        chunks = processor.chunk_text(pages)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from PDF"
            )
        
        # Generate embeddings
        embed_service = get_embedding_service()
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embed_service.encode_batch(texts, show_progress=False)
        
        # Store in vector database
        vector_store = get_vector_store()
        vector_store.insert_chunks(chunks, embeddings, pdf_id=file_id)
        
        # Invalidate cache (new content added)
        cache = get_cache_service()
        cache.invalidate()
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Processed {file.filename}: {len(chunks)} chunks in {processing_time:.2f}s"
        )
        
        return UploadResponse(
            status="success",
            filename=file.filename,
            file_id=file_id,
            chunks_created=len(chunks),
            total_pages=len(pages),
            processing_time_seconds=round(processing_time, 2)
        )
        
    except Exception as e:
        # Clean up on failure
        if file_path.exists():
            os.remove(file_path)
        
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )


@router.delete("/documents/{pdf_name}")
async def delete_document(pdf_name: str):
    """Delete a document and its chunks from the system."""
    try:
        vector_store = get_vector_store()
        deleted = vector_store.delete_by_pdf(pdf_name)
        
        # Invalidate cache
        cache = get_cache_service()
        cache.invalidate()
        
        if deleted > 0:
            return {
                "status": "success",
                "message": f"Deleted {deleted} chunks for {pdf_name}"
            }
        else:
            return {
                "status": "not_found",
                "message": f"No chunks found for {pdf_name}"
            }
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )
