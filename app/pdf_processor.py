"""PDF processing and text chunking module."""
import fitz  # PyMuPDF
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
from pathlib import Path
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Handles PDF text extraction and intelligent chunking.
    
    Features:
    - Text extraction using PyMuPDF (fast and accurate)
    - Token-aware chunking with overlap
    - Metadata preservation for source attribution
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Maximum tokens per chunk (default from settings)
            chunk_overlap: Token overlap between chunks (default from settings)
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize tokenizer (same as GPT-4/ChatGPT)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize text splitter with token-based length function
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _token_length(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF file with page-level metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dicts with 'text', 'page_num', and 'metadata' keys
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"Not a PDF file: {pdf_path}")
        
        logger.info(f"Extracting text from: {pdf_path.name}")
        
        doc = fitz.open(str(pdf_path))
        pages_data = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Skip empty pages
                if not text.strip():
                    logger.debug(f"Skipping empty page {page_num + 1}")
                    continue
                
                pages_data.append({
                    'text': text,
                    'page_num': page_num + 1,
                    'metadata': {
                        'pdf_name': pdf_path.name,
                        'page': page_num + 1,
                        'total_pages': len(doc)
                    }
                })
            
            logger.info(f"Extracted {len(pages_data)} pages from {pdf_path.name}")
            
        finally:
            doc.close()
        
        return pages_data
    
    def chunk_text(self, pages_data: List[Dict]) -> List[Dict]:
        """
        Split extracted text into overlapping chunks with metadata.
        
        Args:
            pages_data: List of page data from extract_text_from_pdf
            
        Returns:
            List of chunks with text and metadata
        """
        chunks = []
        
        for page_data in pages_data:
            text = page_data['text']
            page_num = page_data['page_num']
            base_metadata = page_data['metadata']
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            for idx, chunk_text in enumerate(text_chunks):
                # Skip very small chunks (likely noise)
                if len(chunk_text.strip()) < 20:
                    continue
                
                token_count = self._token_length(chunk_text)
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **base_metadata,
                        'chunk_index': idx,
                        'token_count': token_count,
                        'char_count': len(chunk_text)
                    }
                })
        
        logger.info(f"Created {len(chunks)} chunks from {len(pages_data)} pages")
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Complete pipeline: extract text and chunk in one call.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of chunks ready for embedding
        """
        pages = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(pages)
        return chunks
    
    def get_pdf_info(self, pdf_path: str) -> Dict:
        """
        Get basic PDF metadata without full text extraction.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with PDF metadata
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        
        try:
            info = {
                'filename': pdf_path.name,
                'total_pages': len(doc),
                'file_size_mb': round(pdf_path.stat().st_size / (1024 * 1024), 2),
                'metadata': doc.metadata
            }
        finally:
            doc.close()
        
        return info


# Convenience function
def process_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """
    Process a PDF file and return chunks.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Token overlap between chunks
        
    Returns:
        List of chunks with text and metadata
    """
    processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return processor.process_pdf(pdf_path)
