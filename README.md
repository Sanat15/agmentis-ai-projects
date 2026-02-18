# Real Estate Document Intelligence System

A semantic search system for real estate documents using vector embeddings and modern NLP techniques.

## 🎯 Overview

This system enables natural language search over real estate PDFs (property listings, contracts, disclosures). Upload PDFs and ask questions like:
- "What is the asking price?"
- "How many bedrooms does the property have?"
- "What schools are nearby?"

The system extracts text, creates semantic embeddings, and returns the most relevant document sections with source attribution.

## 🏗️ Architecture

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Storage
                                                           ↓
User Query → Query Embedding → Similarity Search → Ranked Results
                                         ↑
                                    Redis Cache
```

**Key Components:**
- **PDF Processing**: PyMuPDF for text extraction, LangChain for intelligent chunking
- **Embeddings**: sentence-transformers (all-mpnet-base-v2, 768 dimensions)
- **Vector DB**: Qdrant with HNSW indexing
- **Caching**: Redis for query result caching
- **API**: FastAPI with automatic documentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- 4GB+ RAM (for embedding model)

### Option 1: Docker (Recommended)

```bash
# Clone and start all services
git clone <repository>
cd real-estate-doc-intelligence

# Start services
docker-compose up -d

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy environment file
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Start infrastructure (Qdrant, Redis)
docker-compose up -d qdrant redis

# Run the API
uvicorn app.main:app --reload --port 8000
```

## 📖 API Usage

### Upload a PDF

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@property_listing.pdf"
```

Response:
```json
{
  "status": "success",
  "filename": "property_listing.pdf",
  "file_id": "abc123",
  "chunks_created": 45,
  "total_pages": 12,
  "processing_time_seconds": 3.2
}
```

### Search Documents

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the asking price?", "top_k": 5}'
```

Response:
```json
{
  "query": "What is the asking price?",
  "results": [
    {
      "text": "The property is listed at $450,000...",
      "pdf_name": "property_listing.pdf",
      "page_number": 1,
      "chunk_index": 3,
      "similarity_score": 0.92
    }
  ],
  "total_results": 5,
  "query_time_ms": 45.2,
  "cached": false
}
```

### API Documentation

Interactive documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧪 Testing

### Run Unit Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html

# Specific test file
pytest tests/test_pdf_processing.py -v
```

### Run Evaluation

```bash
# Ensure services are running
docker-compose up -d

# Upload sample PDFs first
# Then run evaluation
python scripts/evaluate.py --api-url http://localhost:8000
```

### Run Benchmark

```bash
python scripts/benchmark.py --requests 100 --concurrency 5
```

## 📊 Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Top-1 Accuracy | > 60% | Best answer is #1 result |
| Top-3 Accuracy | > 80% | Best answer in top 3 |
| P95 Latency | < 2000ms | 95th percentile response time |
| Avg Latency | < 500ms | Average response time |

## 🔧 Configuration

Key settings in `.env`:

```env
# Embedding Model
EMBEDDING_MODEL=all-mpnet-base-v2  # Best quality/speed balance
EMBEDDING_DIM=768

# Chunking
CHUNK_SIZE=500      # Tokens per chunk
CHUNK_OVERLAP=50    # Overlap between chunks

# Search
TOP_K=5             # Default results count
SIMILARITY_THRESHOLD=0.7  # Minimum similarity

# Cache
REDIS_TTL=3600      # Cache TTL (1 hour)
```

## 📁 Project Structure

```
real-estate-doc-intelligence/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── models.py            # Pydantic schemas
│   ├── pdf_processor.py     # PDF extraction & chunking
│   ├── embedding_service.py # Embedding generation
│   ├── vector_store.py      # Qdrant operations
│   ├── cache.py             # Redis caching
│   └── api/
│       ├── upload.py        # Upload endpoints
│       └── search.py        # Search endpoints
├── tests/
│   ├── conftest.py
│   ├── test_*.py
│   └── evaluation_questions.json
├── scripts/
│   ├── evaluate.py          # Evaluation script
│   └── benchmark.py         # Performance testing
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

## 🎛️ Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Language | Python 3.11 | Modern, async support, ML ecosystem |
| Framework | FastAPI | Async, auto-docs, type safety |
| PDF Processing | PyMuPDF | Fast, accurate text extraction |
| Chunking | LangChain | Smart semantic chunking |
| Embeddings | sentence-transformers | High-quality, offline |
| Vector DB | Qdrant | Fast HNSW search, production-ready |
| Cache | Redis | Sub-ms response times |

## ⚠️ Known Limitations

1. **PDF Quality**: Works best with text-based PDFs. Scanned documents require OCR (not included).
2. **Model Size**: The embedding model requires ~500MB memory.
3. **First Query**: Initial query may be slow while model loads.
4. **Language**: Optimized for English text.

## 🔜 Future Improvements

- [ ] OCR support for scanned documents
- [ ] Re-ranking with cross-encoder for better accuracy
- [ ] Hybrid search (BM25 + vector)
- [ ] Multi-language support
- [ ] Web UI with Streamlit
- [ ] LLM integration for answer synthesis

## 📝 License

MIT License - See LICENSE file for details.

## 👤 Author

Built as part of the Empower AI Internship application.
