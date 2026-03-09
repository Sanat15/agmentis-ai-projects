# 📦 SUBMISSION GUIDE

## What You Need to Submit

### 1. Code Repository (GitHub)

Your repository should contain:

```
smart-document-program/
├── app/                          # ✅ Core application (11 files)
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── pdf_processor.py
│   ├── embedding_service.py
│   ├── vector_store.py
│   ├── cache.py
│   ├── reranker.py              # Optional: Cross-encoder
│   └── api/
│       ├── upload.py
│       └── search.py
│
├── scripts/                      # ✅ Evaluation scripts
│   ├── run_full_evaluation.py
│   ├── evaluate_comprehensive.py
│   └── benchmark_latency.py
│
├── tests/                        # ✅ Test questions
│   ├── real_estate_questions.json   # 100 questions (8 sections)
│   ├── evaluation_questions.json
│   ├── paraphrase_questions.json
│   └── negative_queries.json
│
├── data/                         # ✅ Data folder (with .gitkeep)
│   ├── pdfs/.gitkeep
│   └── qdrant_db/.gitkeep
│
├── results/                      # ✅ Evaluation results (after running)
│   ├── evaluation_results.json
│   └── EVALUATION_SUMMARY.md
│
├── requirements.txt              # ✅ Dependencies
├── README.md                     # ✅ Setup & usage instructions
├── REPORT.md                     # ✅ Performance evaluation report
└── .gitignore                    # ✅ Excludes __pycache__, PDFs, etc.
```

---

## Step-by-Step Submission Process

### Step 1: Download Required PDFs

Go to https://maxestates.in/downloads and download:
- **222 Rajpur** (Dehradun residential property)
- **Max Towers** (Noida commercial office)
- **Max House** (Okhla commercial office)

Save them to `data/pdfs/` folder.

### Step 2: Start the Server

```powershell
cd d:\Documenets\Empower\submit

# Activate virtual environment
..\..venv\Scripts\activate

# Start server
python -m uvicorn app.main:app --port 8000
```

### Step 3: Upload the PDFs

Using Swagger UI (http://localhost:8000/docs):
1. Go to POST /api/upload
2. Click "Try it out"
3. Upload each PDF file

Or using curl:
```bash
curl -X POST http://localhost:8000/api/upload -F "file=@data/pdfs/222_Rajpur.pdf"
curl -X POST http://localhost:8000/api/upload -F "file=@data/pdfs/Max_Towers.pdf"
curl -X POST http://localhost:8000/api/upload -F "file=@data/pdfs/Max_House.pdf"
```

### Step 4: Run Full Evaluation

```powershell
python scripts/run_full_evaluation.py --api-url http://localhost:8000
```

This will:
- Run all 100 test questions
- Calculate Top-1, Top-3, Top-5 accuracy
- Measure latency (avg, P50, P95, P99)
- Test negative queries (false positive rate)
- Test paraphrase robustness
- Save results to `results/` folder

### Step 5: Update Report

After running evaluation, update REPORT.md with actual results:

```markdown
## Performance Results

| Metric | Value | Target |
|--------|-------|--------|
| Top-1 Accuracy | XX% | ≥75% |
| Top-3 Accuracy | XX% | ≥90% |
| P95 Latency | XXXms | <2000ms |
```

### Step 6: Push to GitHub

```powershell
cd d:\Documenets\Empower\submit

# Stage changes
git add .

# Commit
git commit -m "Add evaluation results and updated report"

# Push
git push origin main
```

---

## Required Deliverables Checklist

### Code Repository ✅
- [ ] Core application code (`app/`)
- [ ] Evaluation scripts (`scripts/`)
- [ ] Test questions (`tests/`)
- [ ] README.md with setup instructions
- [ ] requirements.txt

### Report ✅
- [ ] Performance metrics (latency, P95)
- [ ] Retrieval quality (Top-1, Top-3 accuracy, MRR)
- [ ] System behavior analysis (scaling, bottlenecks)

### Evaluation Results ✅
- [ ] Run all 100 test questions
- [ ] Report accuracy metrics
- [ ] Include latency measurements

---

## Report Sections (Required)

Your REPORT.md must include:

### 1. Performance
```
- Average query latency
- P95 latency
- Cache improvement percentage
```

### 2. Retrieval Quality
```
- Test questions used (20+ minimum, you have 100)
- Top-1 accuracy
- Top-3 accuracy
- MRR (optional but good)
```

### 3. System Behavior
```
- What happens as PDFs grow larger?
- What would break first in production?
- Where are the bottlenecks?
```

---

## Demo Commands

### Test Health
```bash
curl http://localhost:8000/health
```

### Test Search
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total area of Max Towers?", "top_k": 5, "score_threshold": 0.2}'
```

### View Collection Stats
```bash
curl http://localhost:8000/debug/collection
```

---

## Tips for Interview

Be prepared to explain:

1. **Why sentence-transformers?**
   - CPU-efficient, no GPU required
   - High quality embeddings for semantic similarity
   - all-mpnet-base-v2 is state-of-the-art for retrieval

2. **Why Qdrant?**
   - Native HNSW indexing
   - Simple API, Python client
   - Supports both in-memory and persistent storage

3. **Why chunking at 1000 chars?**
   - Balances context with embedding quality
   - Larger chunks = more context but diluted embeddings
   - Smaller chunks = precise but may miss context

4. **How to scale this system?**
   - GPU for embeddings (10x faster)
   - Qdrant Cloud for vector storage
   - Redis for distributed caching
   - Kubernetes for API scaling

---

## Your GitHub Repository

**URL:** https://github.com/Sanat15/smart-document-program

Make sure it's **PUBLIC** or share access with the interviewer.
