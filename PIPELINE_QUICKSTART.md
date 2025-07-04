# Malaysian Legal AI Pipeline - Quick Start Guide

## 🇲🇾 Overview

Complete pipeline for processing Malaysian Principal Acts into a searchable AI system with semantic search and Q&A capabilities.

## 📁 Project Structure

```
reg-intel/
├── malaysian_acts/           # Source PDFs (your downloaded acts)
│   ├── EN/                   # English PDFs
│   └── BM/                   # Bahasa Malaysia PDFs
├── pipeline/                 # Processing scripts
│   ├── parse_pdf_folder.py   # PDF → structured text
│   ├── chunk_and_embed.py    # Text → embeddings
│   ├── upload_to_vectordb.py # Embeddings → Qdrant
│   ├── legal_search_api.py   # FastAPI search interface
│   └── run_pipeline.py       # Complete orchestrator
├── parsed/                   # Structured legal documents (JSON)
│   ├── EN/
│   └── BM/
├── embeddings/               # Vector embeddings for search
└── legal_ai_system.py        # Your original legal AI system
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/francis/Desktop/Github/reg-intel
pip install -r requirements.txt
```

Key packages:
- `unstructured[pdf]` - PDF parsing
- `sentence-transformers` - Embeddings
- `qdrant-client` - Vector database
- `fastapi` - Search API
- `langchain` - Text processing

### 2. Run Complete Pipeline

```bash
# Process everything: PDFs → embeddings → search database
python pipeline/run_pipeline.py
```

This will:
1. ✅ Parse all PDFs in `malaysian_acts/EN` and `malaysian_acts/BM`
2. ✅ Extract legal sections and structure
3. ✅ Create semantic chunks optimized for legal search
4. ✅ Generate privacy-friendly embeddings (local models)
5. ✅ Start Qdrant vector database
6. ✅ Upload embeddings for semantic search
7. ✅ Validate the complete system

### 3. Start Search API

```bash
# Start the legal search API
python pipeline/legal_search_api.py
```

Access at: http://localhost:8000/docs

## 🔍 Usage Examples

### Semantic Search
```bash
# Search for employment law
curl "http://localhost:8000/search?q=employment+contract+termination&limit=5"

# Search specific act
curl "http://localhost:8000/search?q=intellectual+property&act_number=456"

# Language-specific search
curl "http://localhost:8000/search?q=kontrak+pekerjaan&language=BM"
```

### Legal Q&A
```bash
# Ask legal questions
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are employee rights for termination?", "language": "EN"}'
```

### Python Integration
```python
from legal_ai_system import LegalAISystem

# Initialize with pipeline data
legal_ai = LegalAISystem()

# Search precedents
result = legal_ai.search_legal_precedents(
    "Employee termination without notice"
)

# Draft documents  
document = legal_ai.draft_legal_document_with_knowledge(
    "Create employment agreement with IP clauses",
    "employment_agreement"
)
```

## 🔧 Individual Pipeline Steps

If you want to run steps individually:

```bash
# Step 1: Parse PDFs
python pipeline/parse_pdf_folder.py

# Step 2: Create embeddings
python pipeline/chunk_and_embed.py

# Step 3: Upload to vector database (requires Qdrant running)
docker run -p 6333:6333 qdrant/qdrant
python pipeline/upload_to_vectordb.py

# Step 4: Start search API
python pipeline/legal_search_api.py
```

## 📊 Expected Results

After running the pipeline:

### Parsed Documents
- ✅ 200+ Malaysian Principal Acts parsed
- ✅ Legal sections identified and structured
- ✅ Bilingual support (EN + BM)
- ✅ Metadata preserved (act numbers, citations)

### Embeddings
- ✅ 5,000+ legal text chunks
- ✅ 768-dimensional embeddings (BAAI/bge-base-en-v1.5)
- ✅ Legal metadata for filtering and citation
- ✅ ~50MB total size

### Search Performance
- ✅ Sub-second semantic search
- ✅ Accurate legal citations
- ✅ Bilingual query support
- ✅ Act-specific filtering

## 🛠️ Troubleshooting

### Missing PDFs
```bash
# Check source files
ls -la malaysian_acts/EN/
ls -la malaysian_acts/BM/

# Should show downloaded PDF files
# If empty, run your scraping notebook first
```

### Dependency Issues
```bash
# Install specific packages
pip install unstructured[pdf]
pip install sentence-transformers
pip install qdrant-client

# For macOS users
brew install poppler tesseract
```

### Qdrant Connection
```bash
# Start Qdrant with Docker
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Check if running
curl http://localhost:6333/collections
```

### Memory Issues
```bash
# Reduce batch sizes in chunk_and_embed.py
# Or process subsets of documents
```

## 🔒 Privacy & Security

- ✅ **Local Processing**: All embeddings generated locally
- ✅ **No Cloud APIs**: Uses open-source models only
- ✅ **Data Control**: All data stays on your infrastructure
- ✅ **Audit Trail**: Complete processing logs
- ✅ **Compliance Ready**: Suitable for legal confidentiality

## 🎯 Production Deployment

For production use:

1. **Scale Qdrant**: Use managed Qdrant Cloud or cluster setup
2. **API Security**: Add authentication to FastAPI
3. **Load Balancing**: Use nginx for API load balancing
4. **Monitoring**: Add metrics and health checks
5. **Backup**: Regular vector database backups

## 📝 API Documentation

Full API documentation available at: http://localhost:8000/docs

Key endpoints:
- `GET /search` - Semantic search
- `POST /ask` - Legal Q&A
- `GET /health` - System status
- `GET /stats` - Database statistics

## 💡 Next Steps

1. **Custom Legal Models**: Fine-tune embeddings on Malaysian legal corpus
2. **Case Law Integration**: Add Malaysian court decisions
3. **Document Generation**: Advanced legal document templates
4. **Workflow Integration**: Connect to legal practice management
5. **Analytics**: Search analytics and user insights

## 📞 Support

For issues or questions:
1. Check logs: `malaysian_legal_pipeline.log`
2. Validate setup: `python pipeline/run_pipeline.py test`
3. Review API docs: http://localhost:8000/docs
