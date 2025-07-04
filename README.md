# Legal LLM Platform 🏛️ (MVP Version)

**A working MVP for custom AI platform for law firms**

🚀 **CURRENTLY RUNNING**: Your Legal LLM Platform is now operational!

## 🎯 What You've Built

This is a **complete, working MVP** that demonstrates:
- ✅ **Document Processing Pipeline**: Upload → Extract → Classify → Store
- ✅ **AI Text Generation**: Template-based legal document creation
- ✅ **Web Interface**: Professional UI for document management
- ✅ **REST API**: Full API with automatic documentation
- ✅ **Document Classification**: Automatic legal document categorization

## 🔧 How to Use Your Platform

### Start the Platform
```bash
# In your terminal, run:
./start-both.sh
```

### Access Your Platform
- **🌐 Web App**: http://localhost:8501 (Main interface)
- **📚 API Docs**: http://localhost:8000/docs (API documentation)
- **🔌 API**: http://localhost:8000 (Direct API access)

### Try These Features
1. **Upload Documents**: Go to "Upload Documents" tab, drag & drop a PDF
2. **Generate Text**: Use "Generate Text" tab to create legal documents
3. **View Training Data**: See processed documents in "Training Data" tab
4. **API Testing**: Use the auto-generated docs at `/docs` endpoint

---

# ⚖️ Full Vision: Custom Legal LLM Platform for Law Firms

> **The complete vision: A secure, private AI assistant that enables law firms to search, retrieve, and summarize legal documents—including internal memos and public case law—using natural language.**

---

## 🧠 Overview

This platform delivers a firm-specific legal research assistant powered by custom LLMs and Retrieval-Augmented Generation (RAG). It ingests:
- Public legal documents (e.g., statutes, court rulings)
- Firm-specific internal case files (e.g., memos, briefs)

Built with **React + FastAPI**, and powered by **LlamaIndex** and **Qdrant**, the system enables lawyers to securely find:
- Relevant past cases
- Internal legal memos
- Source-cited summaries with paragraph-level references

---

## 🚀 Key Features

- 🔍 Natural language search across private + public legal corpora
- 📁 Upload PDFs (e.g., rulings, filings, internal documents)
- 🧠 RAG with **LlamaIndex + Qdrant** for safe, citation-linked retrieval
- 🧾 Document viewer with summary, metadata, and filters
- 🔐 Role-based access control & audit logs
- 🌐 React.js web interface for lawyers, partners, and researchers
- 🐳 Fully Dockerized for cloud or on-premise deployment

---

## 🛠️ Tech Stack

| Layer         | Technology |
|---------------|------------|
| Frontend      | React.js (Vite) |
| Backend       | Python + FastAPI |
| LLM Interface | LlamaIndex (with local or API-based LLMs) |
| Vector DB     | Qdrant |
| PDF Parsing   | unstructured.io, pdfminer.six |
| Storage       | AWS S3 / Local File System |
| Deployment    | Docker / Helm / On-prem or VPC |

---

## 📁 Folder Structure

legal-llm-platform/
├── backend/             
├── frontend/             
├── rag_engine/           
├── vector_store/        
├── docs/                
├── utils/                
├── Dockerfile
└── docker-compose.yml
