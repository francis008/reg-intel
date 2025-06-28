# ⚖️ Custom Legal LLM Platform for Law Firms

> **A secure, private AI assistant that enables law firms to search, retrieve, and summarize legal documents—including internal memos and public case law—using natural language.**

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
├── backend/ # FastAPI server + RAG pipeline
├── frontend/ # React frontend (Vite + Tailwind)
├── rag/ # LlamaIndex setup + chunking logic
├── qdrant/ # Vector DB schema + Docker config
├── parser/ # PDF parsers, metadata enrichers
├── data/ # Sample legal documents (test only)
├── docker-compose.yml
└── README.md
