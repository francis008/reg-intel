# 🇲🇾 MyReg-Intel: AI-Powered Regulatory Intelligence for Malaysia

MyReg-Intel is a prototype AI-powered platform that extracts, summarizes, and indexes legal and regulatory documents from Malaysian government sources. It provides businesses and developers with easy access to structured regulatory intelligence, using LLMs and vector search.

## 🚀 Features

- ✅ Scrape official PDFs from Malaysian regulatory websites (e.g., Bank Negara)
- ✅ Extract and clean full legal text from documents
- ✅ Summarize content using OpenAI GPT via LangChain
- ✅ Tag and assess risk level of each regulation
- ✅ Store documents with embeddings for semantic search
- ✅ Searchable via Streamlit UI or API

## 🧠 Use Case

Startups, law firms, and MNCs expanding into Southeast Asia often struggle with fragmented, inaccessible regulations. MyReg-Intel offers an early step toward a unified, machine-readable regulatory layer.

## 🛠️ Stack

| Component         | Tech Used                      |
|------------------|--------------------------------|
| Language Model    | OpenAI GPT-4 (via LangChain)   |
| Vector Search     | Chroma / FAISS                 |
| Text Extraction   | PyMuPDF                        |
| Web Scraping      | Requests / BeautifulSoup       |
| UI / API          | Streamlit / FastAPI            |
| Language          | Python                         |
