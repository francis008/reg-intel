# src/secure_rag_llamaindex.py - Secure Legal RAG with LlamaIndex

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import hashlib

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext,
    load_index_from_storage,
    Settings,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

class SecureLegalRAGLlamaIndex:
    def __init__(self, firm_id: str):
        self.firm_id = firm_id
        self.firm_db_path = f"./secure_data/{firm_id}"
        self.documents_path = f"{self.firm_db_path}/documents"
        self.index_path = f"{self.firm_db_path}/index"
        self.logs_path = f"{self.firm_db_path}/logs"
        
        self.ensure_secure_directory()
        self.setup_llama_index()
        
    def ensure_secure_directory(self):
        """Create secure directory structure"""
        for path in [self.firm_db_path, self.documents_path, self.index_path, self.logs_path]:
            os.makedirs(path, mode=0o700, exist_ok=True)
    
    def setup_llama_index(self):
        """Configure LlamaIndex with local models for maximum privacy"""
        try:
            # Use local embedding model (no external API calls)
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=f"{self.firm_db_path}/embeddings_cache"
            )
            
            # Configure LlamaIndex settings - NO LLM for now (template-based only)
            Settings.embed_model = embed_model
            Settings.llm = None  # Disable LLM to avoid OpenAI calls
            Settings.node_parser = SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=200,
                separator=" "
            )
            
            self.embed_model = embed_model
            self.log_access("SYSTEM_INITIALIZED", "LlamaIndex setup completed (local-only mode)")
            
        except Exception as e:
            self.log_access("SYSTEM_ERROR", "LlamaIndex setup failed", str(e))
            raise
        
    def process_document_securely(self, document_path: str) -> bool:
        """Process legal document with LlamaIndex"""
        try:
            # Copy document to secure location
            secure_doc_path = os.path.join(self.documents_path, os.path.basename(document_path))
            
            # Read and secure the document
            with open(document_path, 'rb') as src, open(secure_doc_path, 'wb') as dst:
                dst.write(src.read())
            
            # Set secure permissions
            os.chmod(secure_doc_path, 0o600)  # Owner read/write only
            
            # Load documents using LlamaIndex
            documents = SimpleDirectoryReader(
                input_files=[secure_doc_path],
                filename_as_id=True  # Use filename as document ID
            ).load_data()
            
            # Create or update the index
            if self.index_exists():
                # Load existing index and add new documents
                storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
                index = load_index_from_storage(storage_context)
                
                # Insert new documents
                for doc in documents:
                    index.insert(doc)
            else:
                # Create new index
                index = VectorStoreIndex.from_documents(documents)
            
            # Persist the index securely
            index.storage_context.persist(persist_dir=self.index_path)
            
            # Set secure permissions on index files
            for root, dirs, files in os.walk(self.index_path):
                for file in files:
                    os.chmod(os.path.join(root, file), 0o600)
                for dir in dirs:
                    os.chmod(os.path.join(root, dir), 0o700)
            
            # Log the processing
            self.log_access("DOCUMENT_PROCESSED", document_path)
            
            return True
            
        except Exception as e:
            self.log_access("PROCESSING_ERROR", document_path, str(e))
            raise
    
    def index_exists(self) -> bool:
        """Check if index already exists"""
        return os.path.exists(os.path.join(self.index_path, "index_store.json"))
    
    def secure_query(self, query: str, similarity_top_k: int = 5) -> Dict:
        """Query documents with full privacy protection"""
        try:
            if not self.index_exists():
                return {
                    "response": "No documents have been processed yet. Please upload some legal documents first.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Load the index
            storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
            index = load_index_from_storage(storage_context)
            
            # Create query engine with custom settings
            query_engine = index.as_query_engine(
                similarity_top_k=similarity_top_k,
                response_mode="tree_summarize",  # Better for legal documents
                verbose=False
            )
            
            # Execute query (everything stays local)
            response = query_engine.query(query)
            
            # Extract source information
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    sources.append({
                        "document": node.metadata.get('file_name', 'Unknown'),
                        "content_preview": node.text[:200] + "...",
                        "similarity_score": getattr(node, 'score', 0.0)
                    })
            
            # Log the query for audit
            self.log_access("QUERY_EXECUTED", query)
            
            return {
                "response": str(response),
                "sources": sources,
                "confidence": self.calculate_confidence(response, sources),
                "query": query,
                "firm_id": self.firm_id
            }
            
        except Exception as e:
            self.log_access("QUERY_ERROR", query, str(e))
            return {
                "response": f"Error processing query: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def calculate_confidence(self, response, sources) -> float:
        """Calculate confidence score based on response and sources"""
        if not sources:
            return 0.3
        
        # Simple confidence calculation based on number of sources and similarity
        if sources:
            avg_similarity = sum(s.get('similarity_score', 0) for s in sources) / len(sources)
            source_factor = min(len(sources) / 3, 1.0)  # Max confidence with 3+ sources
            return min(avg_similarity * source_factor, 0.95)
        return 0.3
    
    def generate_legal_document_simple(self, prompt: str, doc_type: str) -> Dict:
        """Generate legal document using retrieved context (without local LLM for now)"""
        try:
            # First, search for relevant context
            search_query = f"{doc_type} {prompt}"
            context_result = self.secure_query(search_query, similarity_top_k=3)
            
            # For now, use enhanced templates with context
            template_with_context = self.create_enhanced_template(
                prompt, doc_type, context_result['response']
            )
            
            # Log generation
            self.log_access("DOCUMENT_GENERATED", f"{doc_type}: {prompt}")
            
            return {
                "generated_text": template_with_context,
                "doc_type": doc_type,
                "sources_used": len(context_result['sources']),
                "confidence": context_result['confidence'],
                "firm_style": True,
                "context_used": True
            }
            
        except Exception as e:
            self.log_access("GENERATION_ERROR", f"{doc_type}: {prompt}", str(e))
            return {
                "generated_text": f"Error generating document: {str(e)}",
                "doc_type": doc_type,
                "sources_used": 0,
                "confidence": 0.0,
                "firm_style": False,
                "context_used": False
            }
    
    def create_enhanced_template(self, prompt: str, doc_type: str, context: str) -> str:
        """Create enhanced template using retrieved context"""
        
        # Extract relevant clauses from context
        relevant_clauses = self.extract_relevant_clauses(context, doc_type)
        
        templates = {
            "contract": f"""AGREEMENT

This Agreement is entered into on [DATE] between [PARTY 1] and [PARTY 2].

WHEREAS, {prompt};

Based on firm precedents, the following provisions are recommended:

{relevant_clauses}

NOW THEREFORE, the parties agree as follows:

1. SCOPE OF WORK: [To be defined based on: {prompt}]

2. PAYMENT TERMS: [Standard firm terms apply]

3. INTELLECTUAL PROPERTY: [Per firm standard clauses]

4. TERMINATION: [Include standard termination provisions]

5. GOVERNING LAW: [Specify jurisdiction]

IN WITNESS WHEREOF, the parties have executed this Agreement.

[SIGNATURE BLOCKS]

---
Generated using firm's document precedents and AI assistance.""",

            "legal_memo": f"""LEGAL MEMORANDUM

TO: [CLIENT]
FROM: [ATTORNEY]
DATE: {datetime.now().strftime('%B %d, %Y')}
RE: {prompt}

EXECUTIVE SUMMARY

Based on our firm's experience and relevant precedents:

{relevant_clauses}

ANALYSIS

[Detailed legal analysis based on firm precedents]

CONCLUSION

Based on the above analysis and our firm's prior experience with similar matters, we recommend [specific action].

---
This memo incorporates insights from {len(relevant_clauses.split('\\n'))} firm precedents.""",

            "court_filing": f"""IN THE [COURT NAME]

[CASE CAPTION]
Case No. [NUMBER]

MOTION FOR [RELIEF SOUGHT]

TO THE HONORABLE COURT:

Comes now [PARTY], by and through undersigned counsel, and respectfully submits this Motion regarding {prompt}.

BACKGROUND

[Factual background]

LEGAL STANDARD

Based on firm precedents:
{relevant_clauses}

ARGUMENT

[Legal argument incorporating firm experience]

CONCLUSION

For the foregoing reasons, [PARTY] respectfully requests that this Court grant the requested relief.

Respectfully submitted,

[ATTORNEY SIGNATURE BLOCK]

---
Drafted using firm precedents and legal AI assistance."""
        }
        
        return templates.get(doc_type, f"Legal document regarding: {prompt}\\n\\nContext from firm documents:\\n{relevant_clauses}")
    
    def extract_relevant_clauses(self, context: str, doc_type: str) -> str:
        """Extract relevant clauses from context for the specific document type"""
        if not context or context == "No documents have been processed yet. Please upload some legal documents first.":
            return "[No relevant firm precedents found. Please upload more documents to improve AI suggestions.]"
        
        # Simple extraction - in a full implementation, this would be more sophisticated
        lines = context.split('.')
        relevant_lines = [line.strip() for line in lines if len(line.strip()) > 20][:3]
        
        if relevant_lines:
            return "\\n".join([f"• {line}" for line in relevant_lines])
        else:
            return "[Firm precedents available but no specific clauses extracted for this document type.]"
    
    def get_firm_statistics(self) -> Dict:
        """Get statistics about firm's document database"""
        try:
            stats = {
                "firm_id": self.firm_id,
                "total_documents": 0,
                "index_exists": self.index_exists(),
                "document_types": {},
                "storage_size_mb": 0
            }
            
            # Count documents
            if os.path.exists(self.documents_path):
                documents = [f for f in os.listdir(self.documents_path) if os.path.isfile(os.path.join(self.documents_path, f))]
                stats["total_documents"] = len(documents)
                
                # Analyze document types
                for doc in documents:
                    ext = os.path.splitext(doc)[1].lower()
                    if ext:  # Only count files with extensions
                        stats["document_types"][ext] = stats["document_types"].get(ext, 0) + 1
            
            # Calculate storage size
            total_size = 0
            if os.path.exists(self.firm_db_path):
                for root, dirs, files in os.walk(self.firm_db_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            total_size += os.path.getsize(file_path)
            
            stats["storage_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            self.log_access("STATS_ERROR", "get_firm_statistics", str(e))
            return {"error": str(e)}
    
    def log_access(self, action: str, target: str, error: str = None):
        """Comprehensive audit logging for legal compliance"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "firm_id": self.firm_id,
            "action": action,
            "target": target,
            "error": error,
            "ip_address": "127.0.0.1",  # In production, capture real IP
            "user_agent": "LegalLLM-Platform/1.0"
        }
        
        try:
            log_file = os.path.join(self.logs_path, "access.log")
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\\n")
            
            # Set secure permissions on log file
            os.chmod(log_file, 0o600)
        except Exception as e:
            # Fallback logging to console if file logging fails
            print(f"Log error: {e}, Entry: {log_entry}")
