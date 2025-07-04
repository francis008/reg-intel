# src/api.py - Legal LLM API Platform

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
from datetime import datetime
import uvicorn
import tempfile
import shutil

from text_extractor import LegalTextExtractor
from scraper import LegalDocumentProcessor
from secure_rag_llamaindex import SecureLegalRAGLlamaIndex

# Initialize FastAPI app
app = FastAPI(
    title="Legal LLM Platform",
    description="Custom Legal AI for Law Firms",
    version="1.0.0"
)

# Add CORS middleware to allow connections from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our processors
text_extractor = LegalTextExtractor()
doc_processor = LegalDocumentProcessor("../docs")

# Initialize RAG system for default firm (in production, this would be per-firm)
default_rag = SecureLegalRAGLlamaIndex("default_firm")

# Pydantic models for API requests/responses
class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    doc_type: str
    training_examples: int
    rag_indexed: bool = False

class GenerateTextRequest(BaseModel):
    prompt: str
    doc_type: str = "contract"
    max_length: int = 1000
    use_rag: bool = True

class GenerateTextResponse(BaseModel):
    generated_text: str
    doc_type: str
    confidence: float
    sources_used: int = 0
    context_used: bool = False

class RAGQueryRequest(BaseModel):
    query: str
    similarity_top_k: int = 5

class RAGQueryResponse(BaseModel):
    response: str
    sources: List[Dict]
    confidence: float
    query: str
    firm_id: str

class TrainingDataResponse(BaseModel):
    total_documents: int
    document_types: Dict[str, int]
    total_training_examples: int
    rag_statistics: Dict[str, Any] = None

# Global storage for training data (in production, use a database)
training_data_store = []

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Legal LLM Platform",
        "version": "1.0.0",
        "features": {
            "rag_enabled": True,
            "local_processing": True,
            "privacy_focused": True
        },
        "endpoints": {
            "upload": "/upload-document (Enhanced with RAG indexing)",
            "generate": "/generate-text (Now uses RAG when available)",
            "rag_query": "/rag-query (New: Direct RAG queries)",
            "training-data": "/training-data (Enhanced with RAG stats)",
            "health": "/health"
        }
    }

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a legal document for processing and RAG indexing
    
    This endpoint:
    1. Saves the uploaded file securely
    2. Extracts text content
    3. Classifies the document type
    4. Creates training examples
    5. Indexes document in RAG system for retrieval
    6. Stores everything for future use
    """
    try:
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Save uploaded file to docs directory (legacy)
        file_location = f"../docs/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(content)
        
        # Extract text and create training examples
        extracted_text = text_extractor.extract_text(temp_file_path)
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from document")
        
        # Classify document
        metadata = doc_processor.extract_metadata(temp_file_path)
        
        # Create training examples
        training_examples = text_extractor.create_training_examples(temp_file_path, extracted_text)
        
        # RAG INTEGRATION: Process document with RAG system
        rag_indexed = False
        try:
            rag_indexed = default_rag.process_document_securely(temp_file_path)
        except Exception as rag_error:
            print(f"RAG indexing failed (continuing without): {rag_error}")
        
        # Store training data
        document_data = {
            "document_id": file.filename,
            "metadata": metadata,
            "text": extracted_text[:5000],  # Store first 5000 chars for preview
            "training_examples": training_examples,
            "upload_date": datetime.now().isoformat(),
            "rag_indexed": rag_indexed
        }
        
        training_data_store.append(document_data)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return DocumentUploadResponse(
            message="Document processed and indexed successfully" if rag_indexed else "Document processed (RAG indexing failed)",
            document_id=file.filename,
            doc_type=metadata["doc_type"],
            training_examples=len(training_examples),
            rag_indexed=rag_indexed
        )
        
    except Exception as e:
        # Clean up on error
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/rag-query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Query the RAG system directly for legal information
    
    This endpoint allows direct querying of the firm's legal document database
    using semantic search and retrieval-augmented generation.
    """
    try:
        result = default_rag.secure_query(
            query=request.query,
            similarity_top_k=request.similarity_top_k
        )
        
        return RAGQueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying RAG system: {str(e)}")

@app.post("/generate-text", response_model=GenerateTextResponse)
async def generate_text(request: GenerateTextRequest):
    """
    Generate legal text using RAG-enhanced system
    
    Now uses the firm's indexed documents to provide contextually relevant
    legal document generation based on precedents and similar cases.
    """
    try:
        if request.use_rag:
            # Use RAG-enhanced generation
            result = default_rag.generate_legal_document_simple(
                prompt=request.prompt,
                doc_type=request.doc_type
            )
            
            generated_text = result["generated_text"]
            
            # Truncate to max_length if specified
            if len(generated_text) > request.max_length:
                generated_text = generated_text[:request.max_length] + "..."
            
            return GenerateTextResponse(
                generated_text=generated_text,
                doc_type=request.doc_type,
                confidence=result["confidence"],
                sources_used=result["sources_used"],
                context_used=result["context_used"]
            )
        else:
            # Fall back to template-based generation
            template_responses = {
                "contract": f"AGREEMENT\n\nThis {request.doc_type} is entered into on [DATE] between [PARTY 1] and [PARTY 2].\n\nWHEREAS, {request.prompt}...\n\nNOW THEREFORE, the parties agree as follows:\n\n1. [PROVISION 1]\n2. [PROVISION 2]\n\nIN WITNESS WHEREOF, the parties have executed this agreement.",
                
                "legal_memo": f"LEGAL MEMORANDUM\n\nTO: [CLIENT]\nFROM: [ATTORNEY]\nDATE: {datetime.now().strftime('%B %d, %Y')}\nRE: {request.prompt}\n\nEXECUTIVE SUMMARY\n\nBased on our analysis of the applicable law and relevant precedents, we advise that...\n\nANALYSIS\n\n[Detailed legal analysis would go here]\n\nCONCLUSION\n\n[Legal conclusion and recommendations]",
                
                "court_filing": f"IN THE [COURT NAME]\n\n[CASE CAPTION]\n\nCase No. [NUMBER]\n\nMOTION FOR [RELIEF SOUGHT]\n\nTO THE HONORABLE COURT:\n\nComes now [PARTY], by and through undersigned counsel, and respectfully submits this Motion regarding {request.prompt}.\n\nBACKGROUND\n\n[Factual background]\n\nLEGAL STANDARD\n\n[Applicable legal standard]\n\nARGUMENT\n\n[Legal argument]\n\nCONCLUSION\n\nFor the foregoing reasons, [PARTY] respectfully requests that this Court grant the requested relief.\n\nRespectfully submitted,\n\n[ATTORNEY SIGNATURE BLOCK]"
            }
            
            generated_text = template_responses.get(
                request.doc_type, 
                f"Legal document regarding: {request.prompt}\n\n[This would be generated by your trained model]"
            )
            
            # Truncate to max_length if specified
            if len(generated_text) > request.max_length:
                generated_text = generated_text[:request.max_length] + "..."
            
            return GenerateTextResponse(
                generated_text=generated_text,
                doc_type=request.doc_type,
                confidence=0.85,  # Placeholder confidence score
                sources_used=0,
                context_used=False
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@app.get("/training-data", response_model=TrainingDataResponse)
async def get_training_data():
    """
    Get summary of training data collected and RAG system statistics
    """
    try:
        total_docs = len(training_data_store)
        total_examples = sum(len(doc["training_examples"]) for doc in training_data_store)
        
        # Count document types
        doc_types = {}
        for doc in training_data_store:
            doc_type = doc["metadata"]["doc_type"]
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Get RAG statistics
        rag_stats = None
        try:
            rag_stats = default_rag.get_firm_statistics()
        except Exception as e:
            print(f"Error getting RAG stats: {e}")
        
        return TrainingDataResponse(
            total_documents=total_docs,
            document_types=doc_types,
            total_training_examples=total_examples,
            rag_statistics=rag_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving training data: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint with RAG system status"""
    try:
        rag_healthy = False
        rag_info = {}
        
        try:
            rag_stats = default_rag.get_firm_statistics()
            rag_healthy = True
            rag_info = {
                "rag_enabled": True,
                "indexed_documents": rag_stats.get("total_documents", 0),
                "index_exists": rag_stats.get("index_exists", False),
                "storage_size_mb": rag_stats.get("storage_size_mb", 0)
            }
        except Exception as e:
            rag_info = {
                "rag_enabled": False,
                "error": str(e)
            }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "documents_processed": len(training_data_store),
            "rag_system": rag_info
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/documents")
async def list_documents():
    """List all processed documents with RAG indexing status"""
    return {
        "documents": [
            {
                "id": doc["document_id"],
                "type": doc["metadata"]["doc_type"],
                "size_kb": doc["metadata"]["size_kb"],
                "training_examples": len(doc["training_examples"]),
                "upload_date": doc["upload_date"],
                "rag_indexed": doc.get("rag_indexed", False)
            }
            for doc in training_data_store
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Legal LLM Platform API with RAG...")
    print("📚 Upload legal documents to /upload-document (now with RAG indexing)")
    print("✍️  Generate text at /generate-text (now uses RAG when available)")
    print("🔍 Query RAG system directly at /rag-query")
    print("📊 View training data at /training-data (includes RAG stats)")
    print("� API docs at http://localhost:8000/docs")
    print("🔒 Privacy-focused: All data stays local")
    print()
    
    # Remove reload=True to fix the warning
    # For development with auto-reload, use: uvicorn api:app --reload
    uvicorn.run(app, host="127.0.0.1", port=8000)
