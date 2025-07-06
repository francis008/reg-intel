#!/usr/bin/env python3
"""
Malaysian Legal Search API
==========================

FastAPI-based search interface for Malaysian legal documents.

Features:
- Semantic search across Malaysian Principal Acts
- Bilingual support (EN/BM)
- Legal citation and metadata
- RAG-based answer generation
- Privacy-focused (local models only)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Vector search
try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class SearchQuery(BaseModel):
    query: str = Field(..., description="Legal search query")
    language: Optional[str] = Field("all", description="Language filter: EN, BM, or all")
    act_number: Optional[str] = Field(None, description="Specific Act number filter")
    limit: int = Field(5, description="Number of results to return", ge=1, le=20)

class SearchResult(BaseModel):
    chunk_id: str
    act_number: str
    act_title: str
    section_heading: str
    citation: str
    text_content: str
    relevance_score: float
    language: str
    page_number: Optional[int] = None
    section_number: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int
    search_time_ms: float
    language_filter: str

class LegalQuestion(BaseModel):
    question: str = Field(..., description="Legal question to answer")
    language: Optional[str] = Field("EN", description="Response language: EN or BM")
    include_citations: bool = Field(True, description="Include legal citations")

class LegalAnswer(BaseModel):
    question: str
    answer: str
    citations: List[str]
    relevant_acts: List[str]
    confidence: float
    response_language: str

# Malaysian Legal Search Engine
class MalaysianLegalSearchEngine:
    """
    Privacy-focused legal search engine for Malaysian law
    """
    
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 embedding_model: str = "BAAI/bge-base-en-v1.5"):
        
        self.collection_name = "malaysian_legal_acts"
        self.embedding_model_name = embedding_model
        
        # Initialize components
        self.client = None
        self.embedding_model = None
        
        if SEARCH_AVAILABLE:
            try:
                # Connect to Qdrant
                self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
                logger.info(f"✅ Connected to Qdrant at {qdrant_host}:{qdrant_port}")
                
                # Load embedding model
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"✅ Loaded embedding model: {embedding_model}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize search engine: {e}")
        else:
            logger.error("❌ Search dependencies not available")
    
    def is_ready(self) -> bool:
        """Check if search engine is ready"""
        return self.client is not None and self.embedding_model is not None
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        if not self.embedding_model:
            raise HTTPException(status_code=503, detail="Embedding model not available")
        
        try:
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Failed to embed query: {e}")
            raise HTTPException(status_code=500, detail="Failed to process query")
    
    def search_legal_documents(self, search_request: SearchQuery) -> SearchResponse:
        """
        Perform semantic search across Malaysian legal documents
        """
        start_time = datetime.now()
        
        if not self.is_ready():
            raise HTTPException(status_code=503, detail="Search engine not ready")
        
        try:
            # Generate query embedding
            query_vector = self.embed_query(search_request.query)
            
            # Build search filters
            search_filter = None
            if search_request.language != "all" or search_request.act_number:
                from qdrant_client.http import models
                
                filter_conditions = []
                
                if search_request.language != "all":
                    filter_conditions.append(
                        models.FieldCondition(
                            key="language",
                            match=models.MatchValue(value=search_request.language)
                        )
                    )
                
                if search_request.act_number:
                    filter_conditions.append(
                        models.FieldCondition(
                            key="act_number", 
                            match=models.MatchValue(value=search_request.act_number)
                        )
                    )
                
                search_filter = models.Filter(must=filter_conditions)
            
            # Perform vector search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=search_request.limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Process results
            results = []
            for hit in search_results:
                payload = hit.payload
                
                result = SearchResult(
                    chunk_id=payload.get('chunk_id', str(hit.id)),
                    act_number=payload.get('act_number', ''),
                    act_title=payload.get('act_title', ''),
                    section_heading=payload.get('section_heading', ''),
                    citation=payload.get('citation', ''),
                    text_content=payload.get('chunk_text', ''),
                    relevance_score=hit.score,
                    language=payload.get('language', 'EN'),
                    page_number=payload.get('page_number'),
                    section_number=payload.get('section_number')
                )
                results.append(result)
            
            # Calculate search time
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SearchResponse(
                query=search_request.query,
                results=results,
                total_found=len(results),
                search_time_ms=search_time,
                language_filter=search_request.language
            )
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    def answer_legal_question(self, legal_question: LegalQuestion) -> LegalAnswer:
        """
        Generate answers to legal questions using RAG approach
        """
        if not self.is_ready():
            raise HTTPException(status_code=503, detail="Search engine not ready")
        
        try:
            # First, search for relevant legal content
            search_query = SearchQuery(
                query=legal_question.question,
                language=legal_question.language if legal_question.language != "all" else "EN",
                limit=10
            )
            
            search_response = self.search_legal_documents(search_query)
            
            if not search_response.results:
                return LegalAnswer(
                    question=legal_question.question,
                    answer="No relevant Malaysian legal provisions found for this question.",
                    citations=[],
                    relevant_acts=[],
                    confidence=0.0,
                    response_language=legal_question.language
                )
            
            # Extract relevant content and citations
            relevant_content = []
            citations = []
            relevant_acts = set()
            
            for result in search_response.results[:5]:  # Use top 5 results
                relevant_content.append(f"{result.citation}: {result.text_content}")
                citations.append(result.citation)
                relevant_acts.add(f"Act {result.act_number}")
            
            # Generate structured answer
            answer = self._generate_legal_answer(
                legal_question.question,
                relevant_content,
                legal_question.language
            )
            
            # Calculate confidence based on search scores
            avg_score = sum(r.relevance_score for r in search_response.results[:3]) / min(3, len(search_response.results))
            
            return LegalAnswer(
                question=legal_question.question,
                answer=answer,
                citations=citations[:5] if legal_question.include_citations else [],
                relevant_acts=list(relevant_acts),
                confidence=avg_score,
                response_language=legal_question.language
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to answer legal question: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")
    
    def _generate_legal_answer(self, question: str, relevant_content: List[str], language: str) -> str:
        """
        Generate a structured legal answer from relevant content
        
        Note: This is a simplified implementation. In production, you might want to use
        a local LLM like Llama 2, CodeLlama, or a fine-tuned legal model.
        """
        
        # For now, provide a structured response based on the retrieved content
        intro = "Based on Malaysian legal provisions:" if language == "EN" else "Berdasarkan peruntukan undang-undang Malaysia:"
        
        answer_parts = [intro, ""]
        
        # Summarize key points from relevant content
        for i, content in enumerate(relevant_content[:3], 1):
            # Extract key phrases and structure
            citation, text = content.split(": ", 1) if ": " in content else ("", content)
            
            # Simple summarization - in production, use a proper summarization model
            summary = text[:300] + "..." if len(text) > 300 else text
            
            answer_parts.append(f"{i}. {citation}")
            answer_parts.append(f"   {summary}")
            answer_parts.append("")
        
        conclusion = ("Please consult with a qualified Malaysian lawyer for specific legal advice." 
                     if language == "EN" 
                     else "Sila berunding dengan peguam Malaysia yang berkelayakan untuk nasihat undang-undang khusus.")
        
        answer_parts.append(conclusion)
        
        return "\n".join(answer_parts)

# Initialize FastAPI app
app = FastAPI(
    title="Malaysian Legal Search API",
    description="Semantic search and Q&A for Malaysian Principal Acts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine
search_engine = MalaysianLegalSearchEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🇲🇾 Malaysian Legal Search API starting up...")
    if search_engine.is_ready():
        logger.info("✅ Search engine ready")
    else:
        logger.warning("⚠️ Search engine not ready - check Qdrant and embeddings")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Malaysian Legal Search API",
        "version": "1.0.0",
        "status": "ready" if search_engine.is_ready() else "not_ready",
        "endpoints": {
            "search": "/search",
            "ask": "/ask",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if search_engine.is_ready() else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "search_engine_ready": search_engine.is_ready(),
        "components": {
            "qdrant": search_engine.client is not None,
            "embeddings": search_engine.embedding_model is not None
        }
    }

@app.post("/search", response_model=SearchResponse)
async def search_legal_documents(search_request: SearchQuery):
    """
    Search Malaysian legal documents
    
    Performs semantic search across Malaysian Principal Acts and returns
    relevant sections with legal citations and metadata.
    """
    return search_engine.search_legal_documents(search_request)

@app.get("/search")
async def search_legal_documents_get(
    q: str = Query(..., description="Search query"),
    language: str = Query("all", description="Language filter: EN, BM, or all"),
    act_number: Optional[str] = Query(None, description="Specific Act number"),
    limit: int = Query(5, description="Number of results", ge=1, le=20)
):
    """
    Search Malaysian legal documents (GET version for easy testing)
    """
    search_request = SearchQuery(
        query=q,
        language=language,
        act_number=act_number,
        limit=limit
    )
    return search_engine.search_legal_documents(search_request)

@app.post("/ask", response_model=LegalAnswer)
async def ask_legal_question(legal_question: LegalQuestion):
    """
    Ask a legal question and get an answer based on Malaysian law
    
    Uses RAG (Retrieval-Augmented Generation) to provide answers based on
    relevant Malaysian legal provisions with proper citations.
    """
    return search_engine.answer_legal_question(legal_question)

@app.get("/ask")
async def ask_legal_question_get(
    question: str = Query(..., description="Legal question"),
    language: str = Query("EN", description="Response language: EN or BM"),
    include_citations: bool = Query(True, description="Include citations")
):
    """
    Ask a legal question (GET version for easy testing)
    """
    legal_question = LegalQuestion(
        question=question,
        language=language,
        include_citations=include_citations
    )
    return search_engine.answer_legal_question(legal_question)

@app.get("/stats")
async def get_collection_stats():
    """Get statistics about the legal document collection"""
    if not search_engine.is_ready():
        raise HTTPException(status_code=503, detail="Search engine not ready")
    
    try:
        # Get collection info from Qdrant
        collection_info = search_engine.client.get_collection(search_engine.collection_name)
        
        return {
            "collection_name": search_engine.collection_name,
            "total_documents": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance_metric": str(collection_info.config.params.vectors.distance),
            "status": collection_info.status,
            "embedding_model": search_engine.embedding_model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🇲🇾 MALAYSIAN LEGAL SEARCH API")
    print("=" * 40)
    print("Starting FastAPI server...")
    print("📋 Available endpoints:")
    print("   🔍 Search: http://localhost:8000/search?q=employment+contract")
    print("   ❓ Ask: http://localhost:8000/ask?question=What+are+employment+rights")
    print("   📚 Docs: http://localhost:8000/docs")
    print("   ❤️ Health: http://localhost:8000/health")
    print()
    
    uvicorn.run(
        "legal_search_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
