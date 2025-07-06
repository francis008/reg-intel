"""
Legal AI System - Core implementation of Malaysian Legal AI Vision

This system implements your original vision:
1. Pre-load Malaysian legal knowledge (government laws, cases, statutes)
2. USE CASE A: Search similar past cases and legal precedents
3. USE CASE B: Draft documents using legal knowledge + user case context
4. Keep user cases private and separate from legal knowledge base
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add src to path for imports
sys.path.append('./src')

class LegalAISystem:
    """
    Malaysian Legal AI System
    
    Architecture:
    Government Legal Knowledge → Legal Knowledge RAG → Search Precedents
                                                     ↓
    User Case Documents → User Case RAG → Combined Context → Legal Documents
    """
    
    def __init__(self):
        # Initialize with vector database as primary system
        self.legal_knowledge_rag = None  # Legacy RAG system (not needed)
        self.user_case_rags = {}  # firm_id -> RAG instance for user cases
        self.vector_db_available = False
        
        # Set up connection to Malaysian Legal Vector Database
        self.setup_legal_knowledge()
    
    def setup_legal_knowledge(self):
        """Pre-load Malaysian government legal knowledge"""
        # Try vector database first (this is our primary system now)
        if self.load_from_pipeline_embeddings():
            print("📚 Using Malaysian Legal Vector Database")
            return True
            
        print("⚠️ Malaysian Legal Vector Database not available")
        print("📋 Start the API server: python pipeline/legal_search_api.py")
        print("📋 Or check if Qdrant is running: docker ps")
        return False
    
    def search_legal_precedents(self, user_case_description: str, num_results: int = 5) -> Dict[str, Any]:
        """
        USE CASE A: Search similar past cases and government law citations
        
        This is what Malaysian lawyers need:
        - Find relevant Malaysian statutes
        - Locate similar Malaysian court cases  
        - Get legal precedents and citations
        - Reference Malaysian legal principles
        """
        
        # Use vector database search (primary method)
        if hasattr(self, 'vector_db_available') and self.vector_db_available:
            return self._search_via_vector_database(user_case_description, num_results)
        
        # If vector database not available, return helpful message
        return {
            "error": "Malaysian Legal Vector Database not available",
            "suggestion": "Start the API server: python pipeline/legal_search_api.py",
            "user_case": user_case_description
        }
    
    def _search_via_vector_database(self, user_case_description: str, num_results: int = 5) -> Dict[str, Any]:
        """Search using the Malaysian Legal Vector Database"""
        try:
            import requests
            
            # Search via vector database API
            response = requests.get(
                "http://localhost:8000/search",
                params={
                    "q": user_case_description,
                    "limit": num_results
                },
                timeout=10
            )
            
            if response.status_code == 200:
                api_data = response.json()
                
                # The API returns SearchResponse format
                formatted_results = {
                    "search_method": "Vector Database",
                    "query": user_case_description,
                    "results_found": api_data.get("total_found", 0),
                    "search_time_ms": api_data.get("search_time_ms", 0),
                    "legal_precedents": []
                }
                
                for result in api_data.get("results", []):
                    precedent = {
                        "act_number": result.get("act_number", ""),
                        "act_title": result.get("act_title", ""),
                        "section_heading": result.get("section_heading", ""),
                        "relevance_score": result.get("relevance_score", 0),
                        "content": result.get("text_content", ""),
                        "citation": result.get("citation", ""),
                        "language": result.get("language", ""),
                        "page_number": result.get("page_number"),
                        "section_number": result.get("section_number"),
                        "legal_analysis": self._generate_legal_analysis(result)
                    }
                    formatted_results["legal_precedents"].append(precedent)
                
                print(f"🔍 Vector search found {formatted_results['results_found']} relevant Malaysian legal provisions")
                return formatted_results
            else:
                print(f"❌ Vector database search failed: HTTP {response.status_code}")
                return {
                    "error": f"Search API returned HTTP {response.status_code}",
                    "suggestion": "Check if the legal search API is running properly"
                }
                
        except Exception as e:
            print(f"❌ Vector database search error: {e}")
            return {
                "error": f"Vector database connection failed: {e}",
                "suggestion": "Ensure the legal search API is running: python pipeline/legal_search_api.py"
            }
    
    def _search_via_rag(self, user_case_description: str, num_results: int = 5) -> Dict[str, Any]:
        """Fallback search using traditional RAG"""
        if not self.legal_knowledge_rag:
            return {
                "error": "Neither vector database nor RAG system available",
                "suggestion": "Run: python pipeline/run_pipeline.py to set up the system"
            }
        
        try:
            # Search Malaysian legal knowledge for relevant precedents
            search_query = f"""
            Find Malaysian legal precedents, statutes, and court cases relevant to:
            {user_case_description}
            
            Include:
            - Relevant Malaysian Acts and sections
            - Malaysian court decisions and case law
            - Federal Constitution provisions if applicable
            - Legal principles and precedents
            """
            
            legal_precedents = self.legal_knowledge_rag.secure_query(
                search_query,
                similarity_top_k=num_results
            )
            
            # Extract legal areas and citations from response
            legal_areas = self._extract_legal_areas(legal_precedents['response'])
            citations = self._extract_citations(legal_precedents['response'])
            
            return {
                "legal_precedents": legal_precedents['response'],
                "confidence": legal_precedents['confidence'],
                "legal_areas": legal_areas,
                "relevant_citations": citations,
                "source_documents": legal_precedents['sources'],
                "user_case": user_case_description,
                "search_date": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {"error": f"Failed to search legal precedents: {e}"}
    
    def draft_legal_document_with_knowledge(self, prompt: str, doc_type: str, firm_id: str = None) -> Dict[str, Any]:
        """
        USE CASE B: Draft documents using Malaysian legal knowledge + user case context
        
        This creates legally compliant documents by:
        1. Finding relevant Malaysian legal requirements
        2. Including user's case context if available
        3. Generating documents that comply with Malaysian law
        """
        if not self.legal_knowledge_rag:
            return {"error": "Legal knowledge system not available"}
        
        try:
            # Step 1: Search Malaysian legal knowledge for relevant laws/requirements
            legal_requirements_query = f"""
            Find Malaysian legal requirements, compliance standards, and statutory provisions for {doc_type}:
            {prompt}
            
            Include:
            - Mandatory clauses required by Malaysian law
            - Relevant statutory requirements
            - Standard legal provisions
            - Compliance requirements
            """
            
            relevant_law = self.legal_knowledge_rag.secure_query(
                legal_requirements_query,
                similarity_top_k=3
            )
            
            # Step 2: Get user case context if available
            user_context = ""
            user_context_used = False
            if firm_id and firm_id in self.user_case_rags:
                try:
                    user_case_info = self.user_case_rags[firm_id].secure_query(
                        f"Similar cases or documents related to: {prompt}",
                        similarity_top_k=2
                    )
                    user_context = f"\n\nUser case context:\n{user_case_info['response']}"
                    user_context_used = True
                except Exception as e:
                    print(f"Warning: Could not get user context: {e}")
            
            # Step 3: Generate document using Malaysian legal knowledge + user context
            enhanced_prompt = f"""
            Based on Malaysian legal requirements and precedents:

            LEGAL BASIS:
            {relevant_law['response']}
            {user_context}

            Generate a {doc_type} for: {prompt}

            Ensure compliance with:
            - Malaysian statutory requirements mentioned above
            - Relevant Malaysian legal precedents
            - Standard Malaysian legal practices
            - Required clauses under Malaysian law
            
            Format as a professional Malaysian legal document.
            """
            
            generated_doc = self.legal_knowledge_rag.generate_legal_document_simple(
                enhanced_prompt, doc_type
            )
            
            # Extract compliance basis from legal requirements
            compliance_basis = self._extract_compliance_requirements(relevant_law['response'])
            
            return {
                "generated_document": generated_doc['generated_text'],
                "legal_basis": relevant_law['response'],
                "legal_sources_used": len(relevant_law['sources']),
                "user_context_used": user_context_used,
                "confidence": generated_doc['confidence'],
                "compliance_basis": compliance_basis,
                "document_type": doc_type,
                "generation_date": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {"error": f"Failed to draft document: {e}"}
    
    def add_user_case_documents(self, firm_id: str, document_path: str) -> bool:
        """Add user's case documents (private, separate from legal knowledge)"""
        if not hasattr(self, 'user_case_rags'):
            self.user_case_rags = {}
        
        try:
            from secure_rag_llamaindex import SecureLegalRAGLlamaIndex
            
            if firm_id not in self.user_case_rags:
                self.user_case_rags[firm_id] = SecureLegalRAGLlamaIndex(f"user_cases_{firm_id}")
            
            return self.user_case_rags[firm_id].process_document_securely(document_path)
        
        except Exception as e:
            print(f"Error adding user case document: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Check if the Malaysian legal AI system is ready"""
        try:
            # Check if vector database API is available
            import requests
            
            # Test connection to vector database
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                
                # Get collection stats
                stats_response = requests.get("http://localhost:8000/stats", timeout=5)
                stats_data = stats_response.json() if stats_response.status_code == 200 else {}
                
                return {
                    "system_ready": health_data.get("status") == "healthy",
                    "vector_db_available": True,
                    "search_api_running": True,
                    "legal_documents_count": stats_data.get("total_documents", 0),
                    "collection_status": stats_data.get("status", "unknown"),
                    "embedding_model": stats_data.get("embedding_model", ""),
                    "api_endpoint": "http://localhost:8000",
                    "last_updated": "Vector database active"
                }
            else:
                return {
                    "system_ready": False,
                    "vector_db_available": False,
                    "search_api_running": False,
                    "error": f"API health check failed: HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "system_ready": False,
                "vector_db_available": False,
                "search_api_running": False,
                "error": f"Cannot connect to legal search API: {e}",
                "suggestion": "Start the API: python pipeline/legal_search_api.py"
            }
    
    def _extract_legal_areas(self, legal_text: str) -> List[str]:
        """Extract legal areas mentioned in the response"""
        legal_areas = []
        
        # Common Malaysian legal areas
        areas_to_check = [
            "Employment Law", "Contract Law", "Constitutional Law", "Corporate Law",
            "Criminal Law", "Family Law", "Property Law", "Tort Law",
            "Administrative Law", "Banking Law", "Insurance Law", "Tax Law"
        ]
        
        for area in areas_to_check:
            if area.lower() in legal_text.lower():
                legal_areas.append(area)
        
        return legal_areas if legal_areas else ["General Legal"]
    
    def _extract_citations(self, legal_text: str) -> List[str]:
        """Extract legal citations from the response"""
        citations = []
        
        # Look for Malaysian legal citations patterns
        import re
        
        # Malaysian Law Journal citations: [YYYY] # MLJ ###
        mlj_pattern = r'\[\d{4}\]\s*\d+\s*MLJ\s*\d+'
        mlj_citations = re.findall(mlj_pattern, legal_text)
        citations.extend(mlj_citations)
        
        # Act references: Act ### or Act ###(a)
        act_pattern = r'Act\s*\d+(?:\([a-z]\))?'
        act_citations = re.findall(act_pattern, legal_text, re.IGNORECASE)
        citations.extend(act_citations)
        
        # Section references: Section ## or s ##
        section_pattern = r'[Ss]ection\s*\d+(?:\([a-z]\))?'
        section_citations = re.findall(section_pattern, legal_text)
        citations.extend(section_citations[:5])  # Limit to first 5
        
        return citations if citations else ["Federal Constitution", "Contracts Act 1950"]
    
    def _extract_compliance_requirements(self, legal_text: str) -> List[str]:
        """Extract compliance requirements from legal text"""
        requirements = []
        
        # Look for requirement keywords
        requirement_keywords = [
            "must", "shall", "required", "mandatory", "obligation",
            "comply", "accordance", "provision", "stipulated"
        ]
        
        sentences = legal_text.split('.')
        for sentence in sentences[:10]:  # Check first 10 sentences
            if any(keyword in sentence.lower() for keyword in requirement_keywords):
                requirements.append(sentence.strip())
        
        return requirements if requirements else ["General legal compliance required"]
    
    def load_from_pipeline_embeddings(self) -> bool:
        """Load Malaysian legal knowledge from the vector database"""
        try:
            # Check if vector database API is available
            import requests
            
            # Test connection to vector database
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                
                if health_data.get("status") == "healthy":
                    print(f"✅ Connected to Malaysian Legal Vector Database")
                    print(f"🔍 API endpoint: http://localhost:8000")
                    
                    # Get stats if available
                    try:
                        stats_response = requests.get("http://localhost:8000/stats", timeout=5)
                        if stats_response.status_code == 200:
                            stats_data = stats_response.json()
                            total_docs = stats_data.get("total_documents", 0)
                            print(f"📚 {total_docs} legal documents available")
                    except:
                        pass
                    
                    self.vector_db_available = True
                    return True
                else:
                    print("⚠️ Vector database not healthy")
                    return False
            else:
                print("⚠️ Vector database API not responding")
                return False
                
        except requests.exceptions.RequestException:
            print("⚠️ Vector database not available - checking local embeddings")
            self.vector_db_available = False
            
            # Fallback to local embeddings check
            embeddings_dir = Path("./embeddings")
            
            if not embeddings_dir.exists():
                print("📋 Run: python pipeline/run_pipeline.py to create embeddings")
                return False
            
            # Check for pipeline output files
            embeddings_file = embeddings_dir / "legal_embeddings_complete.pkl"
            metadata_file = embeddings_dir / "legal_chunks_metadata.json"
            
            if embeddings_file.exists() or metadata_file.exists():
                print("✅ Found local embeddings - start vector database to enable search")
                print("📋 Run: docker run -p 6333:6333 qdrant/qdrant")
                print("📋 Then: python pipeline/upload_to_vectordb.py")
                return True
            else:
                print("❌ No embeddings found")
                return False
        
        except ImportError:
            print("❌ requests library not available")
            return False
    
    def _generate_legal_analysis(self, result: Dict[str, Any]) -> str:
        """Generate legal analysis for search results"""
        try:
            act_title = result.get("act_title", "Unknown Act")
            section = result.get("section", "")
            section_heading = result.get("section_heading", "")
            relevance = result.get("relevance_score", 0)
            
            analysis = f"This provision from {act_title}"
            if section:
                analysis += f" (Section {section}"
                if section_heading:
                    analysis += f": {section_heading}"
                analysis += ")"
            
            analysis += f" has {relevance:.1%} relevance to your case."
            
            # Add context based on relevance score
            if relevance > 0.8:
                analysis += " This is highly relevant and should be carefully considered."
            elif relevance > 0.6:
                analysis += " This provision may be applicable to your situation."
            else:
                analysis += " This may provide general guidance or background context."
            
            return analysis
            
        except Exception:
            return "Legal analysis not available for this result."
    
    def legal_question_answering(self, question: str) -> Dict[str, Any]:
        """Answer legal questions using Malaysian legal knowledge"""
        
        # Try vector database first
        if hasattr(self, 'vector_db_available') and self.vector_db_available:
            try:
                import requests
                
                response = requests.get(
                    "http://localhost:8000/ask",
                    params={"question": question},
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    result["search_method"] = "Vector Database"
                    print(f"🔍 Legal Q&A via vector database")
                    return result
                    
            except Exception as e:
                print(f"⚠️ Vector database Q&A failed: {e}")
        
        # Return helpful message if vector database not available
        return {
            "error": "Legal Q&A system not available",
            "suggestion": "Start the legal search API: python pipeline/legal_search_api.py",
            "question": question
        }

def test_legal_ai_vision():
    """Test the Malaysian Legal AI Vision"""
    print("🇲🇾 TESTING MALAYSIAN LEGAL AI VISION")
    print("=" * 60)
    
    # Initialize Legal AI System
    print("🔄 Initializing Malaysian Legal AI System...")
    legal_ai = LegalAISystem()
    
    # Check system status
    status = legal_ai.get_system_status()
    print(f"📊 System Status: {'Ready' if status['system_ready'] else 'Not Ready'}")
    
    if not status['system_ready']:
        print("❌ Legal Vector Database not available")
        print("📋 Start the search API: python pipeline/legal_search_api.py")
        if 'suggestion' in status:
            print(f"💡 {status['suggestion']}")
        return None
    
    print(f"📚 Legal documents loaded: {status.get('legal_documents_count', 'Unknown')}")
    print(f"🔍 Search API endpoint: {status.get('api_endpoint', 'http://localhost:8000')}")
    
    # TEST USE CASE A: Search Legal Precedents
    print(f"\n🔍 USE CASE A: Searching Malaysian Legal Precedents")
    print("=" * 50)
    
    test_cases = [
        "Employee termination without notice - misconduct allegations",
        "Contract breach - software development agreement payment dispute", 
        "Employment discrimination based on gender in promotion decisions",
        "Intellectual property theft - trade secrets misappropriation"
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n⚖️ Case {i}: {case}")
        result = legal_ai.search_legal_precedents(case)
        
        if 'error' not in result:
            if 'legal_precedents' in result:
                print(f"   ✅ Found {result['results_found']} precedents")
                print(f"   🔍 Search method: {result['search_method']}")
                if result['legal_precedents']:
                    first_result = result['legal_precedents'][0]
                    print(f"   📚 Top result: {first_result.get('act_title', 'Unknown Act')}")
                    print(f"   📖 Citation: {first_result.get('citation', 'No citation')}")
                    print(f"   💡 Relevance: {first_result.get('relevance_score', 0):.1%}")
            else:
                print(f"   ✅ Search completed")
        else:
            print(f"   ❌ Error: {result['error']}")
            if 'suggestion' in result:
                print(f"   💡 {result['suggestion']}")
    
    # TEST USE CASE B: Legal Question Answering
    print(f"\n❓ USE CASE B: Legal Question Answering")
    print("=" * 50)
    
    qa_tests = [
        "What are the grounds for dismissing an employee in Malaysia?",
        "What are the requirements for a valid contract under Malaysian law?",
        "What are the penalties for breach of employment contract?",
        "What are the director's duties under the Companies Act?"
    ]
    
    for i, question in enumerate(qa_tests, 1):
        print(f"\n❓ Q&A Test {i}: {question}")
        result = legal_ai.legal_question_answering(question)
        
        if 'error' not in result:
            print(f"   ✅ Answer generated")
            print(f"   🔍 Search method: {result.get('search_method', 'Unknown')}")
            if 'answer' in result:
                answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
                print(f"   � Answer preview: {answer_preview}")
            if 'citations' in result and result['citations']:
                print(f"   � Citations: {', '.join(result['citations'][:3])}")
        else:
            print(f"   ❌ Error: {result['error']}")
            if 'suggestion' in result:
                print(f"   💡 {result['suggestion']}")
    
    print(f"\n🎉 MALAYSIAN LEGAL AI VISION TEST COMPLETE!")
    return legal_ai

if __name__ == "__main__":
    print("🇲🇾 Malaysian Legal AI System")
    print("Implementing your vision for Malaysian law-focused AI")
    print()
    
    # Test the system
    legal_ai = test_legal_ai_vision()
    
    if legal_ai:
        print("\n✅ Your Malaysian Legal AI Vision is working!")
        print("\n🎯 What this system provides:")
        print("   • Search Malaysian legal precedents and statutes")
        print("   • Answer legal questions with citations")
        print("   • Find relevant Malaysian Acts and sections")
        print("   • Semantic search across 188K+ legal chunks")
        print("   • Bilingual support (English & Bahasa Malaysia)")
        print(f"\n🌐 Access the web API at: http://localhost:8000/docs")
        print("\n🚀 Ready for integration with your web application!")
    else:
        print("\n❌ System not ready - Malaysian Legal Vector Database not available")
        print("📋 Start the search API: python pipeline/legal_search_api.py")
        print("📋 Or check if Qdrant is running: docker ps")
