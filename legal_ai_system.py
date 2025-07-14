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
        # Import here to avoid issues if RAG system not available
        try:
            from secure_rag_llamaindex import SecureLegalRAGLlamaIndex
            
            # Two separate RAG systems for your vision:
            self.legal_knowledge_rag = SecureLegalRAGLlamaIndex("malaysian_legal_knowledge")  # Government laws
            self.user_case_rags = {}  # firm_id -> RAG instance for user cases
            
            self.setup_legal_knowledge()
            
        except ImportError as e:
            print(f"⚠️ RAG system not available: {e}")
            self.legal_knowledge_rag = None
            self.user_case_rags = {}
    
    def setup_legal_knowledge(self):
        """Pre-load Malaysian government legal knowledge"""
        if not self.legal_knowledge_rag:
            print("❌ RAG system not available")
            return False
        
        # Try new pipeline embeddings first
        if self.load_from_pipeline_embeddings():
            print("📚 Using pipeline-processed Malaysian legal knowledge")
            return True
            
        knowledge_base_path = Path("./malaysian_legal_knowledge")
        
        if not knowledge_base_path.exists():
            print("⚠️ Malaysian legal knowledge base not found.")
            print("📋 Run: python pipeline/run_pipeline.py for complete processing")
            print("📋 Or run: python build_malaysian_legal_base.py for basic setup")
            return False
        
        print("📚 Loading Malaysian legal knowledge base...")
        
        # Process all Malaysian legal documents
        legal_docs = list(knowledge_base_path.glob("*.txt"))
        if not legal_docs:
            print("❌ No legal documents found in knowledge base")
            return False
        
        processed_count = 0
        for legal_doc in legal_docs:
            try:
                success = self.legal_knowledge_rag.process_document_securely(str(legal_doc))
                if success:
                    processed_count += 1
                    print(f"✅ Processed: {legal_doc.name}")
                else:
                    print(f"❌ Failed to process: {legal_doc.name}")
            except Exception as e:
                print(f"❌ Error processing {legal_doc.name}: {e}")
        
        if processed_count > 0:
            print(f"🎉 Malaysian legal knowledge base ready! ({processed_count} documents)")
            return True
        else:
            print("❌ No documents successfully processed")
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
        if not self.legal_knowledge_rag:
            return {"error": "Legal knowledge system not available"}
        
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
        if not self.legal_knowledge_rag:
            return {
                "system_ready": False,
                "error": "RAG system not available"
            }
        
        try:
            knowledge_stats = self.legal_knowledge_rag.get_firm_statistics()
            
            return {
                "system_ready": knowledge_stats.get('total_documents', 0) > 0,
                "legal_knowledge_ready": knowledge_stats.get('index_exists', False),
                "legal_documents_count": knowledge_stats.get('total_documents', 0),
                "storage_size_mb": knowledge_stats.get('storage_size_mb', 0),
                "user_firms": len(self.user_case_rags),
                "last_updated": knowledge_stats.get('last_updated', 'Never')
            }
        
        except Exception as e:
            return {
                "system_ready": False,
                "error": f"Status check failed: {e}"
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
        """Load Malaysian legal knowledge from the new pipeline embeddings"""
        embeddings_dir = Path("./embeddings")
        
        if not embeddings_dir.exists():
            print("⚠️ Pipeline embeddings not found.")
            print("📋 Run: python pipeline/run_pipeline.py to create embeddings")
            return False
        
        # Check for pipeline output files
        embeddings_file = embeddings_dir / "legal_embeddings_complete.pkl"
        metadata_file = embeddings_dir / "legal_chunks_metadata.json"
        
        if embeddings_file.exists():
            print("✅ Found pipeline embeddings - legal knowledge ready!")
            return True
        elif metadata_file.exists():
            print("✅ Found pipeline metadata - legal knowledge ready!")
            return True
        else:
            print("❌ No pipeline embeddings found")
            return False


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
        print("❌ Legal knowledge base not ready")
        print("📋 Please run: python build_malaysian_legal_base.py")
        return None
    
    print(f"📚 Legal documents loaded: {status['legal_documents_count']}")
    
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
            print(f"   ✅ Found precedents (Confidence: {result['confidence']:.1%})")
            print(f"   📚 Legal Areas: {', '.join(result['legal_areas'])}")
            print(f"   📖 Citations: {', '.join(result['relevant_citations'][:3])}")
            print(f"   💡 Precedents: {result['legal_precedents'][:200]}...")
        else:
            print(f"   ❌ Error: {result['error']}")
    
    # TEST USE CASE B: Draft with Legal Knowledge
    print(f"\n📝 USE CASE B: Drafting with Malaysian Legal Knowledge")
    print("=" * 50)
    
    drafting_tests = [
        ("Create an employment agreement with proper termination clauses", "employment_agreement"),
        ("Draft a software licensing agreement with IP protections", "licensing_agreement"),
        ("Generate a privacy policy compliant with Malaysian consumer protection laws", "legal_memo")
    ]
    
    for i, (prompt, doc_type) in enumerate(drafting_tests, 1):
        print(f"\n📄 Draft Test {i}: {prompt}")
        result = legal_ai.draft_legal_document_with_knowledge(prompt, doc_type)
        
        if 'error' not in result:
            print(f"   ✅ Document generated (Confidence: {result['confidence']:.1%})")
            print(f"   📚 Legal sources used: {result['legal_sources_used']}")
            print(f"   📋 Compliance basis: {len(result['compliance_basis'])} requirements")
            print(f"   📄 Preview: {result['generated_document'][:200]}...")
        else:
            print(f"   ❌ Error: {result['error']}")
    
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
        print("   • Draft documents compliant with Malaysian law")
        print("   • Cite relevant Malaysian cases and acts")
        print("   • Keep user cases private and separate")
        print("\n🚀 Ready for integration with your web application!")
    else:
        print("\n❌ System not ready - need to build Malaysian legal knowledge base")
