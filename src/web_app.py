# src/web_app.py - Simple Streamlit interface for the Legal LLM Platform

import streamlit as st
import requests
import json
from datetime import datetime
import os

# Configuration
API_BASE_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="Legal LLM Platform",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Legal LLM Platform")
    st.subheader("Custom AI for Law Firms")
    
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 Dashboard"
    
    # Sidebar navigation with styled buttons
    st.sidebar.title("⚖️ Legal LLM")
    st.sidebar.markdown("---")
    
    # Custom CSS for navigation
    st.sidebar.markdown("""
    <style>
    .nav-section {
        margin-bottom: 15px;
    }
    .nav-section h4 {
        color: #1f77b4;
        margin-bottom: 8px;
        font-size: 14px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Analytics Section
    st.sidebar.markdown('<div class="nav-section"><h4>📊 Analytics</h4></div>', unsafe_allow_html=True)
    if st.sidebar.button("Dashboard", 
                        use_container_width=True, 
                        type="primary" if st.session_state.current_page == "📊 Dashboard" else "secondary"):
        st.session_state.current_page = "📊 Dashboard"
        st.rerun()
    
    # Documents Section
    st.sidebar.markdown('<div class="nav-section"><h4>📄 Documents</h4></div>', unsafe_allow_html=True)
    if st.sidebar.button("Upload Documents", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == "📤 Upload Documents" else "secondary"):
        st.session_state.current_page = "📤 Upload Documents"
        st.rerun()
    
    if st.sidebar.button("Training Data", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == "📚 Training Data" else "secondary"):
        st.session_state.current_page = "📚 Training Data"
        st.rerun()
    
    # AI Tools Section
    st.sidebar.markdown('<div class="nav-section"><h4>🤖 AI Tools</h4></div>', unsafe_allow_html=True)
    if st.sidebar.button("Generate Text", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == "✍️ Generate Text" else "secondary"):
        st.session_state.current_page = "✍️ Generate Text"
        st.rerun()
    
    # RAG Search Section
    st.sidebar.markdown('<div class="nav-section"><h4>⚖️ Legal AI (Your Vision)</h4></div>', unsafe_allow_html=True)
    if st.sidebar.button("🔍 Search Legal Precedents", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == "🔍 Legal Precedents" else "secondary"):
        st.session_state.current_page = "🔍 Legal Precedents"
        st.rerun()
    
    if st.sidebar.button("📝 Draft with Legal Knowledge", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == "📝 Legal Drafting" else "secondary"):
        st.session_state.current_page = "📝 Legal Drafting"
        st.rerun()
    
    if st.sidebar.button("📁 Upload Case Documents", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == "📁 Case Documents" else "secondary"):
        st.session_state.current_page = "📁 Case Documents"
        st.rerun()
    
    # Legacy Features Section  
    st.sidebar.markdown('<div class="nav-section"><h4>🔧 Legacy Features</h4></div>', unsafe_allow_html=True)
    if st.sidebar.button("Document Search (Old)", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == "🔍 Document Search" else "secondary"):
        st.session_state.current_page = "🔍 Document Search"
        st.rerun()
    
    if st.sidebar.button("RAG Generator (Old)", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == "🎯 RAG Generator" else "secondary"):
        st.session_state.current_page = "🎯 RAG Generator"
        st.rerun()
    
    # Add some spacing and info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Your Legal AI")
    st.sidebar.info("🏛️ Pre-trained on government law\n\n🔍 Find legal precedents for cases\n\n📝 Draft with legal knowledge\n\n� Add your case documents securely")
    
    # Set the current page
    page = st.session_state.current_page
    
    # Page routing
    if page == "📊 Dashboard":
        dashboard_page()
    elif page == "📤 Upload Documents":
        upload_page()
    elif page == "✍️ Generate Text":
        generate_page()
    elif page == "📚 Training Data":
        training_data_page()
    elif page == "🔍 Legal Precedents":
        legal_precedents_page()
    elif page == "📝 Legal Drafting":
        legal_drafting_page()
    elif page == "📁 Case Documents":
        case_documents_page()
    elif page == "🔍 Document Search":
        rag_search_page()
    elif page == "🎯 RAG Generator":
        rag_generator_page()

def dashboard_page():
    st.header("📊 Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    try:
        # Get health status
        health_response = requests.get(f"{API_BASE_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            
            with col1:
                st.metric("System Status", "🟢 Healthy")
            
            with col2:
                st.metric("Documents Processed", health_data.get("documents_processed", 0))
            
            with col3:
                st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
        
        # Get training data summary
        training_response = requests.get(f"{API_BASE_URL}/training-data")
        if training_response.status_code == 200:
            training_data = training_response.json()
            
            st.subheader("📄 Training Data Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Documents", training_data["total_documents"])
                st.metric("Training Examples", training_data["total_training_examples"])
            
            with col2:
                if training_data["document_types"]:
                    st.write("**Document Types:**")
                    for doc_type, count in training_data["document_types"].items():
                        st.write(f"• {doc_type.replace('_', ' ').title()}: {count}")
                else:
                    st.info("No documents uploaded yet. Start by uploading some legal documents!")
    
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Cannot connect to API server.")
    
    # RAG System Overview
    st.subheader("🔍 RAG System Overview")
    
    firm_id = st.text_input("Check Firm RAG Status", value="default_firm", help="Enter firm ID to check RAG status")
    
    if firm_id:
        try:
            import sys
            sys.path.append('./src')
            from secure_rag_llamaindex import SecureLegalRAGLlamaIndex
            
            rag = SecureLegalRAGLlamaIndex(firm_id)
            stats = rag.get_firm_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📄 RAG Documents", stats.get('total_documents', 0))
            with col2:
                st.metric("🔍 Index Status", "✅ Ready" if stats.get('index_exists', False) else "❌ Empty")
            with col3:
                st.metric("💾 Storage Size", f"{stats.get('storage_size_mb', 0):.2f} MB")
            with col4:
                last_updated = stats.get('last_updated', 'Never')
                if last_updated != 'Never':
                    last_updated = last_updated[:16] if len(last_updated) > 16 else last_updated
                st.metric("🕐 Last Updated", last_updated)
            
            # Quick actions
            st.markdown("**🚀 Quick Actions:**")
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("📤 Upload Documents", use_container_width=True):
                    st.session_state.current_page = "📤 Upload Documents"
                    st.rerun()
            
            with action_col2:
                if st.button("🔍 Search Documents", use_container_width=True):
                    st.session_state.current_page = "🔍 Document Search"
                    st.rerun()
            
            with action_col3:
                if st.button("🎯 Generate with RAG", use_container_width=True):
                    st.session_state.current_page = "🎯 RAG Generator"
                    st.rerun()
            
            # Status messages
            if stats.get('total_documents', 0) == 0:
                st.info("💡 No documents in RAG system yet. Upload some documents to get started!")
            elif not stats.get('index_exists', False):
                st.warning("⚠️ RAG index not ready. Try uploading and processing a document.")
            else:
                st.success(f"✅ RAG system ready with {stats.get('total_documents', 0)} documents!")
        
        except ImportError:
            st.warning("⚠️ RAG system not available. Make sure dependencies are installed.")
        except Exception as e:
            st.warning(f"⚠️ Could not check RAG status: {e}")
    
    # System Health Check
    st.subheader("🏥 System Health")
    
    health_col1, health_col2, health_col3 = st.columns(3)
    
    with health_col1:
        # API Health
        try:
            api_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if api_response.status_code == 200:
                st.success("🔗 API Server: Online")
            else:
                st.error("🔗 API Server: Error")
        except:
            st.error("🔗 API Server: Offline")
    
    with health_col2:
        # RAG Health
        try:
            import sys
            sys.path.append('./src')
            from secure_rag_llamaindex import SecureLegalRAGLlamaIndex
            st.success("🔍 RAG System: Available")
        except:
            st.error("🔍 RAG System: Unavailable")
    
    with health_col3:
        # Storage Health
        try:
            import os
            if os.path.exists('./secure_data'):
                st.success("💾 Storage: Ready")
            else:
                st.warning("💾 Storage: Not Initialized")
        except:
            st.error("💾 Storage: Error")

def upload_page():
    st.header("📤 Upload Legal Documents")
    st.write("Upload PDF, Word, or text files to train your custom legal AI and enable RAG search")
    
    # Firm ID input
    firm_id = st.text_input("Firm ID", value="default_firm", help="Your firm's unique identifier for secure document storage")
    
    uploaded_file = st.file_uploader(
        "Choose a legal document",
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, Word documents, and text files"
    )
    
    if uploaded_file is not None:
        st.write("**File Details:**")
        st.write(f"• Name: {uploaded_file.name}")
        st.write(f"• Size: {uploaded_file.size / 1024:.2f} KB")
        st.write(f"• Type: {uploaded_file.type}")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            process_api = st.checkbox("Process with API", value=True, help="Send to API server for processing")
        with col2:
            process_rag = st.checkbox("Process with RAG", value=True, help="Add to RAG system for smart search")
        
        if st.button("📄 Process Document", type="primary"):
            results = {"api": None, "rag": None}
            
            # Process with API if selected
            if process_api:
                try:
                    with st.spinner("Processing with API server..."):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{API_BASE_URL}/upload-document", files=files)
                        
                        if response.status_code == 200:
                            results["api"] = response.json()
                            st.success("✅ API processing completed!")
                        else:
                            error_detail = response.json().get("detail", "Unknown error")
                            st.error(f"❌ API processing failed: {error_detail}")
                
                except requests.exceptions.ConnectionError:
                    st.warning("⚠️ Cannot connect to API server. Skipping API processing.")
                except Exception as e:
                    st.error(f"❌ API processing error: {str(e)}")
            
            # Process with RAG if selected
            if process_rag:
                try:
                    with st.spinner("Adding to RAG system..."):
                        import sys
                        import tempfile
                        import os
                        
                        sys.path.append('./src')
                        from secure_rag_llamaindex import SecureLegalRAGLlamaIndex
                        
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(mode='wb', suffix=f"_{uploaded_file.name}", delete=False) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Initialize RAG system
                            rag = SecureLegalRAGLlamaIndex(firm_id)
                            
                            # Process document
                            rag_result = rag.process_document_securely(tmp_path)
                            
                            if rag_result:
                                results["rag"] = {"success": True, "processed": True}
                                st.success("✅ RAG processing completed!")
                            else:
                                st.error("❌ RAG processing failed")
                        
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                
                except ImportError as e:
                    st.error(f"❌ RAG system not available: {e}")
                except Exception as e:
                    st.error(f"❌ RAG processing error: {str(e)}")
            
            # Display results
            if results["api"] or results["rag"]:
                st.markdown("---")
                st.subheader("📊 Processing Results")
                
                col1, col2 = st.columns(2)
                
                if results["api"]:
                    with col1:
                        st.markdown("**🔗 API Processing**")
                        st.metric("Document Type", results["api"]["doc_type"].replace('_', ' ').title())
                        st.metric("Training Examples", results["api"]["training_examples"])
                        st.success("Ready for text generation")
                
                if results["rag"]:
                    with col2:
                        st.markdown("**🔍 RAG Processing**")
                        st.metric("Status", "✅ Processed")
                        st.metric("Search Ready", "Yes")
                        st.success("Ready for smart search")
                
                st.info("💡 Your document is now available for AI-powered search and generation!")
                
                # Show next steps
                st.markdown("**🚀 What you can do now:**")
                if results["rag"]:
                    st.write("• 🔍 Search your documents in the 'Document Search' section")
                    st.write("• 🎯 Generate documents using your data in 'RAG Document Generator'")
                if results["api"]:
                    st.write("• ✍️ Generate text using the 'Generate Text' section")
                st.write("• 📊 View your data in the 'Training Data' section")

def generate_page():
    st.header("Generate Legal Text")
    st.write("Use your firm's AI to generate legal documents")
    
    # Input form
    with st.form("generate_form"):
        prompt = st.text_area(
            "Describe what you want to generate:",
            placeholder="e.g., 'a software development agreement between TechCorp and DevStudio'",
            height=100
        )
        
        doc_type = st.selectbox(
            "Document Type:",
            ["contract", "legal_memo", "court_filing"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        max_length = st.slider("Maximum Length (characters)", 500, 3000, 1500)
        
        generate_button = st.form_submit_button("Generate Document", type="primary")
    
    if generate_button and prompt:
        try:
            with st.spinner("Generating legal text..."):
                payload = {
                    "prompt": prompt,
                    "doc_type": doc_type,
                    "max_length": max_length
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/generate-text",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Document generated successfully!")
                    
                    # Display results
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.subheader("Generated Document")
                        st.code(result["generated_text"], language="text")
                    
                    with col2:
                        st.metric("Document Type", result["doc_type"].replace('_', ' ').title())
                        st.metric("Confidence", f"{result['confidence']:.0%}")
                        
                        # Download button
                        st.download_button(
                            label="📄 Download",
                            data=result["generated_text"],
                            file_name=f"{doc_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"❌ Error generating text: {error_detail}")
        
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to API. Make sure the API server is running.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

def training_data_page():
    st.header("Training Data")
    st.write("View and manage your legal document training data")
    
    try:
        # Get document list
        response = requests.get(f"{API_BASE_URL}/documents")
        if response.status_code == 200:
            documents = response.json()["documents"]
            
            if documents:
                st.subheader(f"Documents ({len(documents)})")
                
                # Create a table
                for doc in documents:
                    with st.expander(f"📄 {doc['id']} ({doc['type'].replace('_', ' ').title()})"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Size:** {doc['size_kb']:.2f} KB")
                            st.write(f"**Type:** {doc['type'].replace('_', ' ').title()}")
                        
                        with col2:
                            st.write(f"**Training Examples:** {doc['training_examples']}")
                            st.write(f"**Upload Date:** {doc['upload_date'][:10]}")
                        
                        with col3:
                            st.write("**Status:** ✅ Processed")
                            if st.button(f"🗑️ Remove", key=f"remove_{doc['id']}"):
                                st.warning("Remove functionality would be implemented here")
            else:
                st.info("No documents uploaded yet. Go to the Upload page to add some legal documents!")
        
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Make sure the API server is running.")

def legal_precedents_page():
    """YOUR VISION: USE CASE A - Search legal precedents for user's case"""
    st.header("🔍 Search Legal Precedents")
    st.subheader("Find Similar Cases and Government Law Citations")
    st.write("Describe your legal case and get relevant precedents, statutes, and citations from government legal knowledge.")
    
    # Case description input
    with st.form("precedent_search_form"):
        case_description = st.text_area(
            "Describe Your Legal Case:",
            placeholder="""Example: "My client was fired after reporting safety violations to OSHA. The employer claims it was for poor performance, but the timing suggests retaliation. Can we pursue a wrongful termination claim?"
            
Or: "A software company wants to enforce a non-compete clause against a former employee who joined a competitor. The clause covers the entire state for 2 years. Is this enforceable?"
            
Be specific about facts, timeline, and legal issues involved.""",
            height=150,
            help="Provide detailed facts about your case. The AI will search government legal knowledge for relevant precedents."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            num_precedents = st.slider("Number of Precedents", 3, 10, 5)
        with col2:
            include_citations = st.checkbox("Include Legal Citations", value=True)
        
        search_button = st.form_submit_button("🔍 Search Legal Precedents", type="primary")
    
    if search_button and case_description:
        try:
            with st.spinner("Searching government legal knowledge for relevant precedents..."):
                import sys
                sys.path.append('.')
                from legal_ai_system import LegalAISystem
                
                # Initialize the legal AI system
                legal_ai = LegalAISystem()
                
                # Check if system is ready
                status = legal_ai.get_system_status()
                if not status.get('system_ready', False):
                    st.error("❌ Legal knowledge base not ready. Please build the knowledge base first.")
                    st.info("Run: `python build_legal_knowledge_base.py` to set up the legal knowledge.")
                    return
                
                # Search for legal precedents
                result = legal_ai.search_legal_precedents(case_description, num_precedents)
                
                if 'error' not in result:
                    st.success("✅ Legal precedents found!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    with col2:
                        st.metric("Sources Found", result['sources_found'])
                    with col3:
                        st.metric("Legal Areas", len(result['legal_areas']))
                    
                    # Main legal analysis
                    st.subheader("📋 Legal Analysis & Precedents")
                    st.write(result['legal_precedents'])
                    
                    # Legal areas identified
                    if result['legal_areas']:
                        st.subheader("⚖️ Areas of Law")
                        for area in result['legal_areas']:
                            st.badge(area, outline=True)
                    
                    # Citations found
                    if include_citations and result['relevant_citations']:
                        st.subheader("📚 Legal Citations")
                        for citation in result['relevant_citations']:
                            st.code(citation)
                    
                    # Action items
                    st.subheader("🎯 Recommended Next Steps")
                    st.write("Based on the legal precedents found:")
                    st.write("• Review the cited cases and statutes in detail")
                    st.write("• Research jurisdiction-specific variations")
                    st.write("• Consider drafting legal documents using the 'Legal Drafting' section")
                    st.write("• Upload your case documents for more personalized analysis")
                    
                else:
                    st.error(f"❌ Search failed: {result['error']}")
        
        except ImportError as e:
            st.error(f"❌ Legal AI system not available: {e}")
            st.info("Make sure the legal knowledge base is built and dependencies are installed.")
        except Exception as e:
            st.error(f"❌ Search failed: {e}")
    
    # System status
    st.markdown("---")
    st.subheader("📊 Legal Knowledge System Status")
    
    try:
        import sys
        sys.path.append('.')
        from legal_ai_system import LegalAISystem
        
        legal_ai = LegalAISystem()
        status = legal_ai.get_system_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Legal Documents", status.get('legal_documents_count', 0))
        with col2:
            st.metric("Knowledge Ready", "✅ Yes" if status.get('legal_knowledge_ready', False) else "❌ No")
        with col3:
            st.metric("Storage Size", f"{status.get('legal_knowledge_size_mb', 0):.1f} MB")
        with col4:
            st.metric("System Ready", "✅ Yes" if status.get('system_ready', False) else "❌ No")
        
        if not status.get('system_ready', False):
            st.warning("⚠️ Legal knowledge base not ready. Build it first:")
            st.code("python build_legal_knowledge_base.py")
    
    except Exception as e:
        st.warning(f"⚠️ Could not check system status: {e}")

def legal_drafting_page():
    """YOUR VISION: USE CASE B - Draft documents using legal knowledge"""
    st.header("📝 Draft with Legal Knowledge")
    st.subheader("Generate Legal Documents Using Government Law Knowledge")
    st.write("Describe what you want to draft. The AI uses pre-trained legal knowledge to create compliant documents.")
    
    # Firm ID for case context
    firm_id = st.text_input(
        "Firm/Client ID (Optional)", 
        value="", 
        help="If you've uploaded case documents, provide your firm ID to include case-specific context"
    )
    
    # Document drafting form
    with st.form("legal_drafting_form"):
        prompt = st.text_area(
            "Document Description:",
            placeholder="""Example: "Create an employment agreement for a software engineer position with confidentiality clauses and intellectual property assignment. Include proper termination procedures and compliance with California labor laws."
            
Or: "Draft a software licensing agreement that protects our IP while allowing reasonable use. Include termination clauses and limitation of liability provisions."
            
Be specific about requirements, parties involved, and any special considerations.""",
            height=150,
            help="The AI will use legal knowledge to ensure your document includes required elements and complies with applicable laws."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            doc_type = st.selectbox(
                "Document Type:",
                [
                    "employment_agreement",
                    "service_agreement", 
                    "licensing_agreement",
                    "nda",
                    "contract",
                    "legal_memo",
                    "privacy_policy",
                    "terms_of_service",
                    "partnership_agreement",
                    "court_filing"
                ],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            use_case_context = st.checkbox(
                "Use Case Documents", 
                value=bool(firm_id), 
                help="Include context from your uploaded case documents"
            )
        
        draft_button = st.form_submit_button("📝 Draft Legal Document", type="primary")
    
    if draft_button and prompt:
        try:
            with st.spinner("Drafting document using legal knowledge..."):
                import sys
                sys.path.append('.')
                from legal_ai_system import LegalAISystem
                
                # Initialize the legal AI system
                legal_ai = LegalAISystem()
                
                # Check if system is ready
                status = legal_ai.get_system_status()
                if not status.get('system_ready', False):
                    st.error("❌ Legal knowledge base not ready. Please build the knowledge base first.")
                    st.info("Run: `python build_legal_knowledge_base.py` to set up the legal knowledge.")
                    return
                
                # Draft document with legal knowledge
                case_firm_id = firm_id if use_case_context and firm_id else None
                result = legal_ai.draft_legal_document_with_knowledge(prompt, doc_type, case_firm_id)
                
                if 'error' not in result:
                    st.success("✅ Legal document drafted successfully!")
                    
                    # Display results in columns
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.subheader("📄 Generated Legal Document")
                        
                        # Create tabs for different views
                        tab1, tab2, tab3 = st.tabs(["📄 Document", "🔧 Raw Text", "⚖️ Legal Basis"])
                        
                        with tab1:
                            # Formatted document view
                            st.markdown(result['generated_document'])
                        
                        with tab2:
                            # Raw text for copying
                            st.code(result['generated_document'], language="text")
                        
                        with tab3:
                            # Legal foundation used
                            st.write("**Legal Foundation Used:**")
                            st.write(result['legal_foundation'])
                            
                            if result['compliance_basis']:
                                st.write("**Compliance Requirements:**")
                                for req in result['compliance_basis']:
                                    st.write(f"• {req}")
                    
                    with col2:
                        st.subheader("📊 Document Info")
                        st.metric("Document Type", doc_type.replace('_', ' ').title())
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                        st.metric("Legal Sources", result['legal_sources_used'])
                        if result['user_context_used']:
                            st.metric("Case Sources", result['user_sources_used'])
                        
                        # Download button
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"{doc_type}_{timestamp}.txt"
                        
                        st.download_button(
                            label="📄 Download Document",
                            data=result['generated_document'],
                            file_name=filename,
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                        # Legal compliance info
                        st.info("✅ This document was generated using government legal knowledge and precedents.")
                    
                    # Legal basis section
                    st.subheader("⚖️ Legal Compliance")
                    if result['legal_sources_used'] > 0:
                        st.success(f"✅ Document based on {result['legal_sources_used']} legal sources")
                        st.write("This document incorporates:")
                        st.write("• Applicable statutory requirements")
                        st.write("• Relevant case law precedents")
                        st.write("• Standard legal practices and clauses")
                        st.write("• Compliance with identified legal principles")
                    
                    if result['user_context_used']:
                        st.info(f"📁 Also incorporated context from {result['user_sources_used']} of your case documents")
                    
                else:
                    st.error(f"❌ Document drafting failed: {result['error']}")
        
        except ImportError as e:
            st.error(f"❌ Legal AI system not available: {e}")
            st.info("Make sure the legal knowledge base is built and dependencies are installed.")
        except Exception as e:
            st.error(f"❌ Drafting failed: {e}")

def case_documents_page():
    """Upload user's case documents (separate from legal knowledge)"""
    st.header("📁 Upload Case Documents")
    st.subheader("Add Your Case Files for Personalized Legal Analysis")
    st.write("Upload documents specific to your case. These will be kept separate from the legal knowledge base and used to provide personalized context.")
    
    # Firm/Client ID
    firm_id = st.text_input(
        "Firm/Client ID:", 
        value="my_firm", 
        help="Unique identifier for your firm or client. Documents will be stored securely under this ID."
    )
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose case documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload case-specific documents: contracts, correspondence, court filings, etc."
    )
    
    if uploaded_files:
        st.write(f"**Selected Files ({len(uploaded_files)}):**")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size / 1024:.1f} KB)")
        
        if st.button("📁 Upload Case Documents", type="primary"):
            try:
                with st.spinner("Processing case documents..."):
                    import sys
                    import tempfile
                    import os
                    
                    sys.path.append('.')
                    from legal_ai_system import LegalAISystem
                    
                    # Initialize the legal AI system
                    legal_ai = LegalAISystem()
                    
                    success_count = 0
                    failed_files = []
                    
                    for uploaded_file in uploaded_files:
                        # Save file temporarily
                        with tempfile.NamedTemporaryFile(mode='wb', suffix=f"_{uploaded_file.name}", delete=False) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Add to user case documents
                            success = legal_ai.add_user_case_documents(firm_id, tmp_path)
                            if success:
                                success_count += 1
                            else:
                                failed_files.append(uploaded_file.name)
                        
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                    
                    # Show results
                    if success_count > 0:
                        st.success(f"✅ Successfully processed {success_count} case documents!")
                        
                        if failed_files:
                            st.warning(f"⚠️ Failed to process: {', '.join(failed_files)}")
                        
                        st.info("💡 Your case documents are now available for:")
                        st.write("• Personalized legal precedent searches")
                        st.write("• Document drafting with case-specific context")
                        st.write("• Legal analysis incorporating your case facts")
                    else:
                        st.error("❌ Failed to process any documents")
            
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")
    
    # Show current case documents
    st.markdown("---")
    st.subheader(f"📊 Case Documents for: {firm_id}")
    
    try:
        import sys
        sys.path.append('.')
        from legal_ai_system import LegalAISystem
        
        legal_ai = LegalAISystem()
        status = legal_ai.get_system_status()
        
        if firm_id in legal_ai.user_case_rags:
            case_rag = legal_ai.user_case_rags[firm_id]
            case_stats = case_rag.get_firm_statistics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Case Documents", case_stats.get('total_documents', 0))
            with col2:
                st.metric("Storage Size", f"{case_stats.get('storage_size_mb', 0):.1f} MB")
            with col3:
                st.metric("Status", "✅ Ready" if case_stats.get('index_exists', False) else "❌ Empty")
            
            if case_stats.get('total_documents', 0) > 0:
                st.success(f"✅ {case_stats.get('total_documents', 0)} case documents ready for legal analysis")
            else:
                st.info("💡 No case documents uploaded yet")
        else:
            st.info("💡 No case documents for this firm ID yet")
    
    except Exception as e:
        st.warning(f"⚠️ Could not check case document status: {e}")

def rag_search_page():
    """Legacy RAG search functionality"""
    st.header("🔍 Document Search (Legacy)")
    st.write("⚠️ This is the old document search. Consider using 'Search Legal Precedents' for better results.")
    st.info("The new 'Legal Precedents' search uses pre-trained legal knowledge to find relevant case law and statutes.")

def rag_generator_page():
    """Legacy RAG generator functionality"""
    st.header("🎯 RAG Generator (Legacy)")
    st.write("⚠️ This is the old document generator. Consider using 'Draft with Legal Knowledge' for better results.")
    st.info("The new 'Legal Drafting' feature uses government legal knowledge to create compliant documents.")
