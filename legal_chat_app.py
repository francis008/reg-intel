"""
Malaysian Legal AI Chat Interface
=================================

ChatGPT-like web interface for Malaysian Legal AI System
Built with Streamlit for easy deployment and beautiful UI
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, List
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Malaysian Legal AI Chat",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main chat container */
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* User message styling */
    .user-message {
        background: #007bff;
        color: white;
        padding: 15px;
        border-radius: 18px;
        margin: 10px 0;
        margin-left: 50px;
        position: relative;
    }
    
    /* AI message styling */
    .ai-message {
        background: #f8f9fa;
        color: #333;
        padding: 15px;
        border-radius: 18px;
        margin: 10px 0;
        margin-right: 50px;
        border-left: 4px solid #28a745;
        position: relative;
    }
    
    /* Citations styling */
    .citations {
        background: #e9ecef;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.9em;
        border-left: 3px solid #ffc107;
    }
    
    /* Legal analysis styling */
    .legal-analysis {
        background: #f0f8f0;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 3px solid #28a745;
        font-size: 0.9em;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Status indicators */
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Message timestamp */
    .timestamp {
        font-size: 0.8em;
        color: #6c757d;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_status" not in st.session_state:
    st.session_state.system_status = None

def check_system_status() -> Dict[str, Any]:
    """Check if the Malaysian Legal AI system is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Get additional stats but preserve critical health fields
            try:
                stats_response = requests.get("http://localhost:8000/stats", timeout=5)
                if stats_response.status_code == 200:
                    stats_data = stats_response.json()
                    # Only add stats data that doesn't override health data
                    for key, value in stats_data.items():
                        if key not in ["status", "search_engine_ready", "components"]:
                            health_data[key] = value
                    # Update status only if it's not already set to "healthy"
                    if health_data.get("status") != "healthy" and "status" in stats_data:
                        health_data["status"] = stats_data["status"]
            except Exception:
                pass  # Stats are optional
                
            return health_data
        else:
            return {"status": "error", "message": f"API returned HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Cannot connect to API: {str(e)}"}

def search_legal_documents(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search Malaysian legal documents"""
    try:
        response = requests.get(
            "http://localhost:8000/search",
            params={"q": query, "limit": limit},
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Search failed with HTTP {response.status_code}"}
            
    except Exception as e:
        return {"error": f"Search request failed: {str(e)}"}

def ask_legal_question(question: str) -> Dict[str, Any]:
    """Ask a legal question and get an AI-generated answer"""
    try:
        response = requests.get(
            "http://localhost:8000/ask",
            params={"question": question, "include_citations": True},
            timeout=20
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Q&A failed with HTTP {response.status_code}"}
            
    except Exception as e:
        return {"error": f"Q&A request failed: {str(e)}"}

def format_legal_response(response_data: Dict[str, Any]) -> str:
    """Format the legal AI response for display"""
    if "error" in response_data:
        return f"❌ **Error**: {response_data['error']}"
    
    # Check if it's a search response or Q&A response
    if "answer" in response_data:
        # Q&A response
        formatted = f"📋 **Answer:**\n\n{response_data['answer']}\n\n"
        
        if "citations" in response_data and response_data["citations"]:
            formatted += "📚 **Legal Citations:**\n"
            for citation in response_data["citations"][:5]:
                formatted += f"• {citation}\n"
            formatted += "\n"
        
        if "relevant_acts" in response_data and response_data["relevant_acts"]:
            formatted += "⚖️ **Relevant Acts:**\n"
            for act in response_data["relevant_acts"][:3]:
                formatted += f"• {act}\n"
            formatted += "\n"
        
        if "confidence" in response_data:
            confidence = response_data["confidence"] * 100
            formatted += f"🎯 **Confidence**: {confidence:.1f}%\n\n"
            
    else:
        # Search response
        if "results" in response_data and response_data["results"]:
            formatted = f"🔍 **Found {len(response_data['results'])} relevant legal provisions:**\n\n"
            
            for i, result in enumerate(response_data["results"][:3], 1):
                formatted += f"**{i}. {result.get('act_title', 'Unknown Act')}**\n"
                if result.get("section_heading"):
                    formatted += f"📖 Section: {result['section_heading']}\n"
                if result.get("citation"):
                    formatted += f"📋 Citation: {result['citation']}\n"
                
                content = result.get("text_content", "")
                if len(content) > 300:
                    content = content[:300] + "..."
                formatted += f"📄 Content: {content}\n"
                
                if result.get("relevance_score"):
                    relevance = result["relevance_score"] * 100
                    formatted += f"🎯 Relevance: {relevance:.1f}%\n"
                
                formatted += "\n---\n\n"
        else:
            formatted = "❌ No relevant legal provisions found for your query."
    
    return formatted

def display_chat_message(role: str, content: str, timestamp: str = None):
    """Display a chat message with proper styling"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    # Clean content to avoid HTML conflicts
    import html
    clean_content = html.escape(content).replace('\n', '<br>')
    
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>You</strong><br>
            {clean_content}
            <div class="timestamp">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # For AI responses, use Streamlit's markdown renderer instead of custom HTML
        st.markdown("**🇲🇾 Malaysian Legal AI**")
        st.markdown(content)
        st.markdown(f"<small style='color: #6c757d;'>{timestamp}</small>", unsafe_allow_html=True)
        st.markdown("---")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🇲🇾 Malaysian Legal AI Chat</h1>
        <p>ChatGPT-like interface for Malaysian legal knowledge</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # Check system status
        with st.spinner("Checking system status..."):
            st.session_state.system_status = check_system_status()
        
        status = st.session_state.system_status
        
        # Check for both possible status values
        status_value = status.get("status") if status else None
        is_healthy = status_value in ["healthy", "green"] and status.get("search_engine_ready", False)
        
        if status and is_healthy:
            st.markdown('<p class="status-healthy">✅ System Online</p>', unsafe_allow_html=True)
            
            # Show system components status
            if status.get("components", {}).get("qdrant"):
                st.write("🔍 **Vector Database**: Ready")
            if status.get("components", {}).get("embeddings"):
                st.write("🤖 **AI Model**: Ready")
            if status.get("search_engine_ready"):
                st.write("📚 **Search Engine**: Ready")
                
            # Show document count if available
            if status.get("total_documents"):
                st.write(f"📋 **Documents**: {status['total_documents']:,}")
            else:
                st.write("📋 **Documents**: 188,627+ legal provisions")
        else:
            st.markdown('<p class="status-error">❌ System Offline</p>', unsafe_allow_html=True)
            if "message" in status:
                st.error(status["message"])
            st.warning("Start the legal search API:\n```bash\npython pipeline/legal_search_api.py\n```")
        
        st.markdown("---")
        
        # Chat controls
        st.markdown("### 💬 Chat Controls")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Query type selection
        query_type = st.selectbox(
            "Query Type",
            ["Legal Q&A", "Document Search"],
            help="Choose how to process your query"
        )
        
        # Language preference
        language = st.selectbox(
            "Language",
            ["English", "Bahasa Malaysia", "Both"],
            help="Preferred language for results"
        )
        
        st.markdown("---")
        
        # Example queries
        st.markdown("### 💡 Example Queries")
        
        examples = [
            "What are the grounds for dismissing an employee?",
            "Requirements for a valid contract under Malaysian law",
            "Director's duties under Companies Act 2016",
            "Employment termination procedures",
            "Breach of contract remedies"
        ]
        
        for example in examples:
            if st.button(f"💬 {example[:30]}...", key=f"example_{hash(example)}"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": example,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                st.rerun()
    
    # Main chat area
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
                message.get("timestamp")
            )
    
    # Chat input
    st.markdown("---")
    
    # Check if system is ready
    status = st.session_state.system_status
    status_value = status.get("status") if status else None
    is_system_ready = status_value in ["healthy", "green"] and status.get("search_engine_ready", False)
    
    if not is_system_ready:
        st.error("🚫 Malaysian Legal AI system is not available. Please start the legal search API first.")
        st.code("python pipeline/legal_search_api.py")
        return
    
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask a legal question or search for legal documents...",
                placeholder="e.g., What are the grounds for dismissing an employee in Malaysia?",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Send 📤")
    
    # Process user input
    if submit_button and user_input.strip():
        # Add user message
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Show thinking indicator
        with st.spinner("🤔 Malaysian Legal AI is thinking..."):
            # Process the query based on type
            if query_type == "Legal Q&A":
                response_data = ask_legal_question(user_input)
            else:
                response_data = search_legal_documents(user_input)
            
            # Format the response
            ai_response = format_legal_response(response_data)
            
            # Add AI response
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%H:%M")
            })
        
        # Rerun to display new messages
        st.rerun()
    
    # Welcome message for new users
    if not st.session_state.messages:
        st.markdown("""
        <div class="ai-message">
            <strong>🇲🇾 Malaysian Legal AI</strong><br>
            Welcome! I'm your Malaysian Legal AI assistant. I have access to over 188,000 legal provisions from Malaysian Principal Acts.
            <br><br>
            You can:
            <br>• Ask legal questions and get detailed answers with citations
            <br>• Search for specific legal provisions and Acts
            <br>• Get help with Malaysian employment, contract, and corporate law
            <br><br>
            Try asking: <em>"What are the grounds for dismissing an employee in Malaysia?"</em>
            <div class="timestamp">Ready to help</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
