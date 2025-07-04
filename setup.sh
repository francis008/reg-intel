#!/bin/bash

# Legal LLM Platform Startup Script

echo "🏗️ Starting Legal LLM Platform..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create docs directory if it doesn't exist
mkdir -p docs

echo "🚀 Legal LLM Platform is ready!"
echo ""
echo "Available commands:"
echo "  ./start-api.sh     - Start the API server"
echo "  ./start-web.sh     - Start the web interface"
echo "  ./start-both.sh    - Start both API and web interface"
echo ""
echo "Or run manually:"
echo "  python src/api.py       - API server (port 8000)"
echo "  streamlit run src/web_app.py - Web interface (port 8501)"
echo ""
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🌐 Web Interface: http://localhost:8501"
