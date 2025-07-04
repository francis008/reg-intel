#!/bin/bash
echo "🚀 Starting Legal LLM Platform (API + Web)..."

# Start API server in background
echo "Starting API server..."
source .venv/bin/activate
cd src
python api.py &
API_PID=$!

# Wait a moment for API to start
sleep 3

# Start web interface
echo "Starting web interface..."
cd ..
streamlit run src/web_app.py &
WEB_PID=$!

echo ""
echo "✅ Legal LLM Platform is running!"
echo "📚 API Server: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🌐 Web Interface: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for interrupt signal
trap "echo 'Stopping servers...'; kill $API_PID $WEB_PID; exit" INT
wait
