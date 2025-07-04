#!/bin/bash
echo "🚀 Starting Legal LLM API Server..."
source .venv/bin/activate
cd src
python api.py
