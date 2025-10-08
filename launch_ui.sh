#!/bin/bash
# PPE Detection UI Launcher

echo "🚀 Starting PPE Detection Web Interface..."
echo "📍 Make sure you're in the project root directory"
echo ""

# Activate virtual environment and run Streamlit
cd "$(dirname "$0")"
.venv/Scripts/python -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost

echo ""
echo "🌐 Access the interface at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"