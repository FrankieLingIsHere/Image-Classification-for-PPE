@echo off
REM PPE Detection UI Launcher for Windows

echo 🚀 Starting PPE Detection Web Interface...
echo 📍 Make sure you're in the project root directory
echo.

REM Change to script directory
cd /d "%~dp0"

REM Activate virtual environment and run Streamlit
.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost

echo.
echo 🌐 Access the interface at: http://localhost:8501
echo 🛑 Press Ctrl+C to stop the server
pause