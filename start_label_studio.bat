@echo off
echo ================================
echo   PPE Detection - Label Studio
echo ================================
echo.
echo Starting Label Studio for PPE annotation...
echo.
echo 🌐 Opening in browser: http://localhost:8080
echo 📁 Your project will have 129 images ready to annotate
echo 📋 Each image has an annotation guide to help you
echo.
echo ⏹️  To stop: Press Ctrl+C in this window
echo.

REM Enable local file serving and set the data directory
set LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:\Users\User\Documents\GitHub\ImageClassificationPPE\Image-Classification-for-PPE\data

REM Use virtual environment Label Studio
C:\Users\User\Documents\GitHub\ImageClassificationPPE\Image-Classification-for-PPE\.venv\Scripts\label-studio.exe start

echo.
echo Label Studio stopped.
pause
