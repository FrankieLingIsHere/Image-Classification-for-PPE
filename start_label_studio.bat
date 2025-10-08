@echo off
echo ================================
echo   PPE Detection - Label Studio
echo ================================
echo.
echo Starting Label Studio for PPE annotation...
echo.

REM Enable local file serving and set the data directory
set LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:\Users\User\Documents\GitHub\ImageClassificationPPE\Image-Classification-for-PPE\data

REM Start Label Studio and keep it running in this window
echo Starting Label Studio on localhost:8080...
echo.
echo Wait for "Use the following URL:" message, then manually open:
echo http://localhost:8080
echo.
echo ⏹️  To stop: Press Ctrl+C in this window
echo.

C:\Users\User\Documents\GitHub\ImageClassificationPPE\Image-Classification-for-PPE\.venv\Scripts\label-studio.exe start --host localhost --port 8080
