@echo off
cd /d "%~dp0"
echo ============================================================
echo   Obesity Risk Prediction Web Application
echo ============================================================
echo   Starting Flask development server...
echo   Server will be available at: http://127.0.0.1:5000
echo   Press CTRL+C to stop the server
echo ============================================================
echo.
python app.py
pause
