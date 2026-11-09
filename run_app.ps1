# PowerShell script to run the Flask application
Set-Location $PSScriptRoot
Write-Host "Starting Obesity Risk Prediction Web Application..." -ForegroundColor Green
python app.py

