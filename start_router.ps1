# Local AI Beast - Router Service Startup Script
# This script starts the intelligent model router service

Write-Host "🧠 Starting Local AI Beast Router..." -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Navigate to backend directory
$backendDir = Join-Path $PSScriptRoot "backend"
if (-not (Test-Path $backendDir)) {
    Write-Host "❌ Backend directory not found: $backendDir" -ForegroundColor Red
    exit 1
}

Set-Location $backendDir

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "❌ Virtual environment not found. Please run setup first." -ForegroundColor Red
    exit 1
}

# Check if Ollama is running
Write-Host "`n🦙 Checking Ollama..." -ForegroundColor Yellow
try {
    $ollamaCheck = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    Write-Host "   ✅ Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  Ollama not responding. Starting Ollama..." -ForegroundColor Yellow
    Write-Host "   💡 Run 'ollama serve' in another terminal if needed" -ForegroundColor Cyan
}

# Load environment variables from .env if it exists
if (Test-Path ".env") {
    Write-Host "`n📝 Loading .env configuration..." -ForegroundColor Yellow
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

Write-Host "`n🚀 Starting Router Service..." -ForegroundColor Green
Write-Host "   API URL: http://localhost:8001" -ForegroundColor Cyan
Write-Host "   Docs:    http://localhost:8001/docs" -ForegroundColor Cyan
Write-Host "`nEndpoints:" -ForegroundColor White
Write-Host "   POST /v1/chat/completions  - Chat with auto-routing" -ForegroundColor White
Write-Host "   POST /v1/routing/analyze   - Analyze query routing" -ForegroundColor White
Write-Host "   GET  /v1/models           - List available models" -ForegroundColor White
Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "=====================================`n" -ForegroundColor Cyan

# Start the router service
& .\venv\Scripts\python.exe main.py

