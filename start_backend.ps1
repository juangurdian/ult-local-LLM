<# Local AI Beast - Backend Startup Script
   Starts the custom FastAPI backend (not Open WebUI) #>

Write-Host "🔥 Starting Local AI Beast Backend..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Resolve repo root and backend paths
$repoRoot = $PSScriptRoot
$backendDir = Join-Path $repoRoot "backend"
if (-not (Test-Path $backendDir)) {
    Write-Host "❌ Backend directory not found: $backendDir" -ForegroundColor Red
    exit 1
}
Set-Location $repoRoot

# Check virtual environment
$venvActivate = Join-Path $backendDir "venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Host "❌ Virtual environment not found at $venvActivate" -ForegroundColor Red
    Write-Host "💡 Run: cd backend; python -m venv venv; .\\venv\\Scripts\\Activate.ps1; pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "`n📦 Activating virtual environment..." -ForegroundColor Yellow
& $venvActivate

# Check if Ollama is running
Write-Host "`n🦙 Checking Ollama..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop | Out-Null
    Write-Host "  ✅ Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  Ollama not responding. Start Ollama with 'ollama serve' if needed." -ForegroundColor Yellow
}

# Load environment variables from backend/.env if present
$envPath = Join-Path $backendDir ".env"
if (Test-Path $envPath) {
    Write-Host "`n📝 Loading .env configuration..." -ForegroundColor Yellow
    Get-Content $envPath | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# Defaults
if (-not $env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL = "http://localhost:11434" }
if (-not $env:PORT) { $env:PORT = "8001" }
if (-not $env:HOST) { $env:HOST = "0.0.0.0" }

Write-Host "`n🚀 Starting custom FastAPI backend..." -ForegroundColor Green
Write-Host "   Backend URL: http://localhost:$env:PORT" -ForegroundColor Cyan
Write-Host "   Ollama URL:  $env:OLLAMA_BASE_URL" -ForegroundColor Cyan
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

# Start backend with uvicorn using package path
& uvicorn backend.main:app --host $env:HOST --port $env:PORT --reload

