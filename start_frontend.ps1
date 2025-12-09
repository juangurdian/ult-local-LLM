# Local AI Beast - Frontend Startup Script
# This script starts the Open WebUI frontend (requires Node.js)

Write-Host "🎨 Starting Local AI Beast Frontend..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if Node.js is installed
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found!" -ForegroundColor Red
    Write-Host "`nPlease install Node.js:" -ForegroundColor Yellow
    Write-Host "  1. Visit: https://nodejs.org/" -ForegroundColor Cyan
    Write-Host "  2. Download and install Node.js LTS" -ForegroundColor Cyan
    Write-Host "  3. Restart this terminal after installation" -ForegroundColor Cyan
    exit 1
}

# Navigate to open-webui directory
$webuiDir = Join-Path $PSScriptRoot "open-webui"
if (-not (Test-Path $webuiDir)) {
    Write-Host "❌ Open WebUI directory not found: $webuiDir" -ForegroundColor Red
    exit 1
}

Set-Location $webuiDir

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "`n📦 Installing frontend dependencies..." -ForegroundColor Yellow
    Write-Host "   This may take a few minutes..." -ForegroundColor Cyan
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
}

Write-Host "`n🚀 Starting frontend development server..." -ForegroundColor Green
Write-Host "   Frontend URL: http://localhost:3000" -ForegroundColor Cyan
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

# Start the frontend
npm run dev

