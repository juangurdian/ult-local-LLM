<# Local AI Beast - ComfyUI Startup Script
   Starts ComfyUI server for image generation #>

Write-Host "🎨 Starting ComfyUI for Image Generation..." -ForegroundColor Magenta
Write-Host "===========================================" -ForegroundColor Magenta

# Check if ComfyUI directory exists
$comfyuiPath = "C:\Users\jcgus\Documents\ComfyUI"
if (-not (Test-Path $comfyuiPath)) {
    Write-Host "❌ ComfyUI not found at: $comfyuiPath" -ForegroundColor Red
    Write-Host "💡 Run: cd C:\Users\jcgus\Documents; git clone https://github.com/comfyanonymous/ComfyUI.git" -ForegroundColor Yellow
    exit 1
}

Set-Location $comfyuiPath

# Check virtual environment
$venvActivate = Join-Path $comfyuiPath "venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Host "❌ ComfyUI virtual environment not found" -ForegroundColor Red
    Write-Host "💡 Run: python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "`n📦 Activating ComfyUI virtual environment..." -ForegroundColor Yellow
& $venvActivate

# Check for models
$modelsPath = Join-Path $comfyuiPath "models\checkpoints"
if (-not (Test-Path $modelsPath)) {
    New-Item -ItemType Directory -Path $modelsPath -Force | Out-Null
}

$models = Get-ChildItem -Path $modelsPath -Filter "*.safetensors" -ErrorAction SilentlyContinue
if ($models.Count -eq 0) {
    Write-Host "`n⚠️  No image models found in models/checkpoints/" -ForegroundColor Yellow
    Write-Host "💡 Download SDXL from: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0" -ForegroundColor Yellow
    Write-Host "   Place sd_xl_base_1.0.safetensors in: $modelsPath" -ForegroundColor Yellow
} else {
    Write-Host "`n✅ Found $($models.Count) model(s):" -ForegroundColor Green
    foreach ($model in $models) {
        Write-Host "   - $($model.Name)" -ForegroundColor Cyan
    }
}

# Default port
$port = 8188
if ($env:COMFYUI_PORT) {
    $port = $env:COMFYUI_PORT
}

Write-Host "`n🚀 Starting ComfyUI server..." -ForegroundColor Green
Write-Host "   Server URL: http://127.0.0.1:$port" -ForegroundColor Cyan
Write-Host "   Web UI: http://127.0.0.1:$port" -ForegroundColor Cyan
Write-Host "`n===========================================" -ForegroundColor Magenta
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "==========================================`n" -ForegroundColor Magenta

# Start ComfyUI
python main.py --port $port

