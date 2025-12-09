# Local AI Beast - Setup Test Script
# Tests all components to ensure everything is working

Write-Host "🧪 Testing Local AI Beast Setup..." -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

$allTestsPass = $true
$testResults = @()

# Test 1: Ollama Models
Write-Host "`n1️⃣ Testing Ollama Models..." -ForegroundColor Yellow
try {
    $models = ollama list 2>$null | Where-Object { $_ -match "^\w+.*\d.*GB" } | Measure-Object
    if ($models.Count -ge 6) {
        Write-Host "   ✅ Found $($models.Count) models" -ForegroundColor Green
        $testResults += @{Test="Ollama Models"; Status="PASS"; Details="$($models.Count) models available"}
    } else {
        Write-Host "   ❌ Only $($models.Count) models found (need 6+)" -ForegroundColor Red
        $testResults += @{Test="Ollama Models"; Status="FAIL"; Details="Only $($models.Count) models"}
        $allTestsPass = $false
    }
} catch {
    Write-Host "   ❌ Ollama not responding" -ForegroundColor Red
    $testResults += @{Test="Ollama Models"; Status="FAIL"; Details="Ollama not running"}
    $allTestsPass = $false
}

# Test 2: Ollama Server
Write-Host "`n2️⃣ Testing Ollama API..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "   ✅ Ollama API responding" -ForegroundColor Green
    $testResults += @{Test="Ollama API"; Status="PASS"; Details="API accessible"}
} catch {
    Write-Host "   ❌ Ollama API not responding" -ForegroundColor Red
    $testResults += @{Test="Ollama API"; Status="FAIL"; Details="API unreachable"}
    $allTestsPass = $false
}

# Test 3: Python Backend
Write-Host "`n3️⃣ Testing Python Backend..." -ForegroundColor Yellow
try {
    $pythonVersion = & python --version 2>$null
    if ($pythonVersion -match "Python 3\.(1[1-9]|2[0-9])") {
        Write-Host "   ✅ Python available: $pythonVersion" -ForegroundColor Green
        $testResults += @{Test="Python Backend"; Status="PASS"; Details=$pythonVersion}
    } else {
        Write-Host "   ❌ Python version issue: $pythonVersion" -ForegroundColor Red
        $testResults += @{Test="Python Backend"; Status="FAIL"; Details=$pythonVersion}
        $allTestsPass = $false
    }
} catch {
    Write-Host "   ❌ Python not found" -ForegroundColor Red
    $testResults += @{Test="Python Backend"; Status="FAIL"; Details="Python not found"}
    $allTestsPass = $false
}

# Test 4: Node.js Frontend
Write-Host "`n4️⃣ Testing Node.js Frontend..." -ForegroundColor Yellow
$nodePath = "$PSScriptRoot\open-webui\nodejs-v22\node-v22.12.0-win-x64"
$env:Path = "$nodePath;" + $env:Path

try {
    $nodeVersion = & node --version 2>$null
    if ($nodeVersion -match "v2[02]") {
        Write-Host "   ✅ Node.js available: $nodeVersion" -ForegroundColor Green
        $testResults += @{Test="Node.js Frontend"; Status="PASS"; Details=$nodeVersion}
    } else {
        Write-Host "   ❌ Node.js version issue: $nodeVersion" -ForegroundColor Red
        $testResults += @{Test="Node.js Frontend"; Status="FAIL"; Details=$nodeVersion}
        $allTestsPass = $false
    }
} catch {
    Write-Host "   ❌ Node.js not found" -ForegroundColor Red
    $testResults += @{Test="Node.js Frontend"; Status="FAIL"; Details="Node.js not found"}
    $allTestsPass = $false
}

# Test 5: Open WebUI Files
Write-Host "`n5️⃣ Testing Open WebUI Setup..." -ForegroundColor Yellow
$checks = @(
    @{Path="open-webui\backend\venv\Scripts\Activate.ps1"; Name="Python virtual environment"},
    @{Path="open-webui\backend\.env"; Name="Backend configuration"},
    @{Path="open-webui\package.json"; Name="Frontend package.json"},
    @{Path="open-webui\node_modules"; Name="Frontend dependencies"}
)

foreach ($check in $checks) {
    if (Test-Path $check.Path) {
        Write-Host "   ✅ $($check.Name) found" -ForegroundColor Green
        $testResults += @{Test=$check.Name; Status="PASS"; Details="File exists"}
    } else {
        Write-Host "   ❌ $($check.Name) missing" -ForegroundColor Red
        $testResults += @{Test=$check.Name; Status="FAIL"; Details="File missing"}
        $allTestsPass = $false
    }
}

# Test 6: Basic Chat Functionality
Write-Host "`n6️⃣ Testing Chat Functionality..." -ForegroundColor Yellow
try {
    $testMessage = "Hello, please respond with exactly: TEST_OK"
    $response = ollama run qwen3:4b $testMessage --format raw 2>$null | Select-Object -First 1
    if ($response -match "TEST_OK") {
        Write-Host "   ✅ Basic chat working" -ForegroundColor Green
        $testResults += @{Test="Chat Functionality"; Status="PASS"; Details="qwen3:4b responding"}
    } else {
        Write-Host "   ❌ Chat test failed" -ForegroundColor Red
        $testResults += @{Test="Chat Functionality"; Status="FAIL"; Details="Unexpected response"}
        $allTestsPass = $false
    }
} catch {
    Write-Host "   ❌ Chat test error: $($_.Exception.Message)" -ForegroundColor Red
    $testResults += @{Test="Chat Functionality"; Status="FAIL"; Details=$_.Exception.Message}
    $allTestsPass = $false
}

# Summary
Write-Host "`n==================================" -ForegroundColor Cyan
if ($allTestsPass) {
    Write-Host "🎉 ALL TESTS PASSED!" -ForegroundColor Green
    Write-Host "`n🚀 Your Local AI Beast is ready!" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "  1. Start Backend: .\start_backend.ps1" -ForegroundColor White
    Write-Host "  2. Start Frontend: .\start_frontend.ps1" -ForegroundColor White
    Write-Host "  3. Open browser: http://localhost:3000" -ForegroundColor White
} else {
    Write-Host "❌ SOME TESTS FAILED" -ForegroundColor Red
    Write-Host "`nFailed tests:" -ForegroundColor Yellow
    $testResults | Where-Object { $_.Status -eq "FAIL" } | ForEach-Object {
        Write-Host "  - $($_.Test): $($_.Details)" -ForegroundColor Red
    }
}

Write-Host "`n==================================" -ForegroundColor Cyan
