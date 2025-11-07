# KISYSTEM Cache Clear & Run Script
# Clears Python cache before each run to ensure fresh code loads

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "KISYSTEM - Cache Clear & Run" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clear Python cache
Write-Host "[1/3] Clearing Python cache..." -ForegroundColor Yellow

$cacheCount = 0

# Remove __pycache__ directories
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item -Recurse -Force $_.FullName
    $cacheCount++
    Write-Host "  ✓ Removed: $($_.FullName)" -ForegroundColor Gray
}

# Remove .pyc files
Get-ChildItem -Path . -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item -Force $_.FullName
    $cacheCount++
}

if ($cacheCount -eq 0) {
    Write-Host "  ✓ Cache already clean" -ForegroundColor Green
} else {
    Write-Host "  ✓ Removed $cacheCount cache items" -ForegroundColor Green
}

Write-Host ""

# Step 2: Verify Learning DB
Write-Host "[2/3] Checking Learning Module..." -ForegroundColor Yellow

if (Test-Path "learning.db") {
    $dbSize = (Get-Item "learning.db").Length
    Write-Host "  ✓ learning.db exists ($dbSize bytes)" -ForegroundColor Green
} else {
    Write-Host "  → learning.db will be created on first run" -ForegroundColor Gray
}

Write-Host ""

# Step 3: Run test
Write-Host "[3/3] Starting KISYSTEM..." -ForegroundColor Yellow
Write-Host ""

# Run with -B flag (don't write .pyc files)
python -B test_phase6_optimization.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Run complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if learning.db was created
if (Test-Path "learning.db") {
    $dbSize = (Get-Item "learning.db").Length
    Write-Host ""
    Write-Host "✅ learning.db: $dbSize bytes" -ForegroundColor Green
}
