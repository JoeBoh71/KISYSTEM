# KISYSTEM Model Setup - Optimized for qwen2.5-coder:32b

Write-Host "=== KISYSTEM Model Setup ===" -ForegroundColor Cyan
Write-Host ""

# Pull 32B Primary Model
Write-Host "[1/3] Pulling qwen2.5-coder:32b (this will take 10-15 minutes)..." -ForegroundColor Yellow
ollama pull qwen2.5-coder:32b

Write-Host ""
Write-Host "[2/3] Cleaning up small models..." -ForegroundColor Yellow

# Remove small/redundant models
$modelsToRemove = @(
    "codellama:7b",
    "llama3.2:3b",
    "tinyllama:latest",
    "qwen2.5-coder:7b"
)

foreach ($model in $modelsToRemove) {
    Write-Host "  Removing $model..." -ForegroundColor Gray
    ollama rm $model 2>$null
}

Write-Host ""
Write-Host "[3/3] Final model inventory:" -ForegroundColor Yellow
ollama list

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Active Models:" -ForegroundColor Cyan
Write-Host "  - qwen2.5-coder:32b (Primary - Builder/Fixer)" -ForegroundColor White
Write-Host "  - phi4:latest        (Backup - Tester/Search)" -ForegroundColor White
Write-Host ""
Write-Host "Expected Performance:" -ForegroundColor Cyan
Write-Host "  - 32B: 8-15 tokens/s (GPU+RAM offload)" -ForegroundColor White
Write-Host "  - 24h runtime: ~15,000-30,000 lines of code" -ForegroundColor White
Write-Host "  - Equivalent: 75-300 programmer-days/day" -ForegroundColor White
Write-Host ""
Write-Host "Next: Update KISYSTEM config to use qwen2.5-coder:32b" -ForegroundColor Yellow