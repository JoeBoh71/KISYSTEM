# KISYSTEM COMPLETE SETUP
Write-Host "KISYSTEM Setup Starting..." -ForegroundColor Cyan

# Check Ollama
$ollamaRunning = Get-Process ollama -ErrorAction SilentlyContinue
if (-not $ollamaRunning) {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# PHASE 1: Cleanup
Write-Host "`nPHASE 1: Cleanup" -ForegroundColor Cyan
$modelsToRemove = @("codellama:7b", "llama3.2:3b", "tinyllama:latest", "qwen2.5-coder:7b")
foreach ($model in $modelsToRemove) {
    Write-Host "  Removing: $model" -ForegroundColor Gray
    ollama rm $model 2>$null
}

# PHASE 2: Pull Models
Write-Host "`nPHASE 2: Installing Models (30-45 min)" -ForegroundColor Cyan

Write-Host "`n[1/3] qwen2.5-coder:32b - CODING" -ForegroundColor Yellow
ollama pull qwen2.5-coder:32b

Write-Host "`n[2/3] deepseek-r1:32b - DEBUGGING" -ForegroundColor Yellow
ollama pull deepseek-r1:32b

Write-Host "`n[3/3] mistral:7b - DOCS" -ForegroundColor Yellow
ollama pull mistral:7b

# PHASE 3: Create Memory Structure
Write-Host "`nPHASE 3: Creating Memory on D:\" -ForegroundColor Cyan
$memoryRoot = "D:\KISYSTEM_MEMORY"
$dirs = @(
    "$memoryRoot\conversations\builder",
    "$memoryRoot\conversations\debugger",
    "$memoryRoot\conversations\docs",
    "$memoryRoot\conversations\tester",
    "$memoryRoot\code_context",
    "$memoryRoot\learnings",
    "$memoryRoot\project_state"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -Path $dir -ItemType Directory -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Green
    }
}

# Create README
$readme = @"
# KISYSTEM Memory System
Created: $(Get-Date)

## Models
- Builder: qwen2.5-coder:32b (Code Generation)
- Debugger: deepseek-r1:32b (Deep Reasoning)
- Docs: mistral:7b (Documentation)
- Tester: phi4:latest (Testing)

## Performance Targets
- Daily: 15,000-30,000 lines of code
- Equivalent: 75-300 programmer-days/day
"@

Set-Content -Path "$memoryRoot\README.md" -Value $readme

# PHASE 4: Summary
Write-Host "`nPHASE 4: Verification" -ForegroundColor Cyan
ollama list

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nActive Models:" -ForegroundColor Cyan
Write-Host "  BUILDER  : qwen2.5-coder:32b" -ForegroundColor White
Write-Host "  DEBUGGER : deepseek-r1:32b" -ForegroundColor White
Write-Host "  DOCS     : mistral:7b" -ForegroundColor White
Write-Host "  TESTER   : phi4:latest" -ForegroundColor White
Write-Host "`nMemory: D:\KISYSTEM_MEMORY" -ForegroundColor White
Write-Host "`nTest: ollama run qwen2.5-coder:32b" -ForegroundColor Gray
