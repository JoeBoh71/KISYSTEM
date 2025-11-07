# ============================================================================
# KISYSTEM AGENTS UPDATE
# ============================================================================
# Updates agents in C:\KISYSTEM\agents with new LLM-integrated versions
# Uses existing D:\AGENT_MEMORY structure
# ============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   KISYSTEM AGENTS UPDATE - V2.0                      " -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check if downloads directory exists
$downloadsDir = "$env:USERPROFILE\Downloads"
$agentsDir = "C:\KISYSTEM\agents"

if (-not (Test-Path $agentsDir)) {
    Write-Host "Creating agents directory..." -ForegroundColor Yellow
    New-Item -Path $agentsDir -ItemType Directory -Force | Out-Null
}

# Agent files to update
$agentFiles = @(
    "builder_agent_v2.py",
    "fixer_agent_v2.py", 
    "tester_agent_v2.py",
    "docs_agent_v2.py"
)

Write-Host "Looking for agent files in Downloads..." -ForegroundColor Yellow
Write-Host ""

$foundFiles = @()
foreach ($file in $agentFiles) {
    $sourcePath = Join-Path $downloadsDir $file
    if (Test-Path $sourcePath) {
        $foundFiles += $file
        Write-Host "  ✓ Found: $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Missing: $file" -ForegroundColor Red
    }
}

if ($foundFiles.Count -eq 0) {
    Write-Host ""
    Write-Host "No agent files found in Downloads!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please download the agent files from Claude first:" -ForegroundColor Yellow
    Write-Host "  - builder_agent_v2.py" -ForegroundColor White
    Write-Host "  - fixer_agent_v2.py" -ForegroundColor White
    Write-Host "  - tester_agent_v2.py" -ForegroundColor White
    Write-Host "  - docs_agent_v2.py" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "Copying agents to C:\KISYSTEM\agents..." -ForegroundColor Yellow

foreach ($file in $foundFiles) {
    $sourcePath = Join-Path $downloadsDir $file
    $destPath = Join-Path $agentsDir $file
    
    # Backup old version if exists
    if (Test-Path $destPath) {
        $backupPath = $destPath + ".backup"
        Copy-Item $destPath $backupPath -Force
        Write-Host "  Backed up: $file.backup" -ForegroundColor Gray
    }
    
    # Copy new version
    Copy-Item $sourcePath $destPath -Force
    Write-Host "  ✓ Updated: $file" -ForegroundColor Green
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   Verifying D:\AGENT_MEMORY Structure               " -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$memoryRoot = "D:\AGENT_MEMORY"

# Required subdirectories
$requiredDirs = @(
    "conversations\builder",
    "conversations\debugger",
    "conversations\docs",
    "conversations\tester"
)

foreach ($dir in $requiredDirs) {
    $fullPath = Join-Path $memoryRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -Path $fullPath -ItemType Directory -Force | Out-Null
        Write-Host "  ✓ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  ✓ Exists: $dir" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   Testing Agent Configuration                        " -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Test if Ollama models are ready
Write-Host "Checking Ollama models..." -ForegroundColor Yellow
$models = ollama list | Out-String

$requiredModels = @{
    "qwen2.5-coder:32b" = "Builder"
    "deepseek-r1:32b" = "Fixer"
    "phi4:latest" = "Tester"
    "mistral:7b" = "Docs"
}

foreach ($model in $requiredModels.Keys) {
    if ($models -match [regex]::Escape($model)) {
        Write-Host "  ✓ $($requiredModels[$model]): $model" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $($requiredModels[$model]): $model (missing)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "   ✓ AGENTS UPDATE COMPLETE                           " -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""

Write-Host "Agent Configuration:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  🔧 BUILDER   : builder_agent_v2.py  → qwen2.5-coder:32b" -ForegroundColor White
Write-Host "  🐛 FIXER     : fixer_agent_v2.py    → deepseek-r1:32b" -ForegroundColor White
Write-Host "  🧪 TESTER    : tester_agent_v2.py   → phi4:latest" -ForegroundColor White
Write-Host "  📝 DOCS      : docs_agent_v2.py     → mistral:7b" -ForegroundColor White
Write-Host ""
Write-Host "Memory System:" -ForegroundColor Cyan
Write-Host "  📂 Location  : D:\AGENT_MEMORY" -ForegroundColor White
Write-Host "  💾 Database  : D:\AGENT_MEMORY\memory.db (SQLite)" -ForegroundColor White
Write-Host "  📊 Workspace : D:\U3DAW" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Update C:\KISYSTEM\core\supervisor_v2.py to import new agents" -ForegroundColor Gray
Write-Host "  2. Test with: python -c 'from agents.builder_agent_v2 import BuilderAgent'" -ForegroundColor Gray
Write-Host "  3. Run KISYSTEM supervisor" -ForegroundColor Gray
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
