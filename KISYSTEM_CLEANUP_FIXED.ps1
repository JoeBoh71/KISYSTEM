# KISYSTEM CLEANUP SCRIPT
# Removes all failed attempts, installs fixes, creates clean baseline
# Author: Joerg Bohne / Claude
# Date: 2025-11-07

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "KISYSTEM CLEANUP - Phase 1: Backup" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Create timestamped backup
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupPath = "C:\KISYSTEM_BACKUP_$timestamp"

Write-Host "Creating backup: $backupPath"
Copy-Item -Path C:\KISYSTEM -Destination $backupPath -Recurse -Force
Write-Host "[OK] Backup created" -ForegroundColor Green
Write-Host ""

# Phase 2: Clear Cache
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "KISYSTEM CLEANUP - Phase 2: Clear Cache" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

$cacheFiles = Get-ChildItem C:\KISYSTEM -Recurse -Directory -Filter "__pycache__"
$cacheCount = $cacheFiles.Count

if ($cacheCount -gt 0) {
    Write-Host "Removing $cacheCount __pycache__ directories..."
    $cacheFiles | Remove-Item -Recurse -Force
    Write-Host "[OK] Cache cleared" -ForegroundColor Green
}
else {
    Write-Host "[SKIP] No cache to clear" -ForegroundColor Yellow
}
Write-Host ""

# Phase 3: Remove Backups and Duplicates
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "KISYSTEM CLEANUP - Phase 3: Remove Backups and Duplicates" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

$patterns = @("*.backup", "*.old", "*.broken", "*.v20.backup")
$removed = 0

foreach ($pattern in $patterns) {
    $files = Get-ChildItem C:\KISYSTEM -Recurse -Filter $pattern -File
    if ($files.Count -gt 0) {
        Write-Host "Removing $($files.Count) files matching $pattern"
        $files | ForEach-Object { 
            Write-Host "  - $($_.Name)" -ForegroundColor DarkGray
            Remove-Item $_.FullName -Force 
        }
        $removed += $files.Count
    }
}

Write-Host "[OK] Removed $removed backup/duplicate files" -ForegroundColor Green
Write-Host ""

# Phase 4: Remove Reports and Temp Files
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "KISYSTEM CLEANUP - Phase 4: Remove Reports and Temp Files" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

$reports = Get-ChildItem C:\KISYSTEM -Filter "report*.*" -File
if ($reports.Count -gt 0) {
    Write-Host "Removing $($reports.Count) nsys report files..."
    $reports | ForEach-Object {
        Write-Host "  - $($_.Name)" -ForegroundColor DarkGray
        Remove-Item $_.FullName -Force
    }
    Write-Host "[OK] Reports removed" -ForegroundColor Green
}
else {
    Write-Host "[SKIP] No reports to remove" -ForegroundColor Yellow
}
Write-Host ""

# Phase 5: Remove Obsolete Files
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "KISYSTEM CLEANUP - Phase 5: Remove Obsolete Files" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

$obsoleteFiles = @(
    "C:\KISYSTEM\core\supervisor_v2.py",
    "C:\KISYSTEM\core\learning_module.py",
    "C:\KISYSTEM\core\cuda_profiler.py",
    "C:\KISYSTEM\core\execution_module.py",
    "C:\KISYSTEM\core\integration_example.py",
    "C:\KISYSTEM\test_cuda.py",
    "C:\KISYSTEM\test_cuda_hard.py",
    "C:\KISYSTEM\test_cuda_timed.py",
    "C:\KISYSTEM\test_simple.py",
    "C:\KISYSTEM\test_v2.py",
    "C:\KISYSTEM\test_end_to_end.py",
    "C:\KISYSTEM\performance_parser.py"
)

$removedObsolete = 0
foreach ($file in $obsoleteFiles) {
    if (Test-Path $file) {
        Write-Host "Removing obsolete: $(Split-Path $file -Leaf)" -ForegroundColor DarkGray
        Remove-Item $file -Force
        $removedObsolete++
    }
}

Write-Host "[OK] Removed $removedObsolete obsolete files" -ForegroundColor Green
Write-Host ""

# Phase 6: Clean Empty Directories
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "KISYSTEM CLEANUP - Phase 6: Clean Empty Directories" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

$emptyDirs = @("models", "sandbox", "scripts", "templates", "tools", "workspace")
foreach ($dir in $emptyDirs) {
    $path = "C:\KISYSTEM\$dir"
    if ((Test-Path $path) -and ((Get-ChildItem $path).Count -eq 0)) {
        Write-Host "Removing empty directory: $dir" -ForegroundColor DarkGray
        Remove-Item $path -Force
    }
}

Write-Host "[OK] Empty directories cleaned" -ForegroundColor Green
Write-Host ""

# Phase 7: Verification
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "KISYSTEM CLEANUP - Phase 7: Verification" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

$coreFiles = (Get-ChildItem C:\KISYSTEM\core -File).Count
$agentFiles = (Get-ChildItem C:\KISYSTEM\agents -File).Count
$totalFiles = (Get-ChildItem C:\KISYSTEM -Recurse -File).Count

Write-Host "Final file count:"
Write-Host "  Core:   $coreFiles files" -ForegroundColor Cyan
Write-Host "  Agents: $agentFiles files" -ForegroundColor Cyan
Write-Host "  Total:  $totalFiles files" -ForegroundColor Cyan
Write-Host ""

Write-Host "Critical files check:" -ForegroundColor Yellow
$criticalFiles = @(
    "C:\KISYSTEM\core\supervisor_v3_optimization.py",
    "C:\KISYSTEM\core\model_selector.py",
    "C:\KISYSTEM\core\learning_module_v2.py",
    "C:\KISYSTEM\agents\builder_agent.py",
    "C:\KISYSTEM\agents\fixer_agent.py",
    "C:\KISYSTEM\agents\cuda_profiler_agent.py",
    "C:\KISYSTEM\agents\search_agent_v2.py",
    "C:\KISYSTEM\test_phase6_optimization.py"
)

$allPresent = $true
foreach ($file in $criticalFiles) {
    $exists = Test-Path $file
    if ($exists) {
        Write-Host "  [OK] $(Split-Path $file -Leaf)" -ForegroundColor Green
    }
    else {
        Write-Host "  [MISS] $(Split-Path $file -Leaf)" -ForegroundColor Red
        $allPresent = $false
    }
}

Write-Host ""

if ($allPresent) {
    Write-Host "[OK] All critical files present" -ForegroundColor Green
}
else {
    Write-Host "[WARN] Some critical files missing!" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "CLEANUP COMPLETE" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Backup location: $backupPath" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Install fixed fixer_agent.py from Claude output"
Write-Host "  2. Clear cache: Remove-Item C:\KISYSTEM\**\__pycache__ -Recurse -Force"
Write-Host "  3. Run test: python C:\KISYSTEM\test_phase6_optimization.py"
Write-Host ""
