# KISYSTEM v3.9.1 - Diagnose & Fix
# Findet und behebt alle Probleme

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "KISYSTEM v3.9.1 - DIAGNOSE & FIX"
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

cd C:\KISYSTEM

# SCHRITT 1: Zeige welcher Agent fehlschlägt
Write-Host "[1/5] Analysiere Integrity Report..." -ForegroundColor Green
if (Test-Path "Logs\integrity_report.json") {
    $report = Get-Content "Logs\integrity_report.json" | ConvertFrom-Json
    Write-Host ""
    foreach ($entry in $report) {
        if ($entry.ok -eq $false) {
            Write-Host "  ✗ FEHLER: $($entry.module)" -ForegroundColor Red
            Write-Host "    Error: $($entry.error)" -ForegroundColor Yellow
            Write-Host ""
        } else {
            Write-Host "  ✓ OK: $($entry.module)" -ForegroundColor Green
        }
    }
}

# SCHRITT 2: Fixe SyntaxWarnings
Write-Host ""
Write-Host "[2/5] Fixe SyntaxWarnings in review_agent_v2.py..." -ForegroundColor Green
$file = "agents\review_agent_v2.py"
if (Test-Path $file) {
    $content = Get-Content $file -Raw -Encoding UTF8
    $content = $content -replace 'Memory: D:\\AGENT_MEMORY', 'Memory: D:\\\\AGENT_MEMORY'
    [System.IO.File]::WriteAllText((Resolve-Path $file).Path, $content, [System.Text.UTF8Encoding]::new($false))
    Write-Host "  ✓ review_agent_v2.py fixed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[3/5] Fixe SyntaxWarnings in docs_agent_v2.py..." -ForegroundColor Green
$file = "agents\docs_agent_v2.py"
if (Test-Path $file) {
    $content = Get-Content $file -Raw -Encoding UTF8
    $content = $content -replace 'Memory: D:\\AGENT_MEMORY', 'Memory: D:\\\\AGENT_MEMORY'
    [System.IO.File]::WriteAllText((Resolve-Path $file).Path, $content, [System.Text.UTF8Encoding]::new($false))
    Write-Host "  ✓ docs_agent_v2.py fixed" -ForegroundColor Yellow
}

# SCHRITT 3: Checke ob performance_parser existiert
Write-Host ""
Write-Host "[4/5] Checke performance_parser..." -ForegroundColor Green
if (-not (Test-Path "core\performance_parser.py")) {
    Write-Host "  ⚠ performance_parser.py fehlt in core/" -ForegroundColor Yellow
    Write-Host "  → CUDAProfiler Warning ist OK (optionales Modul)" -ForegroundColor Gray
} else {
    Write-Host "  ✓ performance_parser.py existiert" -ForegroundColor Green
}

# SCHRITT 4: Cache löschen und re-test
Write-Host ""
Write-Host "[5/5] Cache löschen und re-test..." -ForegroundColor Green
Remove-Item -Recurse -Force agents\__pycache__ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force core\__pycache__ -ErrorAction SilentlyContinue
Write-Host "  ✓ Cache gelöscht" -ForegroundColor Yellow

Write-Host ""
Write-Host "Re-validiere..." -ForegroundColor Cyan
python -B agents\integrity_agent.py

Write-Host ""
Write-Host "API Key Check..." -ForegroundColor Cyan
python -c "import os; print('API Key:', 'SET ✓' if os.environ.get('BRAVE_API_KEY') else 'NOT SET ✗')"

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "DIAGNOSE ABGESCHLOSSEN"
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Erwartetes Ergebnis:" -ForegroundColor Yellow
Write-Host "  [IntegrityAgent] OK=8 FAIL=0" -ForegroundColor Green
Write-Host "  Keine SyntaxWarnings" -ForegroundColor Green
Write-Host "  CUDAProfiler Warning OK (optionales Modul)" -ForegroundColor Gray
Write-Host ""
