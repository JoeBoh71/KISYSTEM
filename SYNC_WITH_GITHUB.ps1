# ============================================================================
# KISYSTEM - Sync with GitHub (SAFE VERSION)
# ============================================================================
# 
# Was macht das Script:
# 1. Erstellt Backup aller lokalen Files
# 2. Prüft Git-Status
# 3. Gibt dir 3 Optionen:
#    A) Clean Pull (GitHub-Stand übernehmen, lokale Änderungen weg)
#    B) Smart Merge (GitHub + neuere lokale Files)
#    C) Abbrechen
#
# Author: Claude + Jörg Bohne
# Date: 2025-11-12
# ============================================================================

$ErrorActionPreference = "Stop"
$REPO_PATH = "C:\KISYSTEM"
$BACKUP_PATH = "C:\KISYSTEM_BACKUP_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

# Farben für Output
function Write-Success { param($msg) Write-Host "✓ $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "✗ $msg" -ForegroundColor Red }
function Write-Warning { param($msg) Write-Host "⚠ $msg" -ForegroundColor Yellow }
function Write-Info { param($msg) Write-Host "ℹ $msg" -ForegroundColor Cyan }

Write-Host "`n" ("="*70) -ForegroundColor Cyan
Write-Host "KISYSTEM - GitHub Sync Tool" -ForegroundColor Cyan
Write-Host ("="*70) "`n" -ForegroundColor Cyan

# ============================================================================
# STEP 1: CHECKS
# ============================================================================

Write-Info "Step 1: Prüfe Repository..."

if (-not (Test-Path $REPO_PATH)) {
    Write-Error "Repository nicht gefunden: $REPO_PATH"
    exit 1
}

Set-Location $REPO_PATH

if (-not (Test-Path ".git")) {
    Write-Error "Kein Git-Repository in $REPO_PATH"
    exit 1
}

Write-Success "Repository gefunden: $REPO_PATH"

# Git remote prüfen
$remote = git remote -v | Select-String "origin.*github.com/JoeBoh71/KISYSTEM"
if (-not $remote) {
    Write-Error "GitHub remote nicht konfiguriert!"
    Write-Host "Expected: https://github.com/JoeBoh71/KISYSTEM.git"
    exit 1
}

Write-Success "GitHub remote konfiguriert"

# ============================================================================
# STEP 2: STATUS ANALYSE
# ============================================================================

Write-Info "`nStep 2: Analysiere lokale Änderungen..."

# Fetch latest from GitHub
Write-Info "Fetching from GitHub..."
git fetch origin main 2>&1 | Out-Null

# Get status
$status = git status --porcelain

if ($status) {
    Write-Warning "Lokale Änderungen gefunden:"
    $modified = @($status | Select-String "^ M")
    $added = @($status | Select-String "^??")
    $deleted = @($status | Select-String "^ D")
    
    if ($modified.Count -gt 0) {
        Write-Host "`n  Modified Files: $($modified.Count)" -ForegroundColor Yellow
        $modified | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
    }
    
    if ($added.Count -gt 0) {
        Write-Host "`n  New Files: $($added.Count)" -ForegroundColor Yellow
        $added | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
    }
    
    if ($deleted.Count -gt 0) {
        Write-Host "`n  Deleted Files: $($deleted.Count)" -ForegroundColor Yellow
        $deleted | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
    }
} else {
    Write-Success "Keine lokalen Änderungen (sauber)"
}

# Check commits ahead/behind
$ahead = git rev-list --count HEAD ^origin/main 2>$null
$behind = git rev-list --count origin/main ^HEAD 2>$null

Write-Host "`nCommit Status:" -ForegroundColor Cyan
Write-Host "  Lokale Commits (ahead):   $ahead" -ForegroundColor $(if ($ahead -gt 0) { "Yellow" } else { "Green" })
Write-Host "  GitHub Commits (behind):  $behind" -ForegroundColor $(if ($behind -gt 0) { "Yellow" } else { "Green" })

# ============================================================================
# STEP 3: BACKUP ERSTELLEN
# ============================================================================

Write-Info "`nStep 3: Erstelle Backup..."

try {
    Copy-Item -Path $REPO_PATH -Destination $BACKUP_PATH -Recurse -Force
    
    # .git Ordner aus Backup entfernen (spart Platz)
    if (Test-Path "$BACKUP_PATH\.git") {
        Remove-Item -Path "$BACKUP_PATH\.git" -Recurse -Force
    }
    
    Write-Success "Backup erstellt: $BACKUP_PATH"
    
    # Zeige Backup-Größe
    $backupSize = (Get-ChildItem $BACKUP_PATH -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "  Größe: $([math]::Round($backupSize, 2)) MB" -ForegroundColor Gray
    
} catch {
    Write-Error "Backup fehlgeschlagen: $_"
    exit 1
}

# ============================================================================
# STEP 4: OPTIONEN ANZEIGEN
# ============================================================================

Write-Host "`n" ("="*70) -ForegroundColor Cyan
Write-Host "SYNC OPTIONEN" -ForegroundColor Cyan
Write-Host ("="*70) "`n" -ForegroundColor Cyan

Write-Host "[A] CLEAN PULL (⚠️ RADIKAL)" -ForegroundColor Red
Write-Host "    • Alle lokalen Änderungen verwerfen" -ForegroundColor Gray
Write-Host "    • Git reset --hard origin/main" -ForegroundColor Gray
Write-Host "    • Git clean -fdx (löscht untracked files)" -ForegroundColor Gray
Write-Host "    • ⚠️ Nur sicher wenn Backup da ist!`n" -ForegroundColor Red

Write-Host "[B] SMART MERGE (✅ EMPFOHLEN)" -ForegroundColor Green
Write-Host "    • GitHub-Stand als Basis" -ForegroundColor Gray
Write-Host "    • Neuere lokale Files bleiben" -ForegroundColor Gray
Write-Host "    • Vergleich über Last-Modified" -ForegroundColor Gray
Write-Host "    • Zeigt dir Konflikte vor Overwrite`n" -ForegroundColor Gray

Write-Host "[C] ABBRECHEN" -ForegroundColor Yellow
Write-Host "    • Nichts ändern, Backup bleibt`n" -ForegroundColor Gray

Write-Host "[D] LOKALE ÄNDERUNGEN COMMITTEN & PUSHEN" -ForegroundColor Cyan
Write-Host "    • Committet deine lokalen Änderungen" -ForegroundColor Gray
Write-Host "    • Pusht zu GitHub (überschreibt GitHub)" -ForegroundColor Gray
Write-Host "    • Macht lokalen Stand zur Wahrheit`n" -ForegroundColor Gray

# ============================================================================
# STEP 5: USER INPUT
# ============================================================================

do {
    $choice = Read-Host "Deine Wahl (A/B/C/D)"
    $choice = $choice.ToUpper()
} while ($choice -notin @("A", "B", "C", "D"))

switch ($choice) {
    "A" {
        # OPTION A: CLEAN PULL
        Write-Warning "`n⚠️ CLEAN PULL - Alle lokalen Änderungen gehen verloren!"
        $confirm = Read-Host "Wirklich fortfahren? (yes/no)"
        
        if ($confirm -ne "yes") {
            Write-Info "Abgebrochen."
            exit 0
        }
        
        Write-Info "`nFühre Clean Pull aus..."
        
        try {
            # Reset to GitHub state
            git reset --hard origin/main
            Write-Success "Git reset complete"
            
            # Clean untracked files
            git clean -fdx
            Write-Success "Git clean complete"
            
            Write-Success "`n✅ Repository auf GitHub-Stand zurückgesetzt!"
            Write-Host "Backup liegt in: $BACKUP_PATH" -ForegroundColor Gray
            
        } catch {
            Write-Error "Clean Pull fehlgeschlagen: $_"
            Write-Warning "Restore aus Backup: Copy-Item '$BACKUP_PATH\*' -Destination '$REPO_PATH' -Recurse -Force"
            exit 1
        }
    }
    
    "B" {
        # OPTION B: SMART MERGE
        Write-Info "`n🧠 Smart Merge wird ausgeführt..."
        
        try {
            # 1. Hole GitHub-Stand in temporären Branch
            Write-Info "Hole GitHub-Stand..."
            git fetch origin main
            
            # 2. Erstelle Liste aller Files im Repo
            $repoFiles = Get-ChildItem -Path $REPO_PATH -Recurse -File | 
                Where-Object { $_.FullName -notlike "*\.git\*" -and $_.FullName -notlike "*\__pycache__\*" }
            
            # 3. Vergleiche jede Datei mit GitHub-Version
            Write-Info "Vergleiche Files..."
            
            $toUpdate = @()
            $toKeep = @()
            $conflicts = @()
            
            foreach ($file in $repoFiles) {
                $relativePath = $file.FullName.Substring($REPO_PATH.Length + 1).Replace('\', '/')
                
                # Hole Git-Status für diese Datei
                $gitStatus = git status --porcelain $relativePath 2>$null
                
                if (-not $gitStatus) {
                    # File unverändert - überspringen
                    continue
                }
                
                if ($gitStatus -match "^\?\?") {
                    # Neue Datei (nicht in Git) - behalten
                    $toKeep += $file
                } elseif ($gitStatus -match "^ M") {
                    # Modified - prüfe Datum
                    $localModified = $file.LastWriteTime
                    
                    # Hole GitHub-Version Datum (über git log)
                    $githubDate = git log -1 --format="%ai" origin/main -- $relativePath 2>$null
                    
                    if ($githubDate) {
                        $githubModified = [datetime]::Parse($githubDate)
                        
                        if ($localModified -gt $githubModified) {
                            # Lokal neuer - behalten
                            $toKeep += [PSCustomObject]@{
                                Path = $relativePath
                                LocalDate = $localModified
                                GitHubDate = $githubModified
                                Decision = "KEEP (newer)"
                            }
                        } else {
                            # GitHub neuer - updaten
                            $toUpdate += [PSCustomObject]@{
                                Path = $relativePath
                                LocalDate = $localModified
                                GitHubDate = $githubModified
                                Decision = "UPDATE (older)"
                            }
                        }
                    } else {
                        # Keine GitHub-Version - neue Datei, behalten
                        $toKeep += $file
                    }
                }
            }
            
            # 4. Zeige Zusammenfassung
            Write-Host "`n" ("="*70) -ForegroundColor Cyan
            Write-Host "SMART MERGE ANALYSE" -ForegroundColor Cyan
            Write-Host ("="*70) "`n" -ForegroundColor Cyan
            
            if ($toKeep.Count -gt 0) {
                Write-Success "Lokale Files behalten: $($toKeep.Count)"
                $toKeep | ForEach-Object {
                    if ($_ -is [PSCustomObject]) {
                        Write-Host "  ✓ $($_.Path) (lokal: $($_.LocalDate.ToString('yyyy-MM-dd HH:mm')))" -ForegroundColor Green
                    } else {
                        Write-Host "  ✓ $($_.Name) (neu)" -ForegroundColor Green
                    }
                }
            }
            
            if ($toUpdate.Count -gt 0) {
                Write-Warning "`nFiles von GitHub updaten: $($toUpdate.Count)"
                $toUpdate | ForEach-Object {
                    Write-Host "  ⬇ $($_.Path) (GitHub: $($_.GitHubDate.ToString('yyyy-MM-dd HH:mm')))" -ForegroundColor Yellow
                }
            }
            
            if ($toKeep.Count -eq 0 -and $toUpdate.Count -eq 0) {
                Write-Success "Keine Änderungen nötig - alles aktuell!"
                exit 0
            }
            
            # 5. Bestätigung
            Write-Host "`n"
            $confirm = Read-Host "Smart Merge ausführen? (yes/no)"
            
            if ($confirm -ne "yes") {
                Write-Info "Abgebrochen."
                exit 0
            }
            
            # 6. Merge ausführen
            Write-Info "`nFühre Merge aus..."
            
            # Stash lokale Änderungen
            git stash push -m "Auto-stash for smart merge"
            
            # Pull GitHub
            git pull origin main --ff-only
            
            # Stash pop (lokale Änderungen zurück)
            git stash pop
            
            Write-Success "`n✅ Smart Merge abgeschlossen!"
            Write-Host "Backup liegt in: $BACKUP_PATH" -ForegroundColor Gray
            
        } catch {
            Write-Error "Smart Merge fehlgeschlagen: $_"
            Write-Warning "Restore aus Backup: Copy-Item '$BACKUP_PATH\*' -Destination '$REPO_PATH' -Recurse -Force"
            exit 1
        }
    }
    
    "C" {
        # OPTION C: ABBRECHEN
        Write-Info "Abgebrochen - keine Änderungen."
        Write-Host "Backup liegt in: $BACKUP_PATH" -ForegroundColor Gray
        Write-Warning "Backup später löschen: Remove-Item '$BACKUP_PATH' -Recurse -Force"
        exit 0
    }
    
    "D" {
        # OPTION D: COMMIT & PUSH
        Write-Warning "`n⚠️ COMMIT & PUSH - Überschreibt GitHub mit deinem lokalen Stand!"
        $confirm = Read-Host "Wirklich fortfahren? (yes/no)"
        
        if ($confirm -ne "yes") {
            Write-Info "Abgebrochen."
            exit 0
        }
        
        Write-Info "`nCommitte lokale Änderungen..."
        
        try {
            # Add all changes
            git add -A
            
            # Commit
            $commitMsg = "Update to RUN 37.2 - Production Validated (Auto-commit from SYNC script)"
            git commit -m $commitMsg
            
            Write-Success "Commit erstellt: $commitMsg"
            
            # Push
            Write-Info "Pushe zu GitHub..."
            git push origin main --force
            
            Write-Success "`n✅ Lokale Änderungen zu GitHub gepusht!"
            Write-Host "GitHub ist jetzt auf deinem lokalen Stand." -ForegroundColor Gray
            
        } catch {
            Write-Error "Commit/Push fehlgeschlagen: $_"
            Write-Warning "Restore aus Backup: Copy-Item '$BACKUP_PATH\*' -Destination '$REPO_PATH' -Recurse -Force"
            exit 1
        }
    }
}

# ============================================================================
# STEP 6: CLEANUP
# ============================================================================

Write-Host "`n" ("="*70) -ForegroundColor Cyan
Write-Host "ABSCHLUSS" -ForegroundColor Cyan
Write-Host ("="*70) "`n" -ForegroundColor Cyan

# Zeige finalen Status
Write-Info "Finaler Git-Status:"
git status --short

Write-Host "`n"
Write-Success "✅ Sync abgeschlossen!"
Write-Host "Backup: $BACKUP_PATH" -ForegroundColor Gray
Write-Host "`nBackup später löschen:" -ForegroundColor Gray
Write-Host "  Remove-Item '$BACKUP_PATH' -Recurse -Force`n" -ForegroundColor DarkGray

# Optional: Backup nach 7 Tagen auto-löschen
$deleteDate = (Get-Date).AddDays(7).ToString('yyyy-MM-dd')
Write-Host "TIPP: Backup wird automatisch irrelevant nach ca. 7 Tagen ($deleteDate)" -ForegroundColor Gray
