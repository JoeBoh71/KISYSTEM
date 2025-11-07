# KISYSTEM - QUICK START

**5-Minuten Setup & Test**

---

## SCHRITT 1: Install (2min)

```powershell
cd C:\KISYSTEM

# Backup
xcopy /E /I /Y . ..\KISYSTEM_BACKUP_20251107

# Copy Files (from downloaded KISYSTEM_FIXED package)
xcopy /Y path\to\KISYSTEM_FIXED\core\*.py core\
xcopy /Y path\to\KISYSTEM_FIXED\agents\*.py agents\
copy path\to\KISYSTEM_FIXED\test_system.py .
```

---

## SCHRITT 2: Test Imports (30sec)

```powershell
python test_system.py --minimal
```

**Erwartung:**
```
✓ supervisor_v3_optimization import OK
✓ builder_agent import OK
✓ fixer_agent import OK
✓ cuda_profiler_agent import OK
✅ ALL IMPORTS OK
```

---

## SCHRITT 3: Full Test (2min)

```powershell
python test_system.py
```

**Erwartung:**
```
[BuilderAgent] ✓ Auto-added 2 includes
[CUDAProfiler] ✓ Compilation successful
Status: completed
✅ TEST PASSED
```

---

## SCHRITT 4: Git Setup (1min)

```bash
cd C:\KISYSTEM
git init
git add .
git commit -m "KISYSTEM v1.0 - MVP Functional"
git remote add origin https://github.com/YOUR_USERNAME/kisystem.git
git push -u origin main
```

---

## FERTIG!

**Nächste Claude Session:**
```
"KISYSTEM Problem X, Repo: github.com/YOUR_USERNAME/kisystem"
```

**= Kein File-Upload mehr**

---

## Troubleshooting

**Import Error?**
```powershell
# Check files copied:
dir core\supervisor_v3.py
dir agents\builder_agent.py
dir agents\fixer_agent.py
```

**Test fails?**
```powershell
# Check Ollama running:
ollama list

# Check CUDA:
nvcc --version
```

**Still broken?**
Poste Error + Repo Link in Claude Chat.

---

**Ende Quick Start**
