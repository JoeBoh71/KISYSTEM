# KISYSTEM Learning Module V2
## Context-Aware Learning mit Multi-Faktor Confidence

**Datum:** 2025-11-06  
**Author:** Jörg Bohne  
**Status:** Production Ready

---

## 🎯 Was ist neu?

### **Vorher (V1):**
- Binäres Learning ("Fehler nur 1x")
- Nur String-Similarity (70% Threshold)
- Kein Environment-Context
- False Positives bei Context-Änderungen

### **Jetzt (V2):**
- **Context-Aware:** Vollständiger Environment-Context (OS, Language, Hardware, etc.)
- **Multi-Faktor Scoring:** 40% Text, 30% Context, 20% History, 10% Recency
- **Granulares Learning:** Enforce/Suggest/Consider basierend auf Confidence
- **Automatische Klassifikation:** Complexity (simple/medium/complex), Domain (audio_dsp/cuda_kernel/web/system)

---

## 📊 Confidence-Berechnung

### **Faktoren:**

| Faktor | Gewicht | Beschreibung |
|--------|---------|--------------|
| **Text-Similarity** | 40% | String-Match des Fehlers (difflib) |
| **Context-Match** | 30% | Language, Version, OS, Hardware |
| **Success-History** | 20% | Bewährte Lösungen (Bayesian Smoothing) |
| **Recency** | 10% | Alter der Lösung (Linear Decay 6 Monate) |

### **Context Sub-Gewichtung:**

- Language: 40% (Python vs C++ = fundamental)
- Version: 25% (3.10 vs 3.11 = wichtig)
- OS: 20% (Windows vs Linux = Pfade)
- Hardware: 15% (CPU vs GPU = relevant)

### **Thresholds:**

- **≥85%:** ENFORCE (Lösung erzwingen)
- **≥70%:** SUGGEST (Lösung vorschlagen)
- **≥50%:** CONSIDER (Lösung erwähnen)
- **<50%:** IGNORE (Neue Lösung suchen)

---

## 🏗️ Architektur

```
C:\KISYSTEM\core\
├── context_tracker.py      ← Environment-Detection
├── confidence_scorer.py    ← Multi-Faktor Scoring
└── learning_module.py      ← Main Module (V2)

D:\AGENT_MEMORY\
└── memory.db              ← SQLite Database (V2 Schema)
```

---

## 📦 Installation

### **1. Download alle Dateien:**
- `context_tracker.py`
- `confidence_scorer.py`
- `learning_module_v2.py`
- `schema_v2.sql`
- `setup_v2.ps1`

### **2. In Download-Ordner navigieren:**
```powershell
cd C:\Users\Jörg\Downloads
```

### **3. Setup ausführen:**
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_v2.ps1
```

**Das Script macht automatisch:**
- ✓ Backup alte Database
- ✓ Löscht alte Database (Fresh Start)
- ✓ Kopiert neue Module nach `C:\KISYSTEM\core\`
- ✓ Initialisiert neue Database mit V2 Schema
- ✓ Führt Tests aus
- ✓ Zeigt Summary

---

## 🚀 Nach Installation

### **KISYSTEM starten:**
```powershell
cd C:\KISYSTEM
python start_kisystem.py
```

### **Test mit einfachem Task:**
```
erstelle eine Python-Funktion zum Addieren zweier Zahlen
```

### **Statistik anzeigen:**
```
stats
```

---

## ⚙️ Konfiguration

### **Gewichtung anpassen:**

Editiere `C:\KISYSTEM\config\kisystem_config.json`:

```json
{
  "confidence_weights": {
    "text_similarity": 0.40,
    "context_match": 0.30,
    "success_history": 0.20,
    "recency": 0.10
  },
  "confidence_thresholds": {
    "enforce": 0.85,
    "suggest": 0.70,
    "consider": 0.50
  }
}
```

**Alternative für stabilen Stack (weniger Recency):**
```json
{
  "text_similarity": 0.45,
  "context_match": 0.30,
  "success_history": 0.20,
  "recency": 0.05
}
```

---

## 📈 Database Schema V2

### **Haupttabelle: solutions_v2**

```sql
-- Core
error_text, solution, error_type

-- Context
language, language_version, os, hardware, gpu_model, cuda_version
dependencies, compiler

-- Task
complexity, domain, model_used

-- Metrics
success_count, failure_count, avg_solve_time

-- Temporal
created_at, last_used_at, last_success_at, last_failure_at
```

### **Views:**
- `solution_stats` - Aggregierte Statistiken
- `top_solutions` - Top 20 nach Erfolgen
- `recent_activity` - Letzte 20 Activities
- `domain_stats` - Statistiken pro Domain
- `complexity_stats` - Statistiken pro Complexity

---

## 🔍 Debugging

### **Confidence-Details anzeigen:**

```python
from learning_module import LearningModule

learner = LearningModule()
results = learner.find_similar_solutions(
    error="ImportError: No module named 'numpy'",
    code="import numpy",
    model_used="test"
)

for r in results:
    print(f"\nConfidence: {r['confidence']:.1%}")
    print(learner.confidence_scorer.explain_score(r['details']))
```

### **Statistiken:**

```python
stats = learner.get_statistics()
print(stats)
```

### **Export Knowledge:**

```python
learner.export_knowledge("knowledge_backup.json")
```

---

## 🎯 Erwartete Performance

### **Vorher (V1):**
- ~6 Min pro Task
- 5 patterns gespeichert
- Binäres Learning

### **Jetzt (V2):**
- ~2:47 Min pro Task (54% schneller)
- Context-Aware Learning
- Granulares Scoring

### **Nach 3-Tier Model-Routing:**
- Einfach: ~30s (87% schneller)
- Mittel: ~1min (83% schneller)
- Komplex: ~2min (67% schneller)

---

## 🛠️ Troubleshooting

### **"ModuleNotFoundError: No module named 'context_tracker'"**

```powershell
# Check ob Dateien kopiert wurden
ls C:\KISYSTEM\core\

# Sollte zeigen:
# context_tracker.py
# confidence_scorer.py
# learning_module.py
```

### **Database-Fehler**

```powershell
# Database neu initialisieren
rm D:\AGENT_MEMORY\memory.db
cd C:\KISYSTEM
python -c "from core.learning_module import LearningModule; LearningModule()"
```

### **Alte V1 Backup wiederherstellen**

```powershell
# Backups sind in D:\AGENT_MEMORY\
ls D:\AGENT_MEMORY\*.backup*

# Wiederherstellen
cp D:\AGENT_MEMORY\memory.db.backup_YYYYMMDD_HHMMSS D:\AGENT_MEMORY\memory.db
```

---

## 📊 Beispiel-Output

```
=== Similar Solutions Found ===

Solution #1:
  Confidence: 92.5%
  Action: ENFORCE
  
  Score Breakdown:
    Text Similarity:  95.0% (weight: 40%)
    Context Match:    90.0% (weight: 30%)
    Success History:  88.0% (weight: 20%)
    Recency:          85.0% (weight: 10%)
  
  Context Match Details:
    Language: ✓
    Version:  95% match
    OS:       ✓
    Hardware: ✓
  
  Solution: pip install numpy==1.26.2
  
→ Applying enforced solution...
```

---

## ✅ Checklist nach Installation

- [ ] Setup-Script ohne Fehler durchgelaufen
- [ ] Tests passed
- [ ] KISYSTEM startet
- [ ] Einfacher Task funktioniert
- [ ] Stats zeigen neue Database
- [ ] Backup alte Database existiert

---

## 🚀 Nächste Schritte

1. **ModelSelector finalisieren** (wartet auf qwen2.5:32b Download)
2. **HardwareTestAgent** (RME/GPU Tests)
3. **TEP-Agent** (Audio-DSP Spezialist)
4. **Hardware-in-the-Loop** (M-32 Analog Loop)

---

**Bei Fragen:** Check PowerShell-Output oder Python-Errors  
**Bei Problemen:** Backup existiert in `D:\AGENT_MEMORY\memory.db.backup_*`

🎯 **KISYSTEM V2 - Context-Aware Learning Ready!**
