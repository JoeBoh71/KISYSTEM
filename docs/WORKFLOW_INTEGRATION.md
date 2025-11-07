# KISYSTEM Supervisor V2 - Workflow Integration
# Instructions für autonomen Build-Test-Fix Loop

## Was wird hinzugefügt:

1. **WorkflowEngine** - Autonomer Loop
2. **Neue Commands** - "baue X" statt "erstelle X"
3. **Auto-Test-Fix** - Bis zu 3 Versuche

---

## Installation Steps:

### 1. Kopiere Module nach core/:

```powershell
Copy-Item "execution_module.py" "C:\KISYSTEM\core\" -Force
Copy-Item "workflow_engine.py" "C:\KISYSTEM\core\" -Force
```

### 2. Erweitere supervisor_v2.py:

**A) Import hinzufügen (nach Zeile 10):**

```python
from core.workflow_engine import WorkflowEngine
```

**B) In __init__ (nach Zeile 20, nach self.initialize_agents()):**

```python
# Initialize Workflow Engine
self.workflow = WorkflowEngine(self)
self.logger.info("✓ Workflow Engine initialized")
```

**C) In parse_command (nach Zeile 85, vor dem 'test' Pattern):**

```python
# Autonomous build (with auto-test-fix)
if any(w in cmd_lower for w in ['baue', 'build autonomous', 'erstelle autonom']):
    return {
        'agent': None,
        'action': 'autonomous_build',
        'target': command,
        'type': 'workflow'
    }
```

**D) In handle_command (nach Zeile 140, vor 'return self.delegate_to_agent'):**

```python
# Handle autonomous build
if intent.get('action') == 'autonomous_build':
    return await self.handle_autonomous_build(intent)
```

**E) Neue Methode hinzufügen (nach delegate_to_agent, Zeile 155):**

```python
async def handle_autonomous_build(self, intent):
    """
    Handle autonomous build with auto-test-fix loop
    """
    task = intent['target']
    
    # Remove trigger words
    for word in ['baue', 'build', 'erstelle', 'autonom', 'autonomous']:
        task = task.replace(word, '').strip()
    
    self.logger.info(f"Starting autonomous build: {task}")
    
    try:
        # Run workflow
        result = await self.workflow.autonomous_build(task, language='python')
        
        if result['success']:
            report = f"""
╔═══════════════════════════════════════╗
║   AUTONOMOUS BUILD SUCCESS ✅        ║
╚═══════════════════════════════════════╝

Task: {task}
Attempts: {result['attempts']}/{self.workflow.max_attempts}
Duration: {result['duration']:.1f}s

Final Code: {result['file_path']}

"""
            if result['final_result'] and result['final_result'].get('stdout'):
                report += f"Output:\n{result['final_result']['stdout']}\n"
            
            return report
        
        else:
            report = f"""
╔═══════════════════════════════════════╗
║   AUTONOMOUS BUILD FAILED ❌         ║
╚═══════════════════════════════════════╝

Task: {task}
Attempts: {result['attempts']}/{self.workflow.max_attempts}
Duration: {result['duration']:.1f}s

Errors:
"""
            for error in result['errors'][-3:]:  # Last 3 errors
                report += f"  - {error.get('error_type', 'Error')}: {error.get('error', 'Unknown')[:100]}\n"
            
            return report
    
    except Exception as e:
        self.logger.error(f"Autonomous build failed: {e}")
        return f"❌ Autonomous build error: {e}"
```

---

## Usage:

### Alter Weg (nur Builder):
```
erstelle eine Python-Funktion zum Addieren
```
→ Builder generiert Code, fertig (kein Test)

### Neuer Weg (Autonomous):
```
baue eine Python-Funktion zum Addieren
```
→ Builder → Test → Fixer (3x) → Garantiert fehlerfreier Code!

---

## Beispiel Output:

```
[Workflow] 🚀 Starting autonomous build: eine Python-Funktion zum Addieren
[Workflow] === Attempt 1/3 ===
[Workflow] 🔨 Step 1: Builder generating code...
[Workflow] ✓ Code generated: D:\AGENT_MEMORY\code_output\generated_20251106_111234_0.py
[Workflow] 🧪 Step 2: Executing code...
[Workflow] ✅ SUCCESS after 1 attempt(s)!

╔═══════════════════════════════════════╗
║   AUTONOMOUS BUILD SUCCESS ✅        ║
╚═══════════════════════════════════════╝

Task: eine Python-Funktion zum Addieren
Attempts: 1/3
Duration: 2.3s

Final Code: D:\AGENT_MEMORY\code_output\generated_20251106_111234_0.py

Output:
Testing add function...
5 + 3 = 8
✓ All tests passed
```

---

## Test Cases:

### Simple (sollte 1 Attempt):
```
baue eine Funktion die zwei Zahlen multipliziert
```

### Medium (könnte 2 Attempts brauchen):
```
baue ein Programm das die Fibonacci-Folge berechnet
```

### Complex (könnte 3 Attempts brauchen):
```
baue einen drehenden Würfel mit tkinter
```

---

## Wichtig:

- **"erstelle X"** = Alter Weg (nur Builder)
- **"baue X"** = Neuer Weg (Autonomous mit Test-Fix)

Beide parallel verfügbar!
