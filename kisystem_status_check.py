#!/usr/bin/env python3
"""
KISYSTEM Phase 5 - Vollständiger Status-Check
Prüft alle Komponenten auf Funktionsfähigkeit

Author: Jörg Bohne
Date: 2025-11-06
"""

import sys
import asyncio
from pathlib import Path

# KISYSTEM Root
KISYSTEM_ROOT = Path("C:/KISYSTEM")

print("="*70)
print("KISYSTEM PHASE 5 - STATUS CHECK")
print("="*70)
print()

# ============================================================================
# 1. VERZEICHNIS-STRUKTUR
# ============================================================================

print("[1] VERZEICHNIS-STRUKTUR")
print("-"*70)

required_dirs = ["core", "agents", "config", "workspace"]
for dir_name in required_dirs:
    dir_path = KISYSTEM_ROOT / dir_name
    exists = dir_path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {dir_name}/")
    if not exists:
        print(f"      → FEHLT!")

print()

# ============================================================================
# 2. CORE MODULE
# ============================================================================

print("[2] CORE MODULE")
print("-"*70)

sys.path.insert(0, str(KISYSTEM_ROOT / "core"))

core_modules = [
    "model_selector",
    "workflow_engine", 
    "ollama_client",
    "supervisor_v3",
    "learning_module_v2",
    "execution_module"
]

imported_modules = {}

for module_name in core_modules:
    try:
        module = __import__(module_name)
        imported_modules[module_name] = module
        print(f"  ✓ {module_name}.py")
    except Exception as e:
        print(f"  ✗ {module_name}.py")
        print(f"      → Fehler: {e}")

print()

# ============================================================================
# 3. AGENTS
# ============================================================================

print("[3] AGENTS")
print("-"*70)

sys.path.insert(0, str(KISYSTEM_ROOT / "agents"))

agent_modules = [
    "builder_agent",
    "tester_agent",
    "fixer_agent",
    "hardware_test_agent",
    "search_agent_v2"
]

for module_name in agent_modules:
    try:
        module = __import__(module_name)
        imported_modules[module_name] = module
        print(f"  ✓ {module_name}.py")
    except Exception as e:
        print(f"  ✗ {module_name}.py")
        print(f"      → Fehler: {e}")

print()

# ============================================================================
# 4. OLLAMA CONNECTION
# ============================================================================

async def check_ollama():
    print("[4] OLLAMA CONNECTION")
    print("-"*70)
    
    try:
        from ollama_client import OllamaClient
        
        client = OllamaClient()
        
        # Check connection
        models = await client.list_models()
        
        if not models:
            print("  ✗ Ollama läuft nicht oder keine Models")
            print("      → Start: ollama serve")
            return False
        
        print(f"  ✓ Ollama läuft")
        print(f"  ✓ {len(models)} Models verfügbar:")
        
        for model in models:
            print(f"      - {model}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Ollama Connection Error: {e}")
        return False

# ============================================================================
# 5. MODEL SELECTOR
# ============================================================================

def check_model_selector():
    print()
    print("[5] MODEL SELECTOR")
    print("-"*70)
    
    try:
        from model_selector import ModelSelector
        
        selector = ModelSelector()
        
        # Test complexity detection
        test_tasks = [
            ("Write hello world", "simple"),
            ("Implement bubble sort", "medium"),
            ("Create CUDA kernel", "complex"),
        ]
        
        all_ok = True
        for task, expected in test_tasks:
            detected = selector.detector.detect(task)
            ok = detected == expected
            all_ok = all_ok and ok
            
            status = "✓" if ok else "✗"
            print(f"  {status} '{task[:30]}...'")
            print(f"      → Detected: {detected}, Expected: {expected}")
        
        if all_ok:
            print(f"\n  ✓ Model Selector funktioniert korrekt")
        else:
            print(f"\n  ✗ Model Selector hat Fehler in Complexity Detection")
        
        return all_ok
        
    except Exception as e:
        print(f"  ✗ Model Selector Error: {e}")
        return False

# ============================================================================
# 6. WORKFLOW ENGINE
# ============================================================================

async def check_workflow_engine():
    print()
    print("[6] WORKFLOW ENGINE")
    print("-"*70)
    
    try:
        from workflow_engine import WorkflowEngine, WorkflowConfig, SecurityLevel
        
        config = WorkflowConfig(
            security_level=SecurityLevel.BALANCED,
            verbose=False
        )
        
        engine = WorkflowEngine(supervisor=None, config=config)
        
        print(f"  ✓ WorkflowEngine initialisiert")
        print(f"  ✓ Security Level: {config.security_level.value}")
        
        # Check dependency installer
        print(f"\n  Teste Dependency Detection...")
        
        # Numpy sollte auf deinem System sein
        result = await engine.installer.ensure_dependencies(["numpy"])
        
        if result.get("numpy", False):
            print(f"  ✓ Dependency Check funktioniert (numpy found)")
        else:
            print(f"  ⚠ Dependency Check: numpy nicht gefunden")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Workflow Engine Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# 7. LEARNING MODULE
# ============================================================================

def check_learning_module():
    print()
    print("[7] LEARNING MODULE V2")
    print("-"*70)
    
    try:
        from learning_module_v2 import LearningModuleV2
        
        # Test initialization
        learning = LearningModuleV2(db_path=":memory:")  # In-memory für Test
        
        print(f"  ✓ LearningModule V2 initialisiert")
        print(f"  ✓ Database: In-Memory (Test)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Learning Module Error: {e}")
        return False

# ============================================================================
# 8. SUPERVISOR V3
# ============================================================================

def check_supervisor():
    print()
    print("[8] SUPERVISOR V3")
    print("-"*70)
    
    try:
        from supervisor_v3 import SupervisorV3
        
        # Initialize (without actually running)
        supervisor = SupervisorV3(
            learning_module=None,
            search_agent=None,
            max_iterations=5,
            workspace="D:/AGENT_MEMORY"
        )
        
        print(f"  ✓ Supervisor V3 initialisiert")
        print(f"  ✓ Workspace: {supervisor.workspace}")
        print(f"  ✓ Max Iterations: {supervisor.max_iterations}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Supervisor V3 Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# 9. AGENTS FUNKTIONSFÄHIGKEIT
# ============================================================================

async def check_agents():
    print()
    print("[9] AGENT CLASSES")
    print("-"*70)
    
    results = {}
    
    # BuilderAgent
    try:
        from builder_agent import BuilderAgent
        builder = BuilderAgent(learning_module=None)
        print(f"  ✓ BuilderAgent initialisiert")
        results['builder'] = True
    except Exception as e:
        print(f"  ✗ BuilderAgent: {e}")
        results['builder'] = False
    
    # TesterAgent
    try:
        from tester_agent import TesterAgent
        tester = TesterAgent(learning_module=None)
        print(f"  ✓ TesterAgent initialisiert")
        results['tester'] = True
    except Exception as e:
        print(f"  ✗ TesterAgent: {e}")
        results['tester'] = False
    
    # FixerAgent
    try:
        from fixer_agent import FixerAgent
        fixer = FixerAgent(learning_module=None, search_agent=None)
        print(f"  ✓ FixerAgent initialisiert")
        results['fixer'] = True
    except Exception as e:
        print(f"  ✗ FixerAgent: {e}")
        results['fixer'] = False
    
    return all(results.values())

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run all checks"""
    
    results = {}
    
    # Synchronous checks
    results['model_selector'] = check_model_selector()
    results['learning_module'] = check_learning_module()
    results['supervisor'] = check_supervisor()
    
    # Async checks
    results['ollama'] = await check_ollama()
    results['workflow_engine'] = await check_workflow_engine()
    results['agents'] = await check_agents()
    
    # Summary
    print()
    print("="*70)
    print("ZUSAMMENFASSUNG")
    print("="*70)
    
    status_map = {
        'model_selector': 'Model Selector',
        'learning_module': 'Learning Module V2',
        'supervisor': 'Supervisor V3',
        'ollama': 'Ollama Connection',
        'workflow_engine': 'Workflow Engine',
        'agents': 'All Agents'
    }
    
    for key, name in status_map.items():
        status = "✓ OK" if results.get(key, False) else "✗ FEHLER"
        print(f"  {status}: {name}")
    
    print("="*70)
    
    all_ok = all(results.values())
    
    if all_ok:
        print()
        print("✅ KISYSTEM PHASE 5 - VOLLSTÄNDIG FUNKTIONSFÄHIG")
        print()
        print("🚀 Ready für:")
        print("   • Smart Model Routing (8b/16b/32b)")
        print("   • Auto-Dependency Management")
        print("   • Build-Test-Fix Loop")
        print("   • Escalation bei Failures")
        print()
    else:
        print()
        print("⚠️  KISYSTEM PHASE 5 - EINIGE PROBLEME")
        print()
        print("Behebe die oben markierten Fehler.")
        print()
    
    print("="*70)
    
    return all_ok


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nAbgebrochen.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
