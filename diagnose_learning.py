# DIAGNOSE SCRIPT - Why is Meta=0.000?
# Save as: diagnose_learning.py

import json
import sys
from pathlib import Path

print("="*70)
print("KISYSTEM LEARNING DIAGNOSE")
print("="*70)

# 1. Check learning_log.json
learning_log_path = Path("C:/KISYSTEM/projects/U3DAW/learning_log.json")
print(f"\n1. Learning Log Check:")
print(f"   Path: {learning_log_path}")
print(f"   Exists: {learning_log_path.exists()}")

if learning_log_path.exists():
    with open(learning_log_path) as f:
        log = json.load(f)
    
    entries = log.get('entries', [])
    print(f"   Entries: {len(entries)}")
    
    if entries:
        print(f"\n   Sample Entry:")
        sample = entries[0]
        for key in ['domain', 'task', 'model', 'success']:
            print(f"     {key}: {sample.get(key, 'MISSING')}")
    else:
        print("   ⚠️  PROBLEM: Entries list is EMPTY!")
else:
    print("   ❌ PROBLEM: File does not exist!")

# 2. Check MetaSupervisor initialization
print(f"\n2. MetaSupervisor Check:")
try:
    sys.path.insert(0, 'C:/KISYSTEM')
    from core.meta_supervisor import MetaSupervisor
    
    meta = MetaSupervisor()
    print(f"   Created: ✓")
    print(f"   Learning data entries: {len(meta.learning_data)}")
    print(f"   Model biases calculated: {len(meta.model_biases)}")
    
    if meta.model_biases:
        print(f"\n   Sample Bias:")
        domain = list(meta.model_biases.keys())[0]
        bias = meta.model_biases[domain]
        print(f"     Domain: {domain}")
        print(f"     Models: {list(bias.keys())}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# 3. Check if learning_log is passed to Supervisor
print(f"\n3. Supervisor Initialization Check:")
print(f"   In run_autonomous.py, check if learning_module is passed:")
print(f"   supervisor = SupervisorV3(")
print(f"       learning_module=???  ← Should be LearningModuleV2 instance")
print(f"   )")

# 4. Check optimization_config.json
config_path = Path("C:/KISYSTEM/config/optimization_config.json")
print(f"\n4. Optimization Config Check:")
print(f"   Path: {config_path}")
print(f"   Exists: {config_path.exists()}")

if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    
    meta_enabled = config.get('phase7', {}).get('meta_supervisor', {}).get('enabled', False)
    learning_path = config.get('phase7', {}).get('meta_supervisor', {}).get('learning_log_path', '')
    
    print(f"   Meta-Supervisor enabled: {meta_enabled}")
    print(f"   Learning log path: {learning_path}")
    
    if learning_path != str(learning_log_path):
        print(f"   ⚠️  WARNING: Path mismatch!")
        print(f"      Config: {learning_path}")
        print(f"      Actual: {learning_log_path}")

# 5. Summary
print(f"\n{'='*70}")
print("SUMMARY")
print("="*70)
print("\nIf Meta=0.000 everywhere, check:")
print("1. ✓ learning_log.json has entries (you have 42)")
print("2. ? MetaSupervisor loads the learning_log")
print("3. ? Supervisor is initialized WITH learning_module")
print("4. ? optimization_config.json points to correct path")
print("\nRun this script to find the issue:")
print("  python diagnose_learning.py")
print("="*70)
