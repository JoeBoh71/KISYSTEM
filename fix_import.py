#!/usr/bin/env python3
"""
KISYSTEM FIX 6: CUDAProfiler PerformanceParser Import
Fixes import path for performance_parser module
"""

import os

def fix_cuda_profiler_import():
    """Fix performance_parser import in cuda_profiler_agent.py"""
    
    filepath = 'agents/cuda_profiler_agent.py'
    
    print("=" * 60)
    print("KISYSTEM FIX 6: PerformanceParser Import")
    print("=" * 60)
    print(f"\n📝 Processing: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: {filepath} not found!")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Find and fix the import
    old_import = "from performance_parser import PerformanceParser"
    new_import = "from core.performance_parser import PerformanceParser"
    
    if old_import in content:
        print(f"  → Fixing import path")
        content = content.replace(old_import, new_import)
        print(f"    OLD: {old_import}")
        print(f"    NEW: {new_import}")
    else:
        print("  ⚠️  Pattern not found - checking if already correct...")
        if new_import in content:
            print("  ✓ Import already correct!")
            return False
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✅ File updated!")
        return True
    else:
        print("  ✓ No changes needed")
        return False

def main():
    success = fix_cuda_profiler_import()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ COMPLETE: Import path fixed")
    else:
        print("✓ Import already correct")
    print("=" * 60)
    
    print("\n📋 Next: Run test again")
    print("  python -B test_phase6_optimization.py")

if __name__ == '__main__':
    main()
