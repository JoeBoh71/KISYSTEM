#!/usr/bin/env python3
"""
KISYSTEM FIX 4: CUDAProfiler Unicode Encoding
Fixes UnicodeDecodeError when reading nsys output
"""

import re
import os

def fix_cuda_profiler():
    """Fix encoding issue in cuda_profiler_agent.py"""
    
    filepath = 'agents/cuda_profiler_agent.py'
    
    print("=" * 60)
    print("KISYSTEM FIX 4: CUDAProfiler Unicode Encoding")
    print("=" * 60)
    print(f"\n📝 Processing: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: {filepath} not found!")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Find subprocess.run calls and add encoding parameter
    # Pattern: subprocess.run(..., capture_output=True)
    
    if 'encoding="utf-8"' not in content:
        print("  → Adding UTF-8 encoding to subprocess calls")
        
        # Replace all subprocess.run calls
        pattern = r'subprocess\.run\((.*?),\s*capture_output=True\)'
        replacement = r'subprocess.run(\1, capture_output=True, encoding="utf-8", errors="ignore")'
        
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    else:
        print("  ✓ UTF-8 encoding already present")
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✅ File updated!")
        print("\n📋 Change: Added UTF-8 encoding to subprocess calls")
        return True
    else:
        print("  ✓ No changes needed")
        return False

def main():
    success = fix_cuda_profiler()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ COMPLETE: Unicode encoding fixed")
    else:
        print("✓ Already fixed")
    print("=" * 60)
    
    print("\n📋 This fixes:")
    print("  - UnicodeDecodeError: 'charmap' codec errors")
    print("  - Ensures nsys output is read correctly")

if __name__ == '__main__':
    main()
