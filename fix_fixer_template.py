#!/usr/bin/env python3
"""
KISYSTEM FIX 2: FixerAgent CUDA Template
Adds CUDA_TEMPLATE like BuilderAgent to prevent include amnesia
"""

import re
import os

def fix_fixer_agent():
    """Add CUDA_TEMPLATE to fixer_agent.py"""
    
    filepath = 'agents/fixer_agent.py'
    
    print("=" * 60)
    print("KISYSTEM FIX 2: FixerAgent CUDA Template")
    print("=" * 60)
    print(f"\n📝 Processing: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: {filepath} not found!")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # CUDA Template to add (same as BuilderAgent)
    cuda_template = '''
# CUDA Template with guaranteed includes
CUDA_TEMPLATE = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>

{code}
"""
'''
    
    # Step 1: Check if CUDA_TEMPLATE already exists
    if 'CUDA_TEMPLATE' in content:
        print("  ✓ CUDA_TEMPLATE already present")
    else:
        print("  → Adding CUDA_TEMPLATE")
        
        # Find class definition and add template after it
        class_pattern = r'(class FixerAgent:.*?""".*?""")'
        
        if re.search(class_pattern, content, re.DOTALL):
            content = re.sub(
                class_pattern,
                r'\1' + cuda_template,
                content,
                flags=re.DOTALL
            )
        else:
            # Fallback: add after imports
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('class FixerAgent'):
                    lines.insert(i, cuda_template.strip())
                    break
            content = '\n'.join(lines)
    
    # Step 2: Modify generate_fix() to use template
    # Look for the code extraction part in generate_fix
    
    # Pattern: where code gets compiled
    if 'CUDA_TEMPLATE.format' not in content:
        print("  → Adding template injection in generate_fix()")
        
        # Find where we extract code and add template usage
        # Look for pattern like: extracted_code = self.extract_code(...)
        
        pattern = r'(extracted_code = self\.extract_code_from_response\(response, language\))'
        
        if re.search(pattern, content):
            replacement = r'''\1
            
            # Inject CUDA includes if CUDA code
            if language.upper() == "CUDA":
                extracted_code = CUDA_TEMPLATE.format(code=extracted_code)
                print("  → CUDA template injected with guaranteed includes")'''
            
            content = re.sub(pattern, replacement, content)
        else:
            print("  ⚠ Could not find extraction pattern - manual review needed")
    else:
        print("  ✓ Template injection already present")
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✅ File updated!")
        
        print("\n📋 Changes made:")
        print("  1. Added CUDA_TEMPLATE constant")
        print("  2. Added template injection in generate_fix()")
        print("  3. Guaranteed includes for CUDA code")
        
        return True
    else:
        print("  ✓ No changes needed")
        return False

def main():
    success = fix_fixer_agent()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ COMPLETE: FixerAgent CUDA template added")
    else:
        print("✓ FixerAgent already has CUDA template")
    print("=" * 60)
    
    print("\n📋 Next steps:")
    print("1. Review changes: git diff agents/fixer_agent.py")
    print("2. Test compilation: Should not fail in iteration 2+")

if __name__ == '__main__':
    main()
