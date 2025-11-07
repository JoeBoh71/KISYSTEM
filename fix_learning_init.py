#!/usr/bin/env python3
"""
KISYSTEM FIX 1: Learning Module Initialization
Adds ensure_tables() call to all agents' __init__ methods
"""

import re
import os
from pathlib import Path

def fix_agent_file(filepath):
    """Fix a single agent file"""
    print(f"\n📝 Processing: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Step 1: Add import if missing
    if 'from learning_module_v2 import learning_module' not in content:
        print("  → Adding learning_module import")
        
        # Find import section and add after core imports
        if 'from ollama_client import OllamaClient' in content:
            content = content.replace(
                'from ollama_client import OllamaClient',
                'from ollama_client import OllamaClient\nfrom learning_module_v2 import learning_module'
            )
        elif 'import sys' in content:
            # For files without OllamaClient
            content = content.replace(
                'import sys',
                'import sys\nfrom learning_module_v2 import learning_module'
            )
        else:
            # Add at very beginning after docstring
            lines = content.split('\n')
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
                    insert_pos = i
                    break
            lines.insert(insert_pos, 'from learning_module_v2 import learning_module')
            content = '\n'.join(lines)
    else:
        print("  ✓ Import already present")
    
    # Step 2: Add ensure_tables() in __init__
    # Look for __init__ method
    init_pattern = r'(def __init__\(self.*?\):)\n(\s+)'
    
    if re.search(init_pattern, content):
        # Check if ensure_tables already called
        if 'learning_module.ensure_tables()' not in content:
            print("  → Adding ensure_tables() call")
            
            # Add as first line in __init__
            def add_ensure_tables(match):
                method_def = match.group(1)
                indent = match.group(2)
                return f"{method_def}\n{indent}learning_module.ensure_tables()  # Initialize learning DB\n{indent}"
            
            content = re.sub(init_pattern, add_ensure_tables, content)
        else:
            print("  ✓ ensure_tables() already present")
    else:
        print("  ⚠ No __init__ method found")
    
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
    print("=" * 60)
    print("KISYSTEM FIX 1: Learning Module Initialization")
    print("=" * 60)
    
    # Agent files to fix
    agents = [
        'agents/builder_agent.py',
        'agents/tester_agent.py',
        'agents/fixer_agent.py',
        'agents/search_agent_v2.py',
        'agents/cuda_profiler_agent.py'
    ]
    
    # Check if we're in KISYSTEM directory
    if not os.path.exists('agents'):
        print("\n❌ ERROR: 'agents' directory not found!")
        print("   Please run this script from C:\\KISYSTEM\\ directory")
        return
    
    fixed_count = 0
    
    for agent in agents:
        if os.path.exists(agent):
            if fix_agent_file(agent):
                fixed_count += 1
        else:
            print(f"\n⚠️  File not found: {agent}")
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETE: Fixed {fixed_count} files")
    print("=" * 60)
    
    print("\n📋 Next steps:")
    print("1. Review changes: git diff")
    print("2. Test: python test_system.py")
    print("3. Check DB created: dir learning.db")

if __name__ == '__main__':
    main()
