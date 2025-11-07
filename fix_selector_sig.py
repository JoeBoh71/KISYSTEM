#!/usr/bin/env python3
"""
KISYSTEM FIX 7b: ModelSelector agent_type compatibility
Adds agent_type parameter for backward compatibility
"""

import os

def fix_model_selector_signature():
    """Add agent_type parameter to select_model method"""
    
    filepath = 'core/model_selector.py'
    
    print("Fixing ModelSelector signature...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the method signature
    old_sig = 'def select_model(self, task: str, language: str = None) -> Tuple[str, int]:'
    new_sig = 'def select_model(self, task: str, language: str = None, agent_type: str = None) -> Tuple[str, int]:'
    
    content = content.replace(old_sig, new_sig)
    
    # Fix standalone function too
    old_func = 'def select_model(task: str, language: str = None) -> Tuple[str, int]:'
    new_func = 'def select_model(task: str, language: str = None, agent_type: str = None) -> Tuple[str, int]:'
    
    content = content.replace(old_func, new_func)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed!")

if __name__ == '__main__':
    fix_model_selector_signature()
