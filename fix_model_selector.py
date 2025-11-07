#!/usr/bin/env python3
"""
KISYSTEM FIX 7: Smart Model Selector
Intelligent CUDA complexity detection to avoid 32b overkill
"""

import os

def create_smart_model_selector():
    """Create improved model_selector.py with CUDA complexity detection"""
    
    filepath = 'core/model_selector.py'
    
    print("=" * 60)
    print("KISYSTEM FIX 7: Smart Model Selector")
    print("=" * 60)
    print(f"\n📝 Creating: {filepath}")
    
    smart_selector_code = '''"""
Smart Model Selector for KISYSTEM
Routes tasks to appropriate models based on actual complexity
"""

from typing import Dict, Tuple


class ModelSelector:
    """Intelligent model selection based on task complexity"""
    
    # Model configurations with timeouts
    MODELS = {
        'llama3.1:8b': {'timeout': 180, 'description': 'Fast, simple tasks'},
        'deepseek-coder-v2:16b': {'timeout': 300, 'description': 'Medium complexity'},
        'qwen2.5-coder:32b': {'timeout': 1800, 'description': 'Complex, large projects'},
        'deepseek-r1:32b': {'timeout': 1800, 'description': 'Deep debugging'}
    }
    
    # CUDA complexity keywords
    CUDA_SIMPLE = [
        'vector add', 'vector addition', 'element-wise', 'elementwise',
        'scalar', 'simple', 'basic', 'multiply', 'divide',
        'copy', 'fill', 'saxpy', 'daxpy'
    ]
    
    CUDA_MEDIUM = [
        'shared memory', 'matrix multiply', 'matrix multiplication',
        'transpose', 'convolution', 'dot product', 'reduction',
        'prefix sum', 'scan', 'histogram', 'sorting'
    ]
    
    CUDA_COMPLEX = [
        'fft', 'fast fourier', 'multi-kernel', 'multi-pass',
        'dynamic parallelism', 'graph', 'cooperative groups',
        'tensor core', 'sparse', 'optimization pipeline',
        'multi-gpu', 'streams', 'async'
    ]
    
    def __init__(self):
        pass
    
    def select_model(self, task: str, language: str = None) -> Tuple[str, int]:
        """
        Select appropriate model based on task complexity
        
        Args:
            task: Task description
            language: Programming language (cuda, python, cpp, etc)
            
        Returns:
            (model_name, timeout_seconds)
        """
        complexity = self._detect_complexity(task, language)
        
        if complexity == "SIMPLE":
            model = 'llama3.1:8b'
        elif complexity == "MEDIUM":
            model = 'deepseek-coder-v2:16b'
        else:  # COMPLEX
            model = 'qwen2.5-coder:32b'
        
        timeout = self.MODELS[model]['timeout']
        
        return model, timeout
    
    def _detect_complexity(self, task: str, language: str = None) -> str:
        """Detect task complexity"""
        task_lower = task.lower()
        
        # Special handling for CUDA
        if language and language.upper() == "CUDA":
            return self._detect_cuda_complexity(task_lower)
        
        # General complexity detection
        if any(kw in task_lower for kw in ['simple', 'basic', 'hello', 'example']):
            return "SIMPLE"
        
        if any(kw in task_lower for kw in [
            'optimize', 'performance', 'algorithm', 'data structure',
            'class', 'interface', 'architecture'
        ]):
            return "COMPLEX"
        
        # Default to medium
        return "MEDIUM"
    
    def _detect_cuda_complexity(self, task_lower: str) -> str:
        """Detect CUDA-specific complexity"""
        
        # Check for complex patterns first
        complex_score = sum(1 for kw in self.CUDA_COMPLEX if kw in task_lower)
        if complex_score > 0:
            return "COMPLEX"
        
        # Check for medium patterns
        medium_score = sum(1 for kw in self.CUDA_MEDIUM if kw in task_lower)
        if medium_score > 0:
            return "MEDIUM"
        
        # Check for simple patterns
        simple_score = sum(1 for kw in self.CUDA_SIMPLE if kw in task_lower)
        if simple_score > 0:
            return "SIMPLE"
        
        # Default: If no specific keywords, assume medium for CUDA
        return "MEDIUM"
    
    def get_escalation_model(self, current_model: str) -> Tuple[str, int]:
        """
        Get next model in escalation chain
        
        Args:
            current_model: Current model that failed
            
        Returns:
            (next_model, timeout) or (None, 0) if no escalation
        """
        escalation_chain = [
            'llama3.1:8b',
            'deepseek-coder-v2:16b',
            'qwen2.5-coder:32b',
            'deepseek-r1:32b'
        ]
        
        try:
            current_idx = escalation_chain.index(current_model)
            if current_idx < len(escalation_chain) - 1:
                next_model = escalation_chain[current_idx + 1]
                return next_model, self.MODELS[next_model]['timeout']
        except (ValueError, IndexError):
            pass
        
        return None, 0
    
    def format_selection(self, model: str, timeout: int, task: str) -> str:
        """Format model selection for logging"""
        complexity = self._detect_complexity(task)
        desc = self.MODELS[model]['description']
        
        output = []
        output.append(f"[ModelSelector] 🎯 Task complexity: {complexity}")
        output.append(f"[ModelSelector] 🤖 Selected model: {model}")
        output.append(f"[ModelSelector] 📝 Reason: {desc}")
        output.append(f"[ModelSelector] ⏱️ Timeout: {timeout}s (~{timeout//60}min)")
        
        return "\\n".join(output)


# Convenience function for backward compatibility
def select_model(task: str, language: str = None) -> Tuple[str, int]:
    """Select model for task"""
    selector = ModelSelector()
    return selector.select_model(task, language)
'''
    
    # Write the new file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(smart_selector_code)
    
    print("  ✅ File created!")
    print("\n📋 Improvements:")
    print("  1. CUDA Simple (vector add, element-wise) → llama3.1:8b (~30s)")
    print("  2. CUDA Medium (matrix mul, reduction) → deepseek-coder-v2:16b (~1-2min)")
    print("  3. CUDA Complex (FFT, multi-kernel) → qwen2.5-coder:32b (~7min)")
    print("  4. Smart escalation chain on failures")
    
    return True

def main():
    success = create_smart_model_selector()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ COMPLETE: Smart Model Selector installed")
    print("=" * 60)
    
    print("\n📊 Expected improvements:")
    print("  - Vector addition: 7min → 30s (14x faster!)")
    print("  - Simple kernels: Use 8b model first")
    print("  - Auto-escalate on failure")
    
    print("\n📋 Test:")
    print("  python -B test_phase6_optimization.py")

if __name__ == '__main__':
    main()
