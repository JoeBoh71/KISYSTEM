"""
Smart Model Selector for KISYSTEM
Routes tasks to appropriate models based on actual complexity
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    timeout: int
    description: str


class ModelSelector:
    """Intelligent model selection based on task complexity"""
    
    # Model configurations
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
    
    def select_model(self, task: str, language: str = None, 
                    agent_type: str = None, context: str = None) -> ModelConfig:
        """
        Select appropriate model based on task complexity
        
        Args:
            task: Task description
            language: Programming language (cuda, python, cpp, etc)
            agent_type: Type of agent requesting (builder, fixer, etc)
            context: Additional context
            
        Returns:
            ModelConfig with name, timeout, description
        """
        complexity = self._detect_complexity(task, language)
        
        if complexity == "SIMPLE":
            model = 'llama3.1:8b'
        elif complexity == "MEDIUM":
            model = 'deepseek-coder-v2:16b'
        else:  # COMPLEX
            model = 'qwen2.5-coder:32b'
        
        model_info = self.MODELS[model]
        
        # Log selection
        print(f"[ModelSelector] 🎯 Task complexity: {complexity}")
        print(f"[ModelSelector] 🤖 Selected model: {model}")
        print(f"[ModelSelector] ⏱️ Estimated time: ~{model_info['timeout']//60}min")
        
        return ModelConfig(
            name=model,
            timeout=model_info['timeout'],
            description=model_info['description']
        )
    
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
    
    def get_escalation_model(self, current_model: str) -> Optional[ModelConfig]:
        """
        Get next model in escalation chain
        
        Args:
            current_model: Current model that failed
            
        Returns:
            ModelConfig for next model or None if no escalation
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
                model_info = self.MODELS[next_model]
                return ModelConfig(
                    name=next_model,
                    timeout=model_info['timeout'],
                    description=model_info['description']
                )
        except (ValueError, IndexError):
            pass
        
        return None
