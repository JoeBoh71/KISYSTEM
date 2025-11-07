"""
KISYSTEM Context Tracker V2
Erfasst vollständigen Environment-Context für Context-Aware Learning

Author: Jörg Bohne
Date: 2025-11-06
"""

import platform
import sys
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class ContextTracker:
    """
    Erfasst Environment-Context für präzises Learning
    """
    
    # Domain-Keywords für Klassifikation
    DOMAIN_KEYWORDS = {
        'audio_dsp': [
            'fft', 'stft', 'pqmf', 'filter', 'convolution', 'sample_rate',
            'dsp', 'audio', 'signal', 'frequency', 'spectrum', 'filterbank',
            'tep', 'time-energy', 'psychoacoustic', 'bark', 'masking'
        ],
        'cuda_kernel': [
            '__global__', '__device__', '__shared__', 'cudaMalloc', 'cudaMemcpy',
            'CUDA', 'blockIdx', 'threadIdx', 'kernel', 'nvcc', 'cupy', 'gpu'
        ],
        'web': [
            'http', 'https', 'flask', 'fastapi', 'websocket', 'html', 'css',
            'javascript', 'react', 'vue', 'api', 'rest', 'endpoint'
        ],
        'system': [
            'thread', 'process', 'mutex', 'semaphore', 'file', 'socket',
            'pipe', 'ipc', 'multiprocessing', 'concurrent', 'async'
        ]
    }
    
    # Complexity-Keywords
    COMPLEX_KEYWORDS = [
        'template', '__global__', 'async', 'multiprocessing', 'threading',
        'cuda', 'optimization', 'algorithm', 'recursive'
    ]
    
    MEDIUM_KEYWORDS = [
        'class', 'def ', 'struct', 'namespace', 'interface', 'abstract'
    ]
    
    COMPLEX_ERRORS = [
        'SegmentationFault', 'CUDA', 'Linker error', 'Template',
        'Memory leak', 'Race condition', 'Deadlock'
    ]
    
    def __init__(self):
        """Initialize Context Tracker"""
        self._cache = {}
        self._detect_system_info()
    
    def _detect_system_info(self):
        """Detect system information once at initialization"""
        self._cache['os'] = platform.system()
        self._cache['os_version'] = platform.version()
        self._cache['python_version'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # GPU Detection
        self._cache['gpu_model'] = self._detect_gpu()
        self._cache['cuda_version'] = self._detect_cuda()
        
        # Compiler Detection
        self._cache['compiler'] = self._detect_compiler()
    
    def _detect_gpu(self) -> Optional[str]:
        """Detect GPU model"""
        try:
            result = subprocess.run(
    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='ignore',
    timeout=2
)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        return None
    
    def _detect_cuda(self) -> Optional[str]:
        """Detect CUDA version"""
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                match = re.search(r'release (\d+\.\d+)', result.stdout)
                if match:
                    return match.group(1)
        except:
            pass
        return None
    
    def _detect_compiler(self) -> Optional[str]:
        """Detect C++ compiler"""
        try:
            # Try MSVC
            result = subprocess.run(
                ['cl'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if 'Microsoft' in result.stderr:
                match = re.search(r'Version (\d+\.\d+)', result.stderr)
                if match:
                    return f"MSVC {match.group(1)}"
        except:
            pass
        
        try:
            # Try GCC
            result = subprocess.run(
                ['gcc', '--version'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                match = re.search(r'gcc.*?(\d+\.\d+\.\d+)', result.stdout)
                if match:
                    return f"GCC {match.group(1)}"
        except:
            pass
        
        return None
    
    def get_current_context(self) -> Dict:
        """
        Get complete current context
        
        Returns:
            dict: Complete environment context
        """
        return {
            'os': self._cache['os'],
            'os_version': self._cache['os_version'],
            'python_version': self._cache['python_version'],
            'gpu_model': self._cache['gpu_model'],
            'cuda_version': self._cache['cuda_version'],
            'compiler': self._cache['compiler']
        }
    
    def detect_language(self, code: str) -> Tuple[str, Optional[str]]:
        """
        Detect programming language and version
        
        Args:
            code: Source code
            
        Returns:
            (language, version): e.g. ('python', '3.11')
        """
        code_lower = code.lower()
        
        # Python
        if any(kw in code for kw in ['import ', 'def ', 'class ', 'print(']):
            return ('python', self._cache['python_version'])
        
        # C++
        if any(kw in code for kw in ['#include', 'namespace', 'template<', 'std::']):
            return ('cpp', 'C++20')  # Default to C++20 für Jörg
        
        # CUDA
        if any(kw in code for kw in ['__global__', '__device__', 'cudaMalloc']):
            return ('cuda', self._cache['cuda_version'] or 'CUDA 13')
        
        # JavaScript
        if any(kw in code for kw in ['const ', 'let ', 'var ', 'function ', '=>']):
            return ('javascript', None)
        
        # Rust
        if any(kw in code for kw in ['fn ', 'let mut', 'impl ', 'pub ']):
            return ('rust', None)
        
        # Go
        if any(kw in code for kw in ['package ', 'func ', 'import (']):
            return ('go', None)
        
        return ('unknown', None)
    
    def detect_dependencies(self, code: str, language: str) -> List[str]:
        """
        Extract dependencies from code
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            List of dependencies
        """
        deps = []
        
        if language == 'python':
            # Extract imports
            import_pattern = r'^(?:from\s+(\S+)\s+)?import\s+(\S+)'
            for match in re.finditer(import_pattern, code, re.MULTILINE):
                module = match.group(1) or match.group(2)
                module = module.split('.')[0]  # Get top-level module
                if module not in ['os', 'sys', 'json', 're', 'typing']:  # Skip stdlib
                    deps.append(module)
        
        elif language in ['cpp', 'cuda']:
            # Extract includes (non-stdlib)
            include_pattern = r'#include\s+[<"]([^>"]+)[>"]'
            for match in re.finditer(include_pattern, code):
                header = match.group(1)
                if not header.startswith('std') and '/' in header:
                    deps.append(header.split('/')[0])
        
        return list(set(deps))  # Unique
    
    def classify_complexity(self, code: str, error: str) -> str:
        """
        Classify task complexity
        
        Args:
            code: Source code
            error: Error message
            
        Returns:
            'simple', 'medium', or 'complex'
        """
        lines = len([l for l in code.split('\n') if l.strip()])
        text = (code + ' ' + error).lower()
        
        score = 0
        
        # Line count
        if lines > 200:
            score += 2
        elif lines > 50:
            score += 1
        
        # Complex keywords
        for keyword in self.COMPLEX_KEYWORDS:
            if keyword.lower() in text:
                score += 2
                break
        
        # Medium keywords
        if score == 0:  # Only if not already complex
            for keyword in self.MEDIUM_KEYWORDS:
                if keyword.lower() in text:
                    score += 1
                    break
        
        # Error type
        for err_type in self.COMPLEX_ERRORS:
            if err_type.lower() in error.lower():
                score += 1
                break
        
        # Nesting depth (rough estimate)
        max_indent = 0
        for line in code.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent // 4)  # Assume 4-space indent
        
        if max_indent > 5:
            score += 1
        
        # Special overrides
        if any(kw in text for kw in ['cuda', '__global__', 'template<']):
            score = max(score, 4)  # Force at least complex
        
        if any(kw in text for kw in ['fft', 'stft', 'pqmf']):
            score = max(score, 2)  # Force at least medium
        
        # Classification
        if score >= 4:
            return 'complex'
        elif score >= 2:
            return 'medium'
        else:
            return 'simple'
    
    def classify_domain(self, code: str, error: str) -> str:
        """
        Classify problem domain
        
        Args:
            code: Source code
            error: Error message
            
        Returns:
            Domain name
        """
        text = (code + ' ' + error).lower()
        
        # Score each domain
        scores = {domain: 0 for domain in self.DOMAIN_KEYWORDS}
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    scores[domain] += 1
        
        # Get highest score
        max_score = max(scores.values())
        if max_score == 0:
            return 'general'
        
        return max(scores, key=scores.get)
    
    def get_full_context(self, code: str, error: str, model_used: str) -> Dict:
        """
        Get complete context for current task
        
        Args:
            code: Source code
            error: Error message
            model_used: Model name used for task
            
        Returns:
            Complete context dictionary
        """
        language, lang_version = self.detect_language(code)
        
        context = self.get_current_context()
        context.update({
            'language': language,
            'language_version': lang_version,
            'dependencies': json.dumps(self.detect_dependencies(code, language)),
            'complexity': self.classify_complexity(code, error),
            'domain': self.classify_domain(code, error),
            'model_used': model_used,
            'error_type': self._classify_error_type(error)
        })
        
        # Hardware classification
        if self._cache['gpu_model']:
            context['hardware'] = 'GPU'
        else:
            context['hardware'] = 'CPU'
        
        return context
    
    def _classify_error_type(self, error: str) -> str:
        """Classify error type"""
        error_lower = error.lower()
        
        if 'import' in error_lower or 'module' in error_lower:
            return 'ImportError'
        elif 'syntax' in error_lower:
            return 'SyntaxError'
        elif 'name' in error_lower and 'defined' in error_lower:
            return 'NameError'
        elif 'type' in error_lower:
            return 'TypeError'
        elif 'value' in error_lower:
            return 'ValueError'
        elif 'cuda' in error_lower or 'gpu' in error_lower:
            return 'CUDAError'
        elif 'segmentation' in error_lower or 'segfault' in error_lower:
            return 'SegmentationFault'
        elif 'linker' in error_lower or 'undefined reference' in error_lower:
            return 'LinkerError'
        else:
            return 'RuntimeError'


if __name__ == '__main__':
    # Quick test
    tracker = ContextTracker()
    
    print("=== System Context ===")
    ctx = tracker.get_current_context()
    for key, value in ctx.items():
        print(f"{key}: {value}")
    
    print("\n=== Language Detection Test ===")
    test_code = """
import numpy as np
def fft_process(signal):
    return np.fft.fft(signal)
"""
    lang, version = tracker.detect_language(test_code)
    print(f"Language: {lang} {version}")
    print(f"Domain: {tracker.classify_domain(test_code, '')}")
    print(f"Complexity: {tracker.classify_complexity(test_code, '')}")
