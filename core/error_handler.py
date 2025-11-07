"""
KISYSTEM Hybrid Error Handler
Intelligente Error-Behandlung mit Kategorisierung, Confidence-Scoring und Model-Escalation

Author: Jörg Bohne
Date: 2025-11-07
Version: 1.0
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


# ============================================================================
# ERROR CATEGORIZATION
# ============================================================================

class ErrorCategory(Enum):
    """Error categories with different retry strategies"""
    COMPILATION = "compilation"      # Strict: 1 retry
    RUNTIME = "runtime"              # Medium: 2 retries
    PERFORMANCE = "performance"      # Flexible: 4 retries
    LOGIC = "logic"                  # Medium-Flexible: 3 retries
    UNKNOWN = "unknown"              # Conservative: 2 retries


@dataclass
class CategorizedError:
    """Error with category and metadata"""
    category: ErrorCategory
    error_message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    retry_limit: int
    recommended_model_size: str  # '8b', '16b', '32b'
    search_priority: int  # 1-5, higher = search earlier
    
    def __str__(self):
        return f"{self.category.value.upper()} [{self.severity}]: {self.error_message[:80]}..."


class ErrorCategorizer:
    """
    Categorize errors into COMPILATION/RUNTIME/PERFORMANCE/LOGIC
    """
    
    # Category-specific retry limits
    RETRY_LIMITS = {
        ErrorCategory.COMPILATION: 1,    # Compile errors are deterministic
        ErrorCategory.RUNTIME: 2,         # Runtime needs some flexibility
        ErrorCategory.PERFORMANCE: 4,     # Performance can be iterative
        ErrorCategory.LOGIC: 3,           # Logic errors need analysis
        ErrorCategory.UNKNOWN: 2          # Conservative default
    }
    
    # Patterns for each category
    COMPILATION_PATTERNS = [
        r'error C\d+:',                    # MSVC errors
        r'error:.*expected',               # GCC/Clang syntax errors
        r'undefined reference',            # Linker errors
        r'cannot find symbol',             # Java compilation
        r'SyntaxError',                    # Python syntax
        r'identifier .* is undefined',     # CUDA/C++
        r'#include.*No such file',         # Missing headers
        r'nvcc.*error',                    # CUDA compilation
        r'fatal error',                    # Critical compile errors
    ]
    
    RUNTIME_PATTERNS = [
        r'Segmentation fault',
        r'CUDA.*out of memory',
        r'RuntimeError',
        r'cudaError',
        r'exception',
        r'assertion.*failed',
        r'NullPointerException',
        r'IndexError',
        r'KeyError',
        r'ValueError',
        r'TypeError',
        r'AttributeError',
    ]
    
    PERFORMANCE_PATTERNS = [
        r'bank conflict',
        r'low occupancy',
        r'uncoalesced',
        r'shared memory',
        r'register pressure',
        r'bandwidth',
        r'latency',
        r'throughput',
        r'inefficient',
        r'slow',
        r'optimization',
    ]
    
    LOGIC_PATTERNS = [
        r'incorrect result',
        r'wrong output',
        r'test.*failed',
        r'assertion.*false',
        r'expected.*got',
        r'mismatch',
        r'divergence',
    ]
    
    @classmethod
    def categorize(cls, error: str, code: str = "", context: Optional[Dict] = None) -> CategorizedError:
        """
        Categorize error and determine handling strategy
        
        Args:
            error: Error message
            code: Source code (for additional context)
            context: Optional context (language, previous attempts, etc.)
            
        Returns:
            CategorizedError with category and metadata
        """
        
        error_lower = error.lower()
        
        # Check each category (priority order)
        category = ErrorCategory.UNKNOWN
        severity = 'medium'
        search_priority = 3
        
        # 1. Compilation errors (highest priority)
        if any(re.search(pattern, error, re.IGNORECASE) for pattern in cls.COMPILATION_PATTERNS):
            category = ErrorCategory.COMPILATION
            severity = cls._determine_compile_severity(error)
            search_priority = 2 if severity == 'high' else 4
            
        # 2. Runtime errors
        elif any(re.search(pattern, error, re.IGNORECASE) for pattern in cls.RUNTIME_PATTERNS):
            category = ErrorCategory.RUNTIME
            severity = cls._determine_runtime_severity(error)
            search_priority = 2 if 'segmentation' in error_lower else 3
            
        # 3. Performance issues
        elif any(re.search(pattern, error, re.IGNORECASE) for pattern in cls.PERFORMANCE_PATTERNS):
            category = ErrorCategory.PERFORMANCE
            severity = cls._determine_performance_severity(error)
            search_priority = 5  # Search late for performance
            
        # 4. Logic errors
        elif any(re.search(pattern, error, re.IGNORECASE) for pattern in cls.LOGIC_PATTERNS):
            category = ErrorCategory.LOGIC
            severity = 'medium'
            search_priority = 3
        
        # Determine recommended model size
        recommended_model = cls._recommend_model_size(category, severity, context)
        
        return CategorizedError(
            category=category,
            error_message=error,
            severity=severity,
            retry_limit=cls.RETRY_LIMITS[category],
            recommended_model_size=recommended_model,
            search_priority=search_priority
        )
    
    @staticmethod
    def _determine_compile_severity(error: str) -> str:
        """Determine compilation error severity"""
        error_lower = error.lower()
        
        if any(kw in error_lower for kw in ['fatal', 'cannot', 'undefined reference']):
            return 'high'
        elif any(kw in error_lower for kw in ['warning', 'deprecated']):
            return 'low'
        else:
            return 'medium'
    
    @staticmethod
    def _determine_runtime_severity(error: str) -> str:
        """Determine runtime error severity"""
        error_lower = error.lower()
        
        if any(kw in error_lower for kw in ['segmentation', 'cuda.*out of memory', 'fatal']):
            return 'critical'
        elif any(kw in error_lower for kw in ['nullpointer', 'assertion']):
            return 'high'
        else:
            return 'medium'
    
    @staticmethod
    def _determine_performance_severity(error: str) -> str:
        """Determine performance issue severity"""
        error_lower = error.lower()
        
        if any(kw in error_lower for kw in ['critical', 'severe', 'high']):
            return 'high'
        elif any(kw in error_lower for kw in ['low', 'minor']):
            return 'low'
        else:
            return 'medium'
    
    @staticmethod
    def _recommend_model_size(
        category: ErrorCategory, 
        severity: str, 
        context: Optional[Dict]
    ) -> str:
        """Recommend model size based on error characteristics"""
        
        # Start with base recommendation
        if category == ErrorCategory.COMPILATION:
            base = '16b'  # Most compile errors are straightforward
        elif category == ErrorCategory.PERFORMANCE:
            base = '32b'  # Performance needs deep analysis
        elif category == ErrorCategory.LOGIC:
            base = '32b'  # Logic errors need reasoning
        else:
            base = '16b'  # Default
        
        # Adjust for severity
        if severity == 'critical':
            base = '32b'
        elif severity == 'high' and base == '16b':
            base = '16b'  # Keep 16b for high
        
        # Adjust for failure count
        if context:
            failure_count = context.get('failure_count', 0)
            if failure_count >= 3:
                base = '32b'  # Escalate after 3 failures
        
        return base


# ============================================================================
# MODEL ESCALATION CHAINS
# ============================================================================

class ModelEscalationChain:
    """
    Defines model escalation paths for different agents
    """
    
    CHAINS = {
        'builder': [
            'deepseek-coder:16b',      # Fast, good for simple/medium
            'qwen2.5-coder:32b',       # Complex tasks
        ],
        'fixer': [
            'deepseek-coder:16b',      # Initial fix attempts
            'qwen2.5-coder:32b',       # Complex debugging
            'deepseek-r1:32b',         # Deep reasoning (last resort)
        ],
        'tester': [
            'phi4:latest',             # Fast test generation
            'deepseek-coder:16b',      # Complex tests
        ],
        'optimizer': [
            'deepseek-coder:16b',      # Initial optimization
            'qwen2.5-coder:32b',       # Deep optimization
        ]
    }
    
    @classmethod
    def get_model_for_attempt(
        cls, 
        agent_type: str, 
        attempt: int, 
        recommended_size: str = None
    ) -> str:
        """
        Get model for specific attempt number
        
        Args:
            agent_type: Agent type (builder/fixer/tester)
            attempt: Attempt number (0-indexed)
            recommended_size: Optional recommended model size override
            
        Returns:
            Model name
        """
        
        chain = cls.CHAINS.get(agent_type, cls.CHAINS['fixer'])
        
        # Override with recommended size if provided
        if recommended_size:
            if recommended_size == '32b':
                # Force largest model
                return chain[-1]
            elif recommended_size == '16b' and len(chain) >= 2:
                # Use middle model
                return chain[min(1, len(chain)-1)]
        
        # Normal escalation
        idx = min(attempt, len(chain) - 1)
        return chain[idx]


# ============================================================================
# HYBRID ERROR HANDLER
# ============================================================================

class HybridErrorHandler:
    """
    Intelligent error handling with:
    - Error categorization
    - Confidence-based decisions
    - Retry limits per category
    - Model escalation
    - SearchAgent integration
    """
    
    def __init__(
        self, 
        learning_module=None,
        confidence_scorer=None,
        search_agent=None,
        verbose: bool = True
    ):
        """
        Initialize Hybrid Error Handler
        
        Args:
            learning_module: LearningModuleV2 instance
            confidence_scorer: ConfidenceScorer instance
            search_agent: SearchAgent instance
            verbose: Print status messages
        """
        
        self.learning = learning_module
        self.confidence_scorer = confidence_scorer
        self.search = search_agent
        self.verbose = verbose
        
        self.categorizer = ErrorCategorizer()
        self.escalation = ModelEscalationChain()
        
        # Statistics
        self.stats = {
            'total_errors': 0,
            'by_category': {},
            'cache_hits': 0,
            'retries': 0,
            'escalations': 0,
            'search_triggered': 0,
            'resolved': 0,
            'failed': 0
        }
        
        self._print("[HybridErrorHandler] ✓ Initialized")
    
    def _print(self, msg: str):
        """Print if verbose"""
        if self.verbose:
            print(msg)
    
    async def handle_error(
        self,
        error: str,
        code: str,
        language: str,
        agent_type: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Handle error with hybrid strategy
        
        Args:
            error: Error message
            code: Source code
            language: Programming language
            agent_type: Agent type (builder/fixer/tester)
            context: Optional context
            
        Returns:
            {
                'action': 'use_cache' | 'retry' | 'escalate' | 'search',
                'model': str,
                'confidence': float,
                'categorized_error': CategorizedError,
                'similar_solutions': List[Dict],
                'search_results': str | None
            }
        """
        
        self.stats['total_errors'] += 1
        
        # Step 1: Categorize error
        self._print("\n[HybridErrorHandler] Step 1: Categorizing error...")
        categorized = self.categorizer.categorize(error, code, context)
        
        self._print(f"[HybridErrorHandler] Category: {categorized.category.value.upper()}")
        self._print(f"[HybridErrorHandler] Severity: {categorized.severity}")
        self._print(f"[HybridErrorHandler] Retry Limit: {categorized.retry_limit}")
        
        # Update stats
        cat_name = categorized.category.value
        self.stats['by_category'][cat_name] = self.stats['by_category'].get(cat_name, 0) + 1
        
        # Get attempt count
        attempt = context.get('attempt', 0) if context else 0
        
        result = {
            'action': None,
            'model': None,
            'confidence': 0.0,
            'categorized_error': categorized,
            'similar_solutions': [],
            'search_results': None,
            'retry_limit_reached': False
        }
        
        # Step 2: Check if we've exceeded retry limit
        if attempt >= categorized.retry_limit:
            self._print(f"[HybridErrorHandler] ⚠️  Retry limit reached ({attempt}/{categorized.retry_limit})")
            result['retry_limit_reached'] = True
            
            # Last resort: SearchAgent
            if self.search and attempt < categorized.retry_limit + 1:
                self._print("[HybridErrorHandler] Action: SEARCH (last resort)")
                result['action'] = 'search'
                result['model'] = self.escalation.get_model_for_attempt(
                    agent_type, 
                    attempt,
                    categorized.recommended_model_size
                )
                self.stats['search_triggered'] += 1
            else:
                self._print("[HybridErrorHandler] Action: GIVE UP")
                result['action'] = 'give_up'
                self.stats['failed'] += 1
            
            return result
        
        # Step 3: Check learning database for similar solutions
        similar_solutions = []
        if self.learning:
            self._print("\n[HybridErrorHandler] Step 2: Checking learning database...")
            
            # Note: find_similar_solutions is NOT async - returns list directly
            similar_solutions = self.learning.find_similar_solutions(
                error=error,
                code=code,
                model_used=context.get('model_used', 'unknown') if context else 'unknown',
                min_confidence=0.60,
                max_results=3
            )
            
            result['similar_solutions'] = similar_solutions
            
            if similar_solutions:
                best = similar_solutions[0]
                confidence = best['confidence']
                
                self._print(f"[HybridErrorHandler] Found {len(similar_solutions)} similar solutions")
                self._print(f"[HybridErrorHandler] Best confidence: {confidence:.1%}")
                
                # Step 4: Make decision based on confidence
                if confidence >= 0.85:
                    # HIGH CONFIDENCE: Use cached solution
                    self._print("[HybridErrorHandler] Action: USE_CACHE (high confidence)")
                    result['action'] = 'use_cache'
                    result['confidence'] = confidence
                    result['model'] = best['solution'].get('model_used', 'unknown')
                    self.stats['cache_hits'] += 1
                    self.stats['resolved'] += 1
                    return result
                
                elif confidence >= 0.60:
                    # MEDIUM CONFIDENCE: Retry with variation
                    self._print("[HybridErrorHandler] Action: RETRY (medium confidence + prompt variation)")
                    result['action'] = 'retry'
                    result['confidence'] = confidence
                    result['model'] = self.escalation.get_model_for_attempt(
                        agent_type,
                        attempt,
                        categorized.recommended_model_size
                    )
                    self.stats['retries'] += 1
                    return result
        
        # Step 5: No good cached solution - decide between escalate and search
        
        # Check if we should search early (compile errors in CUDA/C++)
        should_search_early = (
            categorized.category == ErrorCategory.COMPILATION and
            language.lower() in ['cuda', 'cu', 'cpp', 'c++', 'c'] and
            self.search
        )
        
        if should_search_early and attempt == 0:
            self._print("[HybridErrorHandler] Action: SEARCH (compile error, first attempt)")
            result['action'] = 'search'
            result['model'] = self.escalation.get_model_for_attempt(
                agent_type,
                attempt,
                categorized.recommended_model_size
            )
            self.stats['search_triggered'] += 1
            return result
        
        # Check if we should search based on priority
        if self.search and attempt >= (5 - categorized.search_priority):
            self._print(f"[HybridErrorHandler] Action: SEARCH (priority {categorized.search_priority}, attempt {attempt})")
            result['action'] = 'search'
            result['model'] = self.escalation.get_model_for_attempt(
                agent_type,
                attempt,
                categorized.recommended_model_size
            )
            self.stats['search_triggered'] += 1
            return result
        
        # Default: Escalate to next model
        self._print(f"[HybridErrorHandler] Action: ESCALATE (attempt {attempt})")
        result['action'] = 'escalate'
        result['model'] = self.escalation.get_model_for_attempt(
            agent_type,
            attempt,
            categorized.recommended_model_size
        )
        result['confidence'] = 0.0
        self.stats['escalations'] += 1
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get handler statistics"""
        return {
            **self.stats,
            'success_rate': (
                self.stats['resolved'] / self.stats['total_errors']
                if self.stats['total_errors'] > 0
                else 0.0
            )
        }


# ============================================================================
# TESTING
# ============================================================================

async def test_hybrid_handler():
    """Test Hybrid Error Handler"""
    
    print("\n" + "="*70)
    print("HYBRID ERROR HANDLER TEST")
    print("="*70)
    
    handler = HybridErrorHandler(verbose=True)
    
    # Test cases
    test_cases = [
        {
            'error': 'error C2019: unexpected preprocessor directive "# 123"',
            'code': '# 123\n#include <cuda_runtime.h>',
            'language': 'cuda',
            'agent_type': 'builder',
            'context': {'attempt': 0}
        },
        {
            'error': 'RuntimeError: CUDA out of memory',
            'code': 'cudaMalloc(&ptr, 1000000000000);',
            'language': 'cuda',
            'agent_type': 'fixer',
            'context': {'attempt': 0}
        },
        {
            'error': 'Performance: Low occupancy detected (15%)',
            'code': '__global__ void kernel() { }',
            'language': 'cuda',
            'agent_type': 'fixer',
            'context': {'attempt': 2}
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}:")
        print(f"  Error: {test['error'][:60]}...")
        print(f"{'='*70}")
        
        result = await handler.handle_error(**test)
        
        print(f"\nResult:")
        print(f"  Action: {result['action']}")
        print(f"  Model: {result['model']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Category: {result['categorized_error'].category.value}")
        print(f"  Severity: {result['categorized_error'].severity}")
    
    print(f"\n{'='*70}")
    print("STATISTICS:")
    print(f"{'='*70}")
    stats = handler.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import asyncio
    asyncio.run(test_hybrid_handler())
