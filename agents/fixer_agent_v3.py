"""
KISYSTEM FixerAgent V3 - With Hybrid Error Handler
Enhanced with intelligent error categorization and handling

Author: Jörg Bohne
Date: 2025-11-07
Version: 3.0
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from model_selector import ModelSelector
from workflow_engine import WorkflowEngine, WorkflowConfig, SecurityLevel
from code_extractor import extract_code
from error_handler import HybridErrorHandler


class FixerAgentV3:
    """
    Intelligent debugging and fixing agent with Hybrid Error Handler
    
    Features V3:
    - Hybrid error handling (categorization + confidence-based decisions)
    - Automatic retry limit enforcement per error category
    - Smart model escalation chains
    - SearchAgent integration based on error category
    - Learning from previous fixes
    """
    
    def __init__(self, learning_module=None, search_agent=None, confidence_scorer=None):
        """Initialize FixerAgent V3"""
        
        # Model selection (for fallback)
        self.model_selector = ModelSelector()
        
        # Auto-dependency management
        self.workflow_engine = WorkflowEngine(
            supervisor=None,
            config=WorkflowConfig(
                security_level=SecurityLevel.BALANCED,
                verbose=True
            )
        )
        
        # Learning and search
        self.learning = learning_module
        self.search = search_agent
        self.confidence_scorer = confidence_scorer
        
        # HYBRID ERROR HANDLER (NEW!)
        self.error_handler = HybridErrorHandler(
            learning_module=learning_module,
            confidence_scorer=confidence_scorer,
            search_agent=search_agent,
            verbose=True
        )
        
        print("[FixerAgent V3] ✓ Initialized with Hybrid Error Handler")
    
    async def fix(
        self,
        code: str,
        error: str,
        language: str = "python",
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Fix code using Hybrid Error Handler
        
        Args:
            code: Broken code
            error: Error message
            language: Programming language
            context: Optional additional context
            
        Returns:
            Result dict with fixed_code, model_used, decision_info, etc.
        """
        
        print(f"\n{'='*70}")
        print(f"[FixerAgent V3] 🔧 Fixing {language} code with Hybrid Handler...")
        print(f"[FixerAgent V3] Error: {error[:80]}...")
        print(f"{'='*70}\n")
        
        # Initialize context if not provided
        if context is None:
            context = {}
        
        # Get attempt number
        attempt = context.get('attempt', 0)
        
        result = {
            "status": "pending",
            "language": language,
            "error": error,
            "fixed_code": None,
            "model_used": None,
            "decision_info": None,
            "dependencies_installed": [],
            "search_triggered": False,
            "errors": []
        }
        
        try:
            # Step 1: Detect missing dependencies from error
            print("[FixerAgent V3] Step 1: Analyzing error for missing dependencies...")
            missing_deps = self._detect_missing_dependencies(error, language)
            
            if missing_deps:
                print(f"[FixerAgent V3] 📦 Detected missing packages: {missing_deps}")
                
                dep_results = await self.workflow_engine.installer.ensure_dependencies(
                    missing_deps
                )
                
                result["dependencies_installed"] = list(dep_results.keys())
                
                # If deps were installed, might be fixed now
                failed = [pkg for pkg, ok in dep_results.items() if not ok]
                if not failed:
                    print(f"[FixerAgent V3] ✓ Dependencies installed - code might be fixed!")
                    result["status"] = "completed"
                    result["fixed_code"] = code  # No code change needed
                    result["fix_method"] = "dependency_installation"
                    return result
            
            # Step 2: Use Hybrid Error Handler to make decision
            print(f"\n[FixerAgent V3] Step 2: Hybrid Error Handler decision (attempt {attempt})...")
            
            decision = await self.error_handler.handle_error(
                error=error,
                code=code,
                language=language,
                agent_type="fixer",
                context={
                    'attempt': attempt,
                    'model_used': context.get('model_used', 'unknown'),
                    **context
                }
            )
            
            result["decision_info"] = {
                'action': decision['action'],
                'category': decision['categorized_error'].category.value,
                'severity': decision['categorized_error'].severity,
                'confidence': decision['confidence'],
                'model': decision['model'],
                'retry_limit_reached': decision['retry_limit_reached']
            }
            
            print(f"\n[FixerAgent V3] Decision: {decision['action'].upper()}")
            print(f"[FixerAgent V3] Category: {decision['categorized_error'].category.value}")
            print(f"[FixerAgent V3] Severity: {decision['categorized_error'].severity}")
            print(f"[FixerAgent V3] Model: {decision['model']}")
            
            # Step 3: Execute decision
            if decision['action'] == 'use_cache':
                # Use cached solution
                print(f"[FixerAgent V3] ✓ Using cached solution (confidence: {decision['confidence']:.1%})")
                
                best_solution = decision['similar_solutions'][0]['solution']
                result["fixed_code"] = best_solution['solution']
                result["model_used"] = best_solution.get('model_used', 'cached')
                result["status"] = "completed"
                result["fix_method"] = "cached_solution"
                
                return result
            
            elif decision['action'] == 'retry':
                # Retry with prompt variation
                print(f"[FixerAgent V3] 🔄 Retrying with prompt variation (confidence: {decision['confidence']:.1%})")
                
                # Use similar solution as context for better prompt
                similar_context = ""
                if decision['similar_solutions']:
                    best = decision['similar_solutions'][0]
                    similar_context = f"\n\nSimilar past solution (confidence {best['confidence']:.0%}):\n{best['solution']['solution'][:500]}\n"
                
                fixed_code = await self._generate_fix(
                    code=code,
                    error=error,
                    language=language,
                    model=decision['model'],
                    context={**context, 'similar_solution_hint': similar_context}
                )
                
                result["fixed_code"] = fixed_code
                result["model_used"] = decision['model']
                result["status"] = "completed"
                result["fix_method"] = "retry_with_variation"
                
            elif decision['action'] == 'escalate':
                # Escalate to bigger model
                print(f"[FixerAgent V3] 📈 Escalating to {decision['model']}")
                
                fixed_code = await self._generate_fix(
                    code=code,
                    error=error,
                    language=language,
                    model=decision['model'],
                    context=context
                )
                
                result["fixed_code"] = fixed_code
                result["model_used"] = decision['model']
                result["status"] = "completed"
                result["fix_method"] = "escalation"
                
            elif decision['action'] == 'search':
                # Trigger web search
                print(f"[FixerAgent V3] 🔍 Triggering web search...")
                result["search_triggered"] = True
                
                if self.search:
                    search_result = await self.search.search(
                        query=f"{language} {error[:100]}"
                    )
                    result["search_results"] = search_result
                    print(f"[FixerAgent V3] 🔍 Search completed")
                    
                    # Use search results in fix
                    fixed_code = await self._generate_fix(
                        code=code,
                        error=error,
                        language=language,
                        model=decision['model'],
                        context={
                            **context,
                            'has_search_results': True,
                            'search_results': search_result
                        }
                    )
                    
                    result["fixed_code"] = fixed_code
                    result["model_used"] = decision['model']
                    result["status"] = "completed"
                    result["fix_method"] = "search_assisted"
                else:
                    # No search agent available
                    print(f"[FixerAgent V3] ⚠️  Search requested but SearchAgent not available")
                    result["status"] = "failed"
                    result["errors"].append("Search requested but SearchAgent not available")
                    return result
            
            elif decision['action'] == 'give_up':
                # All options exhausted
                print(f"[FixerAgent V3] ⛔ All retry options exhausted")
                result["status"] = "failed"
                result["errors"].append("Retry limit reached, unable to fix")
                return result
            
            # Step 4: Store success in learning database
            if result["status"] == "completed" and self.learning:
                # Note: store_solution is NOT async - returns int directly
                self.learning.store_solution(
                    error=error,
                    solution=result["fixed_code"],
                    code=code,
                    model_used=result["model_used"],
                    success=True,
                    solve_time=None
                )
                print("[FixerAgent V3] ✓ Fix stored in learning database")
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            
            print(f"[FixerAgent V3] ✗ Fix failed: {e}")
            
            # Store failure in learning (note: store_solution is NOT async)
            if self.learning:
                self.learning.store_solution(
                    error=error,
                    solution="",
                    code=code,
                    model_used=result.get("model_used", "unknown"),
                    success=False
                )
        
        print(f"\n{'='*70}")
        print(f"[FixerAgent V3] Status: {result['status'].upper()}")
        if result.get("decision_info"):
            print(f"[FixerAgent V3] Method: {result.get('fix_method', 'N/A')}")
        print(f"{'='*70}\n")
        
        return result
    
    async def fix_performance(
        self,
        code: str,
        performance_suggestions: List[Dict],
        language: str = "cuda",
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Fix performance issues based on profiler suggestions
        (This method unchanged from V2 - works well)
        
        Args:
            code: Current code
            performance_suggestions: List of optimization suggestions from profiler
            language: Programming language (default: cuda)
            context: Optional context
            
        Returns:
            Result dict with optimized_code, model_used, etc.
        """
        
        print(f"\n{'='*70}")
        print(f"[FixerAgent V3] ⚡ Optimizing {language} code for performance")
        print(f"[FixerAgent V3] Issues found: {len(performance_suggestions)}")
        print(f"{'='*70}\n")
        
        result = {
            "status": "pending",
            "language": language,
            "optimized_code": None,
            "model_used": None,
            "suggestions_addressed": [],
            "errors": []
        }
        
        try:
            # Analyze suggestions and select model
            print("[FixerAgent V3] Step 1: Analyzing performance issues...")
            
            # Prioritize by severity
            high_severity = [s for s in performance_suggestions if s['severity'] == 'high']
            medium_severity = [s for s in performance_suggestions if s['severity'] == 'medium']
            
            if high_severity:
                print(f"[FixerAgent V3] ⚠️  {len(high_severity)} high-severity issues")
            if medium_severity:
                print(f"[FixerAgent V3] ⚠️  {len(medium_severity)} medium-severity issues")
            
            # Select model based on complexity
            if len(performance_suggestions) > 2 or high_severity:
                complexity = "complex"
            else:
                complexity = "medium"
            
            print("\n[FixerAgent V3] Step 2: Selecting model...")
            model_config = self.model_selector.select_model(
                task=f"Optimize {language} performance",
                agent_type="fixer",
                context={"complexity": complexity, **(context or {})}
            )
            
            result["model_used"] = model_config.name
            print(f"[FixerAgent V3] Model: {model_config.name}")
            
            # Generate optimization
            print("\n[FixerAgent V3] Step 3: Generating optimizations...")
            optimized_code = await self._generate_performance_fix(
                code=code,
                suggestions=performance_suggestions,
                model=model_config.name,
                language=language
            )
            
            result["optimized_code"] = optimized_code
            result["status"] = "completed"
            result["suggestions_addressed"] = [s['issue'] for s in performance_suggestions]
            
            print(f"[FixerAgent V3] ✓ Optimizations generated")
            print(f"[FixerAgent V3] Addressed: {', '.join(result['suggestions_addressed'])}")
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            print(f"[FixerAgent V3] ✗ Optimization failed: {e}")
        
        print(f"\n{'='*70}")
        print(f"[FixerAgent V3] Status: {result['status'].upper()}")
        print(f"{'='*70}\n")
        
        return result
    
    async def _generate_performance_fix(
        self,
        code: str,
        suggestions: List[Dict],
        model: str,
        language: str
    ) -> str:
        """
        Generate performance-optimized code using LLM
        (Unchanged from V2)
        """
        
        # Build detailed prompt with specific fixes
        issues_text = "\n".join([
            f"{i+1}. {s['issue'].upper()} [{s['severity']}]\n"
            f"   Problem: {s['description']}\n"
            f"   Fix: {s['fix']}"
            for i, s in enumerate(suggestions)
        ])
        
        prompt = f"""Optimize this {language} code for performance.

Current Code:
```{language}
{code}
```

Performance Issues Detected by Profiler:
{issues_text}

Requirements:
- Address ALL performance issues listed above
- Maintain functional correctness (same behavior)
- Add comments explaining optimizations
- Keep code structure similar for easy diff

Optimized Code:"""
        
        # Import Ollama client
        from ollama_client import OllamaClient
        
        client = OllamaClient()
        
        # Dynamic timeout
        if "32b" in model.lower():
            timeout = 1800
        elif "16b" in model.lower():
            timeout = 600
        else:
            timeout = 300
        
        try:
            fixed_code = await client.generate(
                model=model,
                prompt=prompt,
                temperature=0.2,
                timeout=timeout
            )
            
            fixed_code = extract_code(fixed_code.strip())
            return fixed_code
            
        except Exception as e:
            print(f"[FixerAgent V3] ✗ Ollama generation failed: {e}")
            return f"""// Performance optimization failed: {e}
// Original code:
{code}
"""
    
    def _detect_missing_dependencies(self, error: str, language: str) -> List[str]:
        """
        Detect missing dependencies from error message
        (Unchanged from V2)
        """
        
        if language != "python":
            return []
        
        packages = []
        error_lower = error.lower()
        
        # Common import errors
        if "no module named" in error_lower or "modulenotfounderror" in error_lower:
            import re
            match = re.search(r"no module named ['\"]([^'\"]+)['\"]", error_lower)
            if match:
                module = match.group(1).split('.')[0]
                packages.append(module)
        
        # Common missing packages patterns
        missing_patterns = {
            'numpy': ['numpy', 'np.'],
            'scipy': ['scipy', 'scipy.'],
            'matplotlib': ['matplotlib', 'pyplot'],
            'pandas': ['pandas', 'pd.'],
            'soundfile': ['soundfile', 'sf.'],
            'pytest': ['pytest'],
            'aiohttp': ['aiohttp', 'asyncio'],
        }
        
        for package, patterns in missing_patterns.items():
            if any(pattern in error_lower for pattern in patterns):
                packages.append(package)
        
        return list(set(packages))
    
    async def _generate_fix(
        self,
        code: str,
        error: str,
        language: str,
        model: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate fixed code using selected model
        
        Args:
            code: Broken code
            error: Error message
            language: Programming language
            model: Model to use
            context: Optional context
            
        Returns:
            Fixed code
        """
        
        # Build prompt with context
        attempt = context.get('attempt', 0) if context else 0
        
        # Base prompt
        if attempt >= 3:
            # Deep debugging mode
            prompt = f"""DEEP DEBUGGING MODE - Root Cause Analysis

The following {language} code has failed {attempt} times.
Perform thorough root cause analysis.

Code:
{code}

Error:
{error}

Previous fixes failed. Analyze:
1. What is the ROOT CAUSE?
2. Why did previous fixes fail?
3. What is the CORRECT solution?

Provide fixed code with explanation:"""
        else:
            # Standard fix prompt
            prompt = f"""Fix the following {language} code that produces an error:

Code:
{code}

Error:
{error}

Provide the corrected code:"""
        
        # Add similar solution hint if available
        if context and context.get('similar_solution_hint'):
            prompt += f"""

HINT - Similar past solution:
{context['similar_solution_hint']}

Use this as guidance but adapt to the current error."""
        
        # Add search results if available
        if context and context.get('has_search_results'):
            search_results = context.get('search_results', '')
            if search_results:
                prompt += f"""

WEB SEARCH RESULTS:
{search_results[:1000]}

Use this information to help fix the error."""
        
        # Import Ollama client
        from ollama_client import OllamaClient
        
        client = OllamaClient()
        
        # Dynamic timeout
        if "32b" in model.lower():
            timeout = 1800
        elif "16b" in model.lower():
            timeout = 600
        else:
            timeout = 300
        
        try:
            fixed_code = await client.generate(
                model=model,
                prompt=prompt,
                temperature=0.3,
                timeout=timeout
            )
            
            fixed_code = extract_code(fixed_code.strip())
            
            # AUTO-ADD REQUIRED INCLUDES FOR CUDA/C++
            if language.lower() in ['cuda', 'cu', 'cpp', 'c++', 'c']:
                sys.path.insert(0, str(Path(__file__).parent))
                from cuda_profiler_agent import ensure_required_includes
                
                fixed_code, added_includes = ensure_required_includes(fixed_code)
                if added_includes:
                    print(f"[FixerAgent V3] ✓ Auto-added {len(added_includes)} includes:")
                    for inc in added_includes:
                        print(f"[FixerAgent V3]   • {inc}")
            
            return fixed_code
            
        except Exception as e:
            print(f"[FixerAgent V3] ✗ Ollama generation failed: {e}")
            return f"""# Fix generation failed: {e}
# Original error: {error[:50]}
# Attempt: {attempt}

{code}
"""
    
    def get_handler_statistics(self) -> Dict:
        """Get Hybrid Error Handler statistics"""
        return self.error_handler.get_statistics()


# ============================================================================
# TESTING
# ============================================================================

async def test_fixer_v3():
    """Test FixerAgent V3 with Hybrid Error Handler"""
    
    print("\n" + "="*70)
    print("FIXERAGENT V3 TEST - With Hybrid Error Handler")
    print("="*70)
    
    fixer = FixerAgentV3()
    
    # Test case: CUDA compile error
    broken_code = """
#include <cuda_runtime.h>

__global__ void testKernel(float* data) {
    int idx = threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}

int main() {
    return 0
}
"""
    
    error = "error C2143: syntax error: missing ';' before '}'"
    
    # Attempt 1
    print("\n=== ATTEMPT 1 ===")
    result1 = await fixer.fix(
        code=broken_code,
        error=error,
        language="cuda",
        context={'attempt': 0}
    )
    
    print(f"\nResult:")
    print(f"  Status: {result1['status']}")
    print(f"  Model: {result1['model_used']}")
    print(f"  Decision: {result1['decision_info']}")
    
    # Show handler stats
    print(f"\n=== HANDLER STATISTICS ===")
    stats = fixer.get_handler_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_fixer_v3())
