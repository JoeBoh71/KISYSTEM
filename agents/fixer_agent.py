"""
KISYSTEM FixerAgent - Phase 5 Complete + RUN 37 Fixes
Enhanced with Smart Escalation + Model Routing + Auto-Dependency Management

RUN 37 Fixes (v2.4):
- failure_count from supervisor context (Line 90-95)
- Dynamic temperature: 0.3 + (failure_count * 0.1) (Line 582)
- Full error printing for debugging (Line 78-84)

RUN 37.1 Improvements (v2.5):
- Minimal surgical fix prompts to prevent aggressive code changes
- Explicit instructions: "Do NOT delete working code"
- Specific examples for common errors (M_PI, headers, APIs)
- Preservation of existing functionality emphasized

Author: Jörg Bohne
Date: 2025-11-11
Version: 2.5
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


class FixerAgent:
    """
    Intelligent debugging and fixing agent
    
    Features:
    - Smart model escalation (3+ failures → deepseek-r1:32b)
    - Auto-dependency detection and installation
    - Learning from previous fixes
    - Root cause analysis for complex bugs
    """
    
    def __init__(self, learning_module=None, search_agent=None):
        """Initialize FixerAgent"""
        
        # Model selection with escalation
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
        
        # Track failures for escalation
        self.failure_history = {}
        
        print("[FixerAgent] ✓ Initialized with Smart Escalation + Auto-Dependencies")
    
    async def fix(
        self,
        code: str,
        error: str,
        language: str = "python",
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Fix code with smart model escalation
        
        Args:
            code: Broken code
            error: Error message
            language: Programming language
            context: Optional additional context
            
        Returns:
            Result dict with fixed_code, model_used, escalation_level, etc.
        """
        
        print(f"\n{'='*70}")
        print(f"[FixerAgent] 🔧 Fixing {language} code...")
        print(f"[FixerAgent] Error (first 200 chars): {error[:200]}...")
        # Always print full error for debugging
        print(f"[FixerAgent] Full error ({len(error)} chars):")
        print(error)
        print(f"{'='*70}\n")
        
        # Create task ID for tracking failures
        task_id = self._create_task_id(code, error)
        
        # Get failure count from supervisor context (preferred) or internal history
        if context and "failure_count" in context:
            failure_count = context["failure_count"]
            print(f"[FixerAgent] Using failure_count from supervisor: {failure_count}")
        else:
            failure_count = self.failure_history.get(task_id, 0)
            print(f"[FixerAgent] Using internal failure_count: {failure_count}")
        
        result = {
            "status": "pending",
            "language": language,
            "error": error,
            "fixed_code": None,
            "model_used": None,
            "escalation_level": failure_count,
            "dependencies_installed": [],
            "search_triggered": False,
            "errors": []
        }
        
        try:
            # Step 1: Detect missing dependencies from error
            print("[FixerAgent] Step 1: Analyzing error for missing dependencies...")
            missing_deps = self._detect_missing_dependencies(error, language)
            
            if missing_deps:
                print(f"[FixerAgent] 📦 Detected missing packages: {missing_deps}")
                
                dep_results = await self.workflow_engine.installer.ensure_dependencies(
                    missing_deps
                )
                
                result["dependencies_installed"] = list(dep_results.keys())
                
                # If deps were installed, might be fixed now
                failed = [pkg for pkg, ok in dep_results.items() if not ok]
                if not failed:
                    print(f"[FixerAgent] ✓ Dependencies installed - code might be fixed!")
                    result["status"] = "completed"
                    result["fixed_code"] = code  # No code change needed
                    result["fix_method"] = "dependency_installation"
                    return result
            
            # Step 2: Check if we need web search
            # For compile errors in CUDA/C++: Search immediately for solutions
            should_search = False
            
            if self.search:
                # PRIORITY 1: API errors (undefined symbols) - search IMMEDIATELY
                api_error_keywords = ['undefined', 'unresolved', 'not declared', 'undeclared']
                if language.lower() in ['cuda', 'cu', 'cpp', 'c++', 'c']:
                    if any(kw in error.lower() for kw in api_error_keywords):
                        should_search = True
                        print(f"\n[FixerAgent] 🔍 API error detected (undefined symbol) - searching web immediately...")
                    # PRIORITY 2: Other compile errors - search after 3 failures
                    elif 'error' in error.lower() and failure_count >= 3:
                        should_search = True
                        print(f"\n[FixerAgent] ⚠️ {failure_count} failures - triggering web search...")
                # PRIORITY 3: Non-compile errors - search after 3 failures
                elif failure_count >= 3:
                    should_search = True
                    print(f"\n[FixerAgent] ⚠️ {failure_count} failures - triggering web search...")
                
                if should_search:
                    search_result = await self.search.search(
                        query=f"{language} {error[:100]}"
                    )
                    result["search_triggered"] = True
                    result["search_results"] = search_result
                    print(f"[FixerAgent] 🔍 Search completed")
            
            # Step 3: Select model with escalation
            print(f"\n[FixerAgent] Step 3: Selecting model (failure count: {failure_count})...")
            
            # Build context with failure count for escalation
            fix_context = {
                "failure_count": failure_count,
                "has_search_results": result.get("search_triggered", False),
                "search_results": result.get("search_results", ""),
                **(context or {})
            }
            
            model_config = self.model_selector.select_model(
                task=f"Fix {language} error: {error[:50]}",
                agent_type="fixer",
                context=fix_context
            )
            
            result["model_used"] = model_config.name
            
            # Log escalation
            if failure_count >= 3:
                print(f"[FixerAgent] 📈 ESCALATED to {model_config.name} (deep debugging mode)")
            
            # Step 4: Check learning database
            if self.learning:
                print("\n[FixerAgent] Step 3: Checking learning database...")
                similar = await self.learning.find_similar(error, threshold=0.8)
                
                if similar:
                    print(f"[FixerAgent] 💡 Found similar error fix!")
                    print(f"[FixerAgent] Similarity: {similar['similarity']*100:.1f}%")
                    result["similar_fix"] = similar
            
            # Step 5: Generate fix
            print(f"\n[FixerAgent] Step 4: Generating fix with {model_config.name}...")
            fixed_code = await self._generate_fix(
                code=code,
                error=error,
                language=language,
                model=model_config.name,
                context=fix_context
            )
            
            result["fixed_code"] = fixed_code
            result["status"] = "completed"
            
            print(f"[FixerAgent] ✓ Fix generated")
            
            # Step 6: Reset failure counter on success
            if task_id in self.failure_history:
                del self.failure_history[task_id]
                print(f"[FixerAgent] ✓ Failure counter reset")
            
            # Step 7: Store in learning database
            if self.learning and result["status"] == "completed":
                await self.learning.store_success(
                    task=error,
                    solution=fixed_code,
                    metadata={
                        "model": model_config.name,
                        "language": language,
                        "escalation_level": failure_count
                    }
                )
                print("[FixerAgent] ✓ Fix stored in learning database")
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            
            # Increment failure counter
            self.failure_history[task_id] = failure_count + 1
            
            print(f"[FixerAgent] ✗ Fix failed: {e}")
            print(f"[FixerAgent] Failure count now: {self.failure_history[task_id]}")
        
        print(f"\n{'='*70}")
        print(f"[FixerAgent] Status: {result['status'].upper()}")
        print(f"[FixerAgent] Escalation Level: {result['escalation_level']}")
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
        
        Args:
            code: Current code
            performance_suggestions: List of optimization suggestions from profiler
            language: Programming language (default: cuda)
            context: Optional context
            
        Returns:
            Result dict with optimized_code, model_used, etc.
        """
        
        print(f"\n{'='*70}")
        print(f"[FixerAgent] ⚡ Optimizing {language} code for performance")
        print(f"[FixerAgent] Issues found: {len(performance_suggestions)}")
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
            # Step 1: Analyze suggestions and select model
            print("[FixerAgent] Step 1: Analyzing performance issues...")
            
            # Prioritize by severity
            high_severity = [s for s in performance_suggestions if s['severity'] == 'high']
            medium_severity = [s for s in performance_suggestions if s['severity'] == 'medium']
            
            if high_severity:
                print(f"[FixerAgent] ⚠️  {len(high_severity)} high-severity issues")
            if medium_severity:
                print(f"[FixerAgent] ⚠️  {len(medium_severity)} medium-severity issues")
            
            # Select model based on complexity
            if len(performance_suggestions) > 2 or high_severity:
                complexity = "complex"
            else:
                complexity = "medium"
            
            print("\n[FixerAgent] Step 2: Selecting model...")
            model_config = self.model_selector.select_model(
                task=f"Optimize {language} performance",
                agent_type="fixer",
                context={"complexity": complexity, **(context or {})}
            )
            
            result["model_used"] = model_config.name
            print(f"[FixerAgent] Model: {model_config.name}")
            
            # Step 3: Build optimization prompt
            print("\n[FixerAgent] Step 3: Generating optimizations...")
            optimized_code = await self._generate_performance_fix(
                code=code,
                suggestions=performance_suggestions,
                model=model_config.name,
                language=language
            )
            
            result["optimized_code"] = optimized_code
            result["status"] = "completed"
            result["suggestions_addressed"] = [s['issue'] for s in performance_suggestions]
            
            print(f"[FixerAgent] ✓ Optimizations generated")
            print(f"[FixerAgent] Addressed: {', '.join(result['suggestions_addressed'])}")
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            print(f"[FixerAgent] ✗ Optimization failed: {e}")
        
        print(f"\n{'='*70}")
        print(f"[FixerAgent] Status: {result['status'].upper()}")
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
        
        Args:
            code: Current code
            suggestions: Performance optimization suggestions
            model: Model to use
            language: Programming language
            
        Returns:
            Optimized code
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

CRITICAL INSTRUCTIONS:
- Output ONLY the optimized code
- NO explanations, NO markdown, NO comments about the changes
- Start directly with #include or code
- Do NOT wrap in ```cpp```, ```cuda```, or any other code blocks

Optimized code:"""
        
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
                temperature=0.2,  # Lower temp for precise optimizations
                timeout=timeout
            )
            
            # Extract code (remove markdown wrapper)
            fixed_code = extract_code(fixed_code.strip())
            return fixed_code
            
        except Exception as e:
            print(f"[FixerAgent] ✗ Ollama generation failed: {e}")
            
            # Log failure to learning module
            if self.learning:
                await self.learning.store_failure(
                    task=f"optimize_{language}_performance",
                    error_type="optimization_failure",
                    error_message=str(e)[:200],
                    context={
                        'model': model,
                        'language': language,
                        'num_suggestions': len(suggestions),
                        'timeout': 'timeout' in str(e).lower()
                    },
                    metadata={
                        'suggestions': [s['issue'] for s in suggestions],
                        'model_used': model
                    }
                )
            
            # Fallback
            return f"""// Performance optimization failed: {e}
// Original code:
{code}
"""
    
    def _create_task_id(self, code: str, error: str) -> str:
        """Create unique ID for tracking failures"""
        import hashlib
        content = f"{code[:100]}{error[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _detect_missing_dependencies(self, error: str, language: str) -> List[str]:
        """
        Detect missing dependencies from error message
        
        Args:
            error: Error message
            language: Programming language
            
        Returns:
            List of potentially missing packages
        """
        
        if language != "python":
            return []
        
        packages = []
        error_lower = error.lower()
        
        # Common import errors
        if "no module named" in error_lower or "modulenotfounderror" in error_lower:
            # Extract module name
            import re
            match = re.search(r"no module named ['\"]([^'\"]+)['\"]", error_lower)
            if match:
                module = match.group(1).split('.')[0]  # Get base module
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
        
        return list(set(packages))  # Unique
    
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
        
        # Build prompt with escalation awareness
        failure_count = context.get("failure_count", 0) if context else 0
        
        if failure_count >= 3:
            # Deep debugging prompt
            prompt = f"""DEEP DEBUGGING MODE - Root Cause Analysis

The following {language} code has failed {failure_count} times.
Perform a thorough root cause analysis.

Code:
{code}

Error:
{error}

Previous fixes have failed. Analyze:
1. What is the root cause?
2. Why did previous fixes fail?
3. What is the correct solution?

CRITICAL INSTRUCTIONS FOR MINIMAL SURGICAL FIX:
- Make the SMALLEST possible change to fix the error
- Do NOT delete working code
- Do NOT restructure or refactor
- ONLY fix the specific error mentioned
- If error is "M_PI undefined" → add #define M_PI, DO NOT rewrite code
- If error is "missing header" → add #include, DO NOT rewrite code
- If error is "wrong API" → replace API call, DO NOT rewrite code
- Preserve ALL existing functionality
- Output ONLY the corrected code
- NO explanations, NO markdown, NO comments about the changes
- Start directly with #include or code
- Do NOT wrap in ```cpp```, ```cuda```, or any other code blocks

Corrected code:"""
        else:
            # Standard fix prompt with minimal change emphasis
            prompt = f"""Fix the following {language} code that produces an error.

Code:
{code}

Error:
{error}

CRITICAL INSTRUCTIONS FOR MINIMAL SURGICAL FIX:
- Make the SMALLEST possible change to fix the error
- Do NOT delete working code
- Do NOT restructure or refactor unless absolutely necessary
- ONLY fix the specific error mentioned
- If error is "identifier undefined" → add #define or #include, DO NOT rewrite
- If error is "missing header" → add #include, DO NOT rewrite
- If error is "expression must be constant" → use constexpr or static, DO NOT delete arrays
- If error is "wrong API" → replace API call only, DO NOT rewrite
- Preserve ALL existing functionality and logic
- Output ONLY the corrected code
- NO explanations, NO markdown, NO comments about the changes
- Start directly with #include or code
- Do NOT wrap in ```cpp```, ```cuda```, or any other code blocks

Corrected code:"""
        
        # Import Ollama client
        from ollama_client import OllamaClient
        
        # Add search results if available
        if context and context.get("has_search_results"):
            search_results = context.get("search_results", "")
            if search_results:
                prompt += f"""

WEB SEARCH RESULTS (StackOverflow, Documentation):
{search_results[:1000]}

Use the information above to help fix the error."""
        
        # Generate with Ollama
        client = OllamaClient()
        
        # Dynamic timeout based on model size
        if "32b" in model.lower():
            timeout = 1800  # 30 minutes for 32b models
        elif "16b" in model.lower():
            timeout = 600  # 10 minutes for 16b models
        else:
            timeout = 300  # 5 minutes for small models
        
        try:
            # Increase temperature with each retry to prevent identical outputs
            retry_temp = 0.3 + (failure_count * 0.1)  # 0.3 → 0.4 → 0.5 → 0.6
            
            fixed_code = await client.generate(
                model=model,
                prompt=prompt,
                temperature=retry_temp,
                timeout=timeout
            )
            
            # Extract code (remove markdown wrapper)
            fixed_code = extract_code(fixed_code.strip())
            
            # AUTO-ADD REQUIRED INCLUDES FOR CUDA/C++
            if language.lower() in ['cuda', 'cu', 'cpp', 'c++', 'c']:
                # Import CUDA profiler's include helper
                sys.path.insert(0, str(Path(__file__).parent))
                from cuda_profiler_agent import ensure_required_includes
                
                fixed_code, added_includes = ensure_required_includes(fixed_code)
                if added_includes:
                    print(f"[FixerAgent] ✓ Auto-added {len(added_includes)} includes:")
                    for inc in added_includes:
                        print(f"[FixerAgent]   • {inc}")
            
            return fixed_code
            
        except Exception as e:
            print(f"[FixerAgent] ✗ Ollama generation failed: {e}")
            
            # Log failure to learning module
            if self.learning:
                await self.learning.store_failure(
                    task=f"fix_{language}_error",
                    error_type="generation_failure",
                    error_message=str(e)[:200],
                    context={
                        'model': model,
                        'language': language,
                        'original_error': error[:100],
                        'failure_count': failure_count,
                        'timeout': 'timeout' in str(e).lower()
                    },
                    metadata={
                        'escalation_level': failure_count,
                        'model_used': model
                    }
                )
            
            # Fallback
            return f"""# Fix generation failed: {e}
# Original error: {error[:50]}
# Escalation level: {failure_count}

{code}
"""
    
    def get_failure_stats(self) -> Dict:
        """Get failure tracking statistics"""
        return {
            "active_failures": len(self.failure_history),
            "max_failures": max(self.failure_history.values()) if self.failure_history else 0,
            "tasks": self.failure_history
        }


# ============================================================================
# TESTING
# ============================================================================

async def test_fixer():
    """Test FixerAgent with escalation"""
    
    print("\n" + "="*70)
    print("FIXERAGENT TEST - Phase 5 Complete")
    print("="*70)
    
    fixer = FixerAgent()
    
    broken_code = """
import numpy as np

def calculate():
    return np.sum([1, 2, 3])
"""
    
    error = "ModuleNotFoundError: No module named 'numpy'"
    
    # First attempt
    result1 = await fixer.fix(broken_code, error)
    print(f"\nAttempt 1:")
    print(f"  Status: {result1['status']}")
    print(f"  Model: {result1['model_used']}")
    print(f"  Escalation: {result1['escalation_level']}")
    print(f"  Dependencies: {result1['dependencies_installed']}")
    
    # Simulate failure and retry (to test escalation)
    if result1['status'] == 'failed':
        result2 = await fixer.fix(broken_code, error)
        print(f"\nAttempt 2 (after failure):")
        print(f"  Model: {result2['model_used']}")
        print(f"  Escalation: {result2['escalation_level']}")
    
    print(f"\nFailure Stats:")
    print(f"  {fixer.get_failure_stats()}")


if __name__ == "__main__":
    asyncio.run(test_fixer())
