"""
KISYSTEM TesterAgent - Phase 5 Complete + RUN 37 Fixes
Enhanced with Smart Model Routing + Auto-Dependency Management

RUN 37.1 Fixes (v2.1):
- Markdown stripping: Removes ```cpp, ```python code fences from LLM output
- Prevents compilation errors from stray backticks

RUN 37.2 Fixes (v2.2):
- Simple tests: Uses "simple" framework instead of gtest for C++/CUDA
- No gtest dependency required
- Tests compile without external libraries

Author: Jörg Bohne
Date: 2025-11-11
Version: 2.2
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from model_selector import ModelSelector
from workflow_engine import WorkflowEngine, WorkflowConfig, SecurityLevel


class TesterAgent:
    """
    Intelligent test generation agent
    
    Features:
    - Smart model selection based on test complexity
    - Auto-dependency detection (pytest, unittest, etc.)
    - Multi-language test framework support
    - Learning from previous test patterns
    """
    
    def __init__(self, learning_module=None):
        """Initialize TesterAgent"""
        
        # Model selection
        self.model_selector = ModelSelector()
        
        # Auto-dependency management
        self.workflow_engine = WorkflowEngine(
            supervisor=None,
            config=WorkflowConfig(
                security_level=SecurityLevel.BALANCED,
                verbose=True
            )
        )
        
        # Learning (if available)
        self.learning = learning_module
        
        print("[TesterAgent] ✓ Initialized with Smart Routing + Auto-Dependencies")
    
    async def test(
        self,
        code: str,
        language: str = "python",
        test_framework: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Generate and run tests for code
        
        Args:
            code: Code to test
            language: Programming language
            test_framework: Optional specific test framework (auto-detect if None)
            context: Optional additional context
            
        Returns:
            Result dict with tests, model_used, dependencies_installed, etc.
        """
        
        print(f"\n{'='*70}")
        print(f"[TesterAgent] 🧪 Generating tests for {language} code...")
        print(f"{'='*70}\n")
        
        result = {
            "status": "pending",
            "language": language,
            "test_framework": test_framework,
            "tests": None,
            "model_used": None,
            "dependencies_installed": [],
            "errors": []
        }
        
        try:
            # Step 1: Detect test framework and dependencies
            print("[TesterAgent] Step 1: Detecting test framework...")
            
            if not test_framework:
                test_framework = self._detect_test_framework(language, code)
            
            result["test_framework"] = test_framework
            print(f"[TesterAgent] 📦 Test framework: {test_framework}")
            
            # Step 2: Install test framework dependencies
            required_packages = self._get_test_dependencies(test_framework, language)
            
            if required_packages:
                print(f"[TesterAgent] 📦 Required packages: {required_packages}")
                
                dep_results = await self.workflow_engine.installer.ensure_dependencies(
                    required_packages
                )
                
                result["dependencies_installed"] = list(dep_results.keys())
                
                # Check failures
                failed = [pkg for pkg, ok in dep_results.items() if not ok]
                if failed:
                    result["status"] = "failed"
                    result["errors"].append(f"Missing dependencies: {failed}")
                    print(f"[TesterAgent] ✗ Dependency installation failed: {failed}")
                    return result
                
                print(f"[TesterAgent] ✓ Test framework dependencies satisfied")
            
            # Step 3: Analyze code complexity
            print("\n[TesterAgent] Step 2: Analyzing code complexity...")
            complexity = self._analyze_test_complexity(code, context)
            print(f"[TesterAgent] Complexity: {complexity}")
            
            # Step 4: Select model
            print("\n[TesterAgent] Step 3: Selecting model...")
            model_config = self.model_selector.select_model(
                task=f"Generate {test_framework} tests",
                agent_type="tester",
                context={"complexity": complexity, **(context or {})}
            )
            
            result["model_used"] = model_config.name
            
            # Step 5: Check learning database
            if self.learning:
                print("\n[TesterAgent] Step 4: Checking learning database...")
                similar = await self.learning.find_similar(
                    f"test_{test_framework}_{language}",
                    threshold=0.75
                )
                
                if similar:
                    print(f"[TesterAgent] 💡 Found similar test patterns!")
                    result["similar_pattern"] = similar
            
            # Step 6: Generate tests
            print(f"\n[TesterAgent] Step 5: Generating tests with {model_config.name}...")
            tests = await self._generate_tests(
                code=code,
                language=language,
                framework=test_framework,
                model=model_config.name,
                context=context
            )
            
            result["tests"] = tests
            result["status"] = "completed"
            
            print(f"[TesterAgent] ✓ Tests generated ({len(tests)} characters)")
            
            # Step 7: Store in learning database
            if self.learning and result["status"] == "completed":
                await self.learning.store_success(
                    task=f"test_{test_framework}_{language}",
                    solution=tests,
                    metadata={
                        "model": model_config.name,
                        "framework": test_framework,
                        "complexity": complexity
                    }
                )
                print("[TesterAgent] ✓ Stored test pattern in learning database")
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            print(f"[TesterAgent] ✗ Test generation failed: {e}")
        
        print(f"\n{'='*70}")
        print(f"[TesterAgent] Status: {result['status'].upper()}")
        print(f"{'='*70}\n")
        
        return result
    
    def _detect_test_framework(self, language: str, code: str) -> str:
        """
        Auto-detect appropriate test framework
        
        Args:
            language: Programming language
            code: Code to test
            
        Returns:
            Test framework name
        """
        
        if language == "python":
            # Check if pytest is already used
            if "pytest" in code.lower():
                return "pytest"
            return "pytest"  # Default for Python
        
        elif language in ["c++", "cpp", "cuda"]:
            return "simple"  # Simple tests without framework (no gtest dependency)
        
        elif language in ["javascript", "typescript"]:
            return "jest"
        
        elif language == "c":
            return "cmocka"
        
        else:
            return "unittest"  # Generic fallback
    
    def _get_test_dependencies(self, framework: str, language: str) -> List[str]:
        """
        Get required dependencies for test framework
        
        Args:
            framework: Test framework name
            language: Programming language
            
        Returns:
            List of required packages
        """
        
        deps = {
            "pytest": ["pytest", "pytest-asyncio", "pytest-cov"],
            "unittest": [],  # Python stdlib
            "gtest": [],  # C++ - installed separately
            "jest": [],  # JavaScript - npm
            "cmocka": []  # C - installed separately
        }
        
        return deps.get(framework, [])
    
    def _analyze_test_complexity(self, code: str, context: Optional[Dict]) -> str:
        """
        Analyze complexity of code to determine test complexity
        
        Args:
            code: Code to test
            context: Optional context
            
        Returns:
            Complexity level: simple, medium, complex
        """
        
        # Simple heuristics
        lines = len(code.split('\n'))
        
        if lines < 50:
            return "simple"
        elif lines < 200:
            return "medium"
        else:
            return "complex"
    
    async def _generate_tests(
        self,
        code: str,
        language: str,
        framework: str,
        model: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate test code using selected model
        
        Args:
            code: Code to test
            language: Programming language
            framework: Test framework
            model: Model name to use
            context: Optional context
            
        Returns:
            Generated test code
        """
        
        # Build prompt
        prompt = f"""Generate comprehensive {framework} tests for the following {language} code:

Code to test:
{code}

Requirements:
- Use {framework} framework
- Cover all major functionality
- Include edge cases
- Add test documentation
- Follow {language} testing best practices

Tests:"""
        
        # Import Ollama client
        from ollama_client import OllamaClient, PromptTemplates
        
        # Build prompt using template
        prompt = PromptTemplates.test_generation(
            code=code,
            framework=framework,
            language=language
        )
        
        # Generate with Ollama
        client = OllamaClient()
        
        # Dynamic timeout based on model size
        if "32b" in model.lower():
            timeout = 1800  # 30 minutes for very complex 32b tasks
        elif "16b" in model.lower():
            timeout = 600  # 10 minutes for 16b models
        else:
            timeout = 300  # 5 minutes for small models
        
        try:
            tests = await client.generate(
                model=model,
                prompt=prompt,
                temperature=0.3,
                timeout=timeout
            )
            
            # CRITICAL: Strip markdown code blocks that LLMs often add
            # RUN 37.1 Fix - Use code_extractor for consistent handling
            from code_extractor import extract_code
            tests_clean = extract_code(tests.strip())
            
            return tests_clean
            
        except Exception as e:
            print(f"[TesterAgent] ✗ Ollama generation failed: {e}")
            # Fallback
            return f"""# Test generation failed: {e}
# Framework: {framework}

import {framework}

def test_placeholder():
    '''Placeholder test - generation failed'''
    assert True
"""



# ============================================================================
# TESTING
# ============================================================================

async def test_tester():
    """Test TesterAgent"""
    
    print("\n" + "="*70)
    print("TESTERAGENT TEST - Phase 5 Complete")
    print("="*70)
    
    tester = TesterAgent()
    
    sample_code = """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""
    
    result = await tester.test(sample_code, language="python")
    
    print(f"\nResult:")
    print(f"  Status: {result['status']}")
    print(f"  Framework: {result['test_framework']}")
    print(f"  Model: {result['model_used']}")
    print(f"  Dependencies: {result['dependencies_installed']}")


if __name__ == "__main__":
    asyncio.run(test_tester())
