"""
KISYSTEM BuilderAgent - Phase 5 Complete
Enhanced with Smart Model Routing + Auto-Dependency Management

Author: Jörg Bohne  
Date: 2025-11-06
Version: 2.0
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


class BuilderAgent:
    """
    Intelligent code building agent
    
    Features:
    - Smart model selection based on complexity
    - Auto-dependency detection and installation
    - Learning from previous builds
    - Multi-language support
    """
    
    def __init__(self, learning_module=None):
        """Initialize BuilderAgent"""
        
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
        
        print("[BuilderAgent] ✓ Initialized with Smart Routing + Auto-Dependencies")
    
    async def build(
        self, 
        task: str, 
        language: str = "python",
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Build code for task with smart model selection and dependency management
        
        Args:
            task: Description of what to build
            language: Programming language
            context: Optional additional context
            
        Returns:
            Result dict with code, model_used, dependencies_installed, etc.
        """
        
        print(f"\n{'='*70}")
        print(f"[BuilderAgent] 🏗️ Building: {task[:60]}...")
        print(f"{'='*70}\n")
        
        result = {
            "status": "pending",
            "task": task,
            "language": language,
            "code": None,
            "model_used": None,
            "dependencies_installed": [],
            "errors": []
        }
        
        try:
            # Step 1: Detect required dependencies
            print("[BuilderAgent] Step 1: Detecting dependencies...")
            required_packages = self._detect_dependencies(task, language)
            
            if required_packages:
                print(f"[BuilderAgent] 📦 Required packages: {required_packages}")
                
                # Install dependencies automatically
                dep_results = await self.workflow_engine.installer.ensure_dependencies(
                    required_packages
                )
                
                result["dependencies_installed"] = list(dep_results.keys())
                
                # Check if any failed
                failed = [pkg for pkg, ok in dep_results.items() if not ok]
                if failed:
                    result["status"] = "failed"
                    result["errors"].append(f"Missing dependencies: {failed}")
                    print(f"[BuilderAgent] ✗ Dependency installation failed: {failed}")
                    return result
                
                print(f"[BuilderAgent] ✓ All dependencies satisfied")
            else:
                print("[BuilderAgent] ✓ No external dependencies needed")
            
            # Step 2: Select appropriate model
            print("\n[BuilderAgent] Step 2: Selecting model...")
            model_config = self.model_selector.select_model(
                task=task,
                agent_type="builder",
                context=context
            )
            
            result["model_used"] = model_config.name
            
            # Step 3: Check learning database
            if self.learning:
                print("\n[BuilderAgent] Step 3: Checking learning database...")
                similar = await self.learning.find_similar(task, threshold=0.8)
                
                if similar:
                    print(f"[BuilderAgent] 💡 Found similar successful build!")
                    print(f"[BuilderAgent] Similarity: {similar['similarity']*100:.1f}%")
                    
                    # Could reuse solution with modifications
                    result["similar_solution"] = similar
            
            # Step 4: Generate code
            print(f"\n[BuilderAgent] Step 4: Generating code with {model_config.name}...")
            code = await self._generate_code(
                task=task,
                language=language,
                model=model_config.name,
                context=context
            )
            
            result["code"] = code
            result["status"] = "completed"
            
            print(f"[BuilderAgent] ✓ Code generated ({len(code)} characters)")
            
            # Step 5: Store in learning database
            if self.learning and result["status"] == "completed":
                await self.learning.store_success(
                    task=task,
                    solution=code,
                    metadata={
                        "model": model_config.name,
                        "language": language,
                        "dependencies": required_packages
                    }
                )
                print("[BuilderAgent] ✓ Stored in learning database")
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            print(f"[BuilderAgent] ✗ Build failed: {e}")
        
        print(f"\n{'='*70}")
        print(f"[BuilderAgent] Status: {result['status'].upper()}")
        print(f"{'='*70}\n")
        
        return result
    
    def _detect_dependencies(self, task: str, language: str) -> List[str]:
        """
        Detect required packages from task description
        
        Args:
            task: Task description
            language: Programming language
            
        Returns:
            List of required package names
        """
        
        if language != "python":
            return []  # Only Python for now
        
        packages = []
        task_lower = task.lower()
        
        # Audio processing
        if any(kw in task_lower for kw in ['audio', 'sound', 'wav', 'mp3', 'flac']):
            packages.extend(['soundfile', 'numpy', 'scipy'])
        
        # DSP / Signal processing
        if any(kw in task_lower for kw in ['fft', 'filter', 'spectrum', 'frequency', 'dsp']):
            packages.extend(['scipy', 'numpy'])
        
        # Plotting / Visualization
        if any(kw in task_lower for kw in ['plot', 'visualize', 'graph', 'chart']):
            packages.append('matplotlib')
        
        # Data processing
        if any(kw in task_lower for kw in ['dataframe', 'csv', 'data']):
            packages.extend(['pandas', 'numpy'])
        
        # Machine learning
        if any(kw in task_lower for kw in ['ml', 'machine learning', 'neural', 'train']):
            packages.append('scikit-learn')
        
        # CUDA / GPU
        if any(kw in task_lower for kw in ['cuda', 'gpu', 'cupy']):
            packages.extend(['cupy', 'numpy'])
        
        # Testing
        if 'test' in task_lower:
            packages.append('pytest')
        
        # Remove duplicates, maintain order
        seen = set()
        unique_packages = []
        for pkg in packages:
            if pkg not in seen:
                seen.add(pkg)
                unique_packages.append(pkg)
        
        return unique_packages
    
    async def _generate_code(
        self,
        task: str,
        language: str,
        model: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate code using selected model
        
        Args:
            task: What to build
            language: Programming language
            model: Model name to use
            context: Optional additional context
            
        Returns:
            Generated code as string
        """
        
        # Import Ollama client
        from ollama_client import OllamaClient, PromptTemplates
        
        # Build prompt using template
        additional_context = context.get("additional_info", "") if context else ""
        prompt = PromptTemplates.code_generation(
            task=task,
            language=language,
            context=additional_context
        )
        
        # Generate with Ollama
        client = OllamaClient()
        
        # Dynamic timeout based on model size
        # Large models (32b) need more time
        if "32b" in model.lower():
            timeout = 1800  # 30 minutes for 32b models
        elif "16b" in model.lower():
            timeout = 180  # 3 minutes for 16b models
        else:
            timeout = 120  # 2 minutes for small models
        
        print(f"[BuilderAgent] Timeout set to {timeout}s for {model}")
        
        try:
            code = await client.generate(
                model=model,
                prompt=prompt,
                temperature=0.3,  # Lower temp for more deterministic code
                timeout=timeout
            )
            
            # Extract code (remove markdown wrapper)
            code = extract_code(code.strip())
            
            # AUTO-ADD REQUIRED INCLUDES FOR CUDA/C++
            if language.lower() in ['cuda', 'cu', 'cpp', 'c++', 'c']:
                # Import CUDA profiler's include helper
                sys.path.insert(0, str(Path(__file__).parent))
                from cuda_profiler_agent import ensure_required_includes
                
                code, added_includes = ensure_required_includes(code)
                if added_includes:
                    print(f"[BuilderAgent] ✓ Auto-added {len(added_includes)} includes:")
                    for inc in added_includes:
                        print(f"[BuilderAgent]   • {inc}")
            
            return code
            
        except Exception as e:
            print(f"[BuilderAgent] ✗ Ollama generation failed: {e}")
            # Fallback to placeholder
            return f"""# Generation failed: {e}
# Task: {task}

def placeholder():
    '''Placeholder - generation failed'''
    pass
"""


# ============================================================================
# TESTING
# ============================================================================

async def test_builder():
    """Test BuilderAgent with various tasks"""
    
    print("\n" + "="*70)
    print("BUILDERAGENT TEST - Phase 5 Complete")
    print("="*70)
    
    builder = BuilderAgent()
    
    # Test cases
    test_cases = [
        ("Write a simple hello world function", "python"),
        ("Implement FFT audio spectrum analyzer with matplotlib", "python"),
        ("Create CUDA kernel for matrix multiplication", "cuda"),
    ]
    
    for task, lang in test_cases:
        print(f"\n{'='*70}")
        print(f"Test: {task}")
        print(f"{'='*70}")
        
        result = await builder.build(task, lang)
        
        print(f"\nResult:")
        print(f"  Status: {result['status']}")
        print(f"  Model: {result['model_used']}")
        print(f"  Dependencies: {result['dependencies_installed']}")
        if result['errors']:
            print(f"  Errors: {result['errors']}")
        
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(test_builder())
