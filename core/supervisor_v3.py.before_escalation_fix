"""
KISYSTEM Supervisor V3 - Complete Orchestration with Phase 7 Optimization
Build-Test-Fix Loop with Smart Routing, Auto-Dependencies & Learning

Phase 7 Features:
- Meta-Supervisor: Data-driven task prioritization
- Hybrid Decision Logic: 40% Meta + 30% Complexity + 30% Failure
- 7-Model-Routing with domain-specific escalation
- Stop-Loss: 2 failures → escalate
- Two-Tier-Profiling support (Tier1: 100ms, Tier2: 1000ms)
- Cost-Aware Queue (ROI-based scheduling)

Author: Jörg Bohne
Date: 2025-11-10
Version: 3.1 (Phase 7)
"""

import asyncio
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from model_selector import ModelSelector
from workflow_engine import WorkflowEngine, WorkflowConfig, SecurityLevel

# Phase 7 imports
try:
    from meta_supervisor import MetaSupervisor
    from hybrid_decision import HybridDecision
    PHASE7_AVAILABLE = True
except ImportError:
    PHASE7_AVAILABLE = False
    print("[Supervisor V3] ⚠ Phase 7 modules not available - using Phase 6 mode")


class SupervisorV3:
    """
    Master orchestrator for KISYSTEM
    
    Features:
    - Complete Build-Test-Fix Loop
    - Smart Model Routing (3-tier or 7-tier with Phase 7)
    - Auto-Dependency Management
    - Learning from failures
    - SearchAgent trigger on repeated failures
    - Max iteration protection
    
    Phase 7 Features:
    - Meta-Supervisor for data-driven prioritization
    - Hybrid Decision Logic (Meta + Complexity + Failure)
    - 7-Model-Routing with escalation paths
    - Stop-Loss mechanism (2 failures → escalate)
    """
    
    def __init__(
        self,
        learning_module=None,
        search_agent=None,
        max_iterations: int = 5,
        workspace: str = "D:/AGENT_MEMORY",
        optimization_config_path: Optional[str] = None
    ):
        """
        Initialize Supervisor V3
        
        Args:
            learning_module: Optional learning module for storing success patterns
            search_agent: Optional search agent for web research
            max_iterations: Max Build-Test-Fix iterations
            workspace: Workspace directory for output files
            optimization_config_path: Path to optimization_config.json (Phase 7)
        """
        
        self.learning = learning_module
        self.search = search_agent
        self.max_iterations = max_iterations
        self.workspace = Path(workspace)
        
        # Create workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Phase 7: Load optimization config
        self.phase7_enabled = False
        self.optimization_config = None
        self.meta_supervisor = None
        self.hybrid_decision = None
        
        if optimization_config_path is None:
            # Default path
            optimization_config_path = Path(__file__).parent.parent / "config" / "optimization_config.json"
        else:
            optimization_config_path = Path(optimization_config_path)
        
        if PHASE7_AVAILABLE and optimization_config_path.exists():
            try:
                self.optimization_config = self._load_optimization_config(optimization_config_path)
                
                if self.optimization_config.get('meta_supervisor', {}).get('enabled', False):
                    # Initialize Meta-Supervisor
                    learning_log_path = Path(self.optimization_config['meta_supervisor']['learning_log_path'])
                    if learning_log_path.exists():
                        self.meta_supervisor = MetaSupervisor(learning_log_path)
                        print("[Supervisor V3] ✓ Meta-Supervisor initialized")
                    else:
                        print(f"[Supervisor V3] ⚠ Learning log not found: {learning_log_path}")
                
                if self.optimization_config.get('hybrid_decision', {}).get('enabled', False):
                    # Initialize Hybrid Decision Logic
                    self.hybrid_decision = HybridDecision(meta_supervisor=self.meta_supervisor)
                    print("[Supervisor V3] ✓ Hybrid Decision Logic initialized")
                
                self.phase7_enabled = True
                print("[Supervisor V3] ✓ Phase 7 Optimization ENABLED")
                
            except Exception as e:
                print(f"[Supervisor V3] ⚠ Failed to load Phase 7 config: {e}")
                print("[Supervisor V3] Falling back to Phase 6 mode")
        
        # Fallback: Model selector for all agents (Phase 6 mode)
        self.model_selector = ModelSelector()
        
        if not self.phase7_enabled:
            print("[Supervisor V3] Running in Phase 6 mode (ModelSelector)")
        
        # Workflow engine for dependencies
        self.workflow_engine = WorkflowEngine(
            supervisor=self,
            config=WorkflowConfig(
                security_level=SecurityLevel.BALANCED,
                verbose=True
            )
        )
        
        # Statistics
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_iterations": 0,
            "total_fixes": 0,
            "models_used": {},
            "dependencies_installed": set(),
            # Phase 7 stats
            "phase7_enabled": self.phase7_enabled,
            "hybrid_decisions": 0,
            "model_escalations": 0,
            "meta_supervisor_hits": 0,
            "fallback_to_phase6": 0
        }
        
        print("[Supervisor V3] ✓ Initialized")
        print(f"[Supervisor V3] Workspace: {self.workspace}")
        print(f"[Supervisor V3] Max iterations: {max_iterations}")
        print(f"[Supervisor V3] Phase 7: {'ENABLED' if self.phase7_enabled else 'DISABLED'}")
    
    def _load_optimization_config(self, config_path: Path) -> Dict:
        """Load optimization config from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"[Supervisor V3] ✓ Loaded optimization config: {config_path}")
            return config
        except Exception as e:
            print(f"[Supervisor V3] ✗ Failed to load config: {e}")
            return {}
    
    def _select_model_for_phase(
        self,
        phase: str,
        task_description: str,
        code_snippet: str = "",
        domain: Optional[str] = None
    ) -> str:
        """
        Select model for a phase using Hybrid Decision Logic (Phase 7) or ModelSelector (Phase 6).
        
        Args:
            phase: Phase name ('build', 'test', 'fix')
            task_description: Task description for complexity analysis
            code_snippet: Optional code snippet
            domain: Optional explicit domain
        
        Returns:
            Model name
        """
        
        # Phase 7: Use Hybrid Decision Logic
        if self.phase7_enabled and self.hybrid_decision:
            try:
                decision = self.hybrid_decision.decide_model(
                    task_description=task_description,
                    code_snippet=code_snippet,
                    domain=domain
                )
                
                self.stats["hybrid_decisions"] += 1
                
                # Check if Meta-Supervisor contributed
                if decision.meta_score > 0.0:
                    self.stats["meta_supervisor_hits"] += 1
                
                print(f"[Supervisor V3] Phase 7 Model Selection:")
                print(f"  Selected: {decision.selected_model}")
                print(f"  Confidence: {decision.confidence:.3f} (weighted: 0.40*Meta + 0.30*Complexity + 0.30*Failure)")
                print(f"  Reasoning: {decision.reasoning}")
                print(f"  Raw Scores: Meta={decision.meta_score:.3f}, "
                      f"Complexity={decision.complexity_score:.3f}, "
                      f"Failure={decision.failure_score:.3f}")
                
                return decision.selected_model
                
            except Exception as e:
                print(f"[Supervisor V3] ⚠ Phase 7 decision failed: {e}")
                print("[Supervisor V3] Falling back to Phase 6 ModelSelector")
                self.stats["fallback_to_phase6"] += 1
        
        # Phase 6: Use ModelSelector (Fallback)
        model = self.model_selector.select_model(
            task_type=phase,
            domain=domain or "generic",
            complexity="medium"
        )
        
        print(f"[Supervisor V3] Phase 6 Model Selection: {model} (for {phase})")
        
        return model
    
    def _record_phase_outcome(
        self,
        phase: str,
        model: str,
        domain: str,
        success: bool
    ):
        """
        Record outcome of a phase for learning and failure tracking.
        
        Args:
            phase: Phase name
            model: Model used
            domain: Task domain
            success: Whether phase succeeded
        """
        
        if not self.phase7_enabled or not self.hybrid_decision:
            return
        
        if success:
            # Clear failure history on success
            self.hybrid_decision.clear_failures(domain)
            print(f"[Supervisor V3] ✓ Success recorded: {domain} / {model}")
        else:
            # Record failure for Stop-Loss
            self.hybrid_decision.record_failure(domain, model)
            self.stats["model_escalations"] += 1
            print(f"[Supervisor V3] ⚠ Failure recorded: {domain} / {model}")
    
    async def execute_task(
        self,
        task: str,
        language: str = "python",
        context: Optional[Dict] = None,
        domain: Optional[str] = None
    ) -> Dict:
        """
        Execute complete task: Build → Test → Fix loop until success
        
        Args:
            task: Task description
            language: Programming language
            context: Optional additional context
            domain: Optional explicit domain (for Phase 7)
            
        Returns:
            Result dict with final_code, tests, iterations, status, etc.
        """
        
        print("\n" + "="*70)
        print(f"[Supervisor V3] 🎯 NEW TASK")
        print("="*70)
        print(f"Task: {task}")
        print(f"Language: {language}")
        print(f"Domain: {domain or 'auto-detect'}")
        print(f"Phase 7: {'ENABLED' if self.phase7_enabled else 'DISABLED'}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        result = {
            "status": "pending",
            "task": task,
            "language": language,
            "domain": domain,
            "final_code": None,
            "tests": None,
            "iterations": 0,
            "fixes": 0,
            "errors": [],
            "models_used": [],
            "dependencies_installed": [],
            "timeline": [],
            "phase7_decisions": []
        }
        
        current_code = None
        current_tests = None
        iteration = 0
        
        try:
            # PHASE 1: BUILD
            print(f"\n{'='*70}")
            print(f"[Supervisor V3] PHASE 1: BUILD")
            print(f"{'='*70}\n")
            
            iteration += 1
            result["iterations"] = iteration
            
            # Select model for build phase
            build_model = self._select_model_for_phase(
                phase='build',
                task_description=task,
                domain=domain
            )
            
            build_result = await self._build_phase(
                task,
                language,
                context,
                model_override=build_model
            )
            
            if build_result["status"] != "completed":
                result["status"] = "failed"
                result["errors"].append("Build phase failed")
                self._record_phase_outcome('build', build_model, domain or 'generic', success=False)
                return result
            
            current_code = build_result["code"]
            result["models_used"].append(build_result["model_used"])
            result["dependencies_installed"].extend(build_result["dependencies_installed"])
            result["timeline"].append({
                "phase": "build",
                "iteration": iteration,
                "status": "success",
                "model": build_result["model_used"]
            })
            
            self._record_phase_outcome('build', build_model, domain or 'generic', success=True)
            
            print(f"[Supervisor V3] ✓ Build phase completed")
            print(f"[Supervisor V3] Code: {len(current_code)} characters")
            
            # PHASE 2: TEST
            print(f"\n{'='*70}")
            print(f"[Supervisor V3] PHASE 2: TEST")
            print(f"{'='*70}\n")
            
            # Select model for test phase
            test_model = self._select_model_for_phase(
                phase='test',
                task_description=f"Generate tests for: {task}",
                code_snippet=current_code[:500],
                domain=domain
            )
            
            test_result = await self._test_phase(
                current_code,
                language,
                context,
                model_override=test_model
            )
            
            if test_result["status"] != "completed":
                result["status"] = "failed"
                result["errors"].append("Test generation failed")
                self._record_phase_outcome('test', test_model, domain or 'generic', success=False)
                return result
            
            current_tests = test_result["tests"]
            result["models_used"].append(test_result["model_used"])
            result["dependencies_installed"].extend(test_result["dependencies_installed"])
            result["timeline"].append({
                "phase": "test",
                "iteration": iteration,
                "status": "success",
                "model": test_result["model_used"]
            })
            
            self._record_phase_outcome('test', test_model, domain or 'generic', success=True)
            
            print(f"[Supervisor V3] ✓ Test phase completed")
            print(f"[Supervisor V3] Tests: {len(current_tests)} characters")
            
            # PHASE 3: VALIDATE (run tests)
            print(f"\n{'='*70}")
            print(f"[Supervisor V3] PHASE 3: VALIDATE")
            print(f"{'='*70}\n")
            
            validation_result = await self._validate_phase(
                current_code,
                current_tests,
                language
            )
            
            if validation_result["passed"]:
                # SUCCESS!
                result["status"] = "completed"
                result["final_code"] = current_code
                result["tests"] = current_tests
                
                print(f"\n{'='*70}")
                print(f"[Supervisor V3] ✅ TASK COMPLETED SUCCESSFULLY")
                print(f"{'='*70}")
                print(f"Iterations: {iteration}")
                print(f"Models used: {', '.join(set(result['models_used']))}")
                print(f"Dependencies: {len(set(result['dependencies_installed']))}")
                if self.phase7_enabled:
                    print(f"Phase 7 Decisions: {self.stats['hybrid_decisions']}")
                    print(f"Meta-Supervisor Hits: {self.stats['meta_supervisor_hits']}")
                print(f"{'='*70}\n")
                
                # Store in learning database
                if self.learning:
                    await self._store_success(task, current_code, result)
                
                self.stats["tasks_completed"] += 1
                return result
            
            # PHASE 4: FIX LOOP
            print(f"\n{'='*70}")
            print(f"[Supervisor V3] PHASE 4: FIX LOOP")
            print(f"{'='*70}\n")
            
            errors = validation_result.get("errors", ["Unknown error"])
            
            while iteration < self.max_iterations:
                iteration += 1
                result["iterations"] = iteration
                result["fixes"] += 1
                
                print(f"\n[Supervisor V3] Fix iteration {iteration}/{self.max_iterations}")
                print(f"[Supervisor V3] Errors to fix: {len(errors)}")
                
                # Trigger search on 5+ failures
                if iteration >= 5 and self.search:
                    print(f"[Supervisor V3] 🔍 Triggering SearchAgent...")
                    search_results = await self.search.search(
                        query=f"{language} {errors[0][:100]}"
                    )
                    context = context or {}
                    context['search_results'] = search_results
                
                # Select model for fix phase
                fix_model = self._select_model_for_phase(
                    phase='fix',
                    task_description=f"Fix error: {errors[0][:200]}",
                    code_snippet=current_code[:500],
                    domain=domain
                )
                
                fix_result = await self._fix_phase(
                    current_code,
                    errors[0],
                    language,
                    iteration,
                    context,
                    model_override=fix_model
                )
                
                if fix_result["status"] != "completed":
                    result["timeline"].append({
                        "phase": "fix",
                        "iteration": iteration,
                        "status": "failed",
                        "model": fix_model
                    })
                    self._record_phase_outcome('fix', fix_model, domain or 'generic', success=False)
                    continue
                
                current_code = fix_result["fixed_code"]
                result["models_used"].append(fix_result["model_used"])
                result["timeline"].append({
                    "phase": "fix",
                    "iteration": iteration,
                    "status": "success",
                    "model": fix_result["model_used"]
                })
                
                self._record_phase_outcome('fix', fix_model, domain or 'generic', success=True)
                
                # Re-validate
                validation_result = await self._validate_phase(
                    current_code,
                    current_tests,
                    language
                )
                
                if validation_result["passed"]:
                    # Fixed!
                    result["status"] = "completed"
                    result["final_code"] = current_code
                    result["tests"] = current_tests
                    
                    print(f"\n{'='*70}")
                    print(f"[Supervisor V3] ✅ FIXED AFTER {iteration} ITERATIONS")
                    print(f"{'='*70}\n")
                    
                    if self.learning:
                        await self._store_success(task, current_code, result)
                    
                    self.stats["tasks_completed"] += 1
                    return result
                
                # Still errors - continue loop
                errors = validation_result.get("errors", ["Unknown error"])
            
            # Max iterations reached
            result["status"] = "failed"
            result["errors"].append(f"Max iterations ({self.max_iterations}) reached")
            result["final_code"] = current_code
            result["tests"] = current_tests
            
            print(f"\n{'='*70}")
            print(f"[Supervisor V3] ⚠️ MAX ITERATIONS REACHED")
            print(f"{'='*70}\n")
            
            self.stats["tasks_failed"] += 1
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Exception: {str(e)}")
            print(f"\n[Supervisor V3] ❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            
            self.stats["tasks_failed"] += 1
        
        finally:
            # Update stats
            self.stats["total_iterations"] += result["iterations"]
            self.stats["total_fixes"] += result["fixes"]
            
            for model in result["models_used"]:
                self.stats["models_used"][model] = self.stats["models_used"].get(model, 0) + 1
            
            for dep in result["dependencies_installed"]:
                self.stats["dependencies_installed"].add(dep)
        
        return result
    
    async def _build_phase(
        self,
        task: str,
        language: str,
        context: Optional[Dict],
        model_override: Optional[str] = None
    ) -> Dict:
        """Execute build phase"""
        
        # Import here to avoid circular imports
        sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))
        from builder_agent import BuilderAgent
        from types import SimpleNamespace
        
        builder = BuilderAgent(learning_module=self.learning)
        
        # Override model if provided (Phase 7)
        if model_override and hasattr(builder, 'model_selector'):
            # Wrap string in object with .name attribute for compatibility
            model_obj = SimpleNamespace(name=model_override)
            original_select = builder.model_selector.select_model
            builder.model_selector.select_model = lambda *args, **kwargs: model_obj
        
        result = await builder.build(task, language, context)
        
        # Ensure model_used is set
        if 'model_used' not in result and model_override:
            result['model_used'] = model_override
        
        return result
    
    async def _test_phase(
        self,
        code: str,
        language: str,
        context: Optional[Dict],
        model_override: Optional[str] = None
    ) -> Dict:
        """Execute test phase"""
        
        from tester_agent import TesterAgent
        from types import SimpleNamespace
        
        tester = TesterAgent(learning_module=self.learning)
        
        # Override model if provided (Phase 7)
        if model_override and hasattr(tester, 'model_selector'):
            # Wrap string in object with .name attribute for compatibility
            model_obj = SimpleNamespace(name=model_override)
            original_select = tester.model_selector.select_model
            tester.model_selector.select_model = lambda *args, **kwargs: model_obj
        
        result = await tester.test(code, language, context=context)
        
        # Ensure model_used is set
        if 'model_used' not in result and model_override:
            result['model_used'] = model_override
        
        return result
    
    async def _validate_phase(
        self,
        code: str,
        tests: str,
        language: str
    ) -> Dict:
        """
        Execute validation phase: Compile and run tests.
        
        Args:
            code: Source code to validate
            tests: Test code to run
            language: Programming language ('cuda', 'cpp', 'python')
        
        Returns:
            Dict with:
                - passed: bool (True if all tests pass)
                - errors: List[str] (compilation/runtime errors)
                - output: str (test output)
        """
        
        print(f"[Supervisor V3] Validation: ENABLED")
        print(f"[Supervisor V3] Language: {language}")
        
        errors = []
        output_text = ""
        
        try:
            # Create temporary workspace
            temp_dir = Path(tempfile.mkdtemp(prefix="kisystem_validate_"))
            print(f"[Supervisor V3] Workspace: {temp_dir}")
            
            # Determine file extensions
            if language.lower() in ['cuda', 'cu']:
                code_ext = '.cu'
                test_ext = '.cpp'
                compile_with_nvcc = True
            elif language.lower() in ['cpp', 'c++']:
                code_ext = '.cpp'
                test_ext = '.cpp'
                compile_with_nvcc = False
            elif language.lower() == 'python':
                code_ext = '.py'
                test_ext = '.py'
                compile_with_nvcc = False
            else:
                code_ext = '.cpp'
                test_ext = '.cpp'
                compile_with_nvcc = False
            
            # Write files
            code_file = temp_dir / f"code{code_ext}"
            test_file = temp_dir / f"test{test_ext}"
            
            code_file.write_text(code, encoding='utf-8')
            test_file.write_text(tests, encoding='utf-8')
            
            print(f"[Supervisor V3] ✓ Wrote code: {code_file.name} ({len(code)} chars)")
            print(f"[Supervisor V3] ✓ Wrote tests: {test_file.name} ({len(tests)} chars)")
            
            # STEP 1: Compile main code
            if compile_with_nvcc:
                # CUDA compilation with nvcc
                print(f"[Supervisor V3] Compiling CUDA code with nvcc...")
                
                compile_cmd = [
                    'nvcc',
                    '-arch=sm_89',  # RTX 4070
                    '-O3',
                    '--use_fast_math',
                    '-Xcompiler', '/EHsc',  # Windows: Enable C++ exceptions
                    '-o', str(temp_dir / 'code.exe'),
                    str(code_file)
                ]
                
                result = subprocess.run(
                    compile_cmd,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    errors.append(f"NVCC compilation failed:\n{result.stderr}")
                    print(f"[Supervisor V3] ✗ NVCC compilation failed")
                    return {
                        "passed": False,
                        "errors": errors,
                        "output": result.stderr
                    }
                
                print(f"[Supervisor V3] ✓ NVCC compilation successful")
            
            # STEP 2: Compile and run tests
            if language.lower() == 'python':
                # Python: Run with pytest
                print(f"[Supervisor V3] Running Python tests with pytest...")
                
                test_cmd = [
                    'python', '-m', 'pytest',
                    str(test_file),
                    '-v',
                    '--tb=short'
                ]
                
            else:
                # C++/CUDA tests: Compile with g++ and run
                print(f"[Supervisor V3] Compiling tests with g++...")
                
                # Check if we need to link CUDA libraries
                link_flags = []
                if compile_with_nvcc:
                    # Link against CUDA runtime
                    link_flags = [
                        '-L', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64',
                        '-lcudart'
                    ]
                
                compile_test_cmd = [
                    'g++',
                    '-std=c++17',
                    '-O2',
                    '-o', str(temp_dir / 'test.exe'),
                    str(test_file),
                    str(code_file) if not compile_with_nvcc else ''
                ] + link_flags
                
                # Remove empty strings
                compile_test_cmd = [x for x in compile_test_cmd if x]
                
                result = subprocess.run(
                    compile_test_cmd,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    errors.append(f"Test compilation failed:\n{result.stderr}")
                    print(f"[Supervisor V3] ✗ Test compilation failed")
                    return {
                        "passed": False,
                        "errors": errors,
                        "output": result.stderr
                    }
                
                print(f"[Supervisor V3] ✓ Test compilation successful")
                
                # Run tests
                test_cmd = [str(temp_dir / 'test.exe')]
            
            # Execute tests
            print(f"[Supervisor V3] Running tests...")
            
            result = subprocess.run(
                test_cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output_text = result.stdout + result.stderr
            
            if result.returncode != 0:
                # Tests failed
                errors.append(f"Tests failed:\n{output_text}")
                print(f"[Supervisor V3] ✗ Tests failed (returncode={result.returncode})")
                print(f"[Supervisor V3] Output:\n{output_text[:500]}")
                
                return {
                    "passed": False,
                    "errors": errors,
                    "output": output_text
                }
            
            # Tests passed!
            print(f"[Supervisor V3] ✅ All tests passed!")
            print(f"[Supervisor V3] Output:\n{output_text[:300]}")
            
            return {
                "passed": True,
                "errors": [],
                "output": output_text
            }
            
        except subprocess.TimeoutExpired:
            errors.append("Validation timeout (compilation or test execution took >60s)")
            print(f"[Supervisor V3] ✗ Timeout")
            return {
                "passed": False,
                "errors": errors,
                "output": "Timeout"
            }
            
        except FileNotFoundError as e:
            # nvcc or g++ not found
            errors.append(f"Compiler not found: {e}. Please ensure nvcc and g++ are in PATH.")
            print(f"[Supervisor V3] ✗ Compiler not found: {e}")
            return {
                "passed": False,
                "errors": errors,
                "output": str(e)
            }
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            print(f"[Supervisor V3] ✗ Validation error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "passed": False,
                "errors": errors,
                "output": str(e)
            }
        
        finally:
            # Cleanup temp directory (optional)
            try:
                import shutil
                # Keep temp dir for debugging in case of failure
                if not errors:
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
    
    async def _fix_phase(
        self,
        code: str,
        error: str,
        language: str,
        iteration: int,
        context: Optional[Dict],
        model_override: Optional[str] = None
    ) -> Dict:
        """Execute fix phase"""
        
        from fixer_agent import FixerAgent
        from types import SimpleNamespace
        
        fixer = FixerAgent(
            learning_module=self.learning,
            search_agent=self.search
        )
        
        # Override model if provided (Phase 7)
        if model_override and hasattr(fixer, 'model_selector'):
            # Wrap string in object with .name attribute for compatibility
            model_obj = SimpleNamespace(name=model_override)
            original_select = fixer.model_selector.select_model
            fixer.model_selector.select_model = lambda *args, **kwargs: model_obj
        
        fix_context = {
            "failure_count": iteration - 1,  # 0-indexed
            **(context or {})
        }
        
        result = await fixer.fix(code, error, language, fix_context)
        
        # Ensure model_used is set
        if 'model_used' not in result and model_override:
            result['model_used'] = model_override
        
        return result
    
    async def _store_success(self, task: str, code: str, result: Dict):
        """Store successful result in learning database"""
        
        if not self.learning:
            return
        
        try:
            await self.learning.store_success(
                task=task,
                solution=code,
                metadata={
                    "iterations": result["iterations"],
                    "fixes": result["fixes"],
                    "models": result["models_used"],
                    "dependencies": list(set(result["dependencies_installed"])),
                    "phase7_enabled": self.phase7_enabled
                }
            )
            print("[Supervisor V3] ✓ Stored in learning database")
        except Exception as e:
            print(f"[Supervisor V3] ⚠️ Failed to store in learning: {e}")
    
    def get_stats(self) -> Dict:
        """Get supervisor statistics"""
        return {
            **self.stats,
            "dependencies_installed": list(self.stats["dependencies_installed"]),
            "success_rate": (
                self.stats["tasks_completed"] / 
                (self.stats["tasks_completed"] + self.stats["tasks_failed"])
                if (self.stats["tasks_completed"] + self.stats["tasks_failed"]) > 0
                else 0.0
            )
        }
    
    def print_phase7_summary(self):
        """Print Phase 7 statistics summary"""
        
        if not self.phase7_enabled:
            print("[Supervisor V3] Phase 7 not enabled")
            return
        
        print("\n" + "="*70)
        print("PHASE 7 STATISTICS")
        print("="*70)
        print(f"Hybrid Decisions: {self.stats['hybrid_decisions']}")
        print(f"Meta-Supervisor Hits: {self.stats['meta_supervisor_hits']}")
        print(f"Model Escalations: {self.stats['model_escalations']}")
        print(f"Fallback to Phase 6: {self.stats['fallback_to_phase6']}")
        
        if self.meta_supervisor:
            print("\n--- META-SUPERVISOR STATUS ---")
            print(f"Learning Entries: {len(self.meta_supervisor.learning_data)}")
            print(f"Domains Tracked: {len(self.meta_supervisor.model_biases)}")
            
            top_priorities = self.meta_supervisor.get_top_priorities(top_n=3)
            if top_priorities:
                print("\n--- TOP 3 PRIORITY TASKS ---")
                for i, priority in enumerate(top_priorities, 1):
                    print(f"{i}. {priority.domain} / {priority.model}")
                    print(f"   Priority: {priority.priority:.3f}")
                    print(f"   Success Rate: {priority.success_rate:.1%}")
        
        print("="*70 + "\n")


# ============================================================================
# TESTING
# ============================================================================

async def test_supervisor():
    """Test Supervisor V3 with real task"""
    
    print("\n" + "="*70)
    print("SUPERVISOR V3 - END-TO-END TEST")
    print("="*70 + "\n")
    
    supervisor = SupervisorV3(max_iterations=3)
    
    # Simple task
    result = await supervisor.execute_task(
        task="Write a Python function that calculates fibonacci numbers recursively",
        language="python"
    )
    
    print("\n" + "="*70)
    print("FINAL RESULT:")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Fixes: {result['fixes']}")
    print(f"Models: {result['models_used']}")
    print(f"Dependencies: {len(set(result['dependencies_installed']))}")
    
    if result['final_code']:
        print(f"\nCode ({len(result['final_code'])} chars):")
        print("-" * 70)
        print(result['final_code'][:500])
        if len(result['final_code']) > 500:
            print("...")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("SUPERVISOR STATS:")
    print("="*70)
    stats = supervisor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*70 + "\n")
    
    # Phase 7 summary
    supervisor.print_phase7_summary()


if __name__ == "__main__":
    asyncio.run(test_supervisor())
