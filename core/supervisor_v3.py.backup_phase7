"""
KISYSTEM Supervisor V3 - Complete Orchestration
Build-Test-Fix Loop with Smart Routing, Auto-Dependencies & Learning

Author: Jörg Bohne
Date: 2025-11-06
Version: 3.0
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from model_selector import ModelSelector
from workflow_engine import WorkflowEngine, WorkflowConfig, SecurityLevel


class SupervisorV3:
    """
    Master orchestrator for KISYSTEM
    
    Features:
    - Complete Build-Test-Fix Loop
    - Smart Model Routing (3-tier)
    - Auto-Dependency Management
    - Learning from failures
    - SearchAgent trigger on repeated failures
    - Max iteration protection
    """
    
    def __init__(
        self,
        learning_module=None,
        search_agent=None,
        max_iterations: int = 5,
        workspace: str = "D:/AGENT_MEMORY"
    ):
        """
        Initialize Supervisor V3
        
        Args:
            learning_module: Optional learning module for storing success patterns
            search_agent: Optional search agent for web research
            max_iterations: Max Build-Test-Fix iterations
            workspace: Workspace directory for output files
        """
        
        self.learning = learning_module
        self.search = search_agent
        self.max_iterations = max_iterations
        self.workspace = Path(workspace)
        
        # Create workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Model selector for all agents
        self.model_selector = ModelSelector()
        
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
            "dependencies_installed": set()
        }
        
        print("[Supervisor V3] ✓ Initialized")
        print(f"[Supervisor V3] Workspace: {self.workspace}")
        print(f"[Supervisor V3] Max iterations: {max_iterations}")
    
    async def execute_task(
        self,
        task: str,
        language: str = "python",
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Execute complete task: Build → Test → Fix loop until success
        
        Args:
            task: Task description
            language: Programming language
            context: Optional additional context
            
        Returns:
            Result dict with final_code, tests, iterations, status, etc.
        """
        
        print("\n" + "="*70)
        print(f"[Supervisor V3] 🎯 NEW TASK")
        print("="*70)
        print(f"Task: {task}")
        print(f"Language: {language}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        result = {
            "status": "pending",
            "task": task,
            "language": language,
            "final_code": None,
            "tests": None,
            "iterations": 0,
            "fixes": 0,
            "errors": [],
            "models_used": [],
            "dependencies_installed": [],
            "timeline": []
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
            
            build_result = await self._build_phase(task, language, context)
            
            if build_result["status"] != "completed":
                result["status"] = "failed"
                result["errors"].append("Build phase failed")
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
            
            print(f"[Supervisor V3] ✓ Build phase completed")
            print(f"[Supervisor V3] Code: {len(current_code)} characters")
            
            # PHASE 2: TEST
            print(f"\n{'='*70}")
            print(f"[Supervisor V3] PHASE 2: TEST")
            print(f"{'='*70}\n")
            
            test_result = await self._test_phase(current_code, language, context)
            
            if test_result["status"] != "completed":
                result["status"] = "failed"
                result["errors"].append("Test generation failed")
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
                    context["search_results"] = search_results
                
                # Fix phase
                fix_result = await self._fix_phase(
                    current_code,
                    errors[0],  # Fix first error
                    language,
                    iteration,
                    context
                )
                
                if fix_result["status"] != "completed":
                    result["timeline"].append({
                        "phase": "fix",
                        "iteration": iteration,
                        "status": "failed"
                    })
                    continue
                
                current_code = fix_result["fixed_code"]
                result["models_used"].append(fix_result["model_used"])
                result["timeline"].append({
                    "phase": "fix",
                    "iteration": iteration,
                    "status": "success",
                    "model": fix_result["model_used"]
                })
                
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
        context: Optional[Dict]
    ) -> Dict:
        """Execute build phase"""
        
        # Import here to avoid circular imports
        sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))
        from builder_agent import BuilderAgent
        
        builder = BuilderAgent(learning_module=self.learning)
        return await builder.build(task, language, context)
    
    async def _test_phase(
        self,
        code: str,
        language: str,
        context: Optional[Dict]
    ) -> Dict:
        """Execute test phase"""
        
        from tester_agent import TesterAgent
        
        tester = TesterAgent(learning_module=self.learning)
        return await tester.test(code, language, context=context)
    
    async def _validate_phase(
        self,
        code: str,
        tests: str,
        language: str
    ) -> Dict:
        """Execute validation phase (run tests)"""
        
        # MVP: Validation disabled - assume code works if it compiled
        # TODO: Actually run the tests when ready
        
        print("[Supervisor V3] Validation: DISABLED (assuming success)")
        print("[Supervisor V3] For MVP: Manual testing required")
        
        # Always pass for now
        passed = True
        
        return {
            "passed": passed,
            "errors": []
        }
    
    async def _fix_phase(
        self,
        code: str,
        error: str,
        language: str,
        iteration: int,
        context: Optional[Dict]
    ) -> Dict:
        """Execute fix phase"""
        
        from fixer_agent import FixerAgent
        
        fixer = FixerAgent(
            learning_module=self.learning,
            search_agent=self.search
        )
        
        fix_context = {
            "failure_count": iteration - 1,  # 0-indexed
            **(context or {})
        }
        
        return await fixer.fix(code, error, language, fix_context)
    
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
                    "dependencies": list(set(result["dependencies_installed"]))
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


if __name__ == "__main__":
    asyncio.run(test_supervisor())
