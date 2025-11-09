"""
KISYSTEM Supervisor V3 - Optimization Loop Extension
Hardware-in-the-Loop Performance Optimization

UPDATED: 2025-11-07 - Now using FixerAgentV3 with Hybrid Error Handler

Author: Jörg Bohne
Date: 2025-11-07
Version: 3.1
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent))

from supervisor_v3 import SupervisorV3


class SupervisorV3WithOptimization(SupervisorV3):
    """
    Extended Supervisor with Hardware-in-the-Loop Optimization
    
    NEW in V3.1:
    - Uses FixerAgentV3 with Hybrid Error Handler
    - Intelligent error categorization
    - Confidence-based retry decisions
    - Category-specific retry limits
    
    Flow:
    1. Build → Generate Code
    2. Profile → Run on Hardware + Measure Performance
    3. Optimize → Fix Performance Issues (with Hybrid Handler)
    4. Repeat (max iterations)
    """
    
    def __init__(self, *args, max_optimization_iterations: int = 10, **kwargs):
        """
        Initialize with optimization loop
        
        Args:
            max_optimization_iterations: Max optimize iterations (default: 10)
        """
        # Extract verbose if present (don't pass to parent)
        self.verbose = kwargs.pop('verbose', True)  # Default True for optimization mode
        
        super().__init__(*args, **kwargs)
        self.max_optimization_iterations = max_optimization_iterations
        
        # Import profiler agent (FIX: correct path to agents/)
        try:
            # Add agents directory to path
            sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))
            from cuda_profiler_agent import CUDAProfilerAgent
            self.profiler = CUDAProfilerAgent(verbose=self.verbose)
            if self.verbose:
                print("[Supervisor V3+] ✓ CUDA Profiler enabled")
        except ImportError as e:
            self.profiler = None
            if self.verbose:
                print(f"[Supervisor V3+] ⚠️  CUDA Profiler not available: {e}")
        except Exception as e:
            self.profiler = None
            if self.verbose:
                print(f"[Supervisor V3+] ✗ CUDA Profiler initialization failed: {e}")
        
        # Import search agent for error research
        try:
            from search_agent_v2 import SearchAgent
            self.search_agent = SearchAgent()
            if self.verbose:
                print("[Supervisor V3+] ✓ Search Agent enabled")
        except ImportError:
            self.search_agent = None
            if self.verbose:
                print("[Supervisor V3+] ⚠️  Search Agent not available")

        # Initialize persistent FixerAgentV3 (NEW!)
        sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))
        try:
            from fixer_agent_v3 import FixerAgentV3
            
            # Try to import confidence scorer
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
                from confidence_scorer import ConfidenceScorer
                confidence_scorer = ConfidenceScorer()
            except ImportError:
                confidence_scorer = None
                if self.verbose:
                    print("[Supervisor V3+] ⚠️  ConfidenceScorer not available")
            
            self.fixer_agent = FixerAgentV3(
                learning_module=self.learning,
                search_agent=self.search_agent,
                confidence_scorer=confidence_scorer
            )
            self.fixer_failure_count = 0
            if self.verbose:
                print("[Supervisor V3+] ✓ Persistent FixerAgentV3 initialized (Hybrid Handler)")
                
        except ImportError as e:
            if self.verbose:
                print(f"[Supervisor V3+] ⚠️  FixerAgentV3 not available, falling back to V2: {e}")
            # Fallback to V2
            from fixer_agent import FixerAgent
            self.fixer_agent = FixerAgent(
                learning_module=self.learning,
                search_agent=self.search_agent
            )
            self.fixer_failure_count = 0
            if self.verbose:
                print("[Supervisor V3+] ✓ Persistent FixerAgent (V2) initialized")
    
    def _print(self, msg: str):
        """Print if verbose enabled"""
        if self.verbose:
            print(msg)
    
    async def execute_with_optimization(
        self,
        task: str,
        language: str = "cuda",
        context: Optional[Dict] = None,
        performance_target: float = 80.0
    ) -> Dict:
        """
        Execute task with hardware-in-the-loop optimization
        
        Args:
            task: Task description
            language: Programming language (default: cuda)
            context: Optional context
            performance_target: Target performance score (0-100)
            
        Returns:
            Complete result with optimization history
        """
        
        self._print("\n" + "="*70)
        self._print("[Supervisor V3+] 🚀 HARDWARE-IN-THE-LOOP OPTIMIZATION")
        self._print("="*70)
        self._print(f"Task: {task[:80]}...")
        self._print(f"Language: {language}")
        self._print(f"Performance Target: {performance_target}/100")
        self._print(f"Max Optimization Iterations: {self.max_optimization_iterations}")
        self._print("="*70 + "\n")
        
        result = {
            "status": "pending",
            "task": task,
            "language": language,
            "final_code": None,
            "final_performance": None,
            "optimization_history": [],
            "iterations": 0,
            "errors": [],
            "hybrid_handler_stats": None
        }
        
        try:
            # Phase 1: Initial Build
            self._print("\n" + "="*70)
            self._print("[Supervisor V3+] PHASE 1: INITIAL BUILD")
            self._print("="*70)
            
            build_result = await self._build_phase(task, language, context)
            
            if build_result['status'] != 'completed':
                result['status'] = 'build_failed'
                result['errors'].append("Initial build failed")
                return result
            
            current_code = build_result['code']
            result['optimization_history'].append({
                'iteration': 0,
                'phase': 'initial_build',
                'code': current_code,
                'model_used': build_result.get('model_used')
            })
            
            self._print(f"\n[Supervisor V3+] ✓ Initial code generated")
            self._print(f"[Supervisor V3+] Code length: {len(current_code)} chars")
            
            # Phase 2: Profile + Optimize Loop
            if self.profiler and language in ['cuda', 'cu', 'cpp', 'c++']:
                self._print("\n" + "="*70)
                self._print("[Supervisor V3+] PHASE 2: OPTIMIZATION LOOP")
                self._print("="*70)
                
                for iteration in range(1, self.max_optimization_iterations + 1):
                    self._print(f"\n[Supervisor V3+] === Iteration {iteration}/{self.max_optimization_iterations} ===\n")
                    
                    # Step 2a: Profile on hardware
                    self._print(f"[Supervisor V3+] Step 2.{iteration}a: Profiling on RTX 4070...")
                    profile_result = await self.profiler.profile_cuda(
                        code=current_code,
                        profile=True,
                        compile_only=False
                    )
                    
                    if profile_result['status'] == 'compile_error':
                        self._print(f"[Supervisor V3+] ⚠️  Compilation failed")
                        result['errors'].append(f"Iteration {iteration}: Compile error")
                        
                        # Log error to learning module (note: store_solution is NOT async)
                        if self.learning:
                            self.learning.store_solution(
                                error=f"cuda_compilation_{language}",
                                solution="",
                                code=current_code,
                                model_used="profiler",
                                success=False
                            )
                        
                        # Try to fix compile error with FixerAgentV3 (uses Hybrid Handler!)
                        if iteration < self.max_optimization_iterations:
                            self._print(f"[Supervisor V3+] Attempting to fix compile error with Hybrid Handler...")
                            
                            fix_result = await self.fixer_agent.fix(
                                code=current_code,
                                error=profile_result['compile_output'],
                                language=language,
                                context={'attempt': iteration - 1}
                            )
                            
                            if fix_result['status'] == 'completed':
                                current_code = fix_result['fixed_code']
                                self._print(f"[Supervisor V3+] ✓ Compile error fixed")
                                
                                # Log the fix decision
                                if fix_result.get('decision_info'):
                                    decision = fix_result['decision_info']
                                    self._print(f"[Supervisor V3+] Decision: {decision['action']}")
                                    self._print(f"[Supervisor V3+] Category: {decision['category']}")
                                    self._print(f"[Supervisor V3+] Model: {decision['model']}")
                                
                                result['optimization_history'].append({
                                    'iteration': iteration,
                                    'phase': 'compile_fix',
                                    'code': current_code,
                                    'decision_info': fix_result.get('decision_info')
                                })
                                continue
                            else:
                                self._print(f"[Supervisor V3+] ✗ Could not fix compile error")
                        break
                    
                    if profile_result['status'] == 'runtime_error':
                        self._print(f"[Supervisor V3+] ⚠️  Runtime error")
                        result['errors'].append(f"Iteration {iteration}: Runtime error")
                        
                        # Log error to learning module (note: store_solution is NOT async)
                        if self.learning:
                            self.learning.store_solution(
                                error=f"cuda_runtime_{language}",
                                solution="",
                                code=current_code,
                                model_used="profiler",
                                success=False
                            )
                        
                        break
                    
                    # Step 2b: Analyze performance
                    performance = profile_result.get('performance')
                    suggestions = profile_result.get('suggestions')
                    
                    if performance:
                        score = performance.performance_score
                        bottleneck = performance.bottleneck
                        
                        self._print(f"[Supervisor V3+] Performance Score: {score:.1f}/100")
                        self._print(f"[Supervisor V3+] Bottleneck: {bottleneck}")
                        
                        result['optimization_history'].append({
                            'iteration': iteration,
                            'phase': 'profile',
                            'performance_score': score,
                            'bottleneck': bottleneck,
                            'suggestions': suggestions
                        })
                        
                        # Check if target reached
                        if score >= performance_target:
                            self._print(f"[Supervisor V3+] ✓ Performance target reached!")
                            result['final_performance'] = performance
                            break
                        
                        # Step 2c: Optimize if suggestions available
                        if suggestions:
                            self._print(f"\n[Supervisor V3+] Step 2.{iteration}b: Optimizing...")
                            self._print(f"[Supervisor V3+] Addressing {len(suggestions)} issues")
                            
                            optimize_result = await self.fixer_agent.fix_performance(
                                code=current_code,
                                performance_suggestions=suggestions,
                                language=language,
                                context=context
                            )
                            
                            if optimize_result['status'] == 'completed':
                                current_code = optimize_result['optimized_code']
                                self._print(f"[Supervisor V3+] ✓ Code optimized")
                                
                                result['optimization_history'].append({
                                    'iteration': iteration,
                                    'phase': 'optimize',
                                    'code': current_code,
                                    'model_used': optimize_result.get('model_used'),
                                    'addressed': optimize_result.get('suggestions_addressed')
                                })
                            else:
                                self._print(f"[Supervisor V3+] ⚠️  Optimization failed")
                                break
                        else:
                            self._print(f"[Supervisor V3+] No specific suggestions - code is reasonable")
                            result['final_performance'] = performance
                            break
                    else:
                        self._print(f"[Supervisor V3+] ⚠️  No performance metrics available")
                        break
                    
                    result['iterations'] = iteration
                
                self._print(f"\n[Supervisor V3+] Optimization loop completed: {result['iterations']} iterations")
            else:
                self._print("\n[Supervisor V3+] Profiler not available or language not supported")
                self._print("[Supervisor V3+] Skipping optimization loop")
            
            # Final result
            result['final_code'] = current_code
            result['status'] = 'completed'
            
            # Get Hybrid Handler statistics if using FixerAgentV3
            if hasattr(self.fixer_agent, 'get_handler_statistics'):
                result['hybrid_handler_stats'] = self.fixer_agent.get_handler_statistics()
            
            self._print("\n" + "="*70)
            self._print("[Supervisor V3+] ✅ OPTIMIZATION COMPLETE")
            self._print("="*70)
            
            if result['final_performance']:
                perf = result['final_performance']
                self._print(f"Final Performance: {perf.performance_score:.1f}/100")
                self._print(f"Bottleneck: {perf.bottleneck}")
            
            self._print(f"Total Iterations: {result['iterations']}")
            
            # Show Hybrid Handler stats
            if result.get('hybrid_handler_stats'):
                self._print("\nHybrid Handler Statistics:")
                stats = result['hybrid_handler_stats']
                self._print(f"  Total Errors: {stats.get('total_errors', 0)}")
                self._print(f"  Cache Hits: {stats.get('cache_hits', 0)}")
                self._print(f"  Retries: {stats.get('retries', 0)}")
                self._print(f"  Escalations: {stats.get('escalations', 0)}")
                self._print(f"  Searches: {stats.get('search_triggered', 0)}")
                self._print(f"  Success Rate: {stats.get('success_rate', 0):.1%}")
            
            self._print("="*70)
            
        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(str(e))
            self._print(f"\n[Supervisor V3+] ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def _extract_error_line(self, error_output: str) -> int:
        """
        Extract error line number from compiler output
        
        Args:
            error_output: Compiler error output
            
        Returns:
            Line number or -1 if not found
        """
        import re
        # Pattern: kernel.cu(123): error
        match = re.search(r'\.cu\((\d+)\):', error_output)
        if match:
            return int(match.group(1))
        return -1


if __name__ == '__main__':
    async def test():
        supervisor = SupervisorV3WithOptimization(
            max_optimization_iterations=10
        )
        
        task = """Create optimized CUDA kernel for audio gain processing.
        
Requirements:
- Process 1024 samples
- Apply gain factor
- Use __shared__ memory
- Optimize for RTX 4070"""
        
        result = await supervisor.execute_with_optimization(
            task=task,
            language="cuda",
            performance_target=80.0
        )
        
        print(f"\nResult: {result['status']}")
        print(f"Iterations: {result['iterations']}")
        if result['final_performance']:
            print(f"Final Score: {result['final_performance'].performance_score:.1f}/100")
        
        if result.get('hybrid_handler_stats'):
            print(f"\nHybrid Handler Stats:")
            for key, val in result['hybrid_handler_stats'].items():
                print(f"  {key}: {val}")
    
    asyncio.run(test())
