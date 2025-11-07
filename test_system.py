"""
KISYSTEM Quick Test Script
Tests Build-Fix Loop with CUDA
"""

import sys
import asyncio
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'agents'))

from supervisor_v3_optimization import SupervisorV3WithOptimization


async def quick_test():
    """Quick test: Simple CUDA kernel generation"""
    
    print("\n" + "="*70)
    print("KISYSTEM QUICK TEST - CUDA Kernel Generation")
    print("="*70)
    
    supervisor = SupervisorV3WithOptimization(
        max_optimization_iterations=1,
        verbose=True
    )
    
    task = 'Create simple CUDA kernel that adds 1.0f to each element of a float array'
    
    print(f"\nTask: {task}\n")
    
    result = await supervisor.execute_with_optimization(
        task=task,
        language='cuda',
        performance_target=80.0
    )
    
    print("\n" + "="*70)
    print("=== TEST RESULT ===")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Errors: {result.get('errors', [])}")
    
    if result['final_code']:
        print(f"Code Generated: YES ({len(result['final_code'])} chars)")
        
        # Check for auto-includes
        if '#include <cuda_runtime.h>' in result['final_code']:
            print("✓ CUDA runtime header present")
        if '#include <stdio.h>' in result['final_code']:
            print("✓ stdio.h header present")
    
    print("="*70)
    
    if result['status'] == 'completed':
        print("\n✅ TEST PASSED - System functional!")
        return 0
    else:
        print("\n❌ TEST FAILED - Check errors above")
        return 1


async def minimal_test():
    """Minimal test: Just imports"""
    
    print("\n" + "="*70)
    print("MINIMAL TEST - Import Check")
    print("="*70)
    
    try:
        from supervisor_v3_optimization import SupervisorV3WithOptimization
        print("✓ supervisor_v3_optimization import OK")
        
        from builder_agent import BuilderAgent
        print("✓ builder_agent import OK")
        
        from fixer_agent import FixerAgent
        print("✓ fixer_agent import OK")
        
        from cuda_profiler_agent import CUDAProfilerAgent
        print("✓ cuda_profiler_agent import OK")
        
        print("\n✅ ALL IMPORTS OK")
        return 0
        
    except Exception as e:
        print(f"\n❌ IMPORT FAILED: {e}")
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test KISYSTEM')
    parser.add_argument('--minimal', action='store_true', help='Only test imports')
    args = parser.parse_args()
    
    if args.minimal:
        exit_code = asyncio.run(minimal_test())
    else:
        exit_code = asyncio.run(quick_test())
    
    sys.exit(exit_code)
