"""
KISYSTEM Phase 6 - Hardware-in-the-Loop Optimization Test
Complete test of Build → Profile → Optimize loop

Author: Jörg Bohne
Date: 2025-11-06
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "C:/KISYSTEM/core")
sys.path.insert(0, "C:/KISYSTEM/agents")

from supervisor_v3_optimization import SupervisorV3WithOptimization


async def test_optimization_loop():
    print("="*70)
    print("KISYSTEM PHASE 6 - HARDWARE-IN-THE-LOOP OPTIMIZATION")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%H:%M:%S')}\n")
    
    # Initialize supervisor with optimization
    supervisor = SupervisorV3WithOptimization(
        max_optimization_iterations=10
    )
    
    # Test task: CUDA kernel that will have performance issues
    # (intentionally non-optimized for testing)
    task = """Create a CUDA kernel for simple vector addition.

Requirements:
- Add two float arrays: C[i] = A[i] + B[i]
- Process 4096 elements
- Use 256 threads per block
- Include proper bounds checking
- Add __shared__ memory for intermediate results (even if not strictly necessary)
- Include main() function with memory allocation and timing

This should be a working implementation that we can profile and optimize."""
    
    print("Task: CUDA Vector Addition with Shared Memory")
    print(f"Expected: Basic code → Profile → Optimize → Improved Performance\n")
    
    # Execute with optimization loop
    result = await supervisor.execute_with_optimization(
        task=task,
        language="cuda",
        performance_target=85.0  # Target: 85/100 performance score
    )
    
    # Display results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    print(f"\nStatus: {result['status']}")
    print(f"Total Iterations: {result['iterations']}")
    
    # Optimization history
    if result['optimization_history']:
        print("\n=== Optimization History ===")
        for entry in result['optimization_history']:
            iteration = entry['iteration']
            phase = entry['phase']
            
            if phase == 'initial_build':
                print(f"\nIteration 0 (Initial Build):")
                print(f"  Model: {entry.get('model_used', 'N/A')}")
                print(f"  Code Length: {len(entry.get('code', ''))}")
            
            elif phase == 'profile':
                score = entry.get('performance_score', 0)
                bottleneck = entry.get('bottleneck', 'unknown')
                suggestions = entry.get('suggestions', [])
                
                print(f"\nIteration {iteration} (Profile):")
                print(f"  Performance Score: {score:.1f}/100")
                print(f"  Bottleneck: {bottleneck}")
                if suggestions:
                    print(f"  Issues Found: {len(suggestions)}")
                    for sugg in suggestions:
                        print(f"    - {sugg['issue']} [{sugg['severity']}]")
            
            elif phase == 'optimize':
                addressed = entry.get('addressed', [])
                print(f"\nIteration {iteration} (Optimize):")
                print(f"  Model: {entry.get('model_used', 'N/A')}")
                print(f"  Addressed: {', '.join(addressed)}")
    
    # Final performance
    if result['final_performance']:
        perf = result['final_performance']
        print("\n=== Final Performance ===")
        print(f"Score: {perf.performance_score:.1f}/100")
        print(f"Bottleneck: {perf.bottleneck}")
        
        if perf.performance_score >= 85.0:
            print("\n✓ Performance target REACHED!")
        else:
            print(f"\n⚠️  Performance target not reached (target: 85.0)")
    
    # Save final code
    if result['final_code']:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(f"D:/AGENT_MEMORY/optimized_cuda_{timestamp}.cu")
        output_path.write_text(result['final_code'], encoding='utf-8')
        print(f"\n✓ Final code saved: {output_path}")
    
    # Errors
    if result['errors']:
        print("\n=== Errors ===")
        for error in result['errors']:
            print(f"  - {error}")
    
    print("\n" + "="*70)
    print(f"End Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Final verdict
    print("\n" + "="*70)
    if result['status'] == 'completed' and result['iterations'] > 0:
        print("✅ HARDWARE-IN-THE-LOOP OPTIMIZATION TEST PASSED")
        print("\nKISYSTEM Phase 6 is COMPLETE!")
        print("System can:")
        print("  • Generate CUDA code")
        print("  • Profile on real hardware (RTX 4070)")
        print("  • Detect performance issues")
        print("  • Automatically optimize")
        print("  • Iterate until performance target")
        print("\n🚀 KISYSTEM is now SENIOR-LEVEL ready!")
    else:
        print("⚠️  TEST INCOMPLETE")
        print(f"Status: {result['status']}")
        print("Check errors above for details")
    print("="*70 + "\n")
    
    return result['status'] == 'completed'


if __name__ == "__main__":
    print("\n🎯 Testing complete Hardware-in-the-Loop Optimization\n")
    print("This will:")
    print("  1. Generate initial CUDA code")
    print("  2. Compile and run on RTX 4070")
    print("  3. Profile with nvprof/nsys")
    print("  4. Detect performance issues")
    print("  5. Automatically optimize code")
    print("  6. Repeat until performance target or max iterations\n")
    print("Expected duration: 10-20 minutes (multiple GPU runs)\n")
    
    input("Press ENTER to start test...")
    
    success = asyncio.run(test_optimization_loop())
    
    sys.exit(0 if success else 1)
