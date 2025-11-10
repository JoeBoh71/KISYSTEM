"""
CUDA Convolution Challenge - Controlled Test
Audio FIR Convolution with Hardware-in-Loop Optimization

Author: Jörg Bohne
Date: 2025-11-09
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add KISYSTEM paths
sys.path.insert(0, "C:/KISYSTEM/core")
sys.path.insert(0, "C:/KISYSTEM/agents")

from supervisor_v3_optimization import SupervisorV3WithOptimization


async def test_convolution():
    """Test CUDA convolution with optimization loop"""
    
    print("="*80)
    print("CUDA CONVOLUTION CHALLENGE")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize supervisor
    print("[TEST] Initializing Supervisor V3 with Optimization...")
    supervisor = SupervisorV3WithOptimization(
        max_optimization_iterations=5,  # Limit to 5 for first test
        verbose=True
    )
    print("[TEST] ✓ Supervisor initialized\n")
    
    # Define convolution task
    task = """Create an optimized CUDA kernel for 1D audio convolution (FIR filter).

Requirements:
- Input signal: 8192 samples (float array)
- Filter: 64-tap FIR filter coefficients (float array)
- Output: Convolved signal (8192 samples)
- Use __shared__ memory to cache filter coefficients
- Each block processes 256 samples
- Include proper boundary handling (zero-padding)
- Add timing with CUDA events
- Include main() with:
  - Memory allocation (cudaMalloc)
  - Test data initialization
  - Kernel launch with error checking
  - Result verification
  - Memory cleanup

The convolution formula: y[n] = sum(h[k] * x[n-k]) for k=0 to filter_length-1

Optimize for:
- Memory coalescing (stride-1 access)
- Shared memory usage (cache filter in shared mem)
- Minimal global memory traffic
- RTX 4070 (SM 8.9, 12GB VRAM)"""
    
    print("[TEST] Task defined:")
    print("-" * 80)
    print(task[:400] + "...")
    print("-" * 80)
    print()
    
    # Execute with optimization
    print("[TEST] Starting optimization loop...")
    print("[TEST] Target: 80/100 performance score")
    print("[TEST] Max iterations: 5")
    print()
    
    try:
        result = await supervisor.execute_with_optimization(
            task=task,
            language="cuda",
            performance_target=80.0
        )
        
        # Analyze results
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        
        print(f"\nStatus: {result['status']}")
        print(f"Total Iterations: {result['iterations']}")
        print(f"Errors: {len(result.get('errors', []))}")
        
        if result.get('errors'):
            print("\nErrors encountered:")
            for i, error in enumerate(result['errors'], 1):
                print(f"  {i}. {error}")
        
        # Optimization history
        if result.get('optimization_history'):
            print("\n" + "="*80)
            print("OPTIMIZATION HISTORY")
            print("="*80)
            
            for entry in result['optimization_history']:
                iteration = entry.get('iteration', 0)
                phase = entry.get('phase', 'unknown')
                
                if phase == 'initial_build':
                    print(f"\n[Iteration 0: Initial Build]")
                    print(f"  Model: {entry.get('model_used', 'N/A')}")
                    print(f"  Code Length: {len(entry.get('code', ''))} chars")
                
                elif phase == 'profile':
                    score = entry.get('performance_score', 0)
                    bottleneck = entry.get('bottleneck', 'unknown')
                    suggestions = entry.get('suggestions', [])
                    
                    print(f"\n[Iteration {iteration}: Profile]")
                    print(f"  Performance Score: {score:.1f}/100")
                    print(f"  Bottleneck: {bottleneck}")
                    
                    if suggestions:
                        print(f"  Issues Found: {len(suggestions)}")
                        for sugg in suggestions:
                            severity = sugg.get('severity', 'unknown')
                            issue = sugg.get('issue', 'unknown')
                            desc = sugg.get('description', '')
                            print(f"    [{severity.upper()}] {issue}: {desc}")
                
                elif phase == 'optimize':
                    addressed = entry.get('addressed', [])
                    print(f"\n[Iteration {iteration}: Optimize]")
                    print(f"  Model: {entry.get('model_used', 'N/A')}")
                    if addressed:
                        print(f"  Addressed: {', '.join(addressed)}")
        
        # Final performance
        if result.get('final_performance'):
            perf = result['final_performance']
            print("\n" + "="*80)
            print("FINAL PERFORMANCE")
            print("="*80)
            print(f"Score: {perf.performance_score:.1f}/100")
            print(f"Bottleneck: {perf.bottleneck}")
            print(f"Occupancy: {perf.occupancy:.1%}")
            print(f"Memory Efficiency: {perf.memory_efficiency:.1%}")
            print(f"Compute Efficiency: {perf.compute_efficiency:.1%}")
            
            if perf.performance_score >= 80.0:
                print("\n✅ PERFORMANCE TARGET REACHED!")
            else:
                print(f"\n⚠️  Performance below target (target: 80.0)")
        
        # Save final code
        if result.get('final_code'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f"D:/AGENT_MEMORY/convolution_optimized_{timestamp}.cu")
            output_path.write_text(result['final_code'], encoding='utf-8')
            print(f"\n✓ Final code saved: {output_path}")
        
        # Success/Failure verdict
        print("\n" + "="*80)
        if result['status'] == 'completed':
            print("✅ TEST COMPLETED")
            if result['iterations'] > 0:
                print(f"   Optimization loop executed: {result['iterations']} iterations")
            return True
        else:
            print("❌ TEST FAILED")
            print(f"   Status: {result['status']}")
            return False
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ EXCEPTION OCCURRED")
        print("="*80)
        print(f"Error: {e}")
        
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        
        return False
    
    finally:
        print("\n" + "="*80)
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)


if __name__ == "__main__":
    print("\n🎯 CUDA Convolution Challenge")
    print("   Testing Hardware-in-Loop Optimization")
    print("   Target: Audio FIR filter convolution")
    print()
    
    success = asyncio.run(test_convolution())
    
    sys.exit(0 if success else 1)
