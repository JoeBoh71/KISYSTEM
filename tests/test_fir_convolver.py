"""
Test CUDA FIR Convolver with new PerformanceParser V2
Tests the complete optimization loop with nsys profiling

Author: Jörg Bohne
Date: 2025-11-09
Location: C:\KISYSTEM\tests\test_fir_convolver.py
"""

import asyncio
import sys
from pathlib import Path

# Add KISYSTEM to path
kisystem_root = Path(r'C:\KISYSTEM')
sys.path.insert(0, str(kisystem_root / 'core'))
sys.path.insert(0, str(kisystem_root / 'agents'))

print("="*70)
print("CUDA FIR CONVOLVER TEST - PerformanceParser V2")
print("="*70)
print()

# Import supervisor
try:
    from supervisor_v3_optimization import SupervisorV3WithOptimization
    print("✓ Supervisor V3 imported")
except Exception as e:
    print(f"✗ Failed to import Supervisor: {e}")
    sys.exit(1)

# Import performance parser
try:
    from performance_parser import PerformanceParser
    print("✓ PerformanceParser imported")
    
    # Check if it's V2
    if hasattr(PerformanceParser, '_parse_nsys'):
        print("✓ PerformanceParser V2 detected (nsys support)")
    else:
        print("⚠️  Old PerformanceParser detected (may not work with nsys)")
except Exception as e:
    print(f"✗ Failed to import PerformanceParser: {e}")
    sys.exit(1)

print()
print("="*70)
print("STARTING FIR CONVOLVER TEST")
print("="*70)
print()

async def test_fir_convolver():
    """Test FIR convolution with hardware-in-loop optimization"""
    
    # Create supervisor with optimization
    # Workspace in C:\KISYSTEM\workspace
    supervisor = SupervisorV3WithOptimization(
        max_optimization_iterations=3,
        verbose=True
    )
    
    print(f"[Test] Supervisor created")
    print(f"[Test] Workspace: C:\\KISYSTEM\\workspace")
    print(f"[Test] Max iterations: 3")
    print(f"[Test] Performance target: 80.0/100")
    print()
    
    # Define FIR convolution task
    task = """Create optimized CUDA kernel for FIR (Finite Impulse Response) filter.

Requirements:
- Input: audio signal (float array)
- Filter: FIR coefficients (float array) 
- Output: filtered signal (float array)
- Use shared memory for filter coefficients
- Optimize for RTX 4070 (SM 8.9)
- Signal length: 8192 samples
- Filter length: 256 taps"""
    
    print("[Test] Task defined:")
    print(f"  {task[:100]}...")
    print()
    
    # Execute with optimization
    print("="*70)
    print("EXECUTING WITH HARDWARE-IN-LOOP OPTIMIZATION")
    print("="*70)
    print()
    
    result = await supervisor.execute_with_optimization(
        task=task,
        language='cuda',
        performance_target=80.0
    )
    
    # Print results
    print()
    print("="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    
    if result.get('final_performance'):
        perf = result['final_performance']
        print()
        print("Final Performance Metrics:")
        print(f"  GPU Time:           {perf.gpu_time_ms:.3f} ms")
        print(f"  Occupancy:          {perf.occupancy:.1%}")
        print(f"  Memory Efficiency:  {perf.memory_efficiency:.1%}")
        print(f"  Compute Efficiency: {perf.compute_efficiency:.1%}")
        print(f"  Bottleneck:         {perf.bottleneck}")
        print(f"  Performance Score:  {perf.performance_score:.1f}/100")
        
        # Check if metrics were estimated
        if perf.raw_metrics and perf.raw_metrics.get('metrics_estimated'):
            print()
            print("⚠️  Note: Metrics are estimated (nsys ran without --metrics)")
            print("   For real occupancy/efficiency, run nsys with --metrics flag")
    else:
        print()
        print("⚠️  No performance metrics available")
        print("   Check CUDAProfiler integration")
    
    # Print optimization history
    if result.get('optimization_history'):
        print()
        print("Optimization History:")
        for i, entry in enumerate(result['optimization_history']):
            print(f"  Iteration {entry.get('iteration', i)}: {entry.get('phase', 'unknown')}")
            if entry.get('performance_score'):
                print(f"    Performance: {entry['performance_score']:.1f}/100")
    
    # Print errors if any
    if result.get('errors'):
        print()
        print("Errors encountered:")
        for error in result['errors']:
            print(f"  - {error}")
    
    # Save final code to workspace
    if result.get('final_code'):
        workspace = kisystem_root / 'workspace'
        workspace.mkdir(exist_ok=True)
        
        output_file = workspace / f"fir_convolver_optimized_{result['status']}.cu"
        output_file.write_text(result['final_code'], encoding='utf-8')
        print()
        print(f"✓ Final code saved: {output_file}")
    
    print()
    print("="*70)
    if result['status'] == 'completed':
        print("✅ TEST PASSED")
    else:
        print("⚠️  TEST INCOMPLETE")
    print("="*70)
    
    return result

# Run test
if __name__ == '__main__':
    try:
        result = asyncio.run(test_fir_convolver())
        sys.exit(0 if result['status'] == 'completed' else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
