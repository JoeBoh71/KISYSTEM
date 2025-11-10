"""
KISYSTEM V3.0 - CUDA Challenge: Partitioned FFT Convolution
Real-time audio processing for U3DAW

Challenge: Generate optimized CUDA kernel for low-latency FIR filtering
Using: Partitioned FFT (Overlap-Add) with cuFFT

Author: Jörg Bohne
Date: 2025-11-07
"""

import asyncio
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / 'core'))

from supervisor_v3_optimization import SupervisorV3WithOptimization


async def run_convolution_challenge():
    """
    Run the Partitioned FFT Convolution challenge
    
    This is a real-world U3DAW use case:
    - 1024 samples per block @ 192kHz = 5.33ms per block
    - Target: <5ms processing latency
    - 32-bit float precision
    - Use cuFFT for fast convolution
    """
    
    print("\n" + "="*70)
    print("🎵 KISYSTEM V3.0 - CUDA CONVOLUTION CHALLENGE 🎵")
    print("="*70)
    print("\nTask: Partitioned FFT Convolution for Real-Time Audio")
    print("Use Case: U3DAW FIR Filtering Alternative")
    print("\nPerformance Targets:")
    print("  • Latency: <5ms per block (1024 samples @ 192kHz)")
    print("  • Throughput: >200 MegaSamples/sec")
    print("  • Performance Score: >80/100")
    print("  • Memory Efficiency: >70%")
    print("\nOptimization Strategy:")
    print("  • Use cuFFT library")
    print("  • Overlap-Add algorithm")
    print("  • Shared memory for overlap buffer")
    print("  • Coalesced memory access")
    print("="*70)
    
    # Detailed task specification
    task = """Create optimized CUDA kernel for real-time audio convolution using Partitioned FFT.

REQUIREMENTS:
1. Input: Float array (1024 samples per block)
2. FIR Filter: 2048 taps (impulse response)
3. Algorithm: Overlap-Add with FFT convolution
4. Use cuFFT library for FFT/IFFT operations
5. 32-bit float precision

IMPLEMENTATION DETAILS:
- Block size: 1024 samples
- FFT size: 4096 (next power of 2 for 1024+2048)
- Overlap: 2048 samples (filter length)
- Process: FFT(input) * FFT(filter) -> IFFT -> Overlap-Add

OPTIMIZATION GOALS:
- Minimize kernel launch overhead
- Use shared memory for overlap buffer
- Coalesced global memory access
- Maximize occupancy
- Target: <5ms processing time per block

CUDA FEATURES TO USE:
- cuFFT library (cufftExecR2C, cufftExecC2R)
- __shared__ memory for buffers
- __syncthreads() for synchronization
- Optimized block/grid dimensions

OUTPUT:
- Complete working CUDA code
- Include cuFFT initialization
- Memory allocation/deallocation
- Error checking
- Performance measurement

PERFORMANCE TARGET:
- Latency: <5ms per 1024-sample block
- Suitable for real-time @ 192kHz sample rate
- Memory bandwidth efficient
- High GPU utilization"""

    # Initialize Supervisor with optimization loop
    print("\n[Challenge] Initializing KISYSTEM Supervisor V3.0...")
    
    supervisor = SupervisorV3WithOptimization(
        max_optimization_iterations=10,  # Allow up to 10 optimization passes
        verbose=True
    )
    
    print("[Challenge] ✓ Supervisor ready")
    print("[Challenge] Starting optimization loop...\n")
    
    start_time = time.time()
    
    # Execute with hardware-in-loop optimization
    result = await supervisor.execute_with_optimization(
        task=task,
        language="cuda",
        performance_target=80.0  # Target score 80/100
    )
    
    elapsed_time = time.time() - start_time
    
    # Show results
    print("\n" + "="*70)
    print("🏆 CHALLENGE RESULTS 🏆")
    print("="*70)
    
    print(f"\nStatus: {result['status'].upper()}")
    print(f"Total Time: {elapsed_time:.1f} seconds")
    print(f"Optimization Iterations: {result['iterations']}")
    
    if result.get('final_performance'):
        perf = result['final_performance']
        print(f"\n📊 Final Performance Metrics:")
        print(f"  Performance Score: {perf.performance_score:.1f}/100")
        print(f"  GPU Time: {perf.gpu_time_ms:.2f}ms")
        print(f"  Occupancy: {perf.occupancy:.1%}")
        print(f"  Memory Efficiency: {perf.memory_efficiency:.1%}")
        print(f"  Compute Efficiency: {perf.compute_efficiency:.1%}")
        print(f"  Bottleneck: {perf.bottleneck}")
        
        # Check if targets met
        targets_met = []
        targets_missed = []
        
        if perf.performance_score >= 80.0:
            targets_met.append("✓ Performance Score >80")
        else:
            targets_missed.append("✗ Performance Score <80")
        
        if perf.gpu_time_ms < 5.0:
            targets_met.append("✓ Latency <5ms")
        else:
            targets_missed.append("✗ Latency >5ms")
        
        if perf.memory_efficiency >= 0.7:
            targets_met.append("✓ Memory Efficiency >70%")
        else:
            targets_missed.append("✗ Memory Efficiency <70%")
        
        print(f"\n🎯 Target Achievement:")
        for target in targets_met:
            print(f"  {target}")
        for target in targets_missed:
            print(f"  {target}")
    
    # Show optimization history
    if result.get('optimization_history'):
        print(f"\n📈 Optimization History ({len(result['optimization_history'])} steps):")
        for i, step in enumerate(result['optimization_history']):
            phase = step.get('phase', 'unknown')
            print(f"  Step {i}: {phase}", end='')
            
            if step.get('performance_score'):
                print(f" (score: {step['performance_score']:.1f})", end='')
            
            if step.get('decision_info'):
                decision = step['decision_info']
                print(f" - {decision['action']} ({decision['category']})", end='')
            
            print()
    
    # Show Hybrid Handler statistics
    if result.get('hybrid_handler_stats'):
        print(f"\n🤖 Hybrid Handler Statistics:")
        stats = result['hybrid_handler_stats']
        print(f"  Total Errors Handled: {stats.get('total_errors', 0)}")
        print(f"  Cache Hits: {stats.get('cache_hits', 0)}")
        print(f"  Retries: {stats.get('retries', 0)}")
        print(f"  Escalations: {stats.get('escalations', 0)}")
        print(f"  Searches: {stats.get('search_triggered', 0)}")
        print(f"  Success Rate: {stats.get('success_rate', 0):.1%}")
        
        if stats.get('by_category'):
            print(f"\n  Errors by Category:")
            for cat, count in stats['by_category'].items():
                print(f"    {cat}: {count}")
    
    # Show errors if any
    if result.get('errors'):
        print(f"\n⚠️  Errors Encountered ({len(result['errors'])}):")
        for error in result['errors'][:5]:  # Show max 5
            print(f"  • {error}")
        if len(result['errors']) > 5:
            print(f"  ... and {len(result['errors'])-5} more")
    
    # Final assessment
    print("\n" + "="*70)
    
    if result['status'] == 'completed':
        if result.get('final_performance'):
            if result['final_performance'].performance_score >= 80.0:
                print("✅ CHALLENGE COMPLETED - EXCELLENT PERFORMANCE!")
            elif result['final_performance'].performance_score >= 60.0:
                print("✅ CHALLENGE COMPLETED - GOOD PERFORMANCE")
            else:
                print("⚠️  CHALLENGE COMPLETED - PERFORMANCE BELOW TARGET")
        else:
            print("✅ CHALLENGE COMPLETED - CODE GENERATED")
    else:
        print("❌ CHALLENGE FAILED - SEE ERRORS ABOVE")
    
    print("="*70)
    
    # Save final code if available
    if result.get('final_code'):
        output_file = Path(__file__).parent / 'cuda_convolution_kernel.cu'
        output_file.write_text(result['final_code'])
        print(f"\n💾 Final code saved to: {output_file}")
        print(f"   Lines: {len(result['final_code'].splitlines())}")
        print(f"   Size: {len(result['final_code'])} bytes")
    
    return result


if __name__ == '__main__':
    print("\n🚀 Starting KISYSTEM V3.0 CUDA Convolution Challenge...")
    print("    This will test the complete optimization loop:")
    print("    • BuilderAgent (code generation)")
    print("    • CUDAProfilerAgent (hardware profiling)")
    print("    • FixerAgentV3 (optimization with Hybrid Handler)")
    print("    • LearningModuleV2 (solution caching)")
    print()
    
    result = asyncio.run(run_convolution_challenge())
    
    # Exit code based on result
    if result['status'] == 'completed':
        if result.get('final_performance'):
            if result['final_performance'].performance_score >= 80.0:
                sys.exit(0)  # Perfect!
            else:
                sys.exit(1)  # Completed but below target
        else:
            sys.exit(0)  # Completed without profiling
    else:
        sys.exit(2)  # Failed
