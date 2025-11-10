"""
Debug Test: Check what nsys actually returns
Shows raw output from CUDAProfiler
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, r'C:\KISYSTEM\agents')

from cuda_profiler_agent import CUDAProfilerAgent

# Simple CUDA test code
test_code = """
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void testKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    const int N = 1024;
    float *d_data;
    
    cudaMalloc(&d_data, N * sizeof(float));
    
    testKernel<<<4, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    
    printf("Test completed\\n");
    return 0;
}
"""

async def debug_test():
    print("="*70)
    print("DEBUG: CUDA PROFILER OUTPUT TEST")
    print("="*70)
    print()
    
    profiler = CUDAProfilerAgent(verbose=True)
    
    result = await profiler.profile_cuda(test_code, profile=True, compile_only=False)
    
    print()
    print("="*70)
    print("RAW RESULT INSPECTION")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Compile output length: {len(result.get('compile_output', ''))}")
    print(f"Runtime output length: {len(result.get('runtime_output', ''))}")
    
    # Check profile_output
    profile_out = result.get('profile_output')
    print()
    print(f"Profile output: {type(profile_out)}")
    print(f"Profile output is None: {profile_out is None}")
    if profile_out:
        print(f"Profile output length: {len(profile_out)}")
        print(f"Profile output preview (first 500 chars):")
        print("-"*70)
        print(profile_out[:500])
        print("-"*70)
        
        # Check for key patterns
        print()
        print("Key Pattern Detection:")
        print(f"  Contains 'cuda_gpu_kern_sum': {'cuda_gpu_kern_sum' in profile_out}")
        print(f"  Contains '[6/8]': {'[6/8]' in profile_out}")
        print(f"  Contains 'Total Time (ns)': {'Total Time (ns)' in profile_out}")
        print(f"  Contains 'nvprof': {'nvprof' in profile_out.lower()}")
        print(f"  Contains 'nsys': {'nsys' in profile_out.lower()}")
    else:
        print("  ⚠️  Profile output is EMPTY or None!")
        print()
        print("Checking runtime_output instead:")
        runtime_out = result.get('runtime_output', '')
        if runtime_out:
            print(f"  Runtime output length: {len(runtime_out)}")
            print(f"  Runtime output preview:")
            print("-"*70)
            print(runtime_out[:500])
            print("-"*70)
    
    # Check performance object
    print()
    print(f"Performance object: {result.get('performance')}")
    print(f"Suggestions: {result.get('suggestions')}")
    
    print()
    print("="*70)

asyncio.run(debug_test())
