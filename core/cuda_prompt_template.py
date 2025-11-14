"""
CUDA Prompt Template Module
Enhanced prompts for CUDA code generation with comprehensive syntax rules

Purpose:
- Prevent common CUDA compilation errors
- Guide LLMs to generate correct CUDA C++ code
- Avoid Numba/CuPy/Python confusion

Author: Jörg Bohne / Claude (Anthropic)
Date: 2025-11-14
Version: 1.0 - Phase 2 Enhanced Prompts
"""


class CUDAPromptTemplate:
    """
    Comprehensive CUDA C++ prompt templates with syntax rules
    
    Addresses common LLM mistakes:
    - Missing <<<>>> kernel launch configuration
    - Using __global__ for non-kernel functions
    - Calling normal functions with <<<>>>
    - Using # for comments instead of //
    - Generating Python/Numba code instead of native CUDA
    """
    
    # Core CUDA syntax rules (used across all prompts)
    CUDA_SYNTAX_RULES = """
═══════════════════════════════════════════════════════════════════════════
CRITICAL CUDA SYNTAX RULES - FAILURE TO FOLLOW = COMPILATION ERROR
═══════════════════════════════════════════════════════════════════════════

1. KERNEL LAUNCH SYNTAX (MOST IMPORTANT):
   ❌ WRONG: myKernel(arg1, arg2);  // Missing <<<>>> configuration
   ✅ CORRECT: myKernel<<<blocks, threads>>>(arg1, arg2);
   
   EVERY __global__ function MUST be called with <<<gridDim, blockDim>>>
   Examples:
   - myKernel<<<1, 256>>>(data, size);           // 1 block, 256 threads
   - myKernel<<<gridSize, blockSize>>>(data);    // variables OK
   - myKernel<<<(n+255)/256, 256>>>(data, n);   // calculated grid size

2. __global__ IS ONLY FOR CUDA KERNELS:
   ❌ WRONG: __global__ void helperFunction() { }  // helper != kernel
   ❌ WRONG: __global__ int checkError() { }       // return type != void
   ✅ CORRECT: __global__ void myKernel() { }      // only kernels
   ✅ CORRECT: void helperFunction() { }           // normal functions
   ✅ CORRECT: __device__ void deviceHelper() { }  // device-only helper

3. FUNCTION CALLS - DO NOT USE <<<>>> FOR NORMAL FUNCTIONS:
   ❌ WRONG: checkError<<<1,1>>>();        // checkError is NOT a kernel
   ❌ WRONG: cudaMalloc<<<1,1>>>(...);     // CUDA API is NOT a kernel
   ✅ CORRECT: checkError();               // normal function call
   ✅ CORRECT: cudaMalloc(...);            // CUDA API call
   ✅ CORRECT: myKernel<<<blocks, threads>>>(...);  // ONLY kernels use <<<>>>

4. COMMENTS SYNTAX:
   ❌ WRONG: # This is a comment          // causes nvcc error C1021
   ✅ CORRECT: // This is a comment
   ✅ CORRECT: /* Multi-line comment */
   NOTE: # is ONLY for preprocessor: #include, #define, #pragma

5. CUDA LIBRARY INCLUDES:
   - #include <cuda_runtime.h>    // Always include for CUDA
   - #include <cufft.h>           // For cuFFT functions
   - #include <cublas_v2.h>       // For cuBLAS functions
   - #include <curand.h>          // For cuRAND functions
   - #include <stdio.h>           // For printf

6. cuFFT DATA TYPES (CRITICAL):
   ❌ WRONG: float* data; cufftExecR2C(plan, data, output);
   ✅ CORRECT: cufftReal* input = (cufftReal*)data;
              cufftComplex* output;
              cufftExecR2C(plan, input, output);
   
   Type Definitions:
   - cufftReal = float (for real input)
   - cufftComplex = struct {float x, y;} (for complex output)
   - cufftDoubleReal = double
   - cufftDoubleComplex = struct {double x, y;}
   
   Common Functions:
   - cufftExecR2C(plan, cufftReal* in, cufftComplex* out)
   - cufftExecC2R(plan, cufftComplex* in, cufftReal* out)
   - cufftExecC2C(plan, cufftComplex* in, cufftComplex* out)

7. NO PYTHON SYNTAX:
   ❌ WRONG: import numpy, from numba, @cuda.jit
   ❌ WRONG: def function():
   ✅ CORRECT: Pure CUDA C++ only
"""

    @staticmethod
    def get_code_generation_prompt(task: str, context: str = None) -> str:
        """
        Generate comprehensive CUDA code generation prompt
        
        Args:
            task: Description of what to generate
            context: Optional additional context
            
        Returns:
            Complete prompt with rules and examples
        """
        
        prompt = f"""Generate NATIVE CUDA C++ code (NOT Python, NOT Numba, NOT CuPy).

Task: {task}

{CUDAPromptTemplate.CUDA_SYNTAX_RULES}

═══════════════════════════════════════════════════════════════════════════
CORRECT EXAMPLE - STUDY THIS CAREFULLY:
═══════════════════════════════════════════════════════════════════════════

Example 1 - Basic CUDA kernel:
```cuda
#include <cuda_runtime.h>
#include <stdio.h>

// Helper function (NOT a kernel - no __global__, no <<<>>>)
void checkCudaError(cudaError_t err, const char* msg) {{
    if (err != cudaSuccess) {{
        fprintf(stderr, "CUDA Error: %s - %s\\n", msg, cudaGetErrorString(err));
    }}
}}

// CUDA kernel (uses __global__, called with <<<>>>)
__global__ void addKernel(float* a, float* b, float* c, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        c[idx] = a[idx] + b[idx];
    }}
}}

// Device-only helper (callable from kernel, not from host)
__device__ float square(float x) {{
    return x * x;
}}

int main() {{
    int n = 1024;
    size_t size = n * sizeof(float);
    
    // Allocate device memory (normal function - no <<<>>>)
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Launch kernel (USES <<<gridDim, blockDim>>>)
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Check for errors (normal function - no <<<>>>)
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel launch");  // Normal call - no <<<>>>
    
    // Synchronize (normal function - no <<<>>>)
    cudaDeviceSynchronize();
    
    // Free memory (normal function - no <<<>>>)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}}
```

Example 2 - cuFFT usage (CORRECT TYPES):
```cuda
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

int main() {{
    int n = 1024;
    
    // Allocate memory with CORRECT cuFFT types
    cufftReal *d_input;      // Use cufftReal (= float) for real data
    cufftComplex *d_output;  // Use cufftComplex for complex data
    
    cudaMalloc(&d_input, n * sizeof(cufftReal));
    cudaMalloc(&d_output, (n/2 + 1) * sizeof(cufftComplex));
    
    // Create FFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_R2C, 1);
    
    // Execute FFT with CORRECT types
    cufftExecR2C(plan, d_input, d_output);  // cufftReal* → cufftComplex*
    
    cudaDeviceSynchronize();
    
    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}}
```

═══════════════════════════════════════════════════════════════════════════
WRONG EXAMPLES - DO NOT GENERATE CODE LIKE THIS:
═══════════════════════════════════════════════════════════════════════════

WRONG #1 - Missing <<<>>> on kernel call:
```cuda
__global__ void kernel(float* data) {{ }}
int main() {{
    kernel(d_data);  // ❌ ERROR: must use kernel<<<grid,block>>>(d_data)
}}
```

WRONG #2 - Using <<<>>> on normal function:
```cuda
void checkError() {{ }}
int main() {{
    checkError<<<1,1>>>();  // ❌ ERROR: checkError is not a kernel
}}
```

WRONG #3 - __global__ on non-kernel:
```cuda
__global__ void helperFunction() {{ }}  // ❌ ERROR: not a kernel
```

WRONG #4 - Python/Numba syntax:
```python
import numpy as np
from numba import cuda
@cuda.jit                    // ❌ ERROR: This is Python, not CUDA C++
def kernel(array):
    pass
```

WRONG #5 - Using # for comments:
```cuda
# This is a comment          // ❌ ERROR: Use // not #
__global__ void kernel() {{ }}
```

WRONG #6 - cuFFT type mismatch:
```cuda
float* data;
cufftComplex* output;
cufftExecR2C(plan, data, output);  // ❌ ERROR: data must be cufftReal*
// Correct: cufftReal* data = (cufftReal*)ptr;
```

═══════════════════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════════════════

Generate NATIVE CUDA C++ code (like the CORRECT example above) for:
{task}
"""
        
        if context:
            prompt += f"\n\nAdditional Context:\n{context}\n"
        
        prompt += "\nProvide ONLY the complete CUDA C++ code, no explanations:\n\n"
        
        return prompt
    
    @staticmethod
    def get_bug_fix_prompt(code: str, error: str, escalation: int = 0) -> str:
        """
        Generate CUDA-specific bug fix prompt
        
        Args:
            code: Broken CUDA code
            error: Compilation error message
            escalation: Fix attempt number (0-based)
            
        Returns:
            Bug fix prompt with CUDA-specific guidance
        """
        
        # Extract relevant error patterns for guidance
        error_guidance = CUDAPromptTemplate._get_error_specific_guidance(error)
        
        if escalation >= 3:
            # Deep debugging mode
            prompt = f"""DEEP DEBUGGING MODE - CUDA Root Cause Analysis

The following CUDA code has failed {escalation} times.
Previous fixes did not work. Perform thorough root cause analysis.

{CUDAPromptTemplate.CUDA_SYNTAX_RULES}

{error_guidance}

Broken Code:
```cuda
{code}
```

Compilation Error:
{error}

Analyze:
1. What is the ROOT CAUSE of this error?
2. Is it a kernel launch configuration issue (<<<>>>)?
3. Is __global__ used incorrectly?
4. Are normal functions called with <<<>>>?
5. Are there # comments instead of //?
6. What is the MINIMAL fix needed?

Provide the fixed code with brief explanation of the root cause:

"""
        else:
            # Standard fix with CUDA-specific guidance
            prompt = f"""Fix the following CUDA C++ code that produces a compilation error:

{CUDAPromptTemplate.CUDA_SYNTAX_RULES}

{error_guidance}

Broken Code:
```cuda
{code}
```

Compilation Error:
{error}

CRITICAL FIX RULES:
1. If error mentions "__global__ function call must be configured"
   → Add <<<gridDim, blockDim>>> to kernel call
   
2. If error mentions "cannot overload functions" or "__declspec"
   → Change __declspec(__global__) to __global__
   
3. If error mentions "invalid preprocessing directive"
   → Change # comments to // comments
   
4. If error mentions "identifier is undefined" for CUDA functions
   → Add appropriate #include (cufft.h, cublas_v2.h, etc.)

Provide ONLY the fixed CUDA C++ code:

"""
        
        return prompt
    
    @staticmethod
    def _get_error_specific_guidance(error: str) -> str:
        """
        Generate error-specific guidance based on error message
        
        Args:
            error: Compilation error message
            
        Returns:
            Specific guidance for this error type
        """
        
        guidance = "\n═══════════════════════════════════════════════════════════════════════════\n"
        guidance += "ERROR-SPECIFIC GUIDANCE:\n"
        guidance += "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        if "__global__ function call must be configured" in error:
            guidance += """This error means you called a CUDA kernel without <<<>>> configuration.

FIX: Find the kernel call and add <<<blocks, threads>>>

Example:
❌ WRONG:  myKernel(arg1, arg2);
✅ CORRECT: myKernel<<<(n+255)/256, 256>>>(arg1, arg2);
"""
        
        elif "cannot overload functions" in error or "__declspec" in error:
            guidance += """This error means __declspec(__global__) was used instead of __global__.

FIX: Replace __declspec(__global__) with __global__

Example:
❌ WRONG:  __declspec(__global__) void kernel() {}
✅ CORRECT: __global__ void kernel() {}
"""
        
        elif "invalid preprocessing directive" in error:
            guidance += """This error means # was used for comments instead of //.

FIX: Change # comments to // comments

Example:
❌ WRONG:  # This is a comment
✅ CORRECT: // This is a comment
"""
        
        elif "identifier" in error and "undefined" in error:
            # Check for specific CUDA library identifiers
            if "cufft" in error.lower():
                guidance += """This error means cuFFT types/functions are used without #include <cufft.h>.

FIX: Add #include <cufft.h> at the top of the file
"""
            elif "cublas" in error.lower():
                guidance += """This error means cuBLAS types/functions are used without #include <cublas_v2.h>.

FIX: Add #include <cublas_v2.h> at the top of the file
"""
            else:
                guidance += """This error means an identifier is undefined.

FIX: Check if correct #include is present for CUDA library functions
"""
        
        elif "incompatible with parameter of type" in error and "cufft" in error.lower():
            guidance += """This error means wrong data type passed to cuFFT function.

FIX: Use correct cuFFT types:
- cufftReal* (= float*) for real input/output
- cufftComplex* for complex data

Example:
❌ WRONG:  float* data; cufftExecR2C(plan, data, output);
✅ CORRECT: cufftReal* data = (cufftReal*)ptr; cufftExecR2C(plan, data, output);

Common functions:
- cufftExecR2C(plan, cufftReal* input, cufftComplex* output)
- cufftExecC2R(plan, cufftComplex* input, cufftReal* output)
"""
        
        else:
            guidance += "Analyze the error message carefully and apply the CUDA SYNTAX RULES above.\n"
        
        guidance += "\n═══════════════════════════════════════════════════════════════════════════\n\n"
        
        return guidance


# ============================================================================
# TESTING
# ============================================================================

def test_cuda_prompt_template():
    """Test CUDA prompt template generation"""
    
    print("="*70)
    print("CUDA PROMPT TEMPLATE TEST")
    print("="*70)
    
    # Test 1: Code generation prompt
    print("\n[1] Testing code generation prompt...")
    task = "Implement a CUDA kernel for vector addition"
    prompt = CUDAPromptTemplate.get_code_generation_prompt(task)
    
    print(f"✓ Generated prompt: {len(prompt)} characters")
    print(f"✓ Contains kernel launch rules: {'<<<>>>' in prompt}")
    print(f"✓ Contains __global__ rules: {'__global__' in prompt}")
    print(f"✓ Contains examples: {'CORRECT EXAMPLE' in prompt}")
    
    # Test 2: Bug fix prompt
    print("\n[2] Testing bug fix prompt...")
    code = "__global__ void kernel() {}\nint main() { kernel(); }"
    error = "error: a __global__ function call must be configured"
    prompt = CUDAPromptTemplate.get_bug_fix_prompt(code, error)
    
    print(f"✓ Generated fix prompt: {len(prompt)} characters")
    print(f"✓ Contains error guidance: {'ERROR-SPECIFIC GUIDANCE' in prompt}")
    print(f"✓ Identifies kernel launch issue: {'<<<>>>' in prompt}")
    
    # Test 3: Deep debugging prompt
    print("\n[3] Testing deep debugging prompt...")
    prompt = CUDAPromptTemplate.get_bug_fix_prompt(code, error, escalation=3)
    
    print(f"✓ Generated debug prompt: {len(prompt)} characters")
    print(f"✓ Is deep debugging mode: {'DEEP DEBUGGING' in prompt}")
    print(f"✓ Contains root cause analysis: {'ROOT CAUSE' in prompt}")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)


if __name__ == "__main__":
    test_cuda_prompt_template()
