"""
Quick Test Script for Enhanced CUDA Profiler Agent
Tests validation, auto-include, and error handling
"""

import sys
from pathlib import Path

# Add agents to path
sys.path.insert(0, str(Path(__file__).parent / 'agents'))

from cuda_profiler_agent import CUDAProfilerAgent, validate_and_clean_cuda_code, ensure_required_includes


def test_validation():
    """Test C2019 prevention"""
    print("\n" + "="*60)
    print("TEST 1: C2019 Prevention - Invalid Preprocessor Syntax")
    print("="*60)
    
    bad_code = """
# 1 "some_file.cu"
# 42
#include <cuda_runtime.h>
#unknowndirective test

__global__ void test() {}
"""
    
    cleaned, issues = validate_and_clean_cuda_code(bad_code)
    
    print(f"Found {len(issues)} issues:")
    for issue in issues:
        print(f"  ✓ {issue}")
    
    print(f"\nCleaned code preview:")
    print(cleaned[:200])
    
    assert len(issues) >= 2, "Should detect at least 2 invalid preprocessor lines"
    assert "# 1" not in cleaned, "Should remove '# 1' syntax"
    print("\n✅ TEST 1 PASSED")


def test_auto_include():
    """Test auto-include detection"""
    print("\n" + "="*60)
    print("TEST 2: Auto-Include Detection")
    print("="*60)
    
    code_needs_iostream = """
__global__ void test() {
    printf("test\\n");
}

int main() {
    std::cout << "Hello" << std::endl;
    return 0;
}
"""
    
    enhanced, added = ensure_required_includes(code_needs_iostream)
    
    print(f"Added {len(added)} includes:")
    for inc in added:
        print(f"  ✓ {inc}")
    
    assert "#include <iostream>" in enhanced, "Should add iostream"
    assert "#include <stdio.h>" in enhanced, "Should add stdio.h"
    print("\n✅ TEST 2 PASSED")


def test_compile_invalid():
    """Test compilation with invalid code (should fail gracefully)"""
    print("\n" + "="*60)
    print("TEST 3: Graceful Failure Handling")
    print("="*60)
    
    invalid_code = """
# 1 "bad_file"
#include <cuda_runtime.h>

__global__ void test() {
    UNDEFINED_FUNCTION();  // This should fail
}

int main() {
    return 0;
}
"""
    
    profiler = CUDAProfilerAgent()
    result = profiler.compile(invalid_code)
    
    print(f"Compilation success: {result['success']}")
    print(f"Issues fixed: {len(result.get('issues_fixed', []))}")
    
    if not result['success']:
        print(f"Error message preview: {result['message'][:200]}")
        if result.get('source_file'):
            print(f"Failed source saved at: {result['source_file']}")
    
    assert not result['success'], "Should fail due to undefined function"
    assert len(result.get('issues_fixed', [])) > 0, "Should fix preprocessor issues"
    print("\n✅ TEST 3 PASSED")


def test_compile_valid():
    """Test compilation with valid code"""
    print("\n" + "="*60)
    print("TEST 4: Valid Code Compilation")
    print("="*60)
    
    valid_code = """
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("CUDA kernel compiled successfully\\n");
    return 0;
}
"""
    
    profiler = CUDAProfilerAgent()
    result = profiler.compile(valid_code)
    
    print(f"Compilation success: {result['success']}")
    
    if result['success']:
        print(f"Binary path: {result['binary_path']}")
    else:
        print(f"Unexpected failure: {result['message'][:200]}")
    
    assert result['success'], "Valid code should compile"
    print("\n✅ TEST 4 PASSED")


def main():
    """Run all tests"""
    print("\n" + "🔬"*30)
    print("CUDA PROFILER AGENT - VALIDATION TESTS")
    print("🔬"*30)
    
    try:
        test_validation()
        test_auto_include()
        test_compile_invalid()
        test_compile_valid()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nNew cuda_profiler_agent.py is working correctly.")
        print("You can now run: python test_phase6_optimization.py")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
