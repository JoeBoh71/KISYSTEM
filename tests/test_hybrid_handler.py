"""
KISYSTEM - Hybrid Error Handler Test
Tests the complete error handling flow with categorization and decisions

Author: Jörg Bohne
Date: 2025-11-07
"""

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))


async def test_1_direct_fixer():
    """
    Test 1: Direct FixerAgentV3 test with compile error
    Tests error categorization and decision-making
    """
    print("\n" + "="*70)
    print("TEST 1: DIRECT FIXER AGENT V3 TEST")
    print("="*70)
    
    try:
        from fixer_agent_v3 import FixerAgentV3
        from learning_module_v2 import LearningModuleV2
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Initialize with learning
    learning = LearningModuleV2()
    fixer = FixerAgentV3(learning_module=learning)
    
    # Test case: CUDA compile error (missing semicolon)
    broken_code = """
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void addOne(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] + 1.0f
    }
}

int main() {
    const int N = 1024;
    float *d_data;
    
    cudaMalloc(&d_data, N * sizeof(float));
    
    addOne<<<4, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    
    printf("Test completed\\n");
    return 0;
}
"""
    
    error = "error C2143: syntax error: missing ';' before '}'"
    
    print("\n[Test 1] Fixing CUDA compile error...")
    print(f"[Test 1] Error: {error}")
    
    # Attempt 1
    print("\n--- ATTEMPT 1 ---")
    result1 = await fixer.fix(
        code=broken_code,
        error=error,
        language="cuda",
        context={'attempt': 0}
    )
    
    print(f"\nResult 1:")
    print(f"  Status: {result1['status']}")
    print(f"  Model: {result1['model_used']}")
    if result1.get('decision_info'):
        decision = result1['decision_info']
        print(f"  Decision: {decision['action']}")
        print(f"  Category: {decision['category']}")
        print(f"  Severity: {decision['severity']}")
        print(f"  Confidence: {decision['confidence']:.1%}")
    
    # Show handler stats
    print(f"\n[Test 1] Hybrid Handler Statistics:")
    stats = fixer.get_handler_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    success = result1['status'] == 'completed'
    print(f"\n{'✓' if success else '✗'} Test 1: {'PASSED' if success else 'FAILED'}")
    return success


async def test_2_supervisor_with_iterations():
    """
    Test 2: Full Supervisor test with multiple iterations
    Tests complete optimization loop with error handling
    """
    print("\n" + "="*70)
    print("TEST 2: SUPERVISOR WITH MULTIPLE ITERATIONS")
    print("="*70)
    
    try:
        from supervisor_v3_optimization import SupervisorV3WithOptimization
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    supervisor = SupervisorV3WithOptimization(
        max_optimization_iterations=3,  # Allow 3 fix attempts
        verbose=True
    )
    
    # Task that will likely produce compile error first
    task = """Create CUDA kernel that adds 1.0f to each array element.

Requirements:
- Array size: 1024 elements
- Use __global__ kernel
- Handle memory allocation
- Clean up resources"""
    
    print("\n[Test 2] Executing task with error handling...")
    
    result = await supervisor.execute_with_optimization(
        task=task,
        language="cuda",
        performance_target=80.0
    )
    
    print(f"\n" + "="*70)
    print("TEST 2 RESULTS:")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Errors: {len(result['errors'])}")
    
    if result.get('hybrid_handler_stats'):
        print(f"\nHybrid Handler Statistics:")
        for key, value in result['hybrid_handler_stats'].items():
            print(f"  {key}: {value}")
    
    # Show optimization history
    if result.get('optimization_history'):
        print(f"\nOptimization History ({len(result['optimization_history'])} steps):")
        for step in result['optimization_history']:
            phase = step.get('phase', 'unknown')
            iteration = step.get('iteration', 0)
            print(f"  [{iteration}] {phase}", end='')
            if step.get('decision_info'):
                decision = step['decision_info']
                print(f" - {decision['action']} ({decision['category']})", end='')
            print()
    
    success = result['status'] == 'completed'
    print(f"\n{'✓' if success else '✗'} Test 2: {'PASSED' if success else 'FAILED'}")
    return success


async def test_3_error_categories():
    """
    Test 3: Test all error categories
    Verifies categorization logic works correctly
    """
    print("\n" + "="*70)
    print("TEST 3: ERROR CATEGORIZATION TEST")
    print("="*70)
    
    try:
        from error_handler import ErrorCategorizer, ErrorCategory
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    categorizer = ErrorCategorizer()
    
    # Test cases for each category
    test_cases = [
        {
            'error': 'error C2143: syntax error: missing ";"',
            'expected': ErrorCategory.COMPILATION,
            'name': 'Syntax Error'
        },
        {
            'error': 'Segmentation fault (core dumped)',
            'expected': ErrorCategory.RUNTIME,
            'name': 'Segfault'
        },
        {
            'error': 'CUDA kernel launch error: out of memory',
            'expected': ErrorCategory.RUNTIME,
            'name': 'CUDA OOM'
        },
        {
            'error': 'Performance: Low occupancy detected (15%)',
            'expected': ErrorCategory.PERFORMANCE,
            'name': 'Low Occupancy'
        },
        {
            'error': 'Bank conflict detected in shared memory',
            'expected': ErrorCategory.PERFORMANCE,
            'name': 'Bank Conflict'
        },
        {
            'error': 'Test failed: expected 42, got 0',
            'expected': ErrorCategory.LOGIC,
            'name': 'Logic Error'
        }
    ]
    
    passed = 0
    failed = 0
    
    print("\n[Test 3] Testing error categorization...")
    print()
    
    for test in test_cases:
        categorized = categorizer.categorize(test['error'])
        
        is_correct = categorized.category == test['expected']
        symbol = '✓' if is_correct else '✗'
        
        print(f"{symbol} {test['name']}:")
        print(f"    Error: {test['error'][:60]}...")
        print(f"    Expected: {test['expected'].value}")
        print(f"    Got: {categorized.category.value}")
        print(f"    Severity: {categorized.severity}")
        print(f"    Retry Limit: {categorized.retry_limit}")
        print()
        
        if is_correct:
            passed += 1
        else:
            failed += 1
    
    success = failed == 0
    print(f"Results: {passed} passed, {failed} failed")
    print(f"\n{'✓' if success else '✗'} Test 3: {'PASSED' if success else 'FAILED'}")
    return success


async def test_4_confidence_decisions():
    """
    Test 4: Test confidence-based decision logic
    Verifies handler makes correct decisions based on confidence
    """
    print("\n" + "="*70)
    print("TEST 4: CONFIDENCE-BASED DECISIONS TEST")
    print("="*70)
    
    try:
        from error_handler import HybridErrorHandler
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    handler = HybridErrorHandler(verbose=False)
    
    # Simulate different scenarios
    test_cases = [
        {
            'name': 'No similar solutions (should escalate)',
            'error': 'CompletelyUniqueError: This has never happened before',
            'code': 'int main() { return 0; }',
            'expected_action': 'escalate'
        },
        {
            'name': 'Compile error without SearchAgent (should escalate)',
            'error': 'error C2019: unexpected preprocessor directive',
            'code': '#include <cuda_runtime.h>',
            'expected_action': 'escalate'  # No SearchAgent available, so escalates
        }
    ]
    
    passed = 0
    failed = 0
    
    print("\n[Test 4] Testing decision logic...")
    print()
    
    for test in test_cases:
        decision = await handler.handle_error(
            error=test['error'],
            code=test['code'],
            language='cuda',
            agent_type='fixer',
            context={'attempt': 0}
        )
        
        is_correct = decision['action'] == test['expected_action']
        symbol = '✓' if is_correct else '✗'
        
        print(f"{symbol} {test['name']}:")
        print(f"    Expected: {test['expected_action']}")
        print(f"    Got: {decision['action']}")
        print(f"    Category: {decision['categorized_error'].category.value}")
        print(f"    Severity: {decision['categorized_error'].severity}")
        print()
        
        if is_correct:
            passed += 1
        else:
            failed += 1
    
    success = failed == 0
    print(f"Results: {passed} passed, {failed} failed")
    print(f"\n{'✓' if success else '✗'} Test 4: {'PASSED' if success else 'FAILED'}")
    return success


async def main():
    """Run all tests"""
    
    print("\n" + "="*70)
    print("KISYSTEM HYBRID ERROR HANDLER - COMPLETE TEST SUITE")
    print("="*70)
    print("Date: 2025-11-07")
    print("Version: 1.0")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(('Error Categorization', await test_3_error_categories()))
    results.append(('Confidence Decisions', await test_4_confidence_decisions()))
    results.append(('Direct Fixer V3', await test_1_direct_fixer()))
    results.append(('Supervisor with Iterations', await test_2_supervisor_with_iterations()))
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        symbol = '✓' if result else '✗'
        status = 'PASSED' if result else 'FAILED'
        print(f"{symbol} {name}: {status}")
    
    print("="*70)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*70)
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - HYBRID HANDLER FULLY FUNCTIONAL!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED - CHECK LOGS ABOVE")
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
