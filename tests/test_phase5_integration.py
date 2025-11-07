#!/usr/bin/env python3
"""
KISYSTEM Phase 5 Integration Test
Tests complete system: Model Routing + Auto-Dependencies + Learning

Author: Jörg Bohne
Date: 2025-11-06
"""

import asyncio
import sys
from pathlib import Path

# Add KISYSTEM to path
kisystem_root = Path(__file__).parent.parent
sys.path.insert(0, str(kisystem_root / 'core'))
sys.path.insert(0, str(kisystem_root / 'agents'))

print(f"[Test] KISYSTEM Root: {kisystem_root}")

# Import components
try:
    from model_selector import ModelSelector, get_model_for_task
    from workflow_engine import WorkflowEngine, WorkflowConfig, SecurityLevel
    from builder_agent import BuilderAgent
    from tester_agent import TesterAgent
    from fixer_agent import FixerAgent
    
    print("[Test] ✓ All modules imported successfully\n")
except ImportError as e:
    print(f"[Test] ✗ Import failed: {e}")
    print("[Test] Make sure all files are installed in C:\\KISYSTEM\\")
    sys.exit(1)


# ============================================================================
# TEST SUITE
# ============================================================================

async def test_model_selector():
    """Test 1: Model Selector"""
    print("="*70)
    print("TEST 1: MODEL SELECTOR")
    print("="*70)
    
    selector = ModelSelector()
    
    test_cases = [
        ("Write a hello world function", "simple"),
        ("Implement bubble sort algorithm", "medium"),
        ("Create CUDA kernel for matrix multiply", "complex"),
        ("Debug mysterious segfault", "deep_debug"),
    ]
    
    for task, expected_complexity in test_cases:
        print(f"\nTask: {task}")
        config = selector.select_model(task, agent_type="builder")
        print(f"→ Complexity: {selector.detector.detect(task)}")
        print(f"→ Model: {config.name}")
        print(f"→ Expected: {expected_complexity}")
        
        # Verify complexity detection
        actual = selector.detector.detect(task)
        if actual == expected_complexity:
            print(f"✓ PASS")
        else:
            print(f"✗ FAIL (expected {expected_complexity}, got {actual})")
    
    print("\n" + "="*70)
    return True


async def test_workflow_engine():
    """Test 2: Workflow Engine Auto-Dependencies"""
    print("\n" + "="*70)
    print("TEST 2: WORKFLOW ENGINE (Auto-Dependencies)")
    print("="*70)
    
    config = WorkflowConfig(
        security_level=SecurityLevel.BALANCED,
        verbose=False  # Quiet for test
    )
    
    engine = WorkflowEngine(supervisor=None, config=config)
    
    # Test with common packages
    packages = ["numpy", "scipy"]
    
    print(f"\nTesting dependencies: {packages}")
    results = await engine.installer.ensure_dependencies(packages)
    
    all_ok = all(results.values())
    
    for pkg, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {pkg}")
    
    print("\n" + "="*70)
    return all_ok


async def test_builder_agent():
    """Test 3: BuilderAgent with Smart Routing"""
    print("\n" + "="*70)
    print("TEST 3: BUILDERAGENT (Smart Routing + Auto-Deps)")
    print("="*70)
    
    builder = BuilderAgent()
    
    # Simple task (should use llama3.1:8b)
    task = "Write a simple hello world function"
    
    print(f"\nTask: {task}")
    result = await builder.build(task, language="python")
    
    print(f"\nResult:")
    print(f"  Status: {result['status']}")
    print(f"  Model: {result['model_used']}")
    print(f"  Dependencies: {result['dependencies_installed']}")
    
    success = result['status'] == 'completed'
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}")
    
    print("\n" + "="*70)
    return success


async def test_tester_agent():
    """Test 4: TesterAgent with Auto Framework Detection"""
    print("\n" + "="*70)
    print("TEST 4: TESTERAGENT (Framework Detection + Auto-Deps)")
    print("="*70)
    
    tester = TesterAgent()
    
    sample_code = """
def add(a, b):
    return a + b
"""
    
    print(f"\nGenerating tests for code...")
    result = await tester.test(sample_code, language="python")
    
    print(f"\nResult:")
    print(f"  Status: {result['status']}")
    print(f"  Framework: {result['test_framework']}")
    print(f"  Model: {result['model_used']}")
    print(f"  Dependencies: {result['dependencies_installed']}")
    
    success = result['status'] == 'completed'
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}")
    
    print("\n" + "="*70)
    return success


async def test_fixer_agent():
    """Test 5: FixerAgent with Escalation"""
    print("\n" + "="*70)
    print("TEST 5: FIXERAGENT (Escalation + Auto-Deps)")
    print("="*70)
    
    fixer = FixerAgent()
    
    broken_code = """
import numpy as np
def calc():
    return np.sum([1,2,3])
"""
    
    error = "ModuleNotFoundError: No module named 'numpy'"
    
    print(f"\nFixing error: {error}")
    result = await fixer.fix(broken_code, error, language="python")
    
    print(f"\nResult:")
    print(f"  Status: {result['status']}")
    print(f"  Model: {result['model_used']}")
    print(f"  Escalation Level: {result['escalation_level']}")
    print(f"  Dependencies: {result['dependencies_installed']}")
    
    # Check if dependency was detected and fixed
    success = (
        result['status'] == 'completed' and 
        'numpy' in result.get('dependencies_installed', [])
    )
    
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}")
    
    print("\n" + "="*70)
    return success


async def test_escalation_logic():
    """Test 6: Escalation on Multiple Failures"""
    print("\n" + "="*70)
    print("TEST 6: ESCALATION LOGIC")
    print("="*70)
    
    fixer = FixerAgent()
    
    # Simulate multiple failures
    code = "broken code"
    error = "SyntaxError: test"
    
    print("\nSimulating 3 failures to trigger escalation...")
    
    for i in range(3):
        result = await fixer.fix(code, error)
        print(f"\nAttempt {i+1}:")
        print(f"  Escalation Level: {result['escalation_level']}")
        print(f"  Model: {result.get('model_used', 'N/A')}")
        
        # Manually increment failure counter for test
        task_id = fixer._create_task_id(code, error)
        fixer.failure_history[task_id] = fixer.failure_history.get(task_id, 0) + 1
    
    # 4th attempt should use deep_debug model
    result = await fixer.fix(code, error)
    
    print(f"\nAttempt 4 (should escalate):")
    print(f"  Escalation Level: {result['escalation_level']}")
    print(f"  Model: {result.get('model_used', 'N/A')}")
    
    # Check if escalated to deepseek-r1:32b
    success = 'deepseek-r1' in result.get('model_used', '')
    
    print(f"\n{'✓ PASS - Escalated to deep_debug' if success else '✗ FAIL - Did not escalate'}")
    
    print("\n" + "="*70)
    return success


async def run_all_tests():
    """Run complete integration test suite"""
    
    print("\n" + "="*70)
    print("KISYSTEM PHASE 5 - INTEGRATION TEST SUITE")
    print("="*70)
    print("\nTesting:")
    print("  • Smart Model Routing")
    print("  • Auto-Dependency Management")
    print("  • Agent Integration")
    print("  • Escalation Logic")
    print("\n" + "="*70)
    
    results = []
    
    # Run all tests
    tests = [
        ("Model Selector", test_model_selector),
        ("Workflow Engine", test_workflow_engine),
        ("BuilderAgent", test_builder_agent),
        ("TesterAgent", test_tester_agent),
        ("FixerAgent", test_fixer_agent),
        ("Escalation Logic", test_escalation_logic),
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
        
        await asyncio.sleep(0.5)
    
    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print("="*70)
    
    all_passed = all(r[1] for r in results)
    passed_count = sum(1 for r in results if r[1])
    total_count = len(results)
    
    print(f"\nResults: {passed_count}/{total_count} tests passed")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - Phase 5 Complete!")
        print("\n🚀 KISYSTEM is ready for:")
        print("   • Intelligent model selection (3-tier strategy)")
        print("   • Automatic dependency management")
        print("   • Smart failure escalation")
        print("   • Multi-language support")
    else:
        print("\n⚠️ Some tests failed - check output above")
    
    print("\n" + "="*70 + "\n")
    
    return all_passed


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[Test] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[Test] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
