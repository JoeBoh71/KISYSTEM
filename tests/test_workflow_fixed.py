#!/usr/bin/env python3
"""
KISYSTEM Workflow Engine Test
Tests auto-dependency installation with proper path handling
"""

import asyncio
import sys
from pathlib import Path

# Add KISYSTEM core to path
kisystem_core = Path(__file__).parent.parent / 'core'
sys.path.insert(0, str(kisystem_core))

print(f"[Test] Loading workflow_engine from: {kisystem_core}")

try:
    from workflow_engine import WorkflowEngine, WorkflowConfig, SecurityLevel
    print("[Test] ✓ workflow_engine imported successfully\n")
except ImportError as e:
    print(f"[Test] ✗ Import failed: {e}")
    print(f"[Test] Make sure workflow_engine.py exists in: {kisystem_core}")
    sys.exit(1)


async def test_basic_dependencies():
    """Test 1: Basic whitelisted packages"""
    print("="*70)
    print("TEST 1: WHITELISTED PACKAGES")
    print("="*70)
    
    config = WorkflowConfig(
        security_level=SecurityLevel.BALANCED,
        verbose=True
    )
    
    engine = WorkflowEngine(supervisor=None, config=config)
    
    # Test with common packages that should be already installed
    packages = ["numpy", "scipy"]
    
    print(f"\n[Test] Checking: {packages}\n")
    
    results = await engine.installer.ensure_dependencies(packages)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    
    for pkg, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {pkg}")
    
    return all(results.values())


async def test_audio_packages():
    """Test 2: Audio processing packages (U3DAW specific)"""
    print("\n\n" + "="*70)
    print("TEST 2: U3DAW AUDIO PACKAGES")
    print("="*70)
    
    config = WorkflowConfig(
        security_level=SecurityLevel.BALANCED,
        verbose=True
    )
    
    engine = WorkflowEngine(supervisor=None, config=config)
    
    # U3DAW specific audio packages
    packages = ["numpy", "scipy", "matplotlib"]
    
    print(f"\n[Test] Checking: {packages}\n")
    
    results = await engine.installer.ensure_dependencies(packages)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    
    for pkg, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {pkg}")
    
    return all(results.values())


async def test_validation():
    """Test 3: Package validation (should fail for non-existent package)"""
    print("\n\n" + "="*70)
    print("TEST 3: VALIDATION (Expected to fail for fake package)")
    print("="*70)
    
    config = WorkflowConfig(
        security_level=SecurityLevel.BALANCED,
        verbose=True
    )
    
    engine = WorkflowEngine(supervisor=None, config=config)
    
    # This should fail validation
    packages = ["fake-package-xyz-123-nonexistent"]
    
    print(f"\n[Test] Checking: {packages}\n")
    
    results = await engine.installer.ensure_dependencies(packages)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    
    for pkg, success in results.items():
        # For this test, we EXPECT failure
        status = "✅ CORRECTLY BLOCKED" if not success else "❌ SHOULD HAVE BLOCKED"
        print(f"  {status}: {pkg}")
    
    # Success means it was correctly blocked
    return not any(results.values())


async def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*70)
    print("KISYSTEM WORKFLOW ENGINE - TEST SUITE")
    print("="*70 + "\n")
    
    results = []
    
    # Test 1: Basic
    try:
        result = await test_basic_dependencies()
        results.append(("Basic Dependencies", result))
    except Exception as e:
        print(f"\n[Test] ✗ Test 1 failed with error: {e}")
        results.append(("Basic Dependencies", False))
    
    # Test 2: Audio
    try:
        result = await test_audio_packages()
        results.append(("Audio Packages", result))
    except Exception as e:
        print(f"\n[Test] ✗ Test 2 failed with error: {e}")
        results.append(("Audio Packages", False))
    
    # Test 3: Validation
    try:
        result = await test_validation()
        results.append(("Package Validation", result))
    except Exception as e:
        print(f"\n[Test] ✗ Test 3 failed with error: {e}")
        results.append(("Package Validation", False))
    
    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print("="*70)
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - System is operational!")
    else:
        print("\n⚠️ Some tests failed - check output above")
    
    print("="*70 + "\n")
    
    return all_passed


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
