#!/usr/bin/env python3
"""
Domain Consistency Fix - Quick Verification Test
Tests that detected_domain is properly tracked across phases
"""

import sys
from pathlib import Path

# Add KISYSTEM to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

def test_modeldecision_has_detected_domain():
    """Test 1: ModelDecision dataclass has detected_domain field"""
    print("\n" + "="*60)
    print("TEST 1: ModelDecision has detected_domain field")
    print("="*60)
    
    try:
        from hybrid_decision import ModelDecision
        
        # Check if detected_domain exists
        fields = ModelDecision.__dataclass_fields__
        if 'detected_domain' in fields:
            print("✅ PASS: detected_domain field exists")
            print(f"   Type: {fields['detected_domain'].type}")
            print(f"   Default: {fields['detected_domain'].default}")
            return True
        else:
            print("❌ FAIL: detected_domain field missing!")
            print(f"   Available fields: {list(fields.keys())}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_supervisor_has_current_domain():
    """Test 2: SupervisorV3 has self.current_domain attribute"""
    print("\n" + "="*60)
    print("TEST 2: SupervisorV3 has self.current_domain")
    print("="*60)
    
    try:
        from supervisor_v3 import SupervisorV3
        
        supervisor = SupervisorV3()
        
        if hasattr(supervisor, 'current_domain'):
            print("✅ PASS: self.current_domain exists")
            print(f"   Initial value: {supervisor.current_domain}")
            return True
        else:
            print("❌ FAIL: self.current_domain missing!")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_decide_model_returns_detected_domain():
    """Test 3: decide_model returns detected_domain"""
    print("\n" + "="*60)
    print("TEST 3: decide_model returns detected_domain")
    print("="*60)
    
    try:
        from hybrid_decision import HybridDecision
        
        decision_engine = HybridDecision()
        
        # Test CUDA detection
        decision = decision_engine.decide_model(
            task_description="Implement CUDA vector addition kernel",
            code_snippet="__global__ void vectorAdd(float* a, float* b, float* c)"
        )
        
        if hasattr(decision, 'detected_domain'):
            detected = decision.detected_domain
            print(f"✅ PASS: detected_domain = '{detected}'")
            
            if detected == 'cuda':
                print("   ✓ Correctly detected CUDA domain")
                return True
            else:
                print(f"   ⚠ Expected 'cuda', got '{detected}'")
                return False
        else:
            print("❌ FAIL: detected_domain not in ModelDecision!")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_domain_consistency_simulation():
    """Test 4: Simulate domain consistency across fix iterations"""
    print("\n" + "="*60)
    print("TEST 4: Domain Consistency Simulation")
    print("="*60)
    
    try:
        from hybrid_decision import HybridDecision
        
        decision_engine = HybridDecision()
        
        # Iteration 1: Build phase (auto-detect CUDA)
        decision1 = decision_engine.decide_model(
            task_description="Build CUDA kernel",
            code_snippet="__global__ void kernel()",
            domain=None  # Auto-detect
        )
        
        domain_detected = decision1.detected_domain
        print(f"Iteration 1 (Build): detected_domain = '{domain_detected}'")
        
        # Simulate failure
        decision_engine.record_failure(domain_detected, decision1.selected_model)
        print(f"  Recorded failure: {domain_detected} / {decision1.selected_model}")
        
        # Iteration 2: Fix phase (should use SAME domain)
        decision2 = decision_engine.decide_model(
            task_description="Fix CUDA compilation error",
            code_snippet="__global__ void kernel()",
            domain=None  # Still None, but should use detected domain
        )
        
        domain_detected2 = decision2.detected_domain
        print(f"Iteration 2 (Fix):   detected_domain = '{domain_detected2}'")
        
        # Check failure history
        failure_history = decision_engine.failure_history
        print(f"\nFailure History: {failure_history}")
        
        if domain_detected in failure_history:
            print(f"✅ PASS: Domain '{domain_detected}' has failure history")
            print(f"   Failures: {failure_history[domain_detected]}")
            
            # Check if escalation would happen
            if decision2.failure_score > 0.0:
                print(f"   ✓ Failure score: {decision2.failure_score:.3f} (escalation signal)")
                return True
            else:
                print(f"   ⚠ Failure score: {decision2.failure_score:.3f} (should be > 0)")
                return False
        else:
            print(f"❌ FAIL: Domain '{domain_detected}' not in failure_history!")
            print(f"   Available domains: {list(failure_history.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("DOMAIN CONSISTENCY FIX - VERIFICATION TESTS")
    print("="*60)
    print("Testing supervisor_v3.py v3.4 + hybrid_decision.py v1.1")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("ModelDecision.detected_domain", test_modeldecision_has_detected_domain()))
    results.append(("SupervisorV3.current_domain", test_supervisor_has_current_domain()))
    results.append(("decide_model returns detected_domain", test_decide_model_returns_detected_domain()))
    results.append(("Domain consistency simulation", test_domain_consistency_simulation()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Domain consistency fix is working!")
        print("\nNext steps:")
        print("1. Run autonomous test: python run_autonomous.py")
        print("2. Check that escalation happens at iteration 3 (not 5)")
        print("3. Verify NO MORE 3x same model in a row")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED!")
        print("\nTroubleshooting:")
        print("1. Did you clear __pycache__?")
        print("   → Remove-Item -Recurse -Force core/__pycache__")
        print("2. Did you copy both fixed files?")
        print("   → hybrid_decision.py AND supervisor_v3.py")
        print("3. Are you running from KISYSTEM root?")
        print("   → cd C:\\KISYSTEM")
        return 1

if __name__ == "__main__":
    exit(main())
