#!/usr/bin/env python3
"""
Quick test of workflow_engine auto-install with simulated input
"""

import asyncio
import sys

# Mock async_input to simulate user saying "no"
original_input = input

def mock_input(prompt):
    print(prompt + "no")  # Simulate typing "no"
    return "no"

# Patch input before importing workflow_engine
import builtins
builtins.input = mock_input

# Now import and test
from workflow_engine import WorkflowEngine, WorkflowConfig, SecurityLevel

async def test():
    print("="*70)
    print("WORKFLOW ENGINE AUTO-INSTALL TEST")
    print("="*70)
    
    config = WorkflowConfig(
        auto_install_enabled=True,
        security_level=SecurityLevel.BALANCED,
        verbose=True
    )
    
    engine = WorkflowEngine(supervisor=None, config=config)
    
    # Test with mix of packages
    test_packages = [
        "numpy",              # Should be whitelisted - auto-install
        "fake-pkg-xyz-123",   # Doesn't exist - should fail validation
        "scipy",              # Whitelisted - auto-install
    ]
    
    print(f"\nTesting with packages: {test_packages}\n")
    results = await engine.installer.ensure_dependencies(test_packages)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    for pkg, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {pkg}")
    print("="*70)
    
    return results

if __name__ == "__main__":
    asyncio.run(test())
