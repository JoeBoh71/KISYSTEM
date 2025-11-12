"""
KISYSTEM v3.9 Validation Test
Einfacher CUDA Kernel Test - Vector Addition
"""

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))

from builder_agent import BuilderAgent
from ollama_client import OllamaClient

async def test_cuda_generation():
    print("\n" + "="*70)
    print("KISYSTEM v3.9 - CUDA GENERATION TEST")
    print("="*70)
    
    # Test 1: Model Validation (v1.2 Feature)
    print("\n[Test 1] Model Validation (sollte sofort Fehler zeigen)...")
    client = OllamaClient()
    try:
        await client.generate(
            model="non_existent_model:99b",
            prompt="test",
            timeout=5
        )
        print("  ✗ FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ PASSED: {e}")
    except Exception as e:
        print(f"  ✗ FAILED: Wrong exception: {e}")
    
    # Test 2: CUDA C++ Generation (v1.1 Feature)
    print("\n[Test 2] CUDA C++ Code Generation...")
    builder = BuilderAgent()
    
    task = """Generate a simple CUDA kernel for vector addition:
    - Input: two float arrays (a, b) and output array (c)
    - Each thread adds one element: c[i] = a[i] + b[i]
    - Include proper error checking
    - Keep it simple and well-commented"""
    
    result = await builder.build(
        task=task,
        language="cuda"
    )
    
    if result["status"] == "completed":
        print(f"  ✓ Code generated with {result['model_used']}")
        
        # Check for CUDA C++ (not Python)
        code = result["code"]
        if "__global__" in code and "cudaMalloc" in code:
            print("  ✓ Valid CUDA C++ detected (__global__, cudaMalloc)")
        else:
            print("  ✗ WARNING: Missing CUDA C++ keywords")
        
        if "@cuda.jit" in code or "import numpy" in code:
            print("  ✗ FAILED: Python Numba code detected!")
        else:
            print("  ✓ No Python code detected")
        
        # Save code
        output_file = Path("tests/validation_v3_9/vector_add.cu")
        output_file.write_text(code, encoding='utf-8')
        print(f"  ✓ Saved to: {output_file}")
        
        # Show preview
        print("\n  Code Preview (first 30 lines):")
        print("  " + "-"*66)
        for i, line in enumerate(code.split('\n')[:30], 1):
            print(f"  {i:3d} | {line}")
        print("  " + "-"*66)
        
    else:
        print(f"  ✗ FAILED: {result['errors']}")
    
    print("\n" + "="*70)
    print("TEST COMPLETED")
    print("="*70)
    
    return result

if __name__ == "__main__":
    result = asyncio.run(test_cuda_generation())
    
    if result["status"] == "completed":
        print("\n✓ Next step: Compile with nvcc")
        print(f"  nvcc -arch=sm_89 tests/validation_v3_9/vector_add.cu -o tests/validation_v3_9/vector_add.exe")
        exit(0)
    else:
        print("\n✗ Test failed")
        exit(1)
