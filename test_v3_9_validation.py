"""
KISYSTEM v3.9 Validation Test - Simple CUDA Kernel
"""
import asyncio
import sys
import os
from pathlib import Path

# Set PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'agents'))

print("Python Path:")
for p in sys.path[:3]:
    print(f"  {p}")

print("\nImporting modules...")
try:
    from builder_agent import BuilderAgent
    print("  ✓ BuilderAgent imported")
except Exception as e:
    print(f"  ✗ BuilderAgent import failed: {e}")
    exit(1)

try:
    from ollama_client import OllamaClient
    print("  ✓ OllamaClient imported")
except Exception as e:
    print(f"  ✗ OllamaClient import failed: {e}")
    exit(1)

async def main():
    print("\n" + "="*70)
    print("KISYSTEM v3.9 - VALIDATION TEST")
    print("="*70)
    
    # Test 1: Model Validation (NEW in v1.2)
    print("\n[Test 1] Model Validation...")
    print("  Testing with non-existent model (should fail fast)...")
    
    client = OllamaClient()
    try:
        result = await client.generate(
            model="fake_model_does_not_exist:99b",
            prompt="test",
            timeout=2
        )
        print("  ✗ FAILED: Should have raised ValueError!")
        return False
    except ValueError as e:
        if "not available" in str(e):
            print(f"  ✓ PASSED: Fast error detection")
            print(f"    Message: {e}")
        else:
            print(f"  ✗ Wrong error: {e}")
            return False
    
    # Test 2: Simple CUDA Generation
    print("\n[Test 2] CUDA C++ Generation...")
    print("  Generating simple vector addition kernel...")
    
    builder = BuilderAgent()
    
    task = "Generate CUDA kernel for vector addition: c[i] = a[i] + b[i]. Keep it simple."
    
    result = await builder.build(
        task=task,
        language="cuda"
    )
    
    if result["status"] != "completed":
        print(f"  ✗ FAILED: {result['errors']}")
        return False
    
    print(f"  ✓ Generated with model: {result['model_used']}")
    
    code = result["code"]
    
    # Check CUDA C++ keywords
    has_global = "__global__" in code
    has_cuda_api = "cudaMalloc" in code or "cudaMemcpy" in code
    has_python = "@cuda.jit" in code or "import numba" in code
    
    print(f"  {'✓' if has_global else '✗'} __global__ keyword: {has_global}")
    print(f"  {'✓' if has_cuda_api else '✗'} CUDA API calls: {has_cuda_api}")
    print(f"  {'✓' if not has_python else '✗'} No Python Numba: {not has_python}")
    
    # Save
    output_dir = Path("tests/validation_v3_9")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "vector_add.cu"
    output_file.write_text(code, encoding='utf-8')
    print(f"  ✓ Saved: {output_file}")
    
    # Preview
    print("\n  Code Preview (first 20 lines):")
    print("  " + "-"*66)
    lines = code.split('\n')[:20]
    for i, line in enumerate(lines, 1):
        print(f"  {i:3d} | {line}")
    if len(code.split('\n')) > 20:
        print(f"  ... ({len(code.split('\n')) - 20} more lines)")
    print("  " + "-"*66)
    
    print("\n" + "="*70)
    if has_global and has_cuda_api and not has_python:
        print("✓ ALL TESTS PASSED - KISYSTEM v3.9 VALIDATED")
        print("="*70)
        print("\nNext: Compile with:")
        print(f"  nvcc -arch=sm_89 {output_file} -o tests/validation_v3_9/vector_add.exe")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)