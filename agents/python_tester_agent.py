"""
PythonTesterAgent v1.1 - CuPy GPU Testing
Minimal version for quick deployment

Run standalone: python agents\python_tester_agent.py
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import re
import subprocess
import sys

logger = logging.getLogger(__name__)


class PythonTesterAgent:
    """CuPy-based GPU testing - no compilation needed."""
    
    def __init__(self, ollama_client=None, learning_module=None):
        self.learning_module = learning_module
        self.name = "PythonTesterAgent"
        self.version = "1.1-CuPy"
        
        # Check CuPy
        try:
            import cupy as cp
            print(f"[{self.name}] ✅ CuPy v{cp.__version__} installed")
            device = cp.cuda.Device(0)
            cc = device.compute_capability
            print(f"[{self.name}] ✅ GPU 0: Compute {cc[0]}.{cc[1]}")
        except ImportError:
            print(f"[{self.name}] ❌ CuPy not installed!")
            print(f"[{self.name}] Install: pip install cupy-cuda12x")
        except Exception as e:
            print(f"[{self.name}] ⚠️ GPU warning: {e}")
        
        logger.info(f"{self.name} v{self.version} initialized")
    
    def _extract_kernels(self, cuda_code: str) -> List[Dict]:
        """Extract kernel info from CUDA code."""
        kernels = []
        pattern = r'__global__\s+void\s+(\w+)\s*\((.*?)\)'
        
        for match in re.finditer(pattern, cuda_code, re.DOTALL):
            kernel_name = match.group(1)
            params_str = match.group(2)
            
            params = []
            for param in params_str.split(','):
                param = param.strip()
                if not param:
                    continue
                
                parts = param.split()
                if len(parts) >= 2:
                    param_type = ' '.join(parts[:-1])
                    param_name = parts[-1].strip('*&')
                    is_pointer = '*' in param_type or '*' in param
                    
                    if 'float' in param_type and 'double' not in param_type:
                        dtype = 'float32'
                    elif 'double' in param_type:
                        dtype = 'float64'
                    elif 'int' in param_type:
                        dtype = 'int32'
                    else:
                        dtype = 'float32'
                    
                    params.append({
                        'name': param_name,
                        'type': param_type,
                        'dtype': dtype,
                        'is_pointer': is_pointer
                    })
            
            kernels.append({
                'kernel_name': kernel_name,
                'params': params
            })
        
        return kernels
    
    def _generate_test(self, kernels: List[Dict], cuda_code: str) -> str:
        """Generate Python test with CuPy."""
        
        test_funcs = []
        
        for k in kernels:
            name = k['kernel_name']
            params = k['params']
            
            # Setup params
            setup = []
            args = []
            
            for p in params:
                pname = p['name']
                dtype = p['dtype']
                
                if p['is_pointer']:
                    setup.append(f"        {pname} = cp.zeros(1024, dtype=cp.{dtype})")
                    args.append(pname)
                else:
                    if 'int' in dtype:
                        setup.append(f"        {pname} = cp.int32(1024)")
                    else:
                        setup.append(f"        {pname} = cp.{dtype}(1.0)")
                    args.append(pname)
            
            setup_str = "\n".join(setup)
            args_str = ", ".join(args)
            
            func = f'''
def test_{name}():
    try:
{setup_str}
        
        kernel = cp.RawKernel(cuda_code, '{name}')
        kernel((4,), (256,), ({args_str}))
        cp.cuda.Stream.null.synchronize()
        
        print(f"[PythonTesterAgent]   ✓ {name} executed")
        return True, None
    except Exception as e:
        print(f"[PythonTesterAgent]   ✗ {name} failed: {{e}}")
        return False, str(e)
'''
            test_funcs.append(func)
        
        test_calls = "\n    ".join([
            f"success, error = test_{k['kernel_name']}()\n"
            f"    results.append(('{k['kernel_name']}', success, error))"
            for k in kernels
        ])
        
        code = f'''"""GPU Test - CuPy"""
import cupy as cp
import numpy as np

cuda_code = r"""
{cuda_code}
"""

{''.join(test_funcs)}

def run_tests():
    print("="*70)
    print("[PythonTesterAgent] 🧪 Testing CUDA with CuPy...")
    print("="*70)
    
    results = []
    {test_calls}
    
    passed = sum(1 for _, s, _ in results if s)
    failed = len(results) - passed
    
    print("="*70)
    if failed == 0:
        print(f"[PythonTesterAgent] ✓ ALL TESTS PASSED ({{passed}}/{{len(results)}})")
    else:
        print(f"[PythonTesterAgent] ✗ SOME FAILED ({{passed}}/{{len(results)}})")
    print("="*70)
    
    return {{
        'success': failed == 0,
        'total': len(results),
        'passed': passed,
        'failed': failed
    }}

if __name__ == "__main__":
    result = run_tests()
    exit(0 if result['success'] else 1)
'''
        return code
    
    def generate_and_run_tests(
        self, 
        source_file: Path, 
        build_output: str
    ) -> Tuple[bool, str, Optional[Dict]]:
        """Generate and run GPU tests."""
        try:
            # Check CuPy
            try:
                import cupy as cp
            except ImportError:
                return False, "❌ CuPy not installed. Run: pip install cupy-cuda12x", None
            
            # Read CUDA code
            cuda_code = source_file.read_text()
            
            # Extract kernels
            kernels = self._extract_kernels(cuda_code)
            if not kernels:
                return False, "❌ No __global__ kernels found", None
            
            print(f"[{self.name}] Found {len(kernels)} kernel(s)")
            
            # Generate test
            test_code = self._generate_test(kernels, cuda_code)
            
            # Write test
            test_file = source_file.parent / f"test_{source_file.stem}.py"
            test_file.write_text(test_code)
            
            print(f"[{self.name}] 📝 Generated: {test_file}")
            print(f"[{self.name}] 🚀 Running GPU tests...")
            
            # Execute
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout + result.stderr
            success = result.returncode == 0
            
            test_results = {
                'total_tests': len(kernels),
                'passed': len(kernels) if success else 0,
                'failed': 0 if success else len(kernels)
            }
            
            if success:
                msg = f"✅ GPU tests PASSED ({len(kernels)} kernels)\n{output}"
                print(f"[{self.name}] ✅ All tests passed")
            else:
                msg = f"❌ GPU tests FAILED\n{output}"
                print(f"[{self.name}] ❌ Tests failed")
            
            if self.learning_module and success:
                self.learning_module.log_result(
                    task_type='gpu_test',
                    model_used='cupy',
                    success=True,
                    domain='cuda',
                    error_type=None
                )
            
            return success, msg, test_results
            
        except Exception as e:
            logger.error(f"Test error: {e}")
            return False, f"❌ Test error: {e}", None


def main():
    """Standalone test."""
    
    test_cuda = """
__global__ void vector_add(float* c, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""
    
    test_file = Path("test_kernel.cu")
    test_file.write_text(test_cuda)
    
    print("="*70)
    print("PYTHONTESTERAGENT v1.1 - STANDALONE TEST")
    print("="*70 + "\n")
    
    agent = PythonTesterAgent()
    success, output, results = agent.generate_and_run_tests(test_file, "")
    
    print(f"\n{'='*70}")
    print(f"RESULT: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*70}\n")
    
    test_file.unlink(missing_ok=True)
    Path(f"test_{test_file.stem}.py").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
