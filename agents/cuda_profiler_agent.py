"""
KISYSTEM CUDA Profiler Agent - V2 Fixed
Compile, run, and profile CUDA code for performance optimization

Author: Jörg Bohne
Date: 2025-11-09
Version: 2.0 - Fixed nsys output parsing (stdout not stderr)
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional
import asyncio

# Add path for performance parser
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

try:
    from core.performance_parser import PerformanceParser, PerformanceMetrics
except ImportError:
    print("[CUDAProfiler] Warning: performance_parser not found")
    PerformanceParser = None
    PerformanceMetrics = None


# Standard math constants (auto-injected when needed)
MATH_DEFINES = """
// Standard math constants (auto-injected by KISYSTEM)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
#ifndef M_PI_4
#define M_PI_4 0.78539816339744830962
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
"""


def validate_and_clean_cuda_code(code: str) -> tuple:
    """
    Validates CUDA code and removes problematic preprocessor artifacts.
    
    Prevents C2019 errors by:
    - Removing invalid "# number" syntax (should be "#line number")
    - Filtering unknown preprocessor directives
    - Removing BOM characters
    
    Returns: (cleaned_code, list_of_issues)
    """
    issues = []
    lines = code.split('\n')
    cleaned = []
    
    valid_preprocessor_keywords = {
        'include', 'define', 'undef', 'if', 'ifdef', 'ifndef', 
        'else', 'elif', 'endif', 'line', 'pragma', 'error', 
        'warning', 'import'
    }
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Check for invalid preprocessor syntax
        if stripped.startswith('#'):
            # Split on whitespace to get first token after #
            tokens = stripped[1:].strip().split()
            if tokens:
                keyword = tokens[0]
                
                # Check for invalid "# number" syntax (common C2019 cause)
                if keyword.isdigit():
                    issues.append(f"Line {i}: Invalid preprocessor '# {keyword}' removed (C2019 prevention)")
                    continue  # Skip this line
                
                # Check for unknown preprocessor keyword
                if keyword not in valid_preprocessor_keywords:
                    issues.append(f"Line {i}: Unknown preprocessor '#{keyword}' removed")
                    continue  # Skip this line
        
        # Check for BOM or other invisible characters at start
        if i == 1 and line and ord(line[0]) > 127:
            issues.append(f"Line 1: Non-ASCII character at file start removed (BOM?)")
            # Remove BOM if present
            if line.startswith('\ufeff'):
                line = line[1:]
        
        cleaned.append(line)
    
    return '\n'.join(cleaned), issues


def ensure_required_includes(code: str) -> tuple:
    """
    Ensures critical CUDA includes and math defines are present.
    Returns: (code_with_includes, list_of_added_includes)
    
    v2.1: Added M_PI auto-injection
    """
    added = []
    
    # Required includes and their detection patterns
    required = [
        ('#include <cuda_runtime.h>', ['cudaMalloc', 'cudaMemcpy', 'cudaFree', '__global__']),
        ('#include <iostream>', ['std::cout', 'std::cerr', 'std::endl']),
        ('#include <stdio.h>', ['printf('])
    ]
    
    # Check for math functions (need math.h)
    math_functions = ['sin', 'cos', 'tan', 'sqrt', 'pow', 'exp', 'log', 'fabs', 'ceil', 'floor']
    if any(f'({func}(' in code or f' {func}(' in code for func in math_functions):
        if '#include <math.h>' not in code and '#include <cmath>' not in code:
            required.append(('#include <math.h>', math_functions))
    
    for include_line, patterns in required:
        # Check if include already present
        if include_line in code:
            continue
        
        # Check if any pattern is used in code
        if any(pattern in code for pattern in patterns):
            # Add include at the beginning
            code = include_line + '\n' + code
            added.append(include_line)
    
    # NEW: Check for math constants (M_PI, M_E, etc.)
    math_constants = ['M_PI', 'M_E', 'M_PI_2', 'M_PI_4', 'M_SQRT2']
    needs_math_defines = any(const in code for const in math_constants)
    
    # Check if defines are already present
    has_defines = '#define M_PI' in code or '#ifndef M_PI' in code
    
    if needs_math_defines and not has_defines:
        # Add math defines at the beginning (after includes)
        code = MATH_DEFINES + '\n' + code
        added.append('MATH_DEFINES')
    
    return code, added


class CUDAProfilerAgent:
    """
    Profile CUDA code for performance optimization
    Compiles, runs, profiles with nvprof/nsys
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize CUDA Profiler Agent
        
        Args:
            verbose: Print status messages
        """
        self.verbose = verbose
        self._print("[CUDAProfiler] ✓ Initialized with nvprof/nsys Integration")
    
    def _print(self, msg: str):
        """Print if verbose"""
        if self.verbose:
            print(msg)
    
    async def profile_cuda(
        self,
        code: str,
        profile: bool = True,
        compile_only: bool = False
    ) -> Dict:
        """
        Profile CUDA code on RTX 4070
        
        Args:
            code: CUDA source code
            profile: Run with nvprof for performance metrics
            compile_only: Only compile, don't run
            
        Returns:
            {
                'status': 'success' | 'compile_error' | 'runtime_error',
                'compile_output': str,
                'runtime_output': str,
                'performance': PerformanceMetrics | None,
                'suggestions': List[Dict] | None,
                'executable': Path | None
            }
        """
        self._print("\n" + "="*70)
        self._print("[CUDAProfiler] 🔧 Profiling CUDA Code on RTX 4070")
        self._print("="*70)
        
        # === VALIDATION & AUTO-FIX ===
        self._print("[CUDAProfiler] Validating code...")
        cleaned_code, validation_issues = validate_and_clean_cuda_code(code)
        
        if validation_issues:
            self._print(f"[CUDAProfiler] ⚠️  Fixed {len(validation_issues)} issues:")
            for issue in validation_issues[:5]:  # Show max 5
                self._print(f"[CUDAProfiler]   • {issue}")
            if len(validation_issues) > 5:
                self._print(f"[CUDAProfiler]   ... and {len(validation_issues)-5} more")
        
        cleaned_code, added_includes = ensure_required_includes(cleaned_code)
        
        if added_includes:
            self._print(f"[CUDAProfiler] ✅ Auto-added {len(added_includes)} includes:")
            for inc in added_includes:
                self._print(f"[CUDAProfiler]   • {inc}")
        
        # Use cleaned code
        code = cleaned_code
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Save code
            source_file = tmpdir / "kernel.cu"
            source_file.write_text(code, encoding='utf-8')
            self._print(f"[CUDAProfiler] Source: {source_file}")
            
            # Step 1: Compile
            self._print(f"[CUDAProfiler] Step 1: Compiling with nvcc...")
            compile_result = await self._compile_cuda(source_file, tmpdir)
            
            if compile_result['status'] != 'success':
                self._print(f"[CUDAProfiler] ✗ Compilation failed")
                return {
                    'status': 'compile_error',
                    'compile_output': compile_result['output'],
                    'runtime_output': None,
                    'performance': None,
                    'suggestions': None,
                    'executable': None
                }
            
            self._print(f"[CUDAProfiler] ✓ Compilation successful")
            
            if compile_only:
                return {
                    'status': 'success',
                    'compile_output': compile_result['output'],
                    'runtime_output': None,
                    'performance': None,
                    'suggestions': None,
                    'executable': compile_result['executable']
                }
            
            # Step 2: Run with profiling
            self._print(f"[CUDAProfiler] Step 2: Running on GPU...")
            
            if profile and PerformanceParser:
                run_result = await self._run_with_profiling(compile_result['executable'])
            else:
                run_result = await self._run_cuda(compile_result['executable'])
            
            if run_result['status'] != 'success':
                self._print(f"[CUDAProfiler] ✗ Runtime error")
                return {
                    'status': 'runtime_error',
                    'compile_output': compile_result['output'],
                    'runtime_output': run_result['output'],
                    'performance': None,
                    'suggestions': None,
                    'executable': compile_result['executable']
                }
            
            self._print(f"[CUDAProfiler] ✓ Execution successful")
            
            # Step 3: Parse performance
            performance = None
            suggestions = None
            
            # Try profile_output first, fallback to runtime_output (nsys writes there)
            output_to_parse = run_result.get('profile_output') or run_result.get('output')
            
            if profile and PerformanceParser and output_to_parse:
                self._print(f"[CUDAProfiler] Step 3: Analyzing performance...")
                performance = PerformanceParser.parse_output(output_to_parse)
                
                if performance:
                    self._print(f"[CUDAProfiler] Performance Score: {performance.performance_score:.1f}/100")
                    self._print(f"[CUDAProfiler] Bottleneck: {performance.bottleneck}")
                    
                    # Get optimization suggestions
                    suggestions = PerformanceParser.get_optimization_suggestions(performance)
                    
                    if suggestions:
                        self._print(f"[CUDAProfiler] Found {len(suggestions)} optimization opportunities")
                        for sugg in suggestions:
                            self._print(f"[CUDAProfiler]   - {sugg['issue']}: {sugg['severity']}")
                else:
                    self._print(f"[CUDAProfiler] ⚠️  Could not parse performance metrics")
            
            self._print("="*70)
            
            return {
                'status': 'success',
                'compile_output': compile_result['output'],
                'runtime_output': run_result['output'],
                'performance': performance,
                'suggestions': suggestions,
                'executable': compile_result['executable']
            }
    
    async def _compile_cuda(self, source_file: Path, output_dir: Path) -> Dict:
        """
        Compile CUDA code with nvcc
        
        Args:
            source_file: Path to .cu file
            output_dir: Directory for executable
            
        Returns:
            {'status': 'success' | 'error', 'output': str, 'executable': Path}
        """
        executable = output_dir / "kernel.exe"
        
        # nvcc command - optimized for RTX 4070 (Ada Lovelace, sm_89)
        cmd = [
            'nvcc',
            str(source_file),
            '-o', str(executable),
            '-lcufft',  # Link cuFFT
            '-O3',      # Optimization
            '--use_fast_math',
            '-arch=sm_89',  # RTX 4070 = Ada Lovelace = sm_89
            '--ptxas-options=-v'  # Verbose PTX assembler for register/memory info
        ]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            
            stdout_text = stdout.decode('utf-8', errors='ignore')
            stderr_text = stderr.decode('utf-8', errors='ignore')
            
            if proc.returncode == 0:
                return {
                    'status': 'success',
                    'output': stdout_text + stderr_text,
                    'executable': executable
                }
            else:
                return {
                    'status': 'error',
                    'output': stderr_text or stdout_text,
                    'executable': None
                }
                
        except asyncio.TimeoutError:
            return {
                'status': 'error',
                'output': 'Compilation timeout (60s)',
                'executable': None
            }
        except Exception as e:
            return {
                'status': 'error',
                'output': f'Compilation exception: {str(e)}',
                'executable': None
            }
    
    async def _run_cuda(self, executable: Path) -> Dict:
        """
        Run CUDA executable
        
        Args:
            executable: Path to compiled binary
            
        Returns:
            {'status': 'success' | 'error', 'output': str}
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                str(executable),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            
            stdout_text = stdout.decode('utf-8', errors='ignore')
            stderr_text = stderr.decode('utf-8', errors='ignore')
            
            if proc.returncode == 0:
                return {
                    'status': 'success',
                    'output': stdout_text + stderr_text
                }
            else:
                return {
                    'status': 'error',
                    'output': stderr_text or stdout_text
                }
                
        except asyncio.TimeoutError:
            return {
                'status': 'error',
                'output': 'Runtime timeout (30s)'
            }
        except Exception as e:
            return {
                'status': 'error',
                'output': f'Runtime exception: {str(e)}'
            }
    
    async def _run_with_profiling(self, executable: Path) -> Dict:
        """
        Run CUDA executable with nvprof/nsys
        
        Args:
            executable: Path to compiled binary
            
        Returns:
            {'status': 'success' | 'error', 'output': str, 'profile_output': str}
        """
        # Try nvprof first
        cmd = [
            'nvprof',
            '--metrics', 'gld_efficiency,gst_efficiency,shared_efficiency,achieved_occupancy',
            '--events', 'shared_ld_bank_conflict,shared_st_bank_conflict',
            str(executable)
        ]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            
            # nvprof outputs to stderr
            profile_output = stderr.decode('utf-8', errors='ignore')
            runtime_output = stdout.decode('utf-8', errors='ignore')
            
            if proc.returncode == 0 or 'Profiling result' in profile_output:
                return {
                    'status': 'success',
                    'output': runtime_output,
                    'profile_output': profile_output
                }
            else:
                # Fallback to nsys
                return await self._run_with_nsys(executable)
                
        except FileNotFoundError:
            # nvprof not available
            self._print("[CUDAProfiler] nvprof not found, trying nsys")
            return await self._run_with_nsys(executable)
        except asyncio.TimeoutError:
            return {
                'status': 'error',
                'output': 'Profiling timeout (60s)',
                'profile_output': None
            }
        except Exception as e:
            return {
                'status': 'error',
                'output': f'Profiling exception: {str(e)}',
                'profile_output': None
            }
    
    async def _run_with_nsys(self, executable: Path) -> Dict:
        """
        Run with nsys (newer profiler)
        
        Args:
            executable: Path to compiled binary
            
        Returns:
            {'status': 'success' | 'error', 'output': str, 'profile_output': str}
        """
        nsys_path = r'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe'
        cmd = [nsys_path, 'profile', '--stats=true', str(executable)]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            
            stdout_text = stdout.decode('utf-8', errors='ignore')
            stderr_text = stderr.decode('utf-8', errors='ignore')
            
            if proc.returncode == 0:
                # CRITICAL FIX: nsys writes stats tables to STDOUT, not stderr!
                # stderr only has progress bars [1/8] [====100%]
                # stdout has the actual data [6/8] cuda_gpu_kern_sum tables
                return {
                    'status': 'success',
                    'output': stdout_text,  # Runtime output
                    'profile_output': stdout_text + "\n" + stderr_text  # ← FIXED! Both for parser
                }
            else:
                # Fallback to no profiling
                self._print("[CUDAProfiler] nsys failed, running without profiling")
                fallback = await self._run_cuda(executable)
                fallback['profile_output'] = None
                return fallback
                
        except FileNotFoundError:
            # No profiler available
            self._print("[CUDAProfiler] No profiler available, running without profiling")
            fallback = await self._run_cuda(executable)
            fallback['profile_output'] = None
            return fallback
        except asyncio.TimeoutError:
            return {
                'status': 'error',
                'output': 'nsys timeout (60s)',
                'profile_output': None
            }
        except Exception as e:
            return {
                'status': 'error',
                'output': f'nsys exception: {str(e)}',
                'profile_output': None
            }


if __name__ == '__main__':
    # Simple test
    test_code = """
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void testKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    const int N = 1024;
    float *d_data;
    
    cudaMalloc(&d_data, N * sizeof(float));
    
    testKernel<<<4, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    
    printf("Test completed\\n");
    return 0;
}
"""
    
    async def test():
        profiler = CUDAProfilerAgent()
        result = await profiler.profile_cuda(test_code, profile=True)
        
        print("\n=== Test Result ===")
        print(f"Status: {result['status']}")
        if result['performance']:
            perf = result['performance']
            print(f"Performance Score: {perf.performance_score:.1f}/100")
            print(f"Bottleneck: {perf.bottleneck}")
        if result['suggestions']:
            print(f"\nOptimization Suggestions:")
            for sugg in result['suggestions']:
                print(f"  - {sugg['issue']}: {sugg['fix']}")
    
    asyncio.run(test())
