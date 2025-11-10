"""
Capture nsys output for parser development
Run the existing convolution code and save raw nsys output
"""

import subprocess
import sys
from pathlib import Path

# Paths
cuda_file = Path("D:/AGENT_MEMORY/convolution_optimized_20251109_100554.cu")
output_dir = Path("D:/AGENT_MEMORY")
exe_file = output_dir / "test_convolution.exe"
nsys_path = r'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe'

print("="*70)
print("NSYS OUTPUT CAPTURE")
print("="*70)

# Step 1: Compile
print("\n[1] Compiling CUDA code...")
compile_cmd = [
    'nvcc',
    str(cuda_file),
    '-o', str(exe_file),
    '-arch=sm_89',
    '-O3',
    '--use_fast_math'
]

try:
    result = subprocess.run(
        compile_cmd,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        print(f"✗ Compilation failed:")
        print(result.stderr)
        sys.exit(1)
    
    print("✓ Compilation successful")
    
except Exception as e:
    print(f"✗ Compilation error: {e}")
    sys.exit(1)

# Step 2: Run with nsys
print("\n[2] Running with nsys profiler...")
nsys_cmd = [
    nsys_path,
    'profile',
    '--stats=true',
    '--force-overwrite=true',
    '-o', str(output_dir / 'nsys_report'),
    str(exe_file)
]

try:
    result = subprocess.run(
        nsys_cmd,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # nsys writes to both stdout and stderr
    full_output = result.stdout + "\n" + result.stderr
    
    print("✓ nsys completed")
    
    # Save raw output
    output_file = output_dir / "nsys_raw_output.txt"
    output_file.write_text(full_output, encoding='utf-8')
    
    print(f"\n✓ Raw output saved: {output_file}")
    print(f"   Size: {len(full_output)} chars")
    
    # Show first 2000 chars
    print("\n" + "="*70)
    print("NSYS OUTPUT PREVIEW (first 2000 chars):")
    print("="*70)
    print(full_output[:2000])
    print("...")
    print("="*70)
    
    print(f"\n✓ Full output in: {output_file}")
    print("   Use this to develop better parser patterns!")
    
except FileNotFoundError:
    print(f"✗ nsys not found at: {nsys_path}")
    sys.exit(1)
except Exception as e:
    print(f"✗ nsys error: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("DONE - Check nsys_raw_output.txt for parser development")
print("="*70)