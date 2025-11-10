import sys
from pathlib import Path

file_path = Path('core/supervisor_v3_optimization.py')
backup_path = Path('core/supervisor_v3_optimization.py.backup_20251109')

# Restore backup first
import shutil
shutil.copy(backup_path, file_path)

# Read file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the import block
old_code = '''        # Import profiler agent
        try:
            from cuda_profiler_agent import CUDAProfilerAgent
            self.profiler = CUDAProfilerAgent(verbose=self.verbose)
            if self.verbose:
                print("[Supervisor V3+] ✓ CUDA Profiler enabled")
        except ImportError:
            self.profiler = None
            if self.verbose:
                print("[Supervisor V3+] ⚠️  CUDA Profiler not available")'''

new_code = '''        # Import profiler agent (FIX: correct path to agents/)
        try:
            # Add agents directory to path
            sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))
            from cuda_profiler_agent import CUDAProfilerAgent
            self.profiler = CUDAProfilerAgent(verbose=self.verbose)
            if self.verbose:
                print("[Supervisor V3+] ✓ CUDA Profiler enabled")
        except ImportError as e:
            self.profiler = None
            if self.verbose:
                print(f"[Supervisor V3+] ⚠️  CUDA Profiler not available: {e}")
        except Exception as e:
            self.profiler = None
            if self.verbose:
                print(f"[Supervisor V3+] ✗ CUDA Profiler initialization failed: {e}")'''

# Apply fix
if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fix applied successfully!")
else:
    print("❌ Could not find old code block - manual edit needed")
    print("\nSearching for similar patterns...")
    if "from cuda_profiler_agent import CUDAProfilerAgent" in content:
        print("✓ Found import line")
    
