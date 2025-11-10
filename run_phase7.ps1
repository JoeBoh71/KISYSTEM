# Phase-7 Daily Runner
$ErrorActionPreference = 'Stop'

Write-Host "=== KISYSTEM Integrity Check ==="
python "C:\KISYSTEM\agents\integrity_agent.py"

Write-Host "`n=== Build/Test/Optimize ==="
$pyScript = @'
import asyncio
from core.supervisor_v3_optimization import SupervisorV3WithOptimization, OptimizationConfig

tasks = [
    {"title": "CUDA Hot Kernel", "language": "CUDA", "code_hint": "matrix multiply with shared memory"},
    {"title": "DSP PR-FIR", "language": "cpp", "code_hint": "overlap-save fast convolution 192 kHz"}
]

cfg = OptimizationConfig(
    target_score=80,
    max_iterations=10,
    max_concurrent_builds=3,
    enable_meta_supervisor=True,
    verbose=True
)

asyncio.run(SupervisorV3WithOptimization(cfg).execute_with_optimization(tasks))
'@

# Inline-Python an Interpreter übergeben
python - <<ENDPY
$pyScript
ENDPY

Write-Host "`n=== Report ==="
Get-Content "C:\KISYSTEM\Logs\integrity_report.json"
