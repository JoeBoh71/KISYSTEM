# -*- coding: utf-8 -*-
"""
KISYSTEM Phase-7 Runner (mit Pfad-Bootstrap für core/ und agents/)
"""
import asyncio, json, sys
from pathlib import Path

BASE = Path(__file__).parent.resolve()
# Pfade für Modulimporte hinzufügen (Quick-Fix)
for p in (BASE, BASE / "core", BASE / "agents"):
    pp = str(p)
    if pp not in sys.path:
        sys.path.insert(0, pp)

# jetzt können die bisherigen (nicht paketrelativen) Importe funktionieren
from core.supervisor_v3_optimization import SupervisorV3WithOptimization, OptimizationConfig
from agents.integrity_agent import run as run_integrity

LOG_PATH = BASE / r"Logs\integrity_report.json"

async def main():
    print("=== KISYSTEM Integrity Check ===")
    run_integrity(LOG_PATH)

    print("\n=== Build/Test/Optimize ===")
    tasks = [
        {"title": "CUDA Hot Kernel", "language": "CUDA",
         "code_hint": "matrix multiply with shared memory"},
        {"title": "DSP PR-FIR", "language": "cpp",
         "code_hint": "overlap-save fast convolution 192 kHz"},
    ]

    cfg = OptimizationConfig(
        target_score=80,
        max_iterations=10,
        max_concurrent_builds=3,
        enable_meta_supervisor=True,
        verbose=True,
    )

    supervisor = SupervisorV3WithOptimization(cfg)
    result = await supervisor.execute_with_optimization(tasks)

    print("\n=== Summary ===")
    print(json.dumps(result, indent=2))

    if LOG_PATH.exists():
        print("\n=== Integrity Report ===")
        print(LOG_PATH.read_text(encoding="utf-8"))

if __name__ == "__main__":
    asyncio.run(main())
