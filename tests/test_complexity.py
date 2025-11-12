import asyncio
from core.supervisor_v3 import SupervisorV3

async def main():
    print("STARTING Task 1.2 - cuFFT Wrapper")
    print("="*80)
    
    s = SupervisorV3(max_iterations=5)
    
    task = "cuFFT wrapper for batch R2C/C2R transforms with 32-channel support and pinned memory management"
    
    # Check complexity detection
    complexity = s._analyze_task_complexity(task, "cuda")
    print(f"*** COMPLEXITY DETECTED: {complexity} ***")
    print(f"*** SearchAgent available: {s.search is not None} ***")
    print()
    
    result = await s.execute_task(task=task, language="cuda")
    
    print()
    print("="*80)
    print(f"RESULT: {result['status']}")
    print(f"Iterations: {result['iterations']} | Fixes: {result['fixes']}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
