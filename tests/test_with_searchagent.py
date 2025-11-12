import asyncio
from core.supervisor_v3 import SupervisorV3
from agents.search_agent_v2 import SearchAgent

async def main():
    print("="*80)
    print("U3DAW Task 1.2 - WITH SearchAgent")
    print("="*80)
    print()
    
    # Initialize SearchAgent
    search = SearchAgent()
    
    # Pass it to Supervisor
    s = SupervisorV3(max_iterations=5, search_agent=search)
    
    task = "cuFFT wrapper for batch R2C/C2R transforms with 32-channel support and pinned memory management"
    
    # Check setup
    complexity = s._analyze_task_complexity(task, "cuda")
    print(f"Complexity: {complexity}")
    print(f"SearchAgent: {'ENABLED' if s.search else 'DISABLED'}")
    print()
    
    # Run task
    result = await s.execute_task(task=task, language="cuda")
    
    print()
    print("="*80)
    print(f"RESULT: {result['status']}")
    print(f"Iterations: {result['iterations']} | Fixes: {result['fixes']}")
    print("="*80)

asyncio.run(main())
