import asyncio
import sys
from pathlib import Path

sys.path.insert(0, 'C:/KISYSTEM')
from core.supervisor_v3 import SupervisorV3

# U3DAW Tasks
TASKS = {
    '1.2': 'cuFFT wrapper for batch R2C/C2R transforms with 32-channel support and pinned memory management',
    '1.4': 'TEP Gain/Phase Correction with ±6dB amplitude limit, ±π/4 phase limit, frequency-local processing'
}

async def main():
    print('='*80)
    print('KISYSTEM v3.8 - U3DAW TASK EXECUTION')
    print('='*80)
    print()
    
    # Use Task 1.2 (cuFFT - already attempted in RUN 37.3)
    task_id = '1.2'
    task_desc = TASKS[task_id]
    
    print(f'Task {task_id}: {task_desc}')
    print()
    
    supervisor = SupervisorV3(max_iterations=5)
    
    result = await supervisor.execute_task(
        task=task_desc,
        language='cuda'
    )
    
    print('\n' + '='*80)
    print('RESULT:', result['status'])
    print(f'Iterations: {result["iterations"]} | Fixes: {result["fixes"]}')
    print('='*80)

if __name__ == '__main__':
    asyncio.run(main())
