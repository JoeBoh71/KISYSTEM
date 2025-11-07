import sys
sys.path.insert(0, 'C:\\KISYSTEM')
sys.path.insert(0, 'C:\\KISYSTEM\\core')
sys.path.insert(0, 'C:\\KISYSTEM\\agents')

import asyncio
from supervisor_v3_optimization import SupervisorV3WithOptimization

async def test():
    supervisor = SupervisorV3WithOptimization(max_optimization_iterations=1, verbose=True)
    
    task = 'Create simple CUDA kernel that adds 1.0f to each element of a float array'
    
    result = await supervisor.execute_with_optimization(
        task=task,
        language='cuda',
        performance_target=80.0
    )
    
    print('\n' + '='*70)
    print('=== FINAL RESULT ===')
    print('='*70)
    print('Status:', result['status'])
    print('Iterations:', result['iterations'])
    print('Errors:', result.get('errors', []))
    if result['final_code']:
        print('Code generated: YES')
        print('Code length:', len(result['final_code']), 'chars')
    print('='*70)

asyncio.run(test())
