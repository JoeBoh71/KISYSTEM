from core.meta_supervisor import MetaSupervisor
from pathlib import Path

print('='*80)
print('KISYSTEM v3.8 - META-SUPERVISOR TASK SELECTION')
print('='*80)
print()

meta = MetaSupervisor(learning_log_path='D:/AGENT_MEMORY/learning_log.json')
tasks = meta.get_top_priorities()

if tasks:
    print(f'Found {len(tasks)} prioritized tasks\n')
    print('TOP 5 PRIORITIES:\n')
    for i, t in enumerate(tasks[:5], 1):
        print(f'{i}. {t.get("task_id")} - Priority: {t.get("priority"):.3f}')
        print(f'   SR: {t.get("success_rate"):.0%} ({t.get("attempts")} attempts)')
        print(f'   Complexity: {t.get("complexity")}/10')
        print(f'   Days: {t.get("days_since_attempt"):.1f}')
        print()
    print('='*80)
    print(f'SELECTED: {tasks[0].get("task_id")}')
    print(f'Priority: {tasks[0].get("priority"):.3f}')
    print('='*80)
