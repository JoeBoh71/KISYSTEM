"""
Create Learning Log V2 - Bootstrap learning data for Phase 7

Generates 50 synthetic learning entries with CORRECT FORMAT:
- domain (cuda/python/generic)
- model (deepseek-coder-v2:16b, qwen2.5-coder:32b, etc)
- success (true/false)
- time_hours, iterations, etc.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import random

# Learning data with CORRECT FORMAT
learning_entries = []

# Domain distribution
domains = {
    'cuda': 25,       # 50% CUDA tasks
    'python': 10,     # 20% Python
    'tests': 8,       # 16% Test generation
    'cpp': 4,         # 8% C++
    'docs': 2,        # 4% Documentation
    'generic': 1      # 2% Generic
}

# Model distribution by domain
model_preferences = {
    'cuda': [
        ('qwen2.5-coder:32b', 0.6),      # 60% for complex CUDA
        ('deepseek-coder-v2:16b', 0.3),  # 30% for simple CUDA
        ('qwen2.5:32b', 0.1)             # 10% reasoning
    ],
    'python': [
        ('mistral:7b', 0.5),
        ('phi4:latest', 0.3),
        ('deepseek-coder-v2:16b', 0.2)
    ],
    'tests': [
        ('phi4:latest', 0.7),
        ('deepseek-coder-v2:16b', 0.3)
    ],
    'cpp': [
        ('deepseek-coder-v2:16b', 0.6),
        ('qwen2.5-coder:32b', 0.4)
    ],
    'docs': [
        ('phi4:latest', 0.6),
        ('mistral:7b', 0.4)
    ],
    'generic': [
        ('mistral:7b', 0.7),
        ('phi4:latest', 0.3)
    ]
}

# Task titles by domain
task_titles = {
    'cuda': [
        'cuFFT wrapper implementation',
        'PQMF filterbank with Linkwitz-Riley',
        'Overlap-save convolution',
        'TEP gain/phase correction kernel',
        'Psychoacoustic masking calculation'
    ],
    'python': [
        'Data validation pipeline',
        'Configuration loader',
        'File parser utility',
        'Test fixture generator'
    ],
    'tests': [
        'Unit tests for CUDA kernels',
        'Integration tests',
        'Benchmark suite',
        'Validation tests'
    ],
    'cpp': [
        'ASIO wrapper',
        'Audio buffer manager',
        'Thread pool implementation'
    ],
    'docs': [
        'API documentation',
        'Usage guide'
    ],
    'generic': [
        'Helper function'
    ]
}

# Generate entries
entry_id = 1
base_date = datetime(2025, 11, 1)

for domain, count in domains.items():
    for i in range(count):
        # Select model based on distribution
        models = model_preferences[domain]
        r = random.random()
        cumulative = 0
        selected_model = models[0][0]
        for model, prob in models:
            cumulative += prob
            if r <= cumulative:
                selected_model = model
                break
        
        # Success rate depends on model complexity
        if 'qwen2.5-coder:32b' in selected_model:
            success_rate = 0.85
        elif 'deepseek-coder-v2:16b' in selected_model:
            success_rate = 0.75
        elif 'qwen2.5:32b' in selected_model:
            success_rate = 0.80
        else:
            success_rate = 0.70
        
        success = random.random() < success_rate
        
        # Time depends on success/failure
        if success:
            time_hours = round(random.uniform(0.1, 0.8), 2)
            iterations = random.randint(1, 3)
            fixes = random.randint(0, 2)
        else:
            time_hours = round(random.uniform(0.5, 2.0), 2)
            iterations = random.randint(3, 5)
            fixes = random.randint(2, 5)
        
        # Random task title
        title = random.choice(task_titles[domain])
        
        # Create entry with CORRECT FORMAT
        entry = {
            'id': entry_id,
            'date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
            'phase': 1,
            'task': f'1.{entry_id}',
            'title': title,
            'domain': domain,              # ✓ PRESENT
            'model': selected_model,       # ✓ PRESENT
            'success': success,            # ✓ PRESENT
            'time_hours': time_hours,
            'iterations': iterations,
            'fixes': fixes
        }
        
        learning_entries.append(entry)
        entry_id += 1

# Create learning log
learning_log = {
    'project': 'U3DAW',
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'version': '2.0',
    'entries': learning_entries
}

# Save to file
output_path = Path('C:/KISYSTEM/projects/U3DAW/learning_log.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(learning_log, f, indent=2)

print("="*70)
print("LEARNING LOG V2 CREATED")
print("="*70)
print(f"File: {output_path}")
print(f"Entries: {len(learning_entries)}")
print(f"\nDomain Distribution:")
for domain, count in domains.items():
    success_count = sum(1 for e in learning_entries if e['domain'] == domain and e['success'])
    print(f"  {domain:10s}: {count:2d} entries ({success_count}/{count} success)")

print(f"\nModel Usage:")
model_counts = {}
for entry in learning_entries:
    model = entry['model']
    model_counts[model] = model_counts.get(model, 0) + 1

for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
    print(f"  {model:30s}: {count:2d} times")

print(f"\nSample Entry (first):")
print(json.dumps(learning_entries[0], indent=2))

print("\n" + "="*70)
print("✓ Ready for Meta-Supervisor!")
print("="*70)
