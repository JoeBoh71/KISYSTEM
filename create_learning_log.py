"""
Create Test Learning Log for KISYSTEM Phase 7
==============================================

Generates realistic learning data for Meta-Supervisor testing.

Author: Jörg Bohne / Bohne Audio
Date: 2025-11-10
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import random


def generate_learning_log(output_path: Path, num_entries: int = 50):
    """
    Generate test learning log with realistic data.
    
    Args:
        output_path: Path to save learning_log.json
        num_entries: Number of log entries to generate
    """
    
    # Define realistic domains and models
    domains = ['cuda', 'cpp', 'asio', 'audio_dsp', 'tests', 'docs']
    models = [
        'llama3.1:8b',
        'mistral:7b',
        'phi4:latest',
        'deepseek-coder-v2:16b',
        'qwen2.5:32b',
        'qwen2.5-coder:32b',
        'deepseek-r1:32b'
    ]
    
    # Model success rates per domain (realistic)
    model_success_rates = {
        'cuda': {
            'llama3.1:8b': 0.2,           # Poor for CUDA
            'mistral:7b': 0.4,
            'phi4:latest': 0.5,
            'deepseek-coder-v2:16b': 0.85,  # Good
            'qwen2.5:32b': 0.75,
            'qwen2.5-coder:32b': 0.90,      # Best
            'deepseek-r1:32b': 0.88
        },
        'cpp': {
            'llama3.1:8b': 0.6,
            'mistral:7b': 0.65,
            'phi4:latest': 0.7,
            'deepseek-coder-v2:16b': 0.88,
            'qwen2.5:32b': 0.80,
            'qwen2.5-coder:32b': 0.92,
            'deepseek-r1:32b': 0.85
        },
        'asio': {
            'llama3.1:8b': 0.5,
            'mistral:7b': 0.55,
            'phi4:latest': 0.6,
            'deepseek-coder-v2:16b': 0.90,  # Excellent
            'qwen2.5:32b': 0.75,
            'qwen2.5-coder:32b': 0.88,
            'deepseek-r1:32b': 0.82
        },
        'audio_dsp': {
            'llama3.1:8b': 0.55,
            'mistral:7b': 0.60,
            'phi4:latest': 0.65,
            'deepseek-coder-v2:16b': 0.82,
            'qwen2.5:32b': 0.78,
            'qwen2.5-coder:32b': 0.90,
            'deepseek-r1:32b': 0.87
        },
        'tests': {
            'llama3.1:8b': 0.75,
            'mistral:7b': 0.80,
            'phi4:latest': 0.92,            # Best for tests
            'deepseek-coder-v2:16b': 0.85,
            'qwen2.5:32b': 0.82,
            'qwen2.5-coder:32b': 0.80,
            'deepseek-r1:32b': 0.78
        },
        'docs': {
            'llama3.1:8b': 0.80,
            'mistral:7b': 0.85,
            'phi4:latest': 0.90,            # Best for docs
            'deepseek-coder-v2:16b': 0.80,
            'qwen2.5:32b': 0.88,
            'qwen2.5-coder:32b': 0.75,
            'deepseek-r1:32b': 0.72
        }
    }
    
    # Timing ranges per model (seconds)
    model_timings = {
        'llama3.1:8b': (15, 30),
        'mistral:7b': (20, 40),
        'phi4:latest': (25, 45),
        'deepseek-coder-v2:16b': (40, 80),
        'qwen2.5:32b': (100, 200),
        'qwen2.5-coder:32b': (150, 300),
        'deepseek-r1:32b': (150, 350)
    }
    
    learning_data = []
    now = datetime.now()
    
    for i in range(num_entries):
        # Select domain and model
        domain = random.choice(domains)
        model = random.choice(models)
        
        # Determine success based on model capability for domain
        success_rate = model_success_rates[domain][model]
        is_success = random.random() < success_rate
        
        # Generate timings
        timing_range = model_timings[model]
        build_time = random.uniform(timing_range[0], timing_range[1])
        test_time = random.uniform(10, 20)
        profile_time = random.uniform(50, 200) if is_success and random.random() < 0.3 else 0
        
        # Generate score (higher for successful runs)
        if is_success:
            score = random.randint(80, 98)
            outcome = 'SUCCESS'
            phase = 'PROFILE' if profile_time > 0 else 'TEST'
            reason = 'All tests passed' if phase == 'TEST' else 'Optimization complete'
        else:
            score = random.randint(0, 60)
            outcome = 'FAIL'
            phase = random.choice(['BUILD', 'TEST'])
            reason = random.choice([
                'Compilation error',
                'Test failed',
                'Timeout',
                'Memory error',
                'Syntax error'
            ])
        
        # Generate timestamp (spread over last 30 days)
        days_ago = random.uniform(0, 30)
        timestamp = (now - timedelta(days=days_ago)).isoformat()
        
        entry = {
            'run_id': f'run_{i+1:03d}',
            'domain': domain,
            'model': model,
            'iteration': random.randint(1, 5),
            'score_final': score,
            'outcome': outcome,
            'phase': phase,
            'reason': reason,
            'timings': {
                'build': round(build_time, 1),
                'test': round(test_time, 1),
                'profile': round(profile_time, 1)
            },
            'timestamp': timestamp
        }
        
        learning_data.append(entry)
    
    # Sort by timestamp (oldest first)
    learning_data.sort(key=lambda x: x['timestamp'])
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(learning_data, f, indent=2)
    
    print(f"[CreateLearningLog] ✓ Generated {num_entries} entries")
    print(f"[CreateLearningLog] ✓ Saved to: {output_path}")
    
    # Print statistics
    success_count = sum(1 for e in learning_data if e['outcome'] == 'SUCCESS')
    success_rate = success_count / num_entries
    
    print(f"\n[CreateLearningLog] Statistics:")
    print(f"  Total entries: {num_entries}")
    print(f"  Successful: {success_count} ({success_rate:.1%})")
    print(f"  Failed: {num_entries - success_count} ({1-success_rate:.1%})")
    
    # Domain distribution
    domain_counts = {}
    for entry in learning_data:
        domain = entry['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"\n  Domain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"    {domain}: {count}")
    
    # Model usage
    model_counts = {}
    for entry in learning_data:
        model = entry['model']
        model_counts[model] = model_counts.get(model, 0) + 1
    
    print(f"\n  Model usage:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {model}: {count}")


if __name__ == "__main__":
    # Generate learning log
    output_path = Path("D:/AGENT_MEMORY/learning_log.json")
    
    print("="*70)
    print("CREATING TEST LEARNING LOG FOR KISYSTEM PHASE 7")
    print("="*70 + "\n")
    
    # Generate 50 entries by default (can be changed)
    generate_learning_log(output_path, num_entries=50)
    
    print("\n" + "="*70)
    print("DONE! You can now use Meta-Supervisor.")
    print("="*70)
    print(f"\nTo test Meta-Supervisor:")
    print(f"  python core/meta_supervisor.py")
    print(f"\nTo test Supervisor V3 with Phase 7:")
    print(f"  python core/supervisor_v3.py")
    print("="*70 + "\n")
