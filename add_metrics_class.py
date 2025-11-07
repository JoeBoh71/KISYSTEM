#!/usr/bin/env python3
"""Fix performance_parser.py - add PerformanceMetrics class"""

import os

# Read current file
with open('core/performance_parser.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add PerformanceMetrics after imports
addition = """
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    \"\"\"Performance metrics from profiling\"\"\"
    gpu_time_ms: float = 0.0
    score: float = 0.0
    bottleneck: str = 'unknown'
    suggestion: str = ''

"""

# Insert after imports (after "from typing import Dict, Optional")
insert_pos = content.find('class PerformanceParser:')
if insert_pos > 0:
    content = content[:insert_pos] + addition + content[insert_pos:]
    
    # Write back
    with open('core/performance_parser.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ PerformanceMetrics added to performance_parser.py")
else:
    print("❌ Could not find insertion point")
