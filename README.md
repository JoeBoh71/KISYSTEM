# KISYSTEM

**Status:** MVP Functional (v1.0)  
**Author:** Jörg Bohne  
**Date:** 2025-11-07  
**Language:** Python 3.11+

---

## 🤖 For Claude

**⚡ READ FIRST:** https://raw.githubusercontent.com/JoeBoh71/KISYSTEM/main/CLAUDE_INSTRUCTIONS.md

Complete project context, rules, and current state for all sessions.

---

## Overview

Multi-Agent System für automatisierte Code-Generierung, Testing und Fixing mit Smart Model Routing und Hardware-in-the-Loop CUDA Optimization.

### Autonomous Development Loop:
```
User Task → BuilderAgent → Code Generated
     ↓
CUDA Profiler → Compilation
     ↓
(if error) → FixerAgent → Fixed Code
     ↓
Learning Module → Store Pattern
```

## Features

- 🤖 **Smart Model Routing:** Auto-escalation (8b → 16b → 32b)
- 🔧 **Auto-Fix Loop:** Bis zu 5 Iterations mit Learning
- 🎯 **CUDA-Optimized:** Auto-include injection, Hardware profiling
- 📚 **Learning Module:** Context-aware solution matching
- 🔍 **Web Search:** Integration bei wiederkehrenden Errors

## Requirements

- Python 3.11+
- Ollama mit Models:
  - `llama3.1:8b` (fast, simple tasks)
  - `deepseek-coder-v2:16b` (medium)
  - `qwen2.5-coder:32b` (complex CUDA)
  - `deepseek-r1:32b` (deep debugging)
- CUDA Toolkit 13+ (für CUDA Profiling)
- nvcc im PATH

## Quick Start

```bash
# Clone repo
git clone https://github.com/JoeBoh71/KISYSTEM.git
cd kisystem

# Test installation
python test_system.py --minimal
# Should output:
# ✓ supervisor_v3_optimization import OK
# ✓ builder_agent import OK
# ✓ fixer_agent import OK
# ✓ cuda_profiler_agent import OK
# ✅ ALL IMPORTS OK

python test_system.py
```

Generiert simple CUDA kernel mit auto-includes und testet Compilation.

## Usage Example

```python
import asyncio
from core.supervisor_v3_optimization import SupervisorV3WithOptimization

async def main():
    supervisor = SupervisorV3WithOptimization(
        max_optimization_iterations=2,
        verbose=True
    )
    
    task = 'Create CUDA kernel for vector dot product'
    result = await supervisor.execute_with_optimization(
        task=task,
        language='cuda',
        performance_target=80.0
    )
    
    print(f"Status: {result['status']}")
    print(f"Code: {result['final_code'][:200]}...")

asyncio.run(main())
```

## Project Structure

```
kisystem/
├── core/                    # Basis-Module
│   ├── model_selector.py          # Smart routing
│   ├── ollama_client.py           # Ollama integration
│   ├── learning_module_v2.py      # Context-aware learning
│   ├── supervisor_v3.py           # Main orchestrator
│   └── supervisor_v3_optimization.py # Hardware-in-loop
│
├── agents/                  # Specialized agents
│   ├── builder_agent.py           # Code generation
│   ├── fixer_agent.py             # Error fixing
│   ├── tester_agent.py            # Test generation
│   ├── cuda_profiler_agent.py     # CUDA profiling
│   └── search_agent_v2.py         # Web search
│
└── test_system.py           # Integration tests
```

## Components

### BuilderAgent
- Generiert Code based on task description
- Auto-dependency detection
- CUDA auto-include injection
- Model: `qwen2.5-coder:32b` für CUDA

### FixerAgent
- Error analysis + fixing
- Smart escalation (3+ failures → `deepseek-r1:32b`)
- CUDA auto-include injection
- Web search integration

### Learning Module
- SQLite-based solution storage
- Multi-factor confidence scoring (40/30/20/10)
- Context-aware matching (OS, GPU, compiler)
- 29+ solutions gespeichert

### CUDA Profiler Agent
- C2019 error prevention
- Auto-include detection
- nvprof/nsys integration
- Performance metrics

## Performance

**Success Rates (CUDA Kernels):**
- Simple (array ops): 80-90% in 1 iteration
- Medium (shared memory): 60-70% in 1-2 iterations
- Complex (reductions): 40-50% in 2-3 iterations

**Durchschnitt:**
- Generierung: 1-3 min (abhängig von Model)
- Compilation: 5-10 sec
- Total: 2-5 min per task

## MVP Status

⚠️ **Validation disabled:** Manual testing required  
⚠️ **PerformanceParser missing:** Basic profiling only  
⚠️ **Complex logic:** LLM kann bei komplexen Algorithmen scheitern

**Workarounds:**
- Manual code review nach Generation
- External testing framework
- Iterative refinement mit Feedback

## Documentation

- [INSTALL.md](./INSTALL.md) - Detailed installation
- [CHANGES.md](./CHANGES.md) - Version history + fixes
- [CLAUDE_INSTRUCTIONS.md](./CLAUDE_INSTRUCTIONS.md) - Complete context for Claude
- [Architecture](./docs/architecture.md) - System design (TODO)
- [API Reference](./docs/api.md) - Agent APIs (TODO)

## Development

```bash
# Feature branch
git checkout -b feature/new-agent

# Make changes
git add .
git commit -m "Add: New agent for X"

# Push
git push origin feature/new-agent
```

## Releases

```bash
# Tag release
git tag -a v1.0 -m "MVP Functional"
git push origin v1.0
```

## Usage with Claude

**Nächste Session:**

```
User: "KISYSTEM" or "kis"
Claude: [fetches README → sees URL → fetches CLAUDE_INSTRUCTIONS.md]
Claude: [has complete context]
```

= Kein File-Upload mehr nötig

## Roadmap

- [ ] Real test execution
- [ ] Better error messages
- [ ] Comprehensive logging
- [ ] PerformanceParser implementation
- [ ] Advanced CUDA metrics
- [ ] Optimization suggestions
- [ ] Multi-GPU support
- [ ] Distributed execution
- [ ] Web UI
- [ ] REST API

## License

Private - Jörg Bohne © 2025

## Credits

- Ollama Team - Local LLM infrastructure
- NVIDIA - CUDA Toolkit
- Anthropic - Claude API (für Development)

## Contact

- Issues: github.com/JoeBoh71/KISYSTEM/issues
- Email: [your email]

---

*Built with Claude + 15 years of audio DSP expertise.*
