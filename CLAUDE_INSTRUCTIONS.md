# KISYSTEM - Claude Instructions

**Version:** 3.0 (Run 30)  
**Date:** 2025-11-07  
**Status:** Production Ready with Hybrid Error Handler

---

## CRITICAL: Read This First

This file contains essential instructions for Claude when working on KISYSTEM. 
**ALWAYS read this file at the start of each session.**

Location: `C:\KISYSTEM\CLAUDE_INSTRUCTIONS.md`

---

## System Overview

KISYSTEM is a **multi-agent AI development system** for autonomous software creation with focus on:
- CUDA/C++ code generation with hardware-in-loop optimization
- Intelligent error handling with categorization and confidence-based decisions
- Learning from past solutions (SQLite knowledge base)
- Performance profiling on RTX 4070 GPU
- Multi-model orchestration (Ollama local LLMs)

**Primary Use Case:** Generate, test, and optimize CUDA kernels for U3DAW (Universal 3D Audio Workstation)

---

## Project Context

**Owner:** Jörg Bohne  
**Company:** Bohne Audio (High-end Lautsprecher)  
**Location:** Engelskirchen, Germany  
**Main Project:** U3DAW - GPU-accelerated 3D audio workstation

**User Preferences:**
- Communication: German (code comments in English)
- Style: Direct, technical, no small talk
- Philosophy: "hilf dir selbst, dann hilft dir gott" (self-reliance first)
- Approach: Action > Speculation, Real solutions > Theoretical discussion

---

## Core Architecture (V3.0)

### Component Hierarchy

```
SupervisorV3WithOptimization (core/supervisor_v3_optimization.py)
├── BuilderAgent (agents/builder_agent.py)
├── FixerAgentV3 (agents/fixer_agent_v3.py) ← NEW V3!
│   └── HybridErrorHandler (core/error_handler.py) ← NEW!
│       ├── ErrorCategorizer
│       ├── ModelEscalationChain
│       └── ConfidenceScorer (optional)
├── TesterAgent (agents/tester_agent.py)
├── CUDAProfilerAgent (agents/cuda_profiler_agent.py)
│   └── PerformanceParser (core/performance_parser.py) ← NEW!
├── SearchAgent (agents/search_agent_v2.py)
└── LearningModuleV2 (core/learning_module_v2.py)
```

### NEW in V3.0: Hybrid Error Handler

**Location:** `core/error_handler.py`

**Key Features:**
1. **Error Categorization** - 4 categories with specific retry limits:
   - COMPILATION: 1 retry (deterministic)
   - RUNTIME: 2 retries (medium flexibility)
   - PERFORMANCE: 4 retries (iterative)
   - LOGIC: 3 retries (analysis needed)

2. **Confidence-Based Decisions:**
   - >85% confidence → USE_CACHE (use cached solution)
   - 60-85% confidence → RETRY (with prompt variation)
   - <60% confidence → ESCALATE (bigger model)
   - Retry limit reached → SEARCH (web search) or GIVE_UP

3. **Model Escalation Chains:**
   - Builder: deepseek-coder:16b → qwen2.5-coder:32b
   - Fixer: deepseek-coder:16b → qwen2.5-coder:32b → deepseek-r1:32b
   - Tester: phi4:latest → deepseek-coder:16b

4. **Smart Search Triggering:**
   - Compile errors: Search immediately (priority 2)
   - Runtime errors: Search after 2-3 attempts (priority 3)
   - Performance: Search late (priority 5)

---

## File Structure

```
C:\KISYSTEM\
├── core/
│   ├── supervisor_v3.py                    # Base supervisor
│   ├── supervisor_v3_optimization.py       # With optimization loop
│   ├── model_selector.py                   # Model selection logic
│   ├── workflow_engine.py                  # Dependency management
│   ├── learning_module_v2.py               # SQLite knowledge base
│   ├── error_handler.py                    # NEW: Hybrid Error Handler
│   ├── performance_parser.py               # NEW: Parse nvprof output
│   ├── ollama_client.py                    # Ollama API wrapper
│   └── code_extractor.py                   # Extract code from LLM output
│
├── agents/
│   ├── builder_agent.py                    # Code generation
│   ├── fixer_agent_v3.py                   # NEW: Error fixing with Hybrid Handler
│   ├── tester_agent.py                     # Test generation & execution
│   ├── cuda_profiler_agent.py              # Hardware profiling
│   └── search_agent_v2.py                  # Web search for solutions
│
├── tests/
│   ├── test_system.py                      # Basic system test
│   └── test_hybrid_handler.py              # NEW: Comprehensive V3.0 tests
│
├── prompts/
│   ├── builder_system_prompt.txt
│   ├── fixer_system_prompt.txt
│   └── tester_system_prompt.txt
│
├── data/
│   └── learning_database.db                # SQLite knowledge base
│
├── CLAUDE_INSTRUCTIONS.md                  # This file
├── KISYSTEM_COMPLETE.txt                   # Full system documentation
├── CHANGES.md                              # Change history
└── README.md                               # Project overview
```

---

## Working with KISYSTEM

### Session Start Checklist

1. **Read this file** (CLAUDE_INSTRUCTIONS.md)
2. **Check current status:**
   ```python
   python test_system.py
   ```
3. **Review recent changes:**
   ```
   git log --oneline -5
   ```
4. **Verify Ollama running:**
   ```powershell
   ollama list
   ```

### Code Generation Flow

```
User Request
    ↓
SupervisorV3WithOptimization.execute_with_optimization()
    ↓
[PHASE 1: BUILD]
    ↓
BuilderAgent.build()
    ├── Detect dependencies
    ├── Select model (complexity-based)
    └── Generate code
    ↓
[PHASE 2: OPTIMIZATION LOOP]
    ↓
CUDAProfilerAgent.profile_cuda()
    ├── Compile with nvcc
    ├── Run on RTX 4070
    ├── Parse performance metrics
    └── Identify bottlenecks
    ↓
FixerAgentV3.fix_performance()
    ├── HybridErrorHandler.handle_error()
    │   ├── Categorize error
    │   ├── Check learning database
    │   ├── Calculate confidence
    │   └── Make decision (cache/retry/escalate/search)
    ├── Apply decision
    └── Generate optimized code
    ↓
Repeat until performance_target reached
    ↓
Return optimized code
```

### Error Handling Flow (V3.0)

```
Error Occurs
    ↓
HybridErrorHandler.handle_error()
    ↓
1. CATEGORIZE
   ├── COMPILATION (1 retry)
   ├── RUNTIME (2 retries)
   ├── PERFORMANCE (4 retries)
   └── LOGIC (3 retries)
    ↓
2. CHECK LEARNING DATABASE
   └── Find similar solutions (>60% confidence)
    ↓
3. CALCULATE CONFIDENCE
   ├── >85% → USE_CACHE
   ├── 60-85% → RETRY (with variation)
   └── <60% → ESCALATE
    ↓
4. CHECK RETRY LIMIT
   └── Exceeded → SEARCH or GIVE_UP
    ↓
5. EXECUTE DECISION
   ├── USE_CACHE: Apply cached solution
   ├── RETRY: Generate with variation
   ├── ESCALATE: Use bigger model
   ├── SEARCH: Web search + retry
   └── GIVE_UP: Report failure
    ↓
6. STORE RESULT
   └── LearningModuleV2.store_solution()
```

---

## Model Selection Strategy

### Complexity-Based Selection

**SIMPLE tasks** (phi4:latest or llama3.1:8b):
- Single function implementations
- Basic CUDA kernels (<50 lines)
- Simple bug fixes
- Standard library usage

**MEDIUM tasks** (deepseek-coder:16b):
- Multi-function modules
- CUDA kernels with shared memory
- Error fixing (first attempt)
- Test generation

**COMPLEX tasks** (qwen2.5-coder:32b):
- Full system implementations
- Advanced CUDA optimizations
- Deep debugging (3+ attempts)
- Architecture decisions

**REASONING tasks** (deepseek-r1:32b):
- Last resort for persistent errors
- Complex logic errors
- System design questions

### Escalation Triggers

**Automatic escalation after:**
- 3 failures with same model → Next in chain
- Compile error + no learning match → Search immediately
- Performance < 50% of target → Larger model

---

## Key Rules for Claude

### DO:
✅ Always read CLAUDE_INSTRUCTIONS.md at session start
✅ Use real file paths and actual code
✅ Test changes with `python test_system.py`
✅ Check for existing solutions in learning database
✅ Let Hybrid Error Handler make retry decisions
✅ Use category-specific retry limits
✅ Store all solutions in learning database
✅ Commit working code to git
✅ Use German for communication (unless specified otherwise)

### DON'T:
❌ Simulate or mock functionality
❌ Hardcode retry logic (use Hybrid Handler!)
❌ Skip error categorization
❌ Ignore confidence scores
❌ Use await with non-async functions (LearningModule!)
❌ Modify supervisor without testing
❌ Add features without testing
❌ Use English for general communication

### CRITICAL Anti-Patterns:
🚫 **Vorpreschen** (rushing ahead without understanding)
🚫 **Raten** (guessing instead of checking)
🚫 **Spekulieren** (theorizing instead of testing)
🚫 **Multi-Tasking** (finish one thing before starting another)

---

## Hardware Setup

**GPU:** NVIDIA RTX 4070  
**CPU:** AMD Ryzen 9 7950X (16C/32T)  
**RAM:** 64 GB DDR5  
**OS:** Windows 11 Pro  
**CUDA:** 12.6  
**nvcc:** Available at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe`

### Compilation Settings

**Default nvcc flags:**
```bash
nvcc -arch=sm_89 -O3 --use_fast_math -Xcompiler "/O2 /fp:fast"
```

**For profiling:**
```bash
nvcc -arch=sm_89 -lineinfo -O3
```

---

## Testing

### Quick Test
```powershell
cd C:\KISYSTEM
python test_system.py
```

### Comprehensive Test (V3.0)
```powershell
cd C:\KISYSTEM
python test_hybrid_handler.py
```

**Expected output:**
```
✓ Error Categorization: PASSED
✓ Confidence Decisions: PASSED
✓ Direct Fixer V3: PASSED
✓ Supervisor with Iterations: PASSED

✅ ALL TESTS PASSED - HYBRID HANDLER FULLY FUNCTIONAL!
```

### Manual CUDA Test
```powershell
cd C:\KISYSTEM
python -c "from agents.cuda_profiler_agent import CUDAProfilerAgent; import asyncio; asyncio.run(CUDAProfilerAgent().test_cuda_available())"
```

---

## Troubleshooting

### Common Issues

**1. Ollama not responding:**
```powershell
# Check if running:
ollama list

# Restart if needed:
# Close Ollama Desktop app, then restart
```

**2. Import errors:**
```powershell
# Clear Python cache:
Remove-Item -Recurse -Force core\__pycache__,agents\__pycache__
```

**3. CUDA compilation fails:**
```powershell
# Check nvcc:
nvcc --version

# Test simple compilation:
echo "__global__ void test() {}" > test.cu
nvcc test.cu -o test.exe
```

**4. Learning database locked:**
```powershell
# Close all Python processes, then:
Remove-Item data\learning_database.db-journal -ErrorAction SilentlyContinue
```

**5. Hybrid Handler fails:**
- Check if `error_handler.py` exists in `core/`
- Verify no `await` calls to LearningModule (it's synchronous!)
- Run `python test_hybrid_handler.py` to diagnose

---

## Git Workflow

### Before Changes
```powershell
git status
git pull
```

### After Changes
```powershell
git add .
git commit -m "Clear description of what changed"
git push
```

### Backup Before Major Changes
```powershell
copy core\supervisor_v3_optimization.py core\supervisor_backup.py
```

---

## Performance Targets

**CUDA Kernel Goals:**
- Compilation: <5 seconds
- Performance Score: >80/100
- Occupancy: >50%
- Memory Efficiency: >70%
- Iterations to optimize: <10

**System Goals:**
- Test pass rate: 100%
- Fix success rate: >80%
- Cache hit rate: >30% (after warm-up)
- Average fix time: <2 minutes

---

## Version History

**V3.0 (Run 30 - 2025-11-07):**
- ✅ Hybrid Error Handler with categorization
- ✅ Confidence-based retry decisions
- ✅ Model escalation chains
- ✅ Performance parser with bottleneck detection
- ✅ Smart search triggering
- ✅ 4/4 tests passing

**V2.0 (Run 20-29):**
- Multi-agent system with learning
- Hardware-in-loop optimization
- Search agent integration

**V1.0 (Run 1-19):**
- Basic CUDA generation
- Simple error handling

---

## Contact & Resources

**GitHub:** https://github.com/JoeBoh71/KISYSTEM  
**Owner:** Jörg Bohne  
**Email:** [Contact via GitHub]

**Key Files to Reference:**
- `KISYSTEM_COMPLETE.txt` - Full system documentation
- `CHANGES.md` - Detailed change history
- `test_hybrid_handler.py` - V3.0 test suite

---

## Session End Checklist

Before ending session:
1. ✅ Run tests (`python test_system.py` or `test_hybrid_handler.py`)
2. ✅ Commit changes to git
3. ✅ Update CHANGES.md if significant
4. ✅ Note any pending issues for next session
5. ✅ Clear any temporary files

---

**Last Updated:** 2025-11-07 (Run 30)  
**Next Review:** After Run 35 or major architecture change
