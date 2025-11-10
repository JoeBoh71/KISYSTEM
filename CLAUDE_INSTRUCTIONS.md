# CLAUDE INSTRUCTIONS - KISYSTEM

**Last Updated:** 2025-11-10 22:00 UTC
**Session:** RUN 32
**Status:** Phase 7 Implementation - COMPLETE ✅ (25/25 Tests PASSED)

---

## 🏗️ PROJECT STRUCTURE

**Root:** `C:\KISYSTEM\`

```
C:\KISYSTEM\
├── core/
│   ├── model_selector.py           [v2 - Fixed 2025-11-07 - WORKING]
│   ├── performance_parser.py       [v2 - Fixed 2025-11-07 - WORKING]
│   ├── learning_module_v2.py       [v1 - WORKING]
│   ├── confidence_scorer.py        [v1 - WORKING]
│   ├── context_tracker.py          [v1 - WORKING]
│   ├── ollama_client.py            [v1 - WORKING]
│   ├── supervisor_v3.py            [v3.1 - Phase 7 - WORKING]
│   ├── supervisor_v3_optimization.py [v1 - WORKING]
│   ├── workflow_engine.py          [v1 - WORKING]
│   ├── code_extractor.py           [v1 - WORKING]
│   ├── error_categorizer.py        [v1 - Phase 7 - SEPARATE MODULE]
│   ├── meta_supervisor.py          [v1 - Phase 7 - WORKING]
│   └── hybrid_decision.py          [v1 - Phase 7 - WORKING]
│
├── agents/
│   ├── builder_agent.py            [v2 - Fixed 2025-11-07 - WORKING]
│   ├── fixer_agent.py              [v2 - Fixed 2025-11-07 - WORKING]
│   ├── tester_agent.py             [v2 - Fixed 2025-11-07 - WORKING]
│   ├── search_agent_v2.py          [v2 - Fixed 2025-11-07 - WORKING]
│   ├── review_agent_v2.py          [v1 - WORKING]
│   ├── docs_agent_v2.py            [v1 - WORKING]
│   ├── cuda_profiler_agent.py      [v2 - Fixed 2025-11-07 - PARTIAL]
│   └── hardware_test_agent.py      [v1 - WORKING]
│
├── config/
│   ├── kisystem_config.json
│   └── optimization_config.json    [Phase 7 - WORKING]
│
├── security/
│   └── security_module.py
│
├── tests/
│   ├── test_system.py
│   ├── test_phase6_optimization.py
│   └── test_phase7_meta.py         [Phase 7 - WORKING - 25/25 PASSED]
│
└── docs/
    ├── README.md
    ├── QUICKSTART.md
    └── KISYSTEM_Optimization_Spec_v7.md
```

---

## 💻 SYSTEM HARDWARE SPECIFICATIONS

### Development System
- **CPU:** AMD Ryzen 9 7900 (12-core, 24-thread)
- **GPU:** NVIDIA RTX 4070 (12GB VRAM, CUDA 13.0)
- **RAM:** 64GB DDR5
- **Storage:** Samsung 990 PRO SSD (792GB free)
- **OS:** Windows 10 IoT Enterprise LTSC
- **IDE:** Visual Studio 2022

### Audio Hardware
- **Interface:** RME HDSPe MADI FX (PCIe)
  - 64-channel MADI @ 48/96kHz OR 32-channel @ 192kHz
  - Optical + Coaxial connections
  - ASIO driver with <2ms round-trip latency
  
- **Converters:** 
  - **M-32 AD:** 192.168.10.3 (16 analog inputs)
  - **M-32 DA:** 192.168.10.2 (32 analog outputs)
  - Both: 32-bit/192kHz capable
  
- **Configuration:** 32 channels @ 192kHz via optical MADI

### Speaker System (9.1.6 Immersive)

**Front L/C/R - 4-Way Active:**
- **Bass:** JBL 2242H (18") - 2x per channel
- **Mid:** Audax HM210Z10 (8") - 2x per channel  
- **High:** Bohne Audio patented ribbon tweeters
- **Subs:** 4x RCF LN19S400 (21") in DBA configuration

**Surrounds:**
- 12x Fostex FF165WK fullrange (6 ear-level, 6 height)

**Room:**
- Size: 50m² / 155m³
- Target SPL: 110dB @ Sweet Spot
- System headroom: Running at 1-2% amp power @ 110dB

### Amplifiers
**Custom Class AB Monoblocks (6 identical units):**
- **Topology:** Bridged mono
- **Rails:** ±90V
- **Capacitance:** 200,000µF per amp
- **Transformers:** 2.4kW toroidal per amp
- **Output:** ~1600W RMS per channel
- **Status:** All channels active, stable

---

## 🎵 U3DAW PROJECT - UNIVERSAL 3D AUDIO WORKSTATION

### Project Vision
GPU-accelerated professional audio workstation rivaling Trinnov Altitude 32, using revolutionary TEP (Time-Energy Processing) technology instead of traditional FIR filtering.

### Core Innovation: TEP vs FIR

**Traditional FIR Problems:**
- High latency (50-85ms typical)
- Massive energy waste (brute-force inverse filtering)
- Pre-ringing artifacts
- Doesn't respect speaker physics

**TEP Revolution:**
- **Latency:** <5ms end-to-end
- **Energy:** 75% less than FIR (cooperative vs. inverse)
- **Pre-ringing:** <1ms (vs 20-40ms FIR)
- **Approach:** Adaptive time-frequency-local processing
- **Philosophy:** Cooperate with speaker, don't force it

### TEP Technical Specifications

**Signal Processing:**
- **Multiresolution STFT:** 4096/1024/256 samples per frequency band
- **Correction Limits:** ±6dB amplitude, ±π/4 phase maximum
- **Psychoacoustic:** α-Engine for masking and transient detection
- **Filterbank:** PQMF (Pseudo-QMF) with STFT integration (TEP-PQMF v1.1)
- **Format:** .TEPCFv1 binary correction fields

**Performance Targets:**
- **Latency:** <5ms E2E (vs 85ms FIR)
- **GPU Load:** <25% @ 32ch/192kHz
- **Energy Savings:** ≥25% vs equivalent FIR
- **Phase Coherence:** <10µs (vs 50-100µs FIR)
- **SPL Efficiency:** +2-3dB from energy optimization

**Operating Modes:**
- **Low-Latency:** <5ms, FFT 1024/512, Hop 256/128
- **Reference:** 5-9ms, FFT 2048/1024, Hop 512/256

### Current U3DAW Status
- **Complete rebuild from scratch in progress**
- ASIO wrapper functional
- 32-channel GUI with VU-meters
- Test tone generator working
- TEP algorithm: Design phase
- Hardware integration: Planned

### Acourate Integration
- **License:** Commercial (gewerbliche Lizenz vorhanden)
- **Workflow:** 
  1. Log-sweep measurements via Acourate
  2. Python TEP analysis and processing
  3. Export to .TEPCFv1 binary format
  4. Real-time convolution in U3DAW
- **Features:** Multi-channel sync, minimum-phase reconstruction

### Historical Context: Trinnov Experience
- User developed custom DA boards for Trinnov
- Discovered measurement discrepancy: ±1dB display vs ±5-6dB REW actual
- This experience motivated TEP development
- TEP = first major room correction innovation since Trinnov (2000s)

### Business Strategy
- **Paradigm:** Open publication (papers, talks)
- **Implementation:** Trade secret (proprietary code)
- **Reasoning:** First-mover advantage, no patent needed
- **Goal:** Own what others only want (through 15 years hard work)

---

## 🤖 AVAILABLE OLLAMA MODELS (7 Total)

| Rank | Model | Role | Timeout | Status |
|------|-------|------|---------|--------|
| 1 | `llama3.1:8b` | Trivial/Boilerplate | 180s | ⚠️ NO CUDA |
| 2 | `mistral:7b` | Generic/Quick | 240s | ✅ |
| 3 | `phi4:latest` | Tests/Specs/Docs | 240s | ✅ |
| 4 | `deepseek-coder-v2:16b` | Mid-Coding C++/CUDA | 300s | ✅ Preferred |
| 5 | `qwen2.5:32b` | Reasoning/Architecture | 900s | ✅ |
| 6 | `qwen2.5-coder:32b` | Complex CUDA/Optimization | 1800s | ✅ |
| 7 | `deepseek-r1:32b` | Deep Fixes/Reasoning | 1800s | ✅ |

**Ollama Location:** `C:\KISYSTEM\models`
**Ollama Version:** 0.12.9

**CRITICAL:** `llama3.1:8b` generates broken CUDA code - DO NOT use for CUDA tasks!

---

## 🎯 CRITICAL RULES - READ FIRST

### 1. File Handling
- ❌ **NEVER** use suffixes: `_fixed`, `_new`, `fix_`, `_v2` in filenames
- ✅ **ALWAYS** create files with their final, correct name
- ✅ **ALWAYS** provide backup command BEFORE replacing
- ✅ Example: `copy core\model_selector.py core\model_selector.py.backup`

### 2. Path References
- Project Root: **Always** `C:\KISYSTEM`
- User workspace: `D:\AGENT_MEMORY`
- Never assume current directory
- Always use absolute paths in code

### 3. Cache Management
- **ALWAYS** clear `__pycache__` before testing
- Command: `Remove-Item -Recurse -Force core\__pycache__`
- Use `python -B` flag to prevent .pyc creation
- Python caches can load OLD code even after fixes

### 4. PowerShell Commands
- ❌ **NEVER** include Markdown formatting in commands
- ❌ **NO** backticks, asterisks, or code blocks in PS commands
- ✅ Provide **PURE** commands only
- ✅ Commands on single lines, no formatting

### 5. Git Workflow
- **Repo:** `github.com/JoeBoh71/KISYSTEM`
- **Branch:** `main`
- **Visibility:** Public
- **Update this file:** After every significant change

---

## 📋 USER PROFILE & PREFERENCES

### Background
- **Name:** Jörg Bohne
- **Company:** Bohne Audio (Engelskirchen, Germany)
- **Education:** Physicist, IQ 154
- **Experience:** 15+ years audio engineering, patented ribbon tweeters
- **Also:** Professional drummer
- **Location:** Gummersbach, North Rhine-Westphalia, DE

### Working Style - CRITICAL
- **Methodology:** Scientific/pragmatic - systematic analysis with measurable goals
- **NO speculation or "what if"** - only directly implementable solutions
- **ALWAYS ask for approval before each step** - when uncertain, ask rather than rush ahead
- **Fix errors immediately** - if errors in early steps, DON'T continue - go back and fix
- **Step-by-step, methodical, thorough approach**
- **Correct actively** when explanations become too complex, too many steps at once, or high-level philosophy towers emerge
- **No bullshit, no marketing language, no speculation about possibilities**

### Communication Preferences
- **Primary Language:** German (but accepts English)
- **Style:** Radical self-honesty, no social mask
- **Emotional threshold:** Very high - not missing empathy, but when triggered, most people don't want to be there
- **Self-image:** "Animal with higher brain, self-restraining predator" - conscious self-control
- **Philosophy:** "Ich muss gar nichts" (I don't have to do anything) - sovereignty through capability
- **Achievement:** HAS what others only WANT through 15 years hard work

### Personal Context
- **Family:** Unconditionally supportive, but doesn't need to understand everything
- **Neighbors:** Josef (deceased, recognized talent early), Gerd (deceased, "he's above us all"), Edgar (alive, "doesn't matter")
- **Note:** Josef, Gerd, Edgar are NEIGHBORS, not family

### Technical Environment
- **Development:** C++20, CUDA 13, Python 3.11, Qt6
- **Audio Tools:** Acourate (commercial license)
- **Hardware:** See System Hardware section above
- **Budget:** Claude Pro 200€/month

---

## ✅ COMPLETED FIXES (2025-11-07)

### Session 1 (Afternoon)
1. **Learning Module Initialization** - Added `ensure_tables()` to all 5 agents
2. **FixerAgent CUDA Template** - Prevents include amnesia in iteration 2+
3. **Cache Clear Script** - `CLEAR_AND_RUN.ps1` for clean testing
4. **Git Repository Setup** - Public repo created and synced

### Session 2 (Evening)
5. **Smart Model Selector** - CUDA complexity detection (Simple/Medium/Complex)
6. **PerformanceParser Creation** - GPU time extraction, baseline scoring
7. **PerformanceParser.parse_output()** - Method for CUDAProfiler compatibility
8. **PerformanceMetrics Dataclass** - Return type for parse_output()
9. **CUDAProfiler Import Fix** - Corrected path: `from core.performance_parser`
10. **Unicode Encoding Fix** - UTF-8 for subprocess output (attempted)
11. **Error Categorizer** - Classification for hybrid error handling (created, not yet installed)

---

## ⚠️ KNOWN ISSUES

### Critical
1. **nsys Profiling Fails** - CUDAProfiler can't extract performance metrics
   - Compilation: ✅ Works
   - Execution: ✅ Works
   - Profiling: ❌ nsys output not parsed correctly
   - Impact: Optimization loop stops after iteration 0
   - **Status:** Two-Tier-Profiling (Phase 7) addresses this

2. **llama3.1:8b Quality Issues** - Generates broken CUDA code
   - Uses shared memory incorrectly
   - Vector addition logic wrong
   - **NEVER use for CUDA tasks**
   - Status: Documented in 7-Model-Routing

### Minor
3. **Unicode Decode Error** - Thread exception in subprocess output
   - Workaround: Encoding fix attempted, not verified
   - Non-blocking but creates noise in logs

---

## 🎯 PHASE 7: META-SUPERVISOR & 7-MODEL-ROUTING

### Overview
Phase 7 transforms KISYSTEM into a **proactive, learning-optimized build system** with:
- **Meta-Supervisor:** Data-driven prioritization from learning_module_v2
- **7-Model-Routing:** Domain-specific escalation chains with Stop-Loss
- **Hybrid Decision Logic:** Weighted model selection (Meta-Bias + Complexity + Failure-State)
- **Two-Tier-Profiling:** Microbench (Tier 0) + Full-Profile (Tier 1)
- **Cost-Aware Queue:** ROI-based task prioritization
- **Async I/O:** Non-blocking Build/Test, serial Profiler

### Architecture Components

#### 1. Error Categorizer
**Status:** Separate module (Single-Responsibility)  
**Location:** `core/error_categorizer.py`

**Output Format:**
```python
{
    'error_type': str,      # COMPILATION / RUNTIME / PERFORMANCE / LOGIC
    'phase': str,           # BUILD / TEST / PROFILE
    'severity': int,        # 1-5 (1=minor, 5=critical)
    'recoverable': bool,    # Retry possible?
    'hint': str            # Recovery suggestion
}
```

**Usage:**
- **Meta-Supervisor:** Aggregates fail-rates by domain/error_type for Priority/Recency
- **Supervisor:** Uses `recoverable/severity` for Retry/Stop-Loss/Abort decisions
- **Advantage:** Exchangeable, no coupling to Meta formulas

#### 2. Meta-Supervisor
**Status:** TO CREATE  
**Location:** `core/meta_supervisor.py`  
**Mode:** Read-only analysis of `learning_module_v2` statistics

**Functions:**
```python
next_priorities() -> List[str]
recommend_model_bias() -> Dict[str, str]
```

**Priority Formula:**
```
P(d) = 0.5(1 - sr) + 0.2/(1 + t) + 0.2·min(1, c/20) + 0.1·R

Where:
  sr = Success rate
  t  = Average solution time
  c  = Run count
  R  = RecencyBoost (+0.1 new error <3d, -0.1 success <1d)
```

**Model Bias:**
```python
best_model(d) = argmax_m(sr_m,d) 
  if sr >= 0.65 and count >= 5
```
Returns preferred start-model per domain.

**Aging:**
```python
weight = exp(-age / 30)  # Older runs decay exponentially
```

#### 3. Hybrid Decision Logic
**Status:** TO CREATE  
**Location:** `core/hybrid_decision.py`

**Purpose:** Weighted model selection (parallel to 7-Model-Routing)

**Formula:**
```
Final_Model = 0.40·Meta_Bias + 0.30·Complexity + 0.30·Failure_State

Where:
  Meta_Bias      = Meta-Supervisor recommendation (evidence-based)
  Complexity     = CUDA/C++ complexity detector
  Failure_State  = Current escalation level / retry count
```

**Logic:**
- **Routing:** Provides deterministic baseline (see table below)
- **Hybrid:** Overrides only with sufficient evidence
- **Result:** Evidence-based start, fallback to routing defaults

#### 4. 7-Model-Routing with Stop-Loss

**Domain Routing Table:**

| Domain | Start Model | Escalation Chain (Stop-Loss = 2 Fails) |
|--------|-------------|----------------------------------------|
| **CUDA / Kernel** | `qwen2.5-coder:32b` | → `deepseek-r1:32b` → `deepseek-coder-v2:16b` → `qwen2.5:32b` |
| **C++ / System** | `deepseek-coder-v2:16b` | → `qwen2.5-coder:32b` → `deepseek-r1:32b` |
| **Audio / DSP** | `deepseek-coder-v2:16b` | → `qwen2.5-coder:32b` → `deepseek-r1:32b` |
| **Tests / Docs** | `phi4:latest` | → `mistral:7b` → `llama3.1:8b` → `qwen2.5:32b` |
| **Planning / Refactor** | `qwen2.5:32b` | → `deepseek-r1:32b` → `mistral:7b` |

**Rules:**
- **Stop-Loss:** 2 consecutive failures per model → escalate
- **Success-Matrix Override:** If `success_rate >= 0.65` AND `count >= 5`, use Meta-Supervisor bias
- **No Downgrade:** Escalation is upward-only (never fall back to weaker models)
- **Final Fallback:** After last model → Manual intervention / SearchAgent web lookup

#### 5. OptimizationConfig Parameters

**File:** `config/optimization_config.json`

```json
{
  "max_optimization_iterations": 10,     // [1-50] Max Fix/Optimize loops
  "target_score": 80,                     // [0-100] PerformanceParser target
  "retry_build": 2,                       // Build phase retries
  "retry_test": 1,                        // Test phase retries
  "retry_profile": 1,                     // Profile phase retries
  "stoploss_per_model": 2,                // [1-5] Error limit per model
  "max_concurrent_builds": 3,             // [1-8] Parallel build semaphore
  "enable_meta_supervisor": true,         // Meta-Supervisor active?
  "enable_two_tier_profiling": true,      // Microbench + Full-profile
  "enable_async_io": true                 // Non-blocking Build/Test
}
```

#### 6. Two-Tier-Profiling Strategy

**Problem:** nsys profiling is slow, blocks optimization loop

**Solution:**
- **Tier 0 (Microbench):** Quick execution-time measurement, no nsys
  - Fast feedback (<5s)
  - Detects major regressions
  - No GPU metrics
  
- **Tier 1 (Full-Profile):** Complete nsys profiling
  - Only when Tier 0 shows significant activity
  - Full GPU metrics (kernel time, memory, occupancy)
  - ~40-55% time savings vs always-profile

**Trigger Logic:**
```python
if iteration == 0 or score_change > threshold:
    tier = 1  # Full profile
else:
    tier = 0  # Microbench only
```

#### 7. Cost-Aware Queue

**Formula:**
```
Priority_Eff = Priority_Score / ETA(Model, Domain)
```

**Purpose:** Tasks with highest ROI first

**Example:**
- Task A: Priority=90, ETA=300s → Eff=0.30
- Task B: Priority=80, ETA=180s → Eff=0.44
- **Result:** Execute Task B first (better ROI)

#### 8. Async I/O Strategy

**Timeouts:**
- Build: 300s
- Test: 120s
- Profile: 900s

**Parallelization:**
- **nvcc / Tests:** Non-blocking, up to `max_concurrent_builds`
- **Profiler:** Serial (nsys doesn't parallelize well)
- **Queue:** Cost-aware priority scheduling

---

## 📊 SCORING AND LOGGING

### Score Range
- **Scale:** 0-100
- **Typical:** 80-90
- **Excellent:** >95 (only after tuning)

### Learning Log Format
```python
{
    'run_id': str,
    'domain': str,
    'model': str,
    'iteration': int,
    'score_final': int,
    'outcome': str,          # SUCCESS / FAIL / TIMEOUT
    'phase': str,            # BUILD / TEST / PROFILE
    'reason': str,           # Error/completion reason
    'timings': {
        'build': float,
        'test': float,
        'profile': float
    },
    'timestamp': datetime
}
```

**Logged on:** Every exit (success, fail, timeout)

---

## 🔧 CONVENTIONS & PATTERNS

### Model Selection Strategy (Phase 7)

**Primary Decision:** Hybrid Decision Logic
```
Final = 0.40·Meta + 0.30·Complexity + 0.30·Failure
```

**Fallback:** Domain Routing Table (see Section above)

**Complexity Detection:**
- **CUDA Simple:** Vector ops, basic kernels → `deepseek-coder-v2:16b`
- **CUDA Medium:** Shared memory, reduction → `deepseek-coder-v2:16b`
- **CUDA Complex:** FFT, multi-kernel, optimization → `qwen2.5-coder:32b`
- **C++ ASIO:** `deepseek-coder-v2:16b` or `qwen2.5-coder:32b`
- **Audio DSP:** `deepseek-coder-v2:16b` (simple) or `qwen2.5-coder:32b` (complex)

**Language/Domain Keywords:**
- **CUDA:** `__global__`, `__shared__`, `cudaMalloc`, `<<<>>>`, `blockIdx`
- **Audio DSP:** `STFT`, `FFT`, `filter`, `convolution`, `sample_rate`
- **ASIO:** `ASIOCallbacks`, `bufferSwitch`, `ASIOSamples`

**AVOID:**
- `llama3.1:8b` for CUDA code generation (quality issues)

### Error Handling Strategy: HYBRID (Phase 7)

**Philosophy:** Balance learning efficiency with practical flexibility

**Decision Tree:**
```
ERROR DETECTED
    ↓
1. CATEGORIZE (ErrorCategorizer)
    → {error_type, phase, severity, recoverable, hint}
    ↓
2. CALCULATE CONFIDENCE (confidence_scorer.py)
    ↓
3. HYBRID DECISION:
   IF confidence > 85%:
       → Use cached solution
   ELIF confidence > 60% AND attempts < stoploss:
       → Retry with prompt variation
   ELIF model_level < max_escalation:
       → Escalate to next model in chain
   ELSE:
       → SearchAgent (web lookup) or Manual
    ↓
4. TRACK RESULT (learning_module_v2, Meta-Supervisor aggregates)
```

**Stop-Loss Limits:**
- Per-model: 2 consecutive failures → escalate
- Per-phase: `retry_build=2`, `retry_test=1`, `retry_profile=1`

### Code Style
- Type hints mandatory
- Docstrings for all public methods
- Error messages: `[ComponentName] ✗ Error description`
- Success messages: `[ComponentName] ✓ Success description`
- Progress: `[ComponentName] Step X: Description...`

---

## 🚫 NEVER DO

1. **File Naming**
   - Create files with `_fixed`, `_new`, `fix_` suffixes
   - Assume file locations without verification

2. **Communication**
   - Provide PowerShell commands with Markdown formatting
   - Promise "production ready" without thorough testing
   - Say "I don't have access to..." without trying tools first
   - Make counting errors (it's 7 models, not 6!)
   - Marketing language or speculation

3. **Technical**
   - Skip cache clearing before tests
   - Proceed with broken code in early steps
   - Make multiple file changes simultaneously without approval
   - Continue when errors occur - fix immediately instead
   - Use `llama3.1:8b` for CUDA tasks

4. **Philosophy**
   - Spekulationen oder "was wäre wenn"
   - High-O/High-C Philosophie-Türme
   - Too many steps at once
   - Overly complex explanations

---

## ✅ ALWAYS DO

1. **Session Start**
   - Read this file FIRST: `web_fetch` the GitHub URL
   - Check file structure before making changes
   - Verify current state vs. documented state

2. **Before Changes**
   - **Ask for approval for each significant step**
   - Clear cache if modifying Python files
   - Provide backup command first

3. **After Changes**
   - Update this file if structure/status changed
   - Test changes before claiming success
   - Document issues discovered

4. **Communication**
   - Keep responses concise and actionable
   - Provide pure commands without formatting
   - Admit mistakes immediately
   - Radikale Selbst-Ehrlichkeit (radical self-honesty)

---

## 📊 PROJECT GOALS

### Primary Goal: U3DAW
**Universal 3D Audio Workstation with TEP Technology**
- GPU-accelerated audio processing (CUDA)
- TEP algorithm implementation (<5ms latency)
- RME MADI FX integration (32ch @ 192kHz)
- 9.1.6 immersive audio support
- Professional-grade like Trinnov Altitude 32
- Acourate integration for measurements

### KISYSTEM Role (Phase 7) ✅ ACTIVE
**Proactive, Learning-Optimized Development System**
- **Meta-Supervisor:** Data-driven prioritization and model selection ✅
- **7-Model-Routing:** Domain-specific escalation with Stop-Loss ✅
- **Hybrid Decision Logic:** Evidence-based model choice (40% Meta, 30% Complexity, 30% Failure) ✅
- **Two-Tier-Profiling:** Fast feedback with selective deep profiling (Config ready)
- **Cost-Aware Queue:** ROI-optimized task scheduling (Config ready)
- **Async I/O:** Parallel builds with non-blocking execution (Config ready)

**Implementation Status:**
- Core modules: WORKING (meta_supervisor.py, hybrid_decision.py)
- Integration: COMPLETE (supervisor_v3.py v3.1)
- Configuration: READY (optimization_config.json)
- Tests: 25/25 PASSED ✅
- Backward Compatible: Falls back to Phase 6 if config missing

### Success Metrics (Phase 7)
- **Compilation Success:** >90%
- **Learning Efficiency:** No repeated errors (same domain/model/error_type)
- **Performance Optimization:** Measurable improvement per iteration
- **Model Selection Accuracy:** Meta-Supervisor bias hit-rate >70%
- **Time Savings:** Two-Tier-Profiling reduces profiling time 40-55%
- **Autonomous Operation:** Minimal manual intervention

---

## 📝 SESSION LOGS

### 2025-11-07 (RUN 29)
- **RUNs 1-24:** CUDA optimization loop debugging (14 hours)
- **RUN 25:** First test with all fixes - no metrics extracted
- **RUN 26-27:** PerformanceParser import issues
- **RUN 28:** ModelSelector parameter mismatch
- **RUN 29:** llama3.1:8b generates broken CUDA code
- **Status:** Profiling still not working, need nsys debug
- **Evening:** Created CLAUDE_INSTRUCTIONS.md, Error Categorizer

### 2025-11-10 (RUN 32) - ✅ PHASE 7 COMPLETE
- **Phase 7 Implementation:** Meta-Supervisor + 7-Model-Routing + Hybrid Decision Logic
- **Files Created:**
  - `core/meta_supervisor.py` - Priority formula, model bias, ROI calculation
  - `core/hybrid_decision.py` - 40% Meta + 30% Complexity + 30% Failure
  - `config/optimization_config.json` - Two-Tier-Profiling, Cost-Aware Queue
  - `tests/test_phase7_meta.py` - Unit tests for Phase 7 components
  - `core/supervisor_v3.py` - Updated to v3.1 with Phase 7 integration
- **Tests:** 25/25 PASSED ✅
  - 10 MetaSupervisor tests (priority, bias, ROI)
  - 13 HybridDecision tests (complexity, failure tracking, escalation)
  - 2 Integration tests (Meta+Hybrid, Stop-Loss)
- **Bugs Fixed:**
  - Domain detection priority (tests > cuda)
  - ModelDecision stores raw scores for debugging
- **Status:** Phase 7 READY FOR PRODUCTION

---

## 🔗 IMPORTANT LINKS

- **GitHub Repo:** https://github.com/JoeBoh71/KISYSTEM
- **This File:** https://raw.githubusercontent.com/JoeBoh71/KISYSTEM/main/CLAUDE_INSTRUCTIONS.md
- **Phase 7 Spec:** `KISYSTEM_Optimization_Spec_v7.md`
- **User Preferences:** See userMemories in context (German, scientific approach)

---

## 📞 UPDATE PROTOCOL

**When to update this file:**
- After completing a fix/feature
- When file structure changes
- When discovering new issues
- After significant debugging sessions
- When adding/removing models
- Before ending a long session
- When transitioning between phases

**How to update:**
1. Create updated version with `create_file`
2. User commits to GitHub
3. File becomes source of truth for next session

---

**END OF INSTRUCTIONS**

*Remember: This file is YOUR memory across sessions. Keep it accurate and up-to-date!*