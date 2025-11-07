# CLAUDE INSTRUCTIONS - KISYSTEM

**Last Updated:** 2025-11-07 18:15 UTC
**Session:** RUN 29
**Status:** In Development - Hybrid Error Handling Phase

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
│   ├── supervisor_v3.py            [v1 - WORKING]
│   ├── supervisor_v3_optimization.py [v1 - WORKING]
│   ├── workflow_engine.py          [v1 - WORKING]
│   └── code_extractor.py           [v1 - WORKING]
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
│   └── kisystem_config.json
│
├── security/
│   └── security_module.py
│
├── tests/
│   ├── test_system.py
│   ├── test_phase6_optimization.py
│   └── ...
│
└── docs/
    ├── README.md
    ├── QUICKSTART.md
    └── ...
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

| Model | Size | Use Case | Avg Time | Status |
|-------|------|----------|----------|--------|
| `mistral:7b` | 4.4 GB | Documentation, Quick tasks | ~30s | ✅ |
| `llama3.1:8b` | 4.9 GB | Code Review, Simple tasks | ~30s | ⚠️ Quality issues CUDA |
| `phi4:latest` | 9.1 GB | Testing | ~2min | ✅ |
| `deepseek-coder-v2:16b` | 8.9 GB | Code Gen (medium/complex) | ~2min | ✅ Preferred |
| `qwen2.5:32b` | 19 GB | General reasoning | ~7min | ✅ |
| `qwen2.5-coder:32b` | 19 GB | Complex CUDA/C++ | ~7min | ✅ |
| `deepseek-r1:32b` | 19 GB | Debugging, Reasoning | ~7min | ✅ |

**Ollama Location:** `C:\KISYSTEM\models`
**Ollama Version:** 0.12.9

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

2. **llama3.1:8b Quality Issues** - Generates broken CUDA code
   - Uses shared memory incorrectly
   - Vector addition logic wrong
   - Should NOT be used for CUDA tasks

### Minor
3. **Unicode Decode Error** - Thread exception in subprocess output
   - Workaround: Encoding fix attempted, not verified
   - Non-blocking but creates noise in logs

---

## 🎯 ACTIVE WORK - NEXT STEPS

### Current Task: Professional Error Handling System

**Goal:** Replace "errors only happen once" with intelligent hybrid approach

**Components to Build:**

1. **Error Categorizer** ✅ Created (not yet installed)
   - Classify: COMPILATION / RUNTIME / PERFORMANCE / LOGIC
   - Each category has different retry limits

2. **Hybrid Decision Logic** (Next)
   - Confidence > 85%: Use cached solution
   - Confidence 60-85%: Retry with prompt variation
   - Attempts < limit: Escalate to bigger model
   - Last resort: SearchAgent web lookup

3. **Model Escalation Chains**
   - Builder: `mistral:7b` → `deepseek-coder-v2:16b` → `qwen2.5-coder:32b`
   - Fixer: `deepseek-coder-v2:16b` → `qwen2.5-coder:32b` → `deepseek-r1:32b`
   - Tester: `phi4` → `deepseek-coder-v2:16b`

4. **Category-Based Retry Limits**
   ```python
   LIMITS = {
       'COMPILATION': max_retries=1,     # Strict
       'RUNTIME': max_retries=2,          # Medium
       'PERFORMANCE': max_retries=4,      # Flexible
       'LOGIC': max_retries=3             # Medium-Flexible
   }
   ```

5. **Professional ModelSelector**
   - Agent-type routing
   - Language detection (CUDA/C++/Python/ASIO)
   - Domain knowledge (Audio DSP keywords)
   - All 7 models optimally utilized
   - Performance tracking

**Status:** Error Categorizer done, Hybrid Handler next

---

## 🔧 CONVENTIONS & PATTERNS

### Model Selection Strategy

**By Task Type:**
- Code Generation (simple): `mistral:7b` (30s)
- Code Generation (medium): `deepseek-coder-v2:16b` (2min)
- Code Generation (complex): `qwen2.5-coder:32b` (7min)
- Debugging/Reasoning: `deepseek-r1:32b` (7min)
- Testing: `phi4` (2min)
- Code Review: `llama3.1:8b` (30s)
- Documentation: `mistral:7b` (30s)
- General Reasoning: `qwen2.5:32b` (7min)

**By Language/Domain:**
- **CUDA Simple** (vector ops): `deepseek-coder-v2:16b`
- **CUDA Medium** (shared mem): `deepseek-coder-v2:16b`
- **CUDA Complex** (FFT, multi-kernel): `qwen2.5-coder:32b`
- **C++ ASIO**: `deepseek-coder-v2:16b` or `qwen2.5-coder:32b`
- **Audio DSP**: `deepseek-coder-v2:16b` (simple) or `qwen2.5-coder:32b` (complex)

**AVOID:**
- `llama3.1:8b` for CUDA code generation (quality issues)

### Error Handling Strategy: HYBRID

**Philosophy:** Balance between learning efficiency and practical flexibility

**Decision Tree:**
```
ERROR DETECTED
    ↓
1. CATEGORIZE (Compilation/Runtime/Performance/Logic)
    ↓
2. CALCULATE CONFIDENCE (using existing confidence_scorer.py)
    ↓
3. DECIDE:
   IF confidence > 85%:
       → Use cached solution
   ELIF confidence > 60% AND attempts < category_limit:
       → Retry with prompt variation
   ELIF attempts < escalation_limit:
       → Escalate to next model in chain
   ELSE:
       → SearchAgent (web search)
    ↓
4. TRACK RESULT (update learning DB, success rate)
```

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

4. **Philosophy**
   - Spekulationen oder "was wäre wenn"
   - High-O/High-C Philosophie-Türme
   - Too many steps at once
   - Overly complex explanations

---

## ✅ ALWAYS DO

1. **Session Start**
   - Read this file FIRST: `web_search` + `web_fetch` the GitHub URL
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

### KISYSTEM Role
**Autonomous Development Assistant**
- Generate CUDA kernels for audio DSP
- Test code with actual hardware (RTX 4070, RME MADI)
- Learn from errors (never repeat same mistake)
- Optimize performance automatically
- Eventually: Self-develop U3DAW components

### Success Metrics
- Compilation success rate > 90%
- No repeated errors (learning works)
- Performance improvements per iteration
- Hardware-in-the-loop testing functional
- Autonomous operation (minimal manual intervention)

---

## 📝 SESSION LOGS

### 2025-11-07
- **RUNs 1-24:** CUDA optimization loop debugging (14 hours)
- **RUN 25:** First test with all fixes - no metrics extracted
- **RUN 26-27:** PerformanceParser import issues
- **RUN 28:** ModelSelector parameter mismatch
- **RUN 29:** llama3.1:8b generates broken CUDA code
- **Status:** Profiling still not working, need nsys debug
- **Evening:** Created CLAUDE_INSTRUCTIONS.md, Error Categorizer

---

## 🔗 IMPORTANT LINKS

- **GitHub Repo:** https://github.com/JoeBoh71/KISYSTEM
- **This File:** https://raw.githubusercontent.com/JoeBoh71/KISYSTEM/main/CLAUDE_INSTRUCTIONS.md
- **User Preferences:** See userPreferences in context (German, scientific approach)

---

## 📞 UPDATE PROTOCOL

**When to update this file:**
- After completing a fix/feature
- When file structure changes
- When discovering new issues
- After significant debugging sessions
- When adding/removing models
- Before ending a long session

**How to update:**
1. Create updated version with `create_file`
2. User commits to GitHub
3. File becomes source of truth for next session

---

**END OF INSTRUCTIONS**

*Remember: This file is YOUR memory across sessions. Keep it accurate and up-to-date!*
