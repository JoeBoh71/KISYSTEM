# CLAUDE INSTRUCTIONS - KISYSTEM

**Last Updated:** 2025-11-08 07:00 UTC  
**Session:** RUN 31 (V3.0 Production + All Fixes Applied)  
**Status:** ✅ Production Ready - All Systems Operational
---

## 🏗️ PROJECT STRUCTURE

**Root:** `C:\KISYSTEM\`

```
C:\KISYSTEM\
├── core/
│   ├── model_selector.py           [v2 - WORKING]
│   ├── error_handler.py            [v3 - NEW! - WORKING] ✨
│   ├── performance_parser.py       [v2 - WORKING]
│   ├── learning_module_v2.py       [v2 - WORKING]
│   ├── confidence_scorer.py        [v1 - WORKING]
│   ├── context_tracker.py          [v1 - WORKING]
│   ├── ollama_client.py            [v1 - WORKING]
│   ├── supervisor_v3.py            [v1 - WORKING]
│   ├── supervisor_v3_optimization.py [v1 - WORKING]
│   ├── workflow_engine.py          [v1 - WORKING]
│   └── code_extractor.py           [v1 - WORKING]
│
├── agents/
│   ├── builder_agent.py            [v2 - WORKING]
│   ├── fixer_agent_v3.py           [v3 - NEW! - WORKING] ✨
│   ├── fixer_agent.py              [v2 - DEPRECATED - use V3]
│   ├── tester_agent.py             [v2 - WORKING]
│   ├── search_agent_v2.py          [v2 - WORKING]
│   ├── review_agent_v2.py          [v1 - WORKING]
│   ├── docs_agent_v2.py            [v1 - WORKING]
│   ├── cuda_profiler_agent.py      [v2 - WORKING]
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
│   ├── test_hybrid_handler.py      [v3 - NEW! - 4/4 PASSED] ✨
│   ├── test_phase6_optimization.py
│   └── ...
│
└── docs/
    ├── README.md
    ├── QUICKSTART.md
    ├── KISYSTEM_COMPLETE.txt       [v3.0 - Complete Documentation]
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

## ✅ V3.0 FEATURES - HYBRID ERROR HANDLER (RUN 30)

### **Status: Production Ready - All Tests Passing (4/4)** ✨

**Major Components Added:**

### 1. **HybridErrorHandler** (`core/error_handler.py`, 603 lines)
- ✅ Intelligent error categorization (4 types)
- ✅ Confidence-based decision making
- ✅ Model escalation chains
- ✅ Smart search triggering
- ✅ Statistics tracking

**Error Categories:**
```python
COMPILATION  → max_retries=1  (Strict)
RUNTIME      → max_retries=2  (Medium)
PERFORMANCE  → max_retries=4  (Flexible)
LOGIC        → max_retries=3  (Medium-Flexible)
```

**Decision Logic:**
```python
confidence > 85%  → USE_CACHE (high confidence solution)
confidence 60-85% → RETRY_WITH_VARIATION
attempts < limit  → ESCALATE (bigger model)
last resort       → SEARCH (web lookup)
```

### 2. **FixerAgentV3** (`agents/fixer_agent_v3.py`, 671 lines)
- ✅ Integrated HybridErrorHandler
- ✅ Removed hardcoded retry logic
- ✅ Category-aware error handling
- ✅ Confidence-based caching
- ✅ Performance optimization method

### 3. **PerformanceParser** (`core/performance_parser.py`, 300 lines)
- ✅ Parse nvprof/nsys output
- ✅ Calculate performance score (0-100)
- ✅ Identify bottlenecks (memory/compute/occupancy)
- ✅ Generate optimization suggestions

### 4. **Comprehensive Tests** (`tests/test_hybrid_handler.py`, 364 lines)
- ✅ Error categorization tests (6 cases)
- ✅ Confidence decision tests (2 cases)
- ✅ Direct Fixer V3 test (1 integration test)
- ✅ Full Supervisor optimization loop (1 E2E test)
- ✅ **Result: 4/4 tests PASSED (100%)**

**Test Output:**
```
======================================================================
FINAL RESULTS:
======================================================================
✓ Error Categorization: PASSED
✓ Confidence Decisions: PASSED
✓ Direct Fixer V3: PASSED
✓ Supervisor with Iterations: PASSED
======================================================================
Total: 4/4 tests passed (100%)
======================================================================

✅ ALL TESTS PASSED - HYBRID HANDLER FULLY FUNCTIONAL!
```

---

## 🔮 FUTURE FEATURES - IN DISCUSSION

### **Hardware Protocol Discovery Agent** (Discussed 2025-11-08)

**Concept:** Auto-discover and generate communication code for audio hardware

**Use Case:** 
- M-32 AD/DA converters (192.168.10.2, 192.168.10.3)
- Midas M32 mixer
- Future hardware additions

**Approaches Under Consideration:**

#### **Option A: Manual Research → Agent Implementation** (Pragmatic, Fast)
- Research protocols manually (datasheets, Wireshark, forums)
- Create template agents per device type
- KISYSTEM generates device-specific code from templates
- **Pro:** Quick, reliable, works immediately
- **Con:** Manual initial work per device type

#### **Option B: Discovery Agent with Sniffer-Tools** (Advanced, Complex)
- ProtocolDiscoveryAgent with nmap, Wireshark integration
- Automatic protocol detection (HTTP/REST, OSC, MIDI, Binary)
- Pattern recognition for command structures
- **Pro:** Fully autonomous, future-proof
- **Con:** 2+ weeks development, requires ML/pattern recognition

#### **Option C: Hybrid Approach** (Best of Both)
- Start with Option A for immediate needs
- Develop Option B as separate long-term component
- **Pro:** Pragmatic now, advanced later
- **Con:** Two-phase implementation

**Status:** Concept phase - awaiting decision on approach

**Files:** No implementation yet - design discussion only

---

## ⚠️ KNOWN ISSUES

### **✅ ALL CRITICAL ISSUES RESOLVED!**

**Fixed in RUN 31 (2025-11-09):**
1. ✅ Supervisor profiler import - Path correction applied
2. ✅ V3.0 critical files committed to GitHub  
3. ✅ test_hybrid_handler.py paths fixed - 4/4 tests passing
4. ✅ Optimization loop functional - Hardware-in-loop active

**Previous Issues Resolved in V3.0:**
- ✅ Hardcoded retry logic → Hybrid Handler
- ✅ No error categorization → 4 categories implemented
- ✅ No caching → Confidence-based cache functional
- ✅ Fixed escalation → Smart chains working
- ✅ Late search → Priority-based triggering operational

**Current Status:** No known critical issues. System operational at 100%.

---

## 🔧 CONVENTIONS & PATTERNS

### Model Selection Strategy

**By Task Type:**
- Code Generation (simple): `llama3.1:8b` (30s) - ⚠️ NOT for CUDA
- Code Generation (medium): `deepseek-coder-v2:16b` (2min)
- Code Generation (complex): `qwen2.5-coder:32b` (7min)
- Debugging/Reasoning: `deepseek-r1:32b` (7min)
- Testing: `phi4` (2min)
- Documentation: `mistral:7b` (30s)
- General Reasoning: `qwen2.5:32b` (7min)

**By Language/Domain:**
- **CUDA Simple** (vector ops): `deepseek-coder-v2:16b`
- **CUDA Medium** (shared mem): `deepseek-coder-v2:16b`
- **CUDA Complex** (FFT, multi-kernel): `qwen2.5-coder:32b`
- **C++ ASIO**: `deepseek-coder-v2:16b` or `qwen2.5-coder:32b`
- **Audio DSP**: `deepseek-coder-v2:16b` (simple) or `qwen2.5-coder:32b` (complex)

**Model Escalation Chains (V3.0):**
```python
Builder:  mistral:7b → deepseek-coder-v2:16b → qwen2.5-coder:32b
Fixer:    deepseek-coder-v2:16b → qwen2.5-coder:32b → deepseek-r1:32b
Tester:   phi4:latest → deepseek-coder-v2:16b
```

**AVOID:**
- `llama3.1:8b` for CUDA code generation (quality issues)

### Error Handling Strategy: HYBRID (V3.0)

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
       → Use cached solution (high confidence)
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
   - Read this file FIRST: `web_fetch` the GitHub URL
   - Check KISYSTEM_COMPLETE.txt for comprehensive architecture
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

### 2025-11-07 - RUN 30 (V3.0 Complete)
- ✅ Hybrid Error Handler implemented (603 lines)
- ✅ FixerAgentV3 created (671 lines)
- ✅ PerformanceParser enhanced (300 lines)
- ✅ Comprehensive test suite (364 lines)
- ✅ 4/4 tests PASSED (100%)
- ✅ Committed to Git (7,000+ lines added)
- ✅ Status: Production Ready

### 2025-11-07 - RUNs 1-29
- CUDA optimization loop debugging (14 hours)
- ModelSelector fixes
- PerformanceParser import fixes
- Learning Module fixes
- Auto-include injection fixes

### 2025-11-08 - Future Feature Discussion
- Protocol Discovery Agent concept
- Three approaches evaluated (A/B/C)
- Awaiting decision on implementation strategy

---

## 🔗 IMPORTANT LINKS

- **GitHub Repo:** https://github.com/JoeBoh71/KISYSTEM
- **This File:** https://raw.githubusercontent.com/JoeBoh71/KISYSTEM/main/CLAUDE_INSTRUCTIONS.md
- **Complete Documentation:** KISYSTEM_COMPLETE.txt (in repo)
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
- After major version bumps (V3.0 → V3.1)

**How to update:**
1. Create updated version with `create_file`
2. User commits to GitHub
3. File becomes source of truth for next session

---

**END OF INSTRUCTIONS**

*Remember: This file is YOUR memory across sessions. Keep it accurate and up-to-date!*

**Current Version:** V3.0 (RUN 30+)  
**Next Major Update:** When V3.1 features are implemented or Discovery Agent decision is made
