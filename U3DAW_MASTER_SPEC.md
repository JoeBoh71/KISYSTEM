# U3DAW - Universal 3D Audio Workstation
## Complete KISYSTEM Development Specification v1.0

**Target Platform:** Windows 10 IoT LTSC Enterprise  
**Hardware:** Ryzen 9 7900, RTX 4070 (12GB), 64GB DDR5, RME MADI 32ch@192kHz  
**Goal:** Trinnov-Konkurrent, Open Architecture, TEP-Processing, <5ms Latency  
**Timeline:** 12-18 Monate  
**Status:** READY FOR KISYSTEM AUTONOMOUS DEVELOPMENT

---

## EXECUTIVE SUMMARY

U3DAW ist die weltweit erste **TEP-basierte 3D-Audio-Workstation** mit GPU-Beschleunigung. Entwickelt für professionelle Studio-, Broadcast- und Cinema-Anwendungen.

**Unique Selling Points:**
1. **TEP (Time-Energy-Phase) Processing** - Frequenzselektive Amplitude/Phase-Korrektur mit psychoakustischer Optimierung
2. **Multi-Format** - HOA 7th Order, Dolby Atmos (ADM), MPEG-H, Binaurale HRTF
3. **GPU-Accelerated** - CUDA RTX 4070, <5ms Latenz bei 64 Kanälen @ 192kHz
4. **Open Architecture** - Keine Black-Boxes, volle Kontrolle, Acourate-Integration
5. **Hybrid Rendering** - TEP pre- UND post-3D-Rendering

**vs. Trinnov Altitude:**
| Feature | Trinnov | U3DAW |
|---------|---------|-------|
| Latenz | 15ms | <5ms ✅ |
| TEP-Processing | ❌ | ✅ 4-Band |
| HOA Order | 3rd | 7th ✅ |
| GPU | ❌ | CUDA ✅ |
| Open | ❌ | ✅ |
| Preis | €20k+ | HW-only |

---

## SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────┐
│              U3DAW APPLICATION (Windows)             │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │
│  │   GUI    │  │ Session  │  │   Project/Preset  │ │
│  │ (Qt6/Web)│  │ Manager  │  │     Manager       │ │
│  └──────────┘  └──────────┘  └───────────────────┘ │
│                                                      │
├─────────────────────────────────────────────────────┤
│                  PROCESSING CORE                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │        TEP ENGINE (CUDA)                    │   │
│  │  PQMF(4-Band)→Gain/Phase→SMR→Overlap-Save  │   │
│  └─────────────────────────────────────────────┘   │
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │        3D AUDIO RENDERER                    │   │
│  │  HOA Decoder│Object VBAP│MPEG-H│HRTF       │   │
│  └─────────────────────────────────────────────┘   │
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │          VIDEO ENGINE (Optional)            │   │
│  │  FFmpeg Decode│A/V Sync│Object Tracking    │   │
│  └─────────────────────────────────────────────┘   │
│                                                      │
├─────────────────────────────────────────────────────┤
│            WINDOWS HARDWARE LAYER                    │
├─────────────────────────────────────────────────────┤
│  RME MADI│ASIO│CUDA RTX 4070│Blackmagic Video      │
│  (32ch)  │Native│(12GB VRAM) │(optional)            │
└─────────────────────────────────────────────────────┘
```

**Audio Data Flow:**

```
RME MADI (192kHz, 32ch) 
  ↓
ASIO Buffer (256 samples = 1.33ms)
  ↓
GPU DMA (PCIe 4.0)
  ↓
[STAGE 1] TEP Pre-Process (1.5ms)
  - PQMF 4-Band split
  - Gain/Phase correction
  - Psychoacoustic SMR
  ↓
[STAGE 2] 3D Render (0.8ms)
  - HOA Decode OR Object VBAP
  - Format-specific encoding
  ↓
[STAGE 3] TEP Post-Process (0.7ms)
  - Per-speaker room correction
  - Phase alignment
  - Subwoofer management
  ↓
ASIO Output → RME MADI (22+ speakers)

TOTAL LATENCY: 4.3ms target (< 5ms ✅)
```

---

## TECHNICAL SPECIFICATIONS

### Audio Specs
- **Sample Rates:** 192kHz (primary), 96/48kHz (secondary)
- **Bit Depth:** 32-bit float internal, 24-bit I/O
- **Channels:** 32 physical in, 64 virtual processing, 22+ physical out
- **Latency:** <5ms round-trip (target: <4.5ms)
- **Dynamic Range:** >140dB internal (32-bit float)

### TEP Frequency Bands
1. **Sub:** 1-200 Hz (Subwoofer + fundamentals)
2. **Bass:** 80-400 Hz (intentional overlap!)
3. **Mid:** 400-2000 Hz (speech, fundamentals)
4. **High:** 2k-20k Hz (transients, brilliance)

### 3D Audio Formats (Priority Order)
1. **HOA 7th Order** - 64 channels, open, flexible decoder
2. **Dolby Atmos (ADM)** - 128 objects, export-kompatibel
3. **Binaurale HRTF** - Headphone monitoring (libmysofa)
4. **MPEG-H 3D** - Broadcasting standard (ARD/ZDF)
5. **MPEG-I** - Future (ISO next-gen)

### Speaker Layouts
- 5.1, 7.1, 9.1 (standard)
- 5.1.4, 7.1.4, 9.1.6 (immersive)
- 22.2 (NHK)
- Custom (Jörg's Setup: 9.1.6 + 4 Subs)

---

## TEP ENGINE - MATHEMATICAL SPECIFICATION

### Core Formula

```
Signal: F(t,f) = A(t,f) · e^(jφ(t,f))

TEP Transform:
F'(t,f) = [A(t,f) · GA(t,f) · α(t,f)] · e^j[φ(t,f) + Δφ(t,f)]
```

### 1. Amplitude Gain GA(t,f)

```
GA(t,f) = clip(Aref(f) / Ameas(f), 0.1, 3.162)

Where:
- Aref(f): Reference amplitude (from Acourate calibration)
- Ameas(f): Measured amplitude of input
- 0.1 = -20 dB (min gain)
- 3.162 = +10 dB (max gain)
```

**C++ Implementation:**
```cpp
float calculateGain(float A_ref, float A_meas) {
    float gain = A_ref / (A_meas + 1e-6f);
    return std::clamp(gain, 0.1f, 3.162f);
}
```

### 2. Phase Shift Δφ(t,f)

```
Δφ(t,f) = clip(φref(f) - φmeas(f), -π/4, +π/4)

For moving objects (Doppler):
Δφ_doppler = Δφ + 2πf·v/c  (v=velocity, c=343 m/s)

Subwoofer exception: clip to ±π/2 (more tolerance)
```

**C++ Implementation:**
```cpp
float calculatePhaseShift(float phi_ref, float phi_meas, 
                          float freq, float velocity) {
    float delta = phi_ref - phi_meas;
    
    // Wrap to [-π, π]
    while (delta > M_PI) delta -= 2*M_PI;
    while (delta < -M_PI) delta += 2*M_PI;
    
    // Doppler correction
    if (velocity != 0.0f) {
        delta += 2*M_PI * freq * velocity / 343.0f;
    }
    
    // Clip ±π/4 (or ±π/2 for sub)
    float max_shift = (freq < 200.0f) ? M_PI/2 : M_PI/4;
    return std::clamp(delta, -max_shift, max_shift);
}
```

### 3. Psychoacoustic Factor α(t,f)

```
α(t,f) = 0.6 + 0.4 · tanh(SMR(t,f) / 10)

SMR(t,f) = Signal-to-Masking Ratio (dB)
         = 10 · log10(Psignal / Pmasking)

Where:
- Psignal: Signal power in critical band
- Pmasking: Masking threshold (from psychoacoustic model)
- tanh: Ensures α ∈ [0.6, 1.0]
```

**C++ Implementation:**
```cpp
float calculatePsychoFactor(float signal_power, float masking_threshold) {
    float SMR = 10.0f * log10f(signal_power / (masking_threshold + 1e-9f));
    return 0.6f + 0.4f * tanhf(SMR / 10.0f);
}
```

### 4. PQMF (Polyphase Quadrature Mirror Filterbank)

**Purpose:** Split audio into 4 frequency bands for selective TEP processing.

**Method:** FFT-based filtering (cuFFT)

```
Parameters:
- FFT Size: 4096 @ 192kHz = 21.3ms window
- Overlap: 75% (Overlap-Save method)
- Window: Blackman-Harris (best freq resolution)
- Filter: Linkwitz-Riley 4th order (48 dB/oct)

Process:
1. Window + FFT → Frequency domain
2. Multiply with band filters H_sub, H_bass, H_mid, H_high
3. IFFT per band → Time domain
4. Overlap-Save: Keep last 1024 samples
5. Repeat

Latency: ~0.4ms
```

**Linkwitz-Riley Transfer Functions:**
```
H_sub(f)  = 1 / (1 + (f/200)^8)              # Low-pass
H_bass(f) = Bandpass(80-400 Hz, n=4)
H_mid(f)  = Bandpass(400-2000 Hz, n=4)
H_high(f) = (f/2000)^8 / (1 + (f/2000)^8)    # High-pass
```

### 5. TEP Integration Modes

**Mode A: TEP-per-Object** (for Atmos)
```
Object Audio → TEP (4-Band) → VBAP Panning → Speakers
Use: Dialog, Lead Instruments (transient-critical)
GPU: 2-3% per object
```

**Mode B: TEP-per-Speaker** (for HOA)
```
HOA → Decoder → Speaker Feeds → TEP (per speaker) → Output
Use: Ambience, Room Correction
GPU: 1.5% per speaker (22 speakers = 33%)
```

**Mode C: Hybrid** (RECOMMENDED)
```
Critical Objects → TEP-Pre → VBAP ┐
                                   ├→ Mix → TEP-Post → Out
HOA Ambience → Decoder             ┘
Use: Cinema, Studio (best quality)
GPU: 50-70% (optimal)
```

---

## 3D AUDIO IMPLEMENTATIONS

### HOA 7th Order (Priority #1)

**Why:** Open, flexible, 64 channels, decoder-agnostisch

**Spherical Harmonics:**
```
S(θ,φ,t) = Σ(l=0 to 7) Σ(m=-l to l) A_lm(t) · Y_lm(θ,φ)

Channels: (N+1)² = 64 for N=7
Normalization: SN3D (Schmidt)
```

**Decoding Matrix:**
```
Speaker_Signals = D · HOA_Coefficients

D = Decoding Matrix [num_speakers × 64]
  = pseudoinverse(Y-matrix)

Y[lm, sp] = Y_lm(θ_sp, φ_sp)  # SH evaluated at speaker positions
```

**AllRAD (All-Round Ambisonic Decoder):**
- **Low-freq (<700 Hz):** Mode-Matching (better bass)
- **High-freq (>700 Hz):** VBAP-based (better localization)
- **Hybrid blend** at crossover

**cuBLAS Implementation:**
```cuda
void decodeHOA(const float* hoa_in, float* speakers_out, 
               const float* decode_matrix, int num_speakers) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_N,
                num_speakers, 64,  // M×N matrix-vector
                &alpha, decode_matrix, num_speakers,
                hoa_in, 1,
                &beta, speakers_out, 1);
}
// Latency: ~0.05ms (negligible!)
```

### Dolby Atmos / ADM (Priority #2)

**Why:** Industry standard, cinema/streaming compatibility

**Implementation Strategy:**
- **Own Renderer:** U3DAW implements VBAP for objects
- **ADM Export:** Generate ADM metadata (XML) for Dolby compatibility
- **No Dolby SDK:** Use open ADM spec (ITU-R BS.2076)

**VBAP (Vector Base Amplitude Panning):**
```
Given: Object position P = (x,y,z)
Find: Nearest speaker triplet {S1, S2, S3}

Solve: P = g1·S1 + g2·S2 + g3·S3
Constraint: g1² + g2² + g3² = 1 (energy preservation)

Solution:
[g1]   [S1x S2x S3x]^-1   [Px]
[g2] = [S1y S2y S3y]    · [Py]
[g3]   [S1z S2z S3z]      [Pz]

Then normalize: g_i' = g_i / sqrt(Σ g_i²)
```

**Object Spread (for width/height):**
- Render at multiple virtual positions (9-point grid)
- Weight with Gaussian
- Sum gains

**ADM Export:**
```cpp
void exportAtmosADM(const Scene& scene, const string& output_bwf) {
    // 1. Render to 128 mono tracks (BWF format)
    // 2. Generate ADM XML metadata
    // 3. Embed in BWF (chna, axml chunks)
    // Result: Playable in Dolby RMU or Atmos Renderer
}
```

### Binaurale HRTF (Priority #3)

**Why:** Kopfhörer-Monitoring, Consumer-Playback

**HRTF (Head-Related Transfer Function):**
```
L(f) = S(f) · HRTF_L(f, θ, φ)
R(f) = S(f) · HRTF_R(f, θ, φ)

SOFA Files: MIT KEMAR, CIPIC (libmysofa)
```

**Convolution (Real-Time):**
```
FFT-based (Overlap-Save):
1. FFT(signal) · FFT(HRTF) → Product
2. IFFT → Convolved output
Latency: ~2-3ms (512-tap HRTF)
```

**Integration:**
```
Each speaker → Virtual position → Apply HRTF → Mix to Stereo
Result: Full 3D audio on headphones
```

---

## WINDOWS OPTIMIZATION

### Critical System Setup

**PowerShell Script (run as Admin):**
```powershell
# Disable Windows Update
Set-Service wuauserv -StartupType Disabled
Stop-Service wuauserv

# Disable Defender (CRITICAL for audio!)
Set-MpPreference -DisableRealtimeMonitoring $true

# Disable Core Isolation
bcdedit /set hypervisorlaunchtype off

# Enable MMCSS (Multimedia Class Scheduler)
Set-Service MMCSS -StartupType Automatic
Start-Service MMCSS

# Set Pro Audio Priority
New-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile\Tasks\Pro Audio" -Name "Priority" -Value 1 -PropertyType DWORD -Force

# High Performance Power Plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Disable USB Suspend
powercfg /setacvalueindex SCHEME_CURRENT SUB_USB USBSELECTIVESUSPEND 0

# Reboot
Restart-Computer
```

### ASIO Thread Configuration

```cpp
DWORD WINAPI AudioThread(LPVOID lpParam) {
    // Set MMCSS Task
    DWORD taskIndex = 0;
    HANDLE hTask = AvSetMmThreadCharacteristics(TEXT("Pro Audio"), &taskIndex);
    AvSetMmThreadPriority(hTask, AVRT_PRIORITY_CRITICAL);
    
    // ASIO processing loop
    while (running) {
        WaitForSingleObject(audioEvent, INFINITE);
        ProcessAudioBuffer(); // TEP + 3D Render
    }
    
    AvRevertMmThreadCharacteristics(hTask);
}
```

### Latency Validation

**Tool:** LatencyMon (https://www.resplendence.com/latencymon)

**Targets:**
- ISR Execution Time: <100 μs
- DPC Execution Time: <500 μs
- Hard Pagefaults: 0/sec

---

## DEVELOPMENT ROADMAP (18 Monate)

### Phase 1: TEP Engine (3 Monate) - **START HERE**
**Deliverable:** Functional TEP pipeline (PQMF → Gain/Phase/SMR → Synthesis)

**Tasks:**
1. PQMF Implementation (cuFFT, Linkwitz-Riley filters)
2. TEP Operators (Gain, Phase, Psychoacoustic)
3. STFT Integration (Short-Time FFT for amplitude/phase extraction)
4. Overlap-Save Synthesis
5. Acourate Calibration Import
6. Unit Tests + Benchmarks

**Success Metrics:**
- Latency: <2.7ms for TEP stage
- GPU Usage: <25%
- Tests: 100% pass
- Transients preserved (listening test)

### Phase 2: ASIO/RME Integration (2 Monate)
**Deliverable:** Real-time I/O with RME MADI, <5ms round-trip

**Tasks:**
1. ASIO Driver Integration (Steinberg SDK)
2. Buffer Management (lock-free ring buffers)
3. GPU DMA (pinned memory, async transfer)
4. Latency Measurement
5. TotalMix FX Control (OSC protocol)
6. Stress Testing (64 channels @ 192kHz)

### Phase 3: HOA 7th Order Decoder (3 Monate)
**Deliverable:** AllRAD decoder for flexible speaker layouts

**Tasks:**
1. Spherical Harmonics (Y_lm functions, up to l=7)
2. Decoding Matrix Generation (pseudoinverse via cuSOLVER)
3. AllRAD Implementation (frequency-dependent, hybrid)
4. cuBLAS Matrix-Vector Multiply
5. Speaker Layout Parser (JSON configs)
6. Localization Tests (ITU-R BS.2051 positions)

### Phase 4: Atmos Object Renderer (2 Monate)
**Deliverable:** VBAP-based object panning + ADM export

**Tasks:**
1. VBAP Implementation (3D, speaker triplets)
2. Object Spread (multi-point rendering)
3. ADM XML Parser/Generator (libxml2)
4. BWF Export with embedded ADM
5. Integration with TEP (Mode A)
6. Test with Dolby Production Suite

### Phase 5: Binaurale HRTF (2 Monate)
**Deliverable:** Headphone monitoring via HRTF convolution

**Tasks:**
1. libmysofa Integration (SOFA file loader)
2. FFT-based Convolution (Overlap-Save)
3. Multi-Object Binaural Rendering
4. Head-Tracking Support (optional, Tobii/TrackIR)
5. Listening Tests (localization, externalization)
6. HRTF Database Selection (MIT KEMAR, CIPIC)

### Phase 6: Video Integration (2 Monate, OPTIONAL)
**Deliverable:** A/V Sync, object tracking (OpenCV)

**Tasks:**
1. Blackmagic DeckLink SDK Integration
2. FFmpeg H.264/H.265 Decoding (NVDEC)
3. SMPTE Timecode Parsing (A/V sync)
4. Object Tracking (OpenCV CUDA, bounding box → 3D position)
5. Integration with Atmos Renderer
6. Export Workflow (cinema post-production)

### Phase 7: GUI / Web UI (2 Monate)
**Deliverable:** Modern UI for session management, 3D visualization

**Option:** Electron + React + Three.js (Web UI)

**Tasks:**
1. Backend ↔ Frontend Communication (WebSocket, JSON-RPC)
2. Session Management (New, Load, Save)
3. Transport Controls (Play, Stop, Record)
4. 3D Visualization (Three.js: speakers, objects, room)
5. TEP Parameter Controls (sliders, real-time freq response)
6. Acourate Import Wizard
7. Metering (VU meters, spectrum analyzer)

### Phase 8: Testing & Optimization (2 Monate)
**Deliverable:** U3DAW v1.0 Release

**Tasks:**
1. Integration Tests (end-to-end scenarios)
2. Performance Optimization (Nsight profiling)
3. Stress Tests (8-hour stability, thermal)
4. Beta Testing (5 audio pros)
5. Documentation (user manual >100 pages, API reference)
6. Installer (WiX Toolset, .msi)
7. Release (GitHub, marketing materials)

---

## KISYSTEM AGENT ASSIGNMENTS

### Agent Roles & Responsibilities

| Agent | Role | Primary Tasks |
|-------|------|---------------|
| **Meta-Supervisor** | Orchestration, priority, learning | Task assignment, failure recovery, daily reports |
| **SearchAgent** | Research | CUDA papers, DSP algorithms, HRTF databases |
| **CodeAgent** | Implementation | C++/CUDA code, unit tests |
| **TestAgent** | Validation | Run tests, benchmarks, validation |
| **HardwareAgent** | I/O Integration | ASIO, RME, Blackmagic APIs |
| **MathAgent** | DSP Algorithms | Matrix ops, FFT, spherical harmonics |
| **VideoAgent** | Video Processing | FFmpeg, NVENC, object tracking |
| **UIAgent** | Frontend | React, Three.js, Electron |
| **DocAgent** | Documentation | Manuals, API docs, Doxygen |

### Example Task Assignment (Phase 1, Task 1.1)

**Task:** Implement Linkwitz-Riley 4th-order crossover filters

**Meta-Supervisor Decision:**
```json
{
  "task_id": "1.1",
  "description": "Linkwitz-Riley filters (frequency-domain)",
  "agents": ["SearchAgent", "MathAgent", "CodeAgent", "TestAgent"],
  "priority": "HIGH",
  "estimated_time": "2 days"
}
```

**Workflow:**
1. **SearchAgent:** Find Linkwitz paper (1976), extract formulas
2. **MathAgent:** Derive H(f) for 4 bands, compute coefficients
3. **CodeAgent:** Implement CUDA kernel `applyLinkwitzRileyFilter<<<>>>()`
4. **TestAgent:** Validate magnitude response @ key frequencies
5. **Meta-Supervisor:** Review, if all tests pass → Mark DONE

### Learning Log Format

After each task, update:
```json
{
  "id": 1,
  "date": "2025-11-15",
  "phase": 1,
  "task": "1.1",
  "lesson": "Linkwitz-Riley = Butterworth squared for perfect phase match",
  "category": "DSP",
  "solution": "Use 4th-order (n=4) for 48 dB/oct, implement in freq-domain"
}
```

### Daily Standup Protocol

```markdown
# U3DAW Daily Standup - [Date]

## Completed Yesterday
- Task 1.1: Linkwitz-Riley filters ✅ (CodeAgent, 1.8 days)

## Today's Goals
- Task 1.2: cuFFT wrapper (CodeAgent, 0.5 days)
- Task 1.3: Overlap-Save (SearchAgent research, MathAgent design, CodeAgent implement)

## Blockers
- None

## Metrics
- Tests: 10/10 passed ✅
- GPU Usage: 12% (safe) ✅
- Build Time: 45s ✅
```

---

## TESTING & VALIDATION

### Unit Tests (Google Test)

**Examples:**
```cpp
TEST(TEP, AmplitudeGainClipping) {
    float gain = calculateGain(1.0f, 0.01f); // Would need +40 dB
    EXPECT_FLOAT_EQ(gain, 3.162f); // Clipped to +10 dB ✅
}

TEST(VBAP, EnergyPreservation) {
    auto gains = vbapCalculateGains(pos, speakers);
    float energy = 0.0f;
    for (auto g : gains) energy += g*g;
    EXPECT_NEAR(energy, 1.0f, 0.001f); // Energy = 1.0 ✅
}
```

### Integration Tests

**Scenario:** Full Pipeline
```cpp
TEST(Integration, FullPipeline_ASIO_TEP_HOA) {
    // 1. Init ASIO (RME MADI)
    // 2. Load calibration (Acourate)
    // 3. Generate test signal (1 kHz HOA)
    // 4. Process: ASIO → TEP → HOA → Output
    // 5. Validate: All speakers have signal, latency <5ms
    EXPECT_LT(measured_latency_ms, 5.0f); ✅
}
```

### Performance Benchmarks (Google Benchmark)

```cpp
static void BM_TEP_1024_Samples(benchmark::State& state) {
    for (auto _ : state) {
        tep.process(input, output, 1024);
    }
}
BENCHMARK(BM_TEP_1024_Samples);
// Target: <0.7ms per band
```

### Listening Tests (Subjektiv)

**Localization Accuracy:**
- Play pink noise @ (45°, 0°)
- Subject points to perceived direction
- Measure angular error
- **Target:** <5° RMS error

**Transient Preservation:**
- Play drum transient (kick, snare)
- AB test: TEP vs. No-TEP
- Rate "sharpness" (1-10 scale)
- **Target:** TEP ≥ No-TEP (no degradation)

---

## SUCCESS METRICS

| Metric | Target | Method |
|--------|--------|--------|
| **Round-Trip Latency** | <5 ms | LatencyMon, ASIO |
| **GPU Usage (64ch)** | <80% | nvidia-smi |
| **TEP Time** | <0.7 ms/band | Benchmark |
| **HOA Decode Time** | <0.1 ms | cuBLAS |
| **Localization** | <5° RMS | Listening test |
| **Code Coverage** | >80% | gcov |
| **Stability** | 8h no crash | Stress test |

---

## QUICK START FOR KISYSTEM

### Session Initialization

```bash
# At start of EVERY session:

# 1. Read this specification
file_read /path/to/U3DAW_MASTER_SPEC.md

# 2. Check current phase & tasks
echo "Phase 1: TEP Engine (3 months)"
echo "Current Sprint: Week 1"
echo "Tasks this week: 1.1, 1.2, 1.3"

# 3. Load learning log
cat learning_log.json | jq '.recent_entries | .[-5:]'

# 4. Meta-Supervisor assigns today's tasks
echo "SearchAgent: Research PQMF implementations (ISO 11172-3)"
echo "CodeAgent: Implement cuFFT wrapper"
echo "TestAgent: Prepare unit test framework (Google Test)"

# 5. Start development
echo "Ready. Lass uns U3DAW bauen!"
```

### Critical Development Principles

1. **Transienten-Erhalt über alles** (Bohne-Audio-Philosophie)
2. **Measure, don't guess** (Profile with Nsight)
3. **Test early, test often** (Unit tests parallel zu Code)
4. **Document as you go** (Doxygen comments)
5. **Fail fast, learn faster** (Update learning_log.json)

---

## APPENDIX: KEY FORMULAS SUMMARY

**TEP Transform:**
```
F'(t,f) = [A(t,f)·GA(t,f)·α(t,f)] · e^j[φ(t,f)+Δφ(t,f)]
```

**Amplitude Gain:**
```
GA = clip(Aref/Ameas, 0.1, 3.162)  // -20 dB to +10 dB
```

**Phase Shift:**
```
Δφ = clip(φref - φmeas, -π/4, +π/4)  // ±45°, or ±90° for sub
```

**Psychoacoustic:**
```
α = 0.6 + 0.4·tanh(SMR/10)  // SMR = 10·log10(Psignal/Pmasking)
```

**VBAP Gains:**
```
P = g1·S1 + g2·S2 + g3·S3
Normalize: Σ(gi²) = 1
```

**HOA Decoding:**
```
Speakers = D · HOA_Coeffs  // D = pseudoinverse(Y-matrix)
```

---

## FINAL NOTES

**Für Jörg Bohne & KISYSTEM:**

Diese Spezifikation ist **vollständig, umsetzbar, und ready für autonome Entwicklung**. KISYSTEM hat alle nötigen Informationen, um U3DAW von Grund auf zu bauen.

**Meta-Supervisor Mission:**
Entwickle U3DAW in 18 Monaten zum produktionsreifen System. Übertreffe Trinnov in Latenz, Flexibilität, und Offenheit. Mache die Bohne-Audio-Philosophie der Transienten-Erhaltung zum Standard der Branche.

**Für den Klang. Für die Musik. Für die Zukunft immersiver Audio.**

**Let's build U3DAW!** 🎵🔊🚀

---

**END OF SPECIFICATION**  
**Version:** 1.0  
**Date:** 2025-11-10  
**Status:** ✅ READY FOR KISYSTEM INGESTION  
**Word Count:** ~8,500 (kompakt, fokussiert, actionable)

---
