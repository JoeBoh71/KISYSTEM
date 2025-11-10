# KISYSTEM Phase 7 Guide
## Meta-Supervisor, 7-Model-Routing & Intelligent Optimization

**Version:** Phase 7 (RUN 32)  
**Status:** 🚀 Active Implementation  
**Author:** Jörg Bohne  
**Last Updated:** 2025-11-10

---

## 🎯 Phase 7 Übersicht

Phase 7 transformiert KISYSTEM von einem **reaktiven** zu einem **proaktiven, lernenden Build-System** mit datenbasierter Entscheidungsfindung.

### Hauptziele

1. **Meta-Supervisor** - Datengetriebene Priorisierung und Modellwahl
2. **7-Model-Routing** - Domänen-spezifische Eskalationsketten mit Stop-Loss
3. **Hybrid Decision Logic** - Evidenzbasierte Modellselektion (40% Meta, 30% Complexity, 30% Failure)
4. **Two-Tier-Profiling** - Intelligentes Profiling spart 40-55% Zeit
5. **Cost-Aware Queue** - ROI-optimierte Task-Priorisierung
6. **Async I/O** - Non-blocking Build/Test-Execution

### Warum Phase 7?

**Problem in Phase 6:**
- Statische Modellwahl (feste Regeln)
- Kein Learning-Feedback für Prioritäten
- Profiling immer vollständig (langsam)
- FIFO Task-Queue (suboptimal)
- Fehler-Handling zu starr

**Lösung in Phase 7:**
- Dynamische Modellwahl aus Learning-DB
- Meta-Supervisor berechnet Prioritäten
- Two-Tier-Profiling spart Zeit
- Cost-Aware Queue maximiert ROI
- Stop-Loss-Eskalation für robustes Error-Handling

---

## 1️⃣ Meta-Supervisor

### Konzept

Der Meta-Supervisor ist ein **read-only Analysemodul**, das die Learning-Database auswertet und daraus:
1. **Task-Prioritäten** berechnet (welche Probleme zuerst lösen?)
2. **Modell-Bias** empfiehlt (welches Modell für welche Domain?)

**WICHTIG:** Meta-Supervisor **schreibt nicht** in die DB - nur Analyse!

### Prioritätsberechnung

**Formel:**
```
P(d) = 0.5·(1 - sr) + 0.2/(1 + t) + 0.2·min(1, c/20) + 0.1·R

Wobei:
  d  = Domain (z.B. "CUDA/Kernel", "C++/System")
  sr = Success Rate (Erfolgsquote 0-1)
  t  = Average Solution Time (Sekunden)
  c  = Run Count (Anzahl Versuche)
  R  = Recency Boost (-0.1 bis +0.1)
```

**Faktoren erklärt:**

| Faktor | Gewicht | Bedeutung | Beispiel |
|--------|---------|-----------|----------|
| **1 - sr** | 50% | Hohe Fehlerrate → Hohe Priorität | sr=0.6 → (1-0.6)=0.4 |
| **1/(1+t)** | 20% | Langsame Lösungen → Höhere Priorität | t=100s → 1/101=0.01 |
| **c/20** | 20% | Viele Versuche → Höhere Priorität | c=15 → 15/20=0.75 |
| **R** | 10% | Recency Boost | Neu <3d: +0.1, Erfolg <1d: -0.1 |

**Recency Boost Details:**
```python
if last_error_age < 3 days:
    R = +0.1  # Neue Fehler haben Priorität
elif last_success_age < 1 day:
    R = -0.1  # Kürzlich gelöst → niedrigere Priorität
else:
    R = 0.0   # Neutral
```

**Beispiel-Berechnung:**

```python
# Domain: CUDA/Kernel
sr = 0.85    # 85% Erfolgsquote
t = 120      # Durchschnittlich 120s pro Lösung
c = 12       # 12 Runs bisher
R = +0.05    # Fehler vor 2 Tagen

P = 0.5·(1 - 0.85) + 0.2/(1 + 120) + 0.2·min(1, 12/20) + 0.1·0.05
  = 0.5·0.15 + 0.2/121 + 0.2·0.6 + 0.005
  = 0.075 + 0.00165 + 0.12 + 0.005
  = 0.20165

# Domain: Audio/DSP
sr = 0.94    # 94% Erfolgsquote
t = 80       # Durchschnittlich 80s
c = 25       # 25 Runs
R = -0.1     # Erfolg gestern

P = 0.5·(1 - 0.94) + 0.2/(1 + 80) + 0.2·min(1, 25/20) + 0.1·(-0.1)
  = 0.5·0.06 + 0.2/81 + 0.2·1.0 - 0.01
  = 0.03 + 0.00247 + 0.2 - 0.01
  = 0.22247

→ Audio/DSP hat höhere Priorität (0.222 > 0.202)
```

### Modell-Bias

**Formel:**
```
best_model(d) = argmax_m(sr_m,d)  wenn sr ≥ 0.65 ∧ count ≥ 5

Wobei:
  d     = Domain
  m     = Model
  sr_m,d = Success Rate von Model m in Domain d
```

**Bedingungen:**
- Success Rate ≥ 65% (ausreichend Evidenz)
- Count ≥ 5 (statistisch relevant)
- Nur wenn beide erfüllt → Bias wird verwendet

**Exponential Aging:**
```python
weight = exp(-age / 30)  # Ältere Runs zählen weniger

# Beispiel:
age = 0 days   → weight = 1.00  (100%)
age = 15 days  → weight = 0.61  (61%)
age = 30 days  → weight = 0.37  (37%)
age = 60 days  → weight = 0.14  (14%)
```

**Beispiel:**

```python
# Domain: CUDA/Kernel
# Runs in Learning-DB:

Model              | Runs | Success | Avg Age | Weighted SR
-------------------|------|---------|---------|-------------
qwen2.5-coder:32b  |  15  |  14     | 5 days  | 0.93·0.85 = 0.79
deepseek-r1:32b    |   8  |   7     | 10 days | 0.88·0.72 = 0.63
deepseek-coder:16b |   3  |   2     | 2 days  | 0.67·0.94 = 0.63

→ best_model("CUDA/Kernel") = qwen2.5-coder:32b (SR=0.79 > 0.65, count=15 ≥ 5)
```

### API

```python
from core.meta_supervisor import MetaSupervisor

meta = MetaSupervisor(learning_module)

# 1. Get Task Priorities
priorities = meta.next_priorities(top_n=5)
# Returns: [
#   ("Audio/DSP", 0.222),
#   ("CUDA/Kernel", 0.202),
#   ("C++/System", 0.165),
#   ...
# ]

# 2. Get Model Bias
bias = meta.recommend_model_bias()
# Returns: {
#   "CUDA/Kernel": "qwen2.5-coder:32b",
#   "C++/System": "deepseek-coder-v2:16b",
#   "Audio/DSP": "qwen2.5-coder:32b",
#   ...
# }

# 3. Explain Priority
explanation = meta.explain_priority("CUDA/Kernel")
# Returns: {
#   'priority_score': 0.202,
#   'factors': {
#     'failure_contribution': 0.075,  # (1-sr) component
#     'time_contribution': 0.00165,   # 1/(1+t) component
#     'count_contribution': 0.12,     # c/20 component
#     'recency_boost': 0.005          # R component
#   },
#   'success_rate': 0.85,
#   'avg_time': 120,
#   'run_count': 12
# }
```

### Integration im Workflow

```python
# Supervisor V3 verwendet Meta-Supervisor

async def handle_task(self, task):
    # 1. Check Priority
    priorities = self.meta_supervisor.next_priorities()
    if task.domain in [p[0] for p in priorities[:3]]:
        self.logger.info(f"🔥 High-priority domain: {task.domain}")
    
    # 2. Get Model Bias
    bias = self.meta_supervisor.recommend_model_bias()
    recommended_model = bias.get(task.domain)
    
    if recommended_model:
        self.logger.info(f"📊 Meta-Supervisor recommends: {recommended_model}")
        # Pass to Hybrid Decision Logic
        task.meta_bias = recommended_model
    
    # 3. Proceed with task
    return await self.execute_with_hybrid_decision(task)
```

---

## 2️⃣ 7-Model-Routing mit Stop-Loss

### Konzept

Jede **Domain** hat:
1. Ein **Start-Modell** (Baseline)
2. Eine **Eskalations-Kette** (aufsteigend komplexer/langsamer)
3. Einen **Stop-Loss** (2 Fehler → nächstes Modell)

**Philosophie:** Start schnell, eskaliere bei Bedarf, nie downgrade.

### Routing-Tabelle

| Domain | Start | Eskalation (Stop-Loss = 2 Fails) |
|--------|-------|----------------------------------|
| **CUDA / Kernel** | `qwen2.5-coder:32b` | → `deepseek-r1:32b` → `deepseek-coder-v2:16b` → `qwen2.5:32b` |
| **C++ / System** | `deepseek-coder-v2:16b` | → `qwen2.5-coder:32b` → `deepseek-r1:32b` |
| **Audio / DSP** | `deepseek-coder-v2:16b` | → `qwen2.5-coder:32b` → `deepseek-r1:32b` |
| **Tests / Docs** | `phi4:latest` | → `mistral:7b` → `llama3.1:8b` → `qwen2.5:32b` |
| **Planning / Refactor** | `qwen2.5:32b` | → `deepseek-r1:32b` → `mistral:7b` |

### Eskalations-Logik

```python
class DomainRouter:
    ROUTES = {
        "CUDA/Kernel": [
            "qwen2.5-coder:32b",
            "deepseek-r1:32b",
            "deepseek-coder-v2:16b",
            "qwen2.5:32b"
        ],
        "C++/System": [
            "deepseek-coder-v2:16b",
            "qwen2.5-coder:32b",
            "deepseek-r1:32b"
        ],
        # ...
    }
    
    STOP_LOSS = 2  # Fehler pro Modell
    
    def get_next_model(self, domain, current_model, fail_count):
        route = self.ROUTES.get(domain, ["mistral:7b"])
        
        try:
            current_index = route.index(current_model)
        except ValueError:
            return route[0]  # Fallback zu Start
        
        # Stop-Loss erreicht?
        if fail_count >= self.STOP_LOSS:
            next_index = current_index + 1
            
            if next_index < len(route):
                return route[next_index]  # Eskalation
            else:
                return None  # Alle Modelle versucht → Manual/Search
        
        return current_model  # Weiter mit gleichem Modell
```

### Beispiel-Flow

```
Task: "Implementiere CUDA Kernel für Matrix-Multiplikation"
Domain: CUDA/Kernel

Attempt 1: qwen2.5-coder:32b
  → Build: ✓
  → Test: ✗ (Runtime Error)
  → fail_count = 1

Attempt 2: qwen2.5-coder:32b (same model, fail_count < 2)
  → Build: ✓
  → Test: ✗ (Runtime Error)
  → fail_count = 2 → STOP-LOSS!

Attempt 3: deepseek-r1:32b (eskaliert)
  → Build: ✓
  → Test: ✓
  → SUCCESS!

Learning:
  - qwen2.5-coder: 2 fails, no success
  - deepseek-r1: 1 success
  → Nächstes Mal: Meta-Supervisor könnte deepseek-r1 als Start empfehlen
```

### Success-Matrix Override

Wenn Meta-Supervisor **starke Evidenz** hat (sr ≥ 0.65, count ≥ 5), **überschreibt** er das Start-Modell:

```python
# Routing sagt: Start mit qwen2.5-coder
start = domain_router.get_start_model("CUDA/Kernel")
# → "qwen2.5-coder:32b"

# Aber Meta-Supervisor hat Evidenz:
bias = meta_supervisor.recommend_model_bias()
# → {"CUDA/Kernel": "deepseek-r1:32b"}  (SR=0.91, count=15)

# Hybrid Decision:
if bias["CUDA/Kernel"]:
    start = bias["CUDA/Kernel"]  # Override!
# → "deepseek-r1:32b"
```

### Wichtige Regeln

1. **Nie Downgrade** - Eskalation nur aufwärts (kleinere → größere Modelle)
2. **Stop-Loss = 2** - Nach 2 Fehlern immer eskalieren
3. **Keine Loops** - Jedes Modell max 1x in Kette
4. **Final Fallback** - Nach letztem Modell: SearchAgent oder Manual

---

## 3️⃣ Hybrid Decision Logic

### Konzept

Hybrid Decision Logic **kombiniert 3 Informationsquellen** für optimale Modellwahl:

```
Final_Model = 0.40·Meta_Bias + 0.30·Complexity + 0.30·Failure_State
```

### Faktoren

| Faktor | Gewicht | Source | Beispiel |
|--------|---------|--------|----------|
| **Meta-Bias** | 40% | Learning-DB (Meta-Supervisor) | "deepseek-r1 hat 91% success in CUDA" |
| **Complexity** | 30% | Code-Analyse (Keywords, Structure) | "Shared Memory + FFT → Complex" |
| **Failure-State** | 30% | Current Run (Retry-Count, Errors) | "2 fails → eskaliere" |

### Meta-Bias (40%)

```python
def calculate_meta_bias_score(domain):
    bias = meta_supervisor.recommend_model_bias()
    
    if domain in bias:
        # Starke Evidenz vorhanden
        recommended = bias[domain]
        confidence = get_model_confidence(domain, recommended)
        return confidence  # 0.7 - 1.0
    else:
        # Keine Evidenz → neutral
        return 0.5
```

### Complexity Detection (30%)

```python
def detect_complexity(code, language):
    score = 0.5  # Baseline: Medium
    
    if language == "CUDA":
        # Simple CUDA
        if "__global__" in code and "<<<" in code:
            score += 0.1
        
        # Medium CUDA
        if "__shared__" in code or "blockIdx" in code:
            score += 0.2
        
        # Complex CUDA
        if any(kw in code for kw in ["cufft", "cublas", "cudnn"]):
            score += 0.3
        
        if "__syncthreads" in code and "__shared__" in code:
            score += 0.2  # Shared memory + sync
        
    elif language == "C++":
        # Simple C++
        if "std::vector" in code and "#include" in code:
            score += 0.1
        
        # Medium C++
        if "template" in code or "class" in code:
            score += 0.2
        
        # Complex C++
        if "std::async" in code or "std::thread" in code:
            score += 0.3
    
    return min(1.0, score)
```

### Failure-State (30%)

```python
def calculate_failure_state_score(retry_count, error_type):
    base = 0.5
    
    # Mehr Retries → höherer Score (komplexer)
    retry_factor = min(0.3, retry_count * 0.1)
    
    # Error-Type
    if error_type == "COMPILATION":
        error_factor = 0.0  # Einfach
    elif error_type == "RUNTIME":
        error_factor = 0.1  # Medium
    elif error_type == "PERFORMANCE":
        error_factor = 0.2  # Komplex
    elif error_type == "LOGIC":
        error_factor = 0.15  # Medium-Komplex
    else:
        error_factor = 0.1
    
    return min(1.0, base + retry_factor + error_factor)
```

### Final Decision

```python
def select_model(task):
    # 1. Get Scores
    meta_score = calculate_meta_bias_score(task.domain)     # 0.0 - 1.0
    complexity_score = detect_complexity(task.code, task.language)  # 0.0 - 1.0
    failure_score = calculate_failure_state_score(task.retry, task.error)  # 0.0 - 1.0
    
    # 2. Weighted Combination
    final_score = (
        0.40 * meta_score +
        0.30 * complexity_score +
        0.30 * failure_score
    )
    
    # 3. Map Score to Model
    if final_score < 0.4:
        return "mistral:7b"  # Simple
    elif final_score < 0.6:
        return "deepseek-coder-v2:16b"  # Medium
    elif final_score < 0.8:
        return "qwen2.5-coder:32b"  # Complex
    else:
        return "deepseek-r1:32b"  # Very Complex
```

### Beispiel-Berechnung

```
Task: CUDA Kernel für FFT mit Shared Memory
Domain: CUDA/Kernel
Retry: 1
Error: RUNTIME

1. Meta-Bias:
   - Meta-Supervisor empfiehlt: qwen2.5-coder (SR=0.91, count=15)
   - Confidence: 0.91
   - Score: 0.91

2. Complexity:
   - Code enthält: __global__, __shared__, cufft, __syncthreads
   - Base: 0.5
   - __global__: +0.1 = 0.6
   - __shared__: +0.2 = 0.8
   - cufft: +0.3 = 1.1 → capped at 1.0
   - Score: 1.0

3. Failure-State:
   - Retry: 1 → retry_factor = 0.1
   - Error: RUNTIME → error_factor = 0.1
   - Base: 0.5
   - Score: 0.5 + 0.1 + 0.1 = 0.7

Final Score:
= 0.40·0.91 + 0.30·1.0 + 0.30·0.7
= 0.364 + 0.30 + 0.21
= 0.874

→ final_score = 0.874 > 0.8 → Model: deepseek-r1:32b
```

---

## 4️⃣ Two-Tier-Profiling

### Problem

**nsys** Profiling ist langsam (~30-60s pro Run), aber nicht immer nötig:
- Iteration 0: Ja, Baseline messen
- Iteration 1-N: Nur wenn signifikante Änderung

### Lösung: Two-Tier

**Tier 0 - Microbench (Fast):**
- Einfache Execution-Time-Messung
- Kein nsys overhead
- <5s Feedback
- Erkennt große Regressionen

**Tier 1 - Full Profile (Slow):**
- Complete nsys profiling
- GPU Metrics (Kernel Time, Memory, Occupancy)
- 30-60s pro Run
- Nur bei relevanter Aktivität

### Trigger-Logik

```python
def should_full_profile(iteration, score_change, last_profile_iter):
    # Immer bei Iteration 0 (Baseline)
    if iteration == 0:
        return True
    
    # Score-Änderung > Threshold?
    if abs(score_change) > 5:  # 5% change
        return True
    
    # Letztes Full-Profile zu lange her?
    if iteration - last_profile_iter > 5:
        return True
    
    # Sonst: Tier 0 reicht
    return False
```

### Beispiel-Flow

```
Iteration 0:
  → Tier 1 (Full Profile) - 45s
  → Score: 75
  → Kernel Time: 12.5ms

Iteration 1:
  → Tier 0 (Microbench) - 3s
  → Score: 78 (change: +3, < 5 threshold)
  → Execution Time: 11.8ms

Iteration 2:
  → Tier 0 (Microbench) - 3s
  → Score: 82 (change: +4, < 5 threshold)
  → Execution Time: 10.2ms

Iteration 3:
  → Tier 0 (Microbench) - 3s
  → Score: 88 (change: +6, > 5 threshold!) → TRIGGER
  → Tier 1 (Full Profile) - 48s
  → Score: 88
  → Kernel Time: 8.9ms, Occupancy: 87%

Total Time:
  - With always Tier 1: 4 × 45s = 180s
  - With Two-Tier: 45 + 3 + 3 + 48 = 99s
  → Savings: 45% (81s)
```

### Implementation

```python
class TwoTierProfiler:
    def __init__(self):
        self.last_full_profile_iter = -1
        self.last_score = 0
    
    async def profile(self, code_path, iteration):
        # Decide tier
        score_change = 0
        if iteration > 0:
            score_change = self.calculate_score_change()
        
        should_full = self.should_full_profile(
            iteration,
            score_change,
            self.last_full_profile_iter
        )
        
        if should_full:
            # Tier 1: Full Profile
            result = await self.full_profile(code_path)
            self.last_full_profile_iter = iteration
            return result
        else:
            # Tier 0: Microbench
            result = await self.microbench(code_path)
            return result
    
    async def microbench(self, code_path):
        """Quick execution-time measurement"""
        start = time.time()
        
        # Execute code
        process = await asyncio.create_subprocess_exec(
            code_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        elapsed = time.time() - start
        
        return {
            'tier': 0,
            'execution_time': elapsed,
            'score': self.estimate_score(elapsed),
            'stdout': stdout.decode(),
            'stderr': stderr.decode()
        }
    
    async def full_profile(self, code_path):
        """Complete nsys profiling"""
        # Run nsys
        cmd = f"nsys profile -o profile.nsys-rep {code_path}"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # Parse metrics
        metrics = self.parse_nsys_output(stdout.decode())
        
        return {
            'tier': 1,
            'kernel_time': metrics.get('kernel_time'),
            'memory_throughput': metrics.get('memory_throughput'),
            'occupancy': metrics.get('occupancy'),
            'score': self.calculate_score(metrics),
            'stdout': stdout.decode(),
            'stderr': stderr.decode()
        }
```

### Time Savings

| Scenario | Always Tier 1 | Two-Tier | Savings |
|----------|---------------|----------|---------|
| 5 iterations | 225s | 120s | 47% |
| 10 iterations | 450s | 200s | 56% |
| 20 iterations | 900s | 400s | 56% |

**Typical:** 40-55% Zeitersparnis

---

## 5️⃣ Cost-Aware Queue

### Konzept

Tasks werden nach **ROI** (Return on Investment) priorisiert:

```
Priority_Eff = Priority_Score / ETA(Model, Domain)
```

**Ziel:** Tasks mit höchstem Nutzen pro Zeiteinheit zuerst.

### Beispiel

```
Task Queue:

Task A: Audio DSP Optimization
  - Priority Score: 90 (aus Meta-Supervisor)
  - Model: qwen2.5-coder:32b
  - ETA: 300s (5 min)
  - Priority_Eff: 90 / 300 = 0.30

Task B: Simple Bug Fix
  - Priority Score: 80
  - Model: deepseek-coder-v2:16b
  - ETA: 180s (3 min)
  - Priority_Eff: 80 / 180 = 0.44

Task C: Complex CUDA Kernel
  - Priority Score: 95
  - Model: deepseek-r1:32b
  - ETA: 900s (15 min)
  - Priority_Eff: 95 / 900 = 0.11

Execution Order (by Priority_Eff):
1. Task B (0.44) - 3 min
2. Task A (0.30) - 5 min
3. Task C (0.11) - 15 min

Result:
- After 8 min: 2 tasks done (B, A)
- Traditional FIFO: After 8 min: Only 1 task done (A)
→ 2x throughput!
```

### Implementation

```python
import heapq
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    domain: str
    priority_score: float
    model: str
    eta: float
    
    @property
    def priority_eff(self):
        return self.priority_score / self.eta

class CostAwareQueue:
    def __init__(self):
        self.heap = []
    
    def push(self, task):
        # Negative for max-heap behavior
        heapq.heappush(self.heap, (-task.priority_eff, task))
    
    def pop(self):
        if self.heap:
            _, task = heapq.heappop(self.heap)
            return task
        return None
    
    def peek(self):
        if self.heap:
            return self.heap[0][1]
        return None
```

### ETA Estimation

```python
class ETAEstimator:
    # Historical data from Learning-DB
    MODEL_TIMINGS = {
        "mistral:7b": 30,
        "deepseek-coder-v2:16b": 120,
        "qwen2.5-coder:32b": 300,
        "deepseek-r1:32b": 900
    }
    
    DOMAIN_MULTIPLIERS = {
        "CUDA/Kernel": 1.5,  # CUDA is slower
        "C++/System": 1.0,
        "Audio/DSP": 1.2,
        "Tests/Docs": 0.8    # Tests are faster
    }
    
    def estimate_eta(self, model, domain):
        base = self.MODEL_TIMINGS.get(model, 180)
        multiplier = self.DOMAIN_MULTIPLIERS.get(domain, 1.0)
        
        # Add 30s overhead for build/test
        eta = base * multiplier + 30
        
        return eta
```

---

## 6️⃣ Error Categorizer Integration

### Rolle im System

Error Categorizer ist ein **separates Modul** (Single-Responsibility), das:
1. Fehler klassifiziert (COMPILATION/RUNTIME/PERFORMANCE/LOGIC)
2. Schweregrad bestimmt (1-5)
3. Recovery-Hint gibt

### Output Format

```python
{
    'error_type': str,      # COMPILATION / RUNTIME / PERFORMANCE / LOGIC
    'phase': str,           # BUILD / TEST / PROFILE
    'severity': int,        # 1-5 (1=minor, 5=critical)
    'recoverable': bool,    # Retry möglich?
    'hint': str            # Recovery-Vorschlag
}
```

### Integration Points

**1. Meta-Supervisor:**
```python
# Aggregiert Fail-Rates by error_type
stats = learning_db.get_error_stats()
# → {"COMPILATION": 12, "RUNTIME": 8, "PERFORMANCE": 3}

# Nutzt für Priority-Berechnung (R-Factor)
if stats["RUNTIME"] > threshold:
    R += 0.1  # Recency boost
```

**2. Supervisor:**
```python
# Nutzt für Retry/Stop-Loss/Abort
error_info = error_categorizer.categorize(error)

if error_info['recoverable']:
    if error_info['severity'] < 4:
        # Retry
        return await self.retry_with_next_model()
    else:
        # Eskalate immediately
        return await self.escalate()
else:
    # Abort
    return self.abort_with_manual_intervention()
```

### Beispiel

```
Build Error: "undefined reference to `cudaMalloc`"

Error Categorizer Output:
{
    'error_type': 'COMPILATION',
    'phase': 'BUILD',
    'severity': 2,
    'recoverable': True,
    'hint': 'Add -lcudart to linker flags'
}

Supervisor Decision:
- severity = 2 (low) → Retry
- recoverable = True → Use FixerAgent
- hint provided → Pass hint to FixerAgent

FixerAgent:
- Reads hint
- Adds -lcudart
- Rebuilds → Success!
```

---

## 7️⃣ Configuration

### OptimizationConfig

**File:** `config/optimization_config.json`

```json
{
  "max_optimization_iterations": 10,
  "target_score": 80,
  
  "retry_build": 2,
  "retry_test": 1,
  "retry_profile": 1,
  
  "stoploss_per_model": 2,
  "max_concurrent_builds": 3,
  
  "enable_meta_supervisor": true,
  "enable_two_tier_profiling": true,
  "enable_async_io": true,
  
  "meta_supervisor_weights": {
    "failure_weight": 0.5,
    "time_weight": 0.2,
    "count_weight": 0.2,
    "recency_weight": 0.1
  },
  
  "hybrid_decision_weights": {
    "meta_bias": 0.40,
    "complexity": 0.30,
    "failure_state": 0.30
  },
  
  "two_tier_profiling": {
    "score_change_threshold": 5,
    "max_tier0_streak": 5
  },
  
  "timeouts": {
    "build": 300,
    "test": 120,
    "profile": 900
  }
}
```

### Loading

```python
from core.optimization_config import OptimizationConfig

config = OptimizationConfig.load("config/optimization_config.json")

# Access
print(config.max_optimization_iterations)  # 10
print(config.stoploss_per_model)  # 2
print(config.enable_meta_supervisor)  # True
```

---

## 8️⃣ Migration von Phase 6

### Was ändert sich?

| Aspekt | Phase 6 | Phase 7 |
|--------|---------|---------|
| **Model Selection** | Static rules | Meta-Supervisor + Hybrid |
| **Error Handling** | Simple retry | Stop-Loss Escalation |
| **Profiling** | Always full | Two-Tier |
| **Task Queue** | FIFO | Cost-Aware (ROI) |
| **Config** | Hardcoded | `optimization_config.json` |

### Migration Steps

**1. Update Config:**
```bash
# Create optimization_config.json
cp config/optimization_config.json.template config/optimization_config.json
# Edit as needed
```

**2. Update Code:**
```python
# Old (Phase 6)
model = model_selector.select_simple(language, complexity)

# New (Phase 7)
meta_bias = meta_supervisor.recommend_model_bias()
complexity_score = complexity_detector.detect(code)
failure_score = calculate_failure_score(retry_count)

model = hybrid_decision.select_model(
    domain,
    meta_bias,
    complexity_score,
    failure_score
)
```

**3. Database Update:**
```python
# Learning-DB bleibt kompatibel (kein Schema-Change)
# Aber: Neue Indizes für Performance
learning_db.create_indexes_for_meta()
```

**4. Test:**
```bash
python tests/test_phase7_meta.py
```

---

## 9️⃣ Best Practices

### Meta-Supervisor

✅ **DO:**
- Let Meta-Supervisor run für min. 20 Tasks (statistische Relevanz)
- Check priorities regelmäßig: `meta.next_priorities()`
- Trust the bias wenn count ≥ 5 und SR ≥ 0.65

❌ **DON'T:**
- Override Meta-Bias ohne guten Grund
- Expect accurate predictions bei count < 5
- Ignore recency (alte Daten weniger relevant)

### 7-Model-Routing

✅ **DO:**
- Respect Stop-Loss (verhindert infinite loops)
- Start mit empfohlenem Modell (Routing oder Meta-Bias)
- Log escalations für Analyse

❌ **DON'T:**
- Skip models in escalation chain
- Downgrade bei Failure
- Ignore final fallback (SearchAgent/Manual)

### Two-Tier-Profiling

✅ **DO:**
- Set threshold basierend auf Task-Importance (critical: 3%, normal: 5%)
- Always Tier 1 bei Iteration 0 (Baseline!)
- Check full profile alle 5-10 iterations

❌ **DON'T:**
- Always Tier 0 (miss regressions)
- Always Tier 1 (waste time)
- Ignore score_change

### Cost-Aware Queue

✅ **DO:**
- Update ETA basierend auf actual timings
- Balance priority_score und ETA
- Prefer quick wins wenn unsicher

❌ **DON'T:**
- Ignore ETA (slow tasks block queue)
- Always pick highest priority (may be slowest)
- Forget overhead (build/test takes time)

---

## 🔟 Troubleshooting

### Meta-Supervisor liefert keine Bias

**Symptom:** `recommend_model_bias()` returns `{}`

**Ursache:** Nicht genug Daten (count < 5 oder SR < 0.65)

**Fix:**
```python
# Check statistics
stats = learning_db.get_domain_stats()
for domain, data in stats.items():
    print(f"{domain}: count={data['count']}, SR={data['success_rate']}")

# Wenn count < 5: Mehr Tasks durchführen
# Wenn SR < 0.65: Fehler analysieren und fixen
```

### Stop-Loss eskaliert zu schnell

**Symptom:** Nach 2 Fehlern sofort auf größtes Modell

**Ursache:** Stop-Loss = 2 zu niedrig für deine Tasks

**Fix:**
```json
// optimization_config.json
{
  "stoploss_per_model": 3  // Statt 2
}
```

### Two-Tier-Profiling triggert zu oft

**Symptom:** Meiste Runs sind Tier 1 (langsam)

**Ursache:** Threshold zu niedrig

**Fix:**
```json
// optimization_config.json
{
  "two_tier_profiling": {
    "score_change_threshold": 10,  // Statt 5
    "max_tier0_streak": 10         // Statt 5
  }
}
```

### Cost-Aware Queue priorisiert falsch

**Symptom:** Wichtige Tasks werden zu spät ausgeführt

**Ursache:** ETA-Estimation ungenau

**Fix:**
```python
# Update ETA basierend auf actual timings
eta_estimator.update_from_history(learning_db)

# Oder: Adjust priority_score
task.priority_score *= 1.5  # Boost wichtige Tasks
```

---

## 📊 Success Metrics

### Track These

```python
# Daily
meta_supervisor.get_statistics()
# → Hit rate (Bias verwendet vs ignoriert)
# → Avg priority error (predicted vs actual)

domain_router.get_escalation_stats()
# → Escalations per domain
# → Stop-Loss triggers
# → Model effectiveness

two_tier_profiler.get_time_savings()
# → Tier 0 vs Tier 1 ratio
# → Time saved
# → Regression catch rate

cost_aware_queue.get_efficiency()
# → Avg wait time
# → Throughput (tasks/hour)
# → ROI accuracy
```

### Phase 7 Goals

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Model Selection Accuracy | >70% | Meta-Bias hit rate |
| Profiling Time Reduction | 40-55% | Two-Tier stats |
| Task Efficiency | +30% | Cost-Aware ROI |
| Error Recovery | +25% | Stop-Loss vs Phase 6 |
| Learning Speed | Better recency | Exponential aging |

---

**Status:** 🚀 Phase 7 Implementation Active  
**Next:** Implement Meta-Supervisor → Hybrid Decision → Two-Tier → Cost-Aware  
**Questions?** Check [INDEX.md](INDEX.md) or [CLAUDE_INSTRUCTIONS.md](../CLAUDE_INSTRUCTIONS.md)
