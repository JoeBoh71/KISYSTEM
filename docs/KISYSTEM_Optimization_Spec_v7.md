# KISYSTEM Optimization Spec v7
**Version:** 7.0  **Status:** Phase-7  **Datum:** 2025-11-10  

---

## 1. Zielsetzung
KISYSTEM v7 entwickelt das bestehende Framework zu einem **proaktiven, lernfähigen Build- und Optimierungssystem** weiter.  
Hauptziele:
- Einführung des **Meta-Supervisors** zur datenbasierten Priorisierung und Modellwahl  
- Integration des **7-Modell-Routings** mit Stop-Loss-Eskalation  
- Parametrische Steuerung von Qualität / Durchsatz / Lernverhalten  
- Asynchrone Verarbeitung und Kosten-/Zeit-Optimierung  

---

## 2. Meta-Supervisor
**Aufgabe:** Analyse der Laufstatistiken aus `learning_module_v2`, Berechnung von Prioritäten und Modell-Bias (read-only).

**Ausgaben**
```python
next_priorities() -> List[str]
recommend_model_bias() -> Dict[str, str]


Prioritätsformel
𝑃
(
𝑑
)
=
0.5
(
1
−
𝑠
𝑟
)
+
0.2
/
(
1
+
𝑡
)
+
0.2
min
⁡
(
1
,
𝑐
/
20
)
+
0.1
𝑅
P(d)=0.5(1−sr)+0.2/(1+t)+0.2min(1,c/20)+0.1R

sr – Erfolgsquote

t – Durchschnittliche Lösungszeit

c – Anzahl Durchläufe

R – RecencyBoost (+0.1 bei neuem Fehler < 3 Tage, −0.1 bei Erfolg < 1 Tag)

Modell-Bias
𝑏
𝑒
𝑠
𝑡
_
𝑚
𝑜
𝑑
𝑒
𝑙
(
𝑑
)
=
arg
⁡
max
⁡
𝑚
(
𝑠
𝑟
𝑚
,
𝑑
)
wenn 
𝑠
𝑟
≥
0.65
,
 
𝑐
𝑜
𝑢
𝑛
𝑡
≥
5
best_model(d)=arg
m
max
	​

(sr
m,d
	​

)wenn sr≥0.65,count≥5

→ liefert bevorzugtes Startmodell pro Domäne.

3. Modell-Inventar (7 Modelle)
Rang	Modell	Rolle	Timeout [s]
1	llama3.1:8b	Trivial / Boilerplate	180
2	mistral:7b	Generisch / Kurz	240
3	phi4:latest	Tests / Specs / Docs	240
4	deepseek-coder-v2:16b	Mid-Coding C++ / CUDA	300
5	qwen2.5:32b	Reasoning / Architektur	900
6	qwen2.5-coder:32b	Komplexes Coding / CUDA-Opt	1800
7	deepseek-r1:32b	Deep Fixes / Reasoning	1800
4. Domänen-Routing und Eskalation
Domäne	Start-Modell	Eskalations-Kette (Stop-Loss = 2 Fails)
CUDA / Kernel	qwen2.5-coder	r1 → coder-v2 → qwen2.5
C++ / System	coder-v2	qwen-coder → r1 → qwen2.5
Audio / DSP	coder-v2	qwen-coder → r1
Tests / Docs	phi4	mistral → llama → qwen
Planung / Refactor	qwen2.5	r1 → mistral

Success-Matrix überschreibt Startmodell, wenn success ≥ 0.65 ∧ count ≥ 5.
Ältere Runs → Gewicht = exp(−age / 30).

5. Parametrisierung (OptimizationConfig)
Schlüssel	Typ / Bereich	Default	Bedeutung
max_optimization_iterations	int [1–50]	10	Maximale Fix/Optimize-Schleifen
target_score	int [0–100]	80	Zielwert aus PerformanceParser
retry_build / retry_test / retry_profile	int	2 / 1 / 1	Wiederholungen pro Phase
stoploss_per_model	int [1–5]	2	Fehler-Limit pro Modell
max_concurrent_builds	int [1–8]	3	Parallel-Build-Semaphore
enable_meta_supervisor	bool	True	Priorität / Bias aktivieren
6. Performance-Strategien
Retry-Budget und Stop-Loss

Build/Test/Profile = 2 / 1 / 1 → Eskalation nach 2 Fehlschlägen.

Two-Tier-Profiling

Tier 0 – Microbench (ohne nsys)

Tier 1 – Vollprofil nur bei relevanter Aktivität
→ Profilingzeit −40 bis −55 %

Cost-Aware Queue
𝑃
𝑟
𝑖
𝑜
𝑟
𝑖
𝑡
𝑦
𝐸
𝑓
𝑓
=
𝑃
𝑟
𝑖
𝑜
𝑟
𝑖
𝑡
𝑦
𝑆
𝑐
𝑜
𝑟
𝑒
𝐸
𝑇
𝐴
(
𝑀
𝑜
𝑑
𝑒
𝑙
,
𝐷
𝑜
𝑚
𝑎
¨
𝑛
𝑒
)
PriorityEff=
ETA(Model,Dom
a
¨
ne)
PriorityScore
	​


→ Aufgaben mit höchstem ROI zuerst.

Async I/O

nvcc / Tests non-blocking, Profiler seriell

Timeouts [s] = Build 300, Test 120, Profiler 900

7. Scoring und Logging

Scorebereich 0–100 (typisch 80–90, >95 nur nach Tuning).
Lernlogging bei jedem Exit:


run_id, domain, model, iter, score_final, outcome,
phase, reason, timings:{build,test,profile}, ts


run_id, domain, model, iter, score_final, outcome,
phase, reason, timings:{build,test,profile}, ts

