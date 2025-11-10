# -*- coding: utf-8 -*-
"""
KISYSTEM Phase 7 — Model Selector (lernendes Routing + Heuristik-Fallback)

Author: Jörg Bohne
Date: 2025-11-09
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

# Optionaler Lern-Import (soft dependency)
try:
    from learning_module_v2 import LearningModuleV2
except Exception:
    LearningModuleV2 = None  # Fallback ohne Learning


@dataclass
class ModelConfig:
    name: str
    timeout: int
    description: str


class ModelSelector:
    """Intelligente Modellwahl: Heuristik + (optional) lernendes Routing"""

    MODELS: Dict[str, Dict] = {
        "llama3.1:8b":         {"timeout": 180,  "description": "Schnell, einfache Tasks"},
        "deepseek-coder-v2:16b": {"timeout": 300,  "description": "Mittlere Komplexität"},
        "qwen2.5-coder:32b":   {"timeout": 1800, "description": "Komplexe Projekte / CUDA"},
        "deepseek-r1:32b":     {"timeout": 1800, "description": "Tiefes Debugging / Reasoning"},
    }

    CUDA_SIMPLE = [
        "vector add", "vector addition", "element-wise", "elementwise",
        "scalar", "simple", "basic", "multiply", "divide", "copy", "fill", "saxpy", "daxpy"
    ]
    CUDA_MEDIUM = [
        "shared memory", "matrix multiply", "matrix multiplication", "transpose",
        "convolution", "dot product", "reduction", "prefix sum", "scan", "histogram", "sorting"
    ]
    CUDA_COMPLEX = [
        "fft", "fast fourier", "multi-kernel", "multi-pass", "dynamic parallelism",
        "graph", "cooperative groups", "tensor core", "sparse", "optimization pipeline",
        "multi-gpu", "streams", "async"
    ]

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[ModelSelector] {msg}")

    # ---- öffentliche API -----------------------------------------------------

    def select_model(
        self,
        task: str,
        language: str | None = None,
        agent_type: str | None = None,
        context: dict | None = None
    ) -> ModelConfig:
        complexity = self._detect_complexity(task, language)
        base = self._heuristic_choice(complexity)

        learned = self._learned_choice(
            complexity=complexity,
            language=(language or "unknown").lower(),
            agent_type=(agent_type or "builder").lower(),
            context=context or {}
        )

        name = learned or base
        info = self.MODELS[name]
        self._log(f"🎯 Complexity={complexity} → {name}  (~{info['timeout']//60} min)")
        return ModelConfig(name=name, timeout=info["timeout"], description=info["description"])

    def get_escalation_model(self, current_model: str) -> Optional[ModelConfig]:
        order = ["llama3.1:8b", "deepseek-coder-v2:16b", "qwen2.5-coder:32b", "deepseek-r1:32b"]
        if current_model not in order:
            return None
        idx = min(order.index(current_model) + 1, len(order) - 1)
        nxt = order[idx]
        inf = self.MODELS[nxt]
        return ModelConfig(name=nxt, timeout=inf["timeout"], description=inf["description"])

    # ---- interne Logik -------------------------------------------------------

    def _detect_complexity(self, task: str, language: Optional[str] = None) -> str:
        t = task.lower()

        if language and language.upper() == "CUDA":
            c = self._detect_cuda_complexity(t)
            if c: return c

        if any(k in t for k in ["simple", "basic", "hello", "example"]):
            return "SIMPLE"
        if any(k in t for k in ["optimize", "performance", "algorithm", "data structure",
                                "class", "interface", "architecture"]):
            return "COMPLEX"
        return "MEDIUM"

    def _detect_cuda_complexity(self, t: str) -> str:
        if any(k in t for k in self.CUDA_COMPLEX): return "COMPLEX"
        if any(k in t for k in self.CUDA_MEDIUM):  return "MEDIUM"
        if any(k in t for k in self.CUDA_SIMPLE):  return "SIMPLE"
        return "MEDIUM"

    def _heuristic_choice(self, complexity: str) -> str:
        return {"SIMPLE": "llama3.1:8b", "MEDIUM": "deepseek-coder-v2:16b", "COMPLEX": "qwen2.5-coder:32b"}[complexity]

    def _learned_choice(self, complexity: str, language: str, agent_type: str, context: dict) -> Optional[str]:
        """
        Greift auf aggregierte Erfolgsraten im Learning-Store zu.
        Erwartet, dass LearningModuleV2.get_statistics() u.a. 'by_agent_lang' enthält:
        { "('builder','cuda')": {"best_model":"qwen2.5-coder:32b", "success_rate":0.72}, ... }
        """
        if LearningModuleV2 is None:
            return None
        try:
            lm = LearningModuleV2()
            stats = lm.get_statistics()
            key = str((agent_type, language))
            entry = stats.get("by_agent_lang", {}).get(key)
            if entry and float(entry.get("success_rate", 0.0)) >= 0.65:
                model = entry.get("best_model")
                if model in self.MODELS:
                    self._log(f"🤖 Lern-Bias {(agent_type, language)} → {model} (sr={entry['success_rate']:.2f})")
                    return model
        except Exception as e:
            self._log(f"learned_choice disabled: {e}")
        return None
