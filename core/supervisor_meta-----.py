# -*- coding: utf-8 -*-
"""
KISYSTEM Phase 7 — Meta-Supervisor
Zielsteuerung: priorisiert Domains/Tasks anhand realer KPIs aus LearningModuleV2.

Author: Jörg Bohne
Date: 2025-11-09
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from learning_module_v2 import LearningModuleV2  # vorhandenes Modul


@dataclass
class DomainKPI:
    domain: str
    success_rate: float
    fail_rate: float
    avg_solve_time: float
    volume: int
    last_success_days: Optional[float] = None
    last_failure_days: Optional[float] = None


class MetaSupervisor:
    """
    Liest KPI-Statistiken aus dem Learning-Store und liefert Prioritäten
    für Domains und Modellwahl-Hinweise (soft constraints).
    """

    def __init__(self, db_path: str = "D:/AGENT_MEMORY/memory.db", verbose: bool = True):
        self.learning = LearningModuleV2(db_path)
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[MetaSupervisor] {msg}")

    def fetch_domain_kpis(self) -> List[DomainKPI]:
        s: Dict = self.learning.get_statistics()
        by_domain = s.get("by_domain", [])
        kpis: List[DomainKPI] = []
        for d in by_domain:
            kpis.append(DomainKPI(
                domain=d["domain"],
                success_rate=d.get("success_rate", 0.0) or 0.0,
                fail_rate=1.0 - (d.get("success_rate", 0.0) or 0.0),
                avg_solve_time=d.get("avg_solve_time", 0.0) or 0.0,
                volume=d.get("count", 0) or 0,
                last_success_days=d.get("days_since_success"),
                last_failure_days=d.get("days_since_failure"),
            ))
        return kpis

    def next_priorities(self, kpis: Optional[List[DomainKPI]] = None) -> List[str]:
        """
        Heuristik:
        1) Hohe Fail-Rate zuerst (größter ROI),
        2) bei gleicher Fail-Rate: kürzere Solve-Time bevorzugen,
        3) bei Gleichstand: höheres Volumen zuerst.
        """
        if kpis is None:
            kpis = self.fetch_domain_kpis()
        if not kpis:
            return []

        ranked = sorted(
            kpis,
            key=lambda x: (round(x.fail_rate, 4), -round(x.avg_solve_time or 9e9, 4), x.volume),
            reverse=True,
        )
        order = [k.domain for k in ranked]
        self._log(f"Prioritäten (Domain): {order}")
        return order

    def recommend_model_bias(self) -> Dict[str, str]:
        """
        Liefert optionale Modell-Biases je Domain auf Basis historischer Erfolgsraten.
        Beispiel: {'cuda_kernel': 'qwen2.5-coder:32b', 'audio_dsp': 'deepseek-coder-v2:16b'}
        """
        s: Dict = self.learning.get_statistics()
        by_domain_model = s.get("by_domain_model", [])
        bias: Dict[str, Tuple[str, float]] = {}  # domain -> (best_model, best_rate)

        for row in by_domain_model:
            dom = row["domain"]
            model = row["model_used"]
            rate = row.get("success_rate", 0.0) or 0.0
            if dom not in bias or rate > bias[dom][1]:
                bias[dom] = (model, rate)

        out = {d: m for d, (m, r) in bias.items() if r >= 0.65}
        if out:
            self._log(f"Modell-Bias: {out}")
        return out
