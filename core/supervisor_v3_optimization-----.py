# -*- coding: utf-8 -*-
"""
KISYSTEM Phase 7 — Supervisor V3 mit Optimierung & Parallel-Builds

Build → Test/Profiling → Fix → Re-Test (Iterationslimit, Ziel-Perf-Score)
Neu: Concurrency für Build/Test, Profiler bleibt seriell (GPU-Schutz)

Author: Jörg Bohne
Date: 2025-11-09
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from model_selector import ModelSelector, ModelConfig
from performance_parser import PerformanceParser
from learning_module_v2 import LearningModuleV2
from confidence_scorer import ConfidenceScorer
from context_tracker import ContextTracker

# Agents (vorhanden)
from agents.builder_agent import BuilderAgent
from agents.tester_agent import TesterAgent
from agents.fixer_agent_v3 import FixerAgentV3
from agents.cuda_profiler_agent import CUDAProfilerAgent

# Optional: Meta-Supervisor
try:
    from core.supervisor_meta import MetaSupervisor
except Exception:
    MetaSupervisor = None


@dataclass
class OptimizationConfig:
    target_score: int = 80
    max_iterations: int = 10
    max_concurrent_builds: int = 3
    enable_meta_supervisor: bool = True
    verbose: bool = True


class SupervisorV3WithOptimization:
    def __init__(self, cfg: OptimizationConfig):
        self.cfg = cfg
        self.selector = ModelSelector(verbose=cfg.verbose)
        self.parser = PerformanceParser(verbose=cfg.verbose)
        self.learning = LearningModuleV2()
        self.confidence = ConfidenceScorer()
        self.ctx = ContextTracker()
        self.builder = BuilderAgent()
        self.tester = TesterAgent()
        self.fixer = FixerAgentV3()
        self.profiler = CUDAProfilerAgent()
        self._build_sem = asyncio.Semaphore(self.cfg.max_concurrent_builds)

        self.meta = MetaSupervisor(verbose=cfg.verbose) if (MetaSupervisor and cfg.enable_meta_supervisor) else None

    def _log(self, msg: str):
        if self.cfg.verbose:
            print(f"[SupervisorV3+] {msg}")

    # ---------------------- Öffentliche API ----------------------

    async def execute_with_optimization(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        tasks: Liste von Task-Deskriptoren, z. B. [{'title': 'CUDA Convolver', 'code_hint': 'stft ...', 'language': 'CUDA'}, ...]
        """
        if self.meta:
            prio = self.meta.next_priorities()
            bias = self.meta.recommend_model_bias()
            self._log(f"Meta-Hinweise: prio={prio}, bias={bias}")
        else:
            prio, bias = [], {}

        results = []
        # Parallel: Build/Test (keine GPU-Profiler)
        build_test_jobs = [self._build_and_test_task(t, bias) for t in tasks]
        results_build_test = await asyncio.gather(*build_test_jobs, return_exceptions=False)

        # Seriell: Profiling + Fix-Loop (GPU)
        for res in results_build_test:
            if not res.get("ok"):
                results.append(res)
                continue
            prof_res = await self._profile_and_optimize(res)
            results.append(prof_res)

        return {"ok": True, "results": results}

    # ---------------------- Phasen ----------------------

    async def _build_and_test_task(self, task: Dict[str, Any], bias: Dict[str, str]) -> Dict[str, Any]:
        async with self._build_sem:
            title = task.get("title", "task")
            lang = task.get("language")
            code_hint = task.get("code_hint", "")

            # Modellwahl (+ optionaler Meta-Bias)
            model_cfg: ModelConfig = self.selector.select_model(
                task=code_hint or title,
                language=lang,
                agent_type="builder",
                context={"bias": bias},
            )
            if bias:
                # Soft-Bias durchreichen (Builder kann es nutzen)
                task["model_bias"] = bias

            self._log(f"Build/Test start → {title} @ {model_cfg.name}")

            # Build
            build_res = await self.builder.build(task, model_cfg.name)
            if not build_res.get("ok"):
                return {"ok": False, "phase": "build", "title": title, "error": build_res.get("error")}

            # Tests
            test_res = await self.tester.run_tests(build_res)
            if not test_res.get("ok"):
                # Fix-Loop noch nicht, nur Rückgabe (Profiler seriell danach behandelt)
                return {"ok": False, "phase": "test", "title": title, "error": test_res.get("error")}

            out = {"ok": True, "phase": "test", "title": title, "artifact": test_res.get("artifact"), "language": lang}
            return out

    async def _profile_and_optimize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fährt die GPU-Phase seriell:
        Profiling → Score → ggf. Fix-Loop (Iterationslimit, Zielscore)
        """
        art = result["artifact"]
        lang = result.get("language") or "CUDA"  # konservativ

        # 1) Profiling
        prof_out = await self.profiler.profile(art)
        parsed = self.parser.parse(prof_out.get("stdout", ""), prof_out.get("stderr", ""))

        score = parsed.get("score", 0)
        self._log(f"Profiling-Score: {score}")

        if score >= self.cfg.target_score:
            return {"ok": True, "phase": "profile", "title": result["title"], "score": score, "artifact": art}

        # 2) Optimierungs-Loop
        current_art = art
        it = 0
        while it < self.cfg.max_iterations:
            it += 1
            self._log(f"Optimize Iteration {it}/{self.cfg.max_iterations}")

            fix_res = await self.fixer.optimize(current_art, parsed)  # nutzt Hybrid Error Handler intern
            if not fix_res.get("ok"):
                return {"ok": False, "phase": "optimize", "iter": it, "error": fix_res.get("error")}

            # Rebuild + Test (unter Semaphor)
            async with self._build_sem:
                re_build = await self.builder.rebuild(fix_res)
                if not re_build.get("ok"):
                    return {"ok": False, "phase": "rebuild", "iter": it, "error": re_build.get("error")}
                re_test = await self.tester.run_tests(re_build)
                if not re_test.get("ok"):
                    return {"ok": False, "phase": "retest", "iter": it, "error": re_test.get("error")}
                current_art = re_test.get("artifact")

            # Re-Profile (seriell)
            prof_out = await self.profiler.profile(current_art)
            parsed = self.parser.parse(prof_out.get("stdout", ""), prof_out.get("stderr", ""))
            score = parsed.get("score", 0)
            self._log(f"Score nach Fix {it}: {score}")

            if score >= self.cfg.target_score:
                return {"ok": True, "phase": "optimize", "iter": it, "score": score, "artifact": current_art}

        return {"ok": False, "phase": "optimize", "error": "target_score not reached", "score": score}
