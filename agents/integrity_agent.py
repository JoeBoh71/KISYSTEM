# -*- coding: utf-8 -*-
"""
KISYSTEM Phase 7 — Integrity Agent
Lädt alle Agent-Module, meldet Import-/Syntax-Probleme, schreibt Report.

Author: Jörg Bohne
Date: 2025-11-09
"""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from typing import List, Dict


AGENT_FILES = [
    "builder_agent.py",
    "fixer_agent.py",
    "fixer_agent_v3.py",
    "tester_agent.py",
    "cuda_profiler_agent.py",
    "search_agent_v2.py",
    "review_agent_v2.py",
    "docs_agent_v2.py",
    "hardware_test_agent.py",
]


def _try_import(module_path: Path) -> Dict:
    name = module_path.stem
    try:
        spec = importlib.util.spec_from_file_location(name, str(module_path))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader, "missing loader"
        spec.loader.exec_module(mod)  # noqa
        return {"module": name, "ok": True, "error": None}
    except Exception as e:
        return {"module": name, "ok": False, "error": str(e)}


def run(report_path: Path = Path("C:/KISYSTEM/Logs/integrity_report.json")) -> int:
    agents_dir = Path(__file__).parent
    results: List[Dict] = []

    for f in AGENT_FILES:
        p = agents_dir / f
        if not p.exists():
            results.append({"module": p.stem, "ok": False, "error": "file not found"})
            continue
        results.append(_try_import(p))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    broken = [r for r in results if not r["ok"]]
    print(f"[IntegrityAgent] OK={len(results)-len(broken)}  FAIL={len(broken)}  → {report_path}")
    return 0 if not broken else 1


if __name__ == "__main__":
    raise SystemExit(run())
