"""Structured run-log (JSONL) for the agent's iteration trail.

Each iteration appends one JSON line to ``run_logs/run_log.jsonl`` with a fixed schema,
so both the static report and the dashboard can replay the whole run. The ``LOG_DIR``
constant is shared with :mod:`agent.state`.
"""
from __future__ import annotations

import json
import os
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOG_DIR = os.path.join(_ROOT, "run_logs")
RUN_LOG_PATH = os.path.join(LOG_DIR, "run_log.jsonl")


def log_iteration(*, iteration: int, hypothesis: str, action: dict,
                  metrics: dict | None, tokens: int, gpu_h: float,
                  errors: list[str] | None = None, verdict: str = "",
                  duration_s: float = 0.0) -> None:
    """Append one iteration record (native types only — no numpy scalars)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    record = {
        "iteration": int(iteration),
        "ts": time.time(),
        "hypothesis": hypothesis,
        "action": action,
        "metrics": metrics,
        "verdict": verdict,
        "tokens": int(tokens),
        "gpu_h": round(float(gpu_h), 6),
        "duration_s": round(float(duration_s), 2),
        "errors": list(errors or []),
    }
    with open(RUN_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")


def tail(n: int) -> list[dict]:
    """Last n iteration records (newest last), for the researcher's context window."""
    if not os.path.exists(RUN_LOG_PATH):
        return []
    with open(RUN_LOG_PATH) as f:
        lines = f.readlines()
    return [json.loads(ln) for ln in lines[-n:]]
