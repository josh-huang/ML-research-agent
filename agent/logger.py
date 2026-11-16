"""Structured run-log (JSONL) for the agent's iteration trail.

Each iteration appends one JSON line to ``run_logs/run_log.jsonl`` with a fixed schema,
so both the static report and the dashboard can replay the whole run. The ``LOG_DIR``
constant is shared with :mod:`agent.state`.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOG_DIR = os.path.join(_ROOT, "run_logs")
RUN_LOG_PATH = os.path.join(LOG_DIR, "run_log.jsonl")

_GIT_SHA: str | None = None

# Paths whose uncommitted changes make the recorded commit an inaccurate description of
# the code that ran. Run artifacts (run_logs/, submission/final.csv) and the playbook —
# which the agent itself appends to during a run — are excluded; the playbook's content
# is pinned separately by ``memory.playbook_sha``.
_CODE_PATHS = ["agent", "models", "submission", "dashboard", "kuairand-starter-kit",
               ":(exclude)agent/playbook.md", ":(exclude)submission/final.csv"]


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5,
                         cwd=_ROOT)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return out.stdout.strip()


def git_sha() -> str:
    """HEAD commit of the run's code, cached at first call ("" if not a git repo).

    Suffixed with ``-dirty`` when tracked or untracked code differs from HEAD, so a
    recorded sha never silently misdescribes the code that produced a run. The agent
    never mutates code (only config), so one value per run makes the D3 "code diff"
    traceable without a per-iteration diff.
    """
    global _GIT_SHA
    if _GIT_SHA is None:
        try:
            sha = _git("rev-parse", "--short", "HEAD")
            dirty = bool(_git("status", "--porcelain", "--", *_CODE_PATHS))
            _GIT_SHA = sha + ("-dirty" if sha and dirty else "")
        except Exception:  # noqa: BLE001 — degrade to "" rather than fail the run
            _GIT_SHA = ""
    return _GIT_SHA


def log_iteration(*, iteration: int, hypothesis: str, action: dict,
                  metrics: dict | None, tokens: int, gpu_h: float,
                  errors: list[str] | None = None, verdict: str = "",
                  duration_s: float = 0.0, recovery: str = "",
                  lesson: str = "", tokens_input: int = 0,
                  tokens_output: int = 0, llm_error: str = "") -> None:
    """Append one iteration record (native types only — no numpy scalars)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    record = {
        "iteration": int(iteration),
        "git_sha": git_sha(),
        "ts": time.time(),
        "hypothesis": hypothesis,
        "action": action,
        "metrics": metrics,
        "verdict": verdict,
        "tokens": int(tokens),
        "tokens_input": int(tokens_input),
        "tokens_output": int(tokens_output),
        "gpu_h": round(float(gpu_h), 6),
        "duration_s": round(float(duration_s), 2),
        "errors": list(errors or []),
        "recovery": recovery or None,
        "lesson": lesson or "",
        # LLM/API failure during this iteration's episode (None if the LLM was healthy);
        # distinct from `errors`, which classifies the *experiment's* failure.
        "llm_error": llm_error or None,
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
