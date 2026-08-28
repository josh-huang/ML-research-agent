"""Execute a training config and classify failures for autonomous recovery.

Wraps :func:`models.train.run_experiment` with a try/except + non-finite guard so a bad
config (shape error, OOM, NaN/diverged loss) becomes a *classified, recoverable* error
instead of killing the agent. The reflector routes each error class to a fix strategy.

``wall_s`` is returned so the orchestrator can account GPU-hours (single-GPU assumption:
gpu_h == wall_s / 3600).
"""
from __future__ import annotations

import time
import traceback

import numpy as np

from models.train import run_experiment

ERROR_CLASSES = ("syntax", "shape", "oom", "nan", "runtime")


def classify_error(e: BaseException) -> str:
    msg = str(e).lower()
    if "out of memory" in msg or "cuda" in msg and "alloc" in msg:
        return "oom"
    if any(w in msg for w in ("shape", "size", "dimension", "mismatch", "broadcast")):
        return "shape"
    if isinstance(e, (SyntaxError, NameError, AttributeError, TypeError)):
        return "syntax"
    return "runtime"


def execute(data: dict, config: dict, timeout_s: int = 600) -> dict:
    """Run one config; returns {metrics, error, wall_s, gpu_h, trace} (never raises)."""
    t0 = time.time()
    try:
        res = run_experiment(data, config, verbose=False)
        wall = time.time() - t0
        vp = res["valid"]["primary"]
        if not np.isfinite(vp):
            return {"metrics": None, "error": "nan", "wall_s": wall,
                    "gpu_h": wall / 3600, "trace": "non-finite valid primary"}
        return {"metrics": {"valid": res["valid"], "test": res["test"]},
                "error": None, "wall_s": wall, "gpu_h": wall / 3600, "trace": ""}
    except Exception as e:  # noqa: BLE001 — classify and recover, never crash the agent
        wall = time.time() - t0
        return {"metrics": None, "error": classify_error(e), "wall_s": wall,
                "gpu_h": wall / 3600, "trace": traceback.format_exc(limit=6)}
