"""Execute a training config and classify failures for autonomous recovery.

Wraps :func:`models.train.run_experiment` with a try/except + non-finite guard so a bad
config (shape error, OOM, NaN/diverged loss) becomes a *classified, recoverable* error
instead of killing the agent. The reflector routes each error class to a fix strategy.

``wall_s`` is returned alongside ``gpu_h`` (single-GPU assumption: gpu_h == wall_s / 3600
when the run used CUDA, and 0 on CPU — wall-clock is tracked separately as ``duration_s``).
"""
from __future__ import annotations

import re
import time
import traceback

import numpy as np
import torch

from models.train import run_experiment

ERROR_CLASSES = ("syntax", "shape", "oom", "nan", "runtime")

# Phrases PyTorch/numpy use for tensor-shape problems. Deliberately specific: a bare
# "size" would also match unrelated messages ("batch size", "file size", ...).
_SHAPE_RE = re.compile(
    r"shape|size mismatch|sizes of tensors|dimension|broadcast|"
    r"mat1 and mat2|must match the (?:existing )?size|expected input", re.I)


def classify_error(e: BaseException) -> str:
    msg = str(e).lower()
    if (isinstance(e, MemoryError) or "out of memory" in msg
            or ("cuda" in msg and "alloc" in msg)):
        return "oom"
    if _SHAPE_RE.search(msg):
        return "shape"
    if isinstance(e, (SyntaxError, NameError, AttributeError, TypeError)):
        return "syntax"
    return "runtime"


def _gpu_hours(wall_s: float, device: str | None) -> float:
    """GPU-hours for one run: wall time on a single GPU, and 0 when it ran on CPU."""
    return wall_s / 3600 if str(device or "").startswith("cuda") else 0.0


def degrade_config(config: dict, error_class: str) -> dict:
    """Map a failure class to a safer config variant (graceful degradation).

    One bounded "route around it" step: OOM -> halve batch size, NaN -> shrink LR 10x,
    anything else -> retry with a fresh seed. The orchestrator runs this variant once; if
    it also fails the run records both and moves on (never crashes).
    """
    out = dict(config)
    if error_class == "oom":
        out["bs"] = max(1024, int(config.get("bs", 8192)) // 2)
        out["_degraded"] = "oom->bs/2"
    elif error_class == "nan":
        out["lr"] = max(1e-4, float(config.get("lr", 1e-3)) * 0.1)
        out["_degraded"] = "nan->lr/10"
    else:
        out["seed"] = int(config.get("seed", 0)) + 1
        out["_degraded"] = f"{error_class}->seed+1"
    return out


def execute(data: dict, config: dict, timeout_s: int = 600) -> dict:
    """Run one config; returns {metrics, error, wall_s, gpu_h, trace, timed_out} (never raises).

    ``timeout_s`` is enforced cooperatively by the training loop (checked per epoch): a run
    that exceeds it stops early and is scored on its best epoch so far, flagged
    ``timed_out`` — a slow config costs bounded time instead of stalling the agent.
    """
    t0 = time.time()
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    try:
        res = run_experiment(data, config, verbose=False, time_budget_s=timeout_s)
        wall = time.time() - t0
        gpu_h = _gpu_hours(wall, res.get("device", device))
        timed_out = bool(res.get("timed_out"))
        vp = res["valid"]["primary"]
        if not np.isfinite(vp):
            return {"metrics": None, "error": "nan", "wall_s": wall, "gpu_h": gpu_h,
                    "trace": "non-finite valid primary", "timed_out": timed_out}
        return {"metrics": {"valid": res["valid"], "test": res["test"]},
                "error": None, "wall_s": wall, "gpu_h": gpu_h, "trace": "",
                "timed_out": timed_out}
    except Exception as e:  # noqa: BLE001 — classify and recover, never crash the agent
        wall = time.time() - t0
        return {"metrics": None, "error": classify_error(e), "wall_s": wall,
                "gpu_h": _gpu_hours(wall, device), "trace": traceback.format_exc(limit=6),
                "timed_out": False}
