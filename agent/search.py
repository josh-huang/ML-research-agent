"""Config-space search: seed configs, normalization, mutation.

The agent searches over ``model x loss x hyperparameters``. This module owns the
legal ranges in one place and provides:

- ``SEED_CONFIGS`` — verified-strong starting points (Phase 3d floor, valid primary).
- ``normalize``    — coerce an LLM-proposed config into legal, hashable ranges
                     (also what makes config-dedup sound: every config has the same keys).
- ``mutate``       — perturb one dimension to escape a local optimum (AIDE-style branch).
- ``random_config`` — uniform sample for the no-LLM fallback proposer (robustness).

Hyperparameter ranges are deliberately narrow: the Phase 3 sweep showed embedding
dim / capacity / loss are near-dead-ends, so the agent is encouraged to stay in the
high-evidence region rather than wander.
"""
from __future__ import annotations

import math
import random

# Defaults (a valid config for any model; FM ignores dnn_hidden/dropout).
# `seed` is pinned here so every normalized config carries the same keys — this is
# what makes config-dedup sound (an LLM omitting `seed` must not read as a new config).
_DEFAULT = dict(k=16, lr=1e-3, epochs=40, bs=8192, patience=8,
                dropout=0.0, dnn_hidden=(64, 32), seed=0, aux=None, aux_weight=0.1,
                use_videoside=False, use_userside=False)

# (lo, hi) per hyperparameter. lr is sampled in log-space.
_BOUNDS = {
    "k": (4, 64),
    "lr": (1e-4, 3e-3),
    "epochs": (10, 60),
    "bs": (1024, 32768),
    "patience": (3, 12),
    "dropout": (0.0, 0.5),
}
_DNN_HIDDEN_CHOICES = [(32,), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64)]
_MODELS = ("fm", "deepfm", "din")
_LOSSES = ("bce", "bpr", "listwise")

# Verified-strong seeds (from sweep_phase3d.jsonl + the tuned DIN run). fm+bce is the anchor.
SEED_CONFIGS = [
    dict(model="fm", loss="bce"),      # baseline anchor, valid 0.6020
    dict(model="din", loss="bce"),     # best single (default k=16), valid 0.6038
    dict(model="deepfm", loss="bce"),  # valid 0.6030
    dict(model="fm", loss="bpr"),      # valid 0.6027
    dict(model="din", loss="bce", k=32, lr=3e-4, dropout=0.2),  # tuned, valid 0.6047
]


def normalize(config: dict) -> dict:
    """Coerce a proposed config into a legal, key-complete dict (dedup-safe)."""
    out = dict(_DEFAULT)
    out["model"] = config.get("model", "fm") if config.get("model") in _MODELS else "fm"
    out["loss"] = config.get("loss", "bce") if config.get("loss") in _LOSSES else "bce"

    for key, (lo, hi) in _BOUNDS.items():
        if key in config and config[key] is not None:
            v = config[key]
            if key in ("k", "epochs", "bs", "patience"):
                out[key] = int(min(hi, max(lo, int(v))))
            else:  # lr, dropout -> float
                out[key] = float(min(hi, max(lo, float(v))))

    h = config.get("dnn_hidden")
    if isinstance(h, str):
        parts = tuple(int(x) for x in h.split(",") if x.strip().isdigit())
        out["dnn_hidden"] = parts if parts in _DNN_HIDDEN_CHOICES else (64, 32)
    elif isinstance(h, (list, tuple)):
        t = tuple(int(x) for x in h)
        out["dnn_hidden"] = t if t in _DNN_HIDDEN_CHOICES else (64, 32)

    if "seed" in config and config["seed"] is not None:
        out["seed"] = int(config["seed"])

    aux = config.get("aux")
    out["aux"] = aux if aux in ("cwm",) else None
    aw = config.get("aux_weight", 0.1)
    out["aux_weight"] = float(min(1.0, max(0.0, float(aw))))
    out["use_videoside"] = bool(config.get("use_videoside", False))
    out["use_userside"] = bool(config.get("use_userside", False))
    return out


def mutate(config: dict, rng: random.Random | None = None) -> dict:
    """Perturb exactly one dimension of a config (keep the rest fixed)."""
    rng = rng or random.Random()
    out = dict(config)
    key = rng.choice(["k", "lr", "dropout", "dnn_hidden", "bs", "aux",
                      "use_videoside", "use_userside"])
    if key == "dnn_hidden":
        out["dnn_hidden"] = rng.choice(_DNN_HIDDEN_CHOICES)
    elif key == "aux":
        out["aux"] = "cwm" if not out.get("aux") else None
    elif key == "use_videoside":
        out["use_videoside"] = not bool(out.get("use_videoside"))
    elif key == "use_userside":
        out["use_userside"] = not bool(out.get("use_userside"))
    elif key == "lr":
        lo, hi = _BOUNDS["lr"]
        out["lr"] = round(float(math.exp(rng.uniform(math.log(lo), math.log(hi)))), 6)
    elif key == "dropout":
        out["dropout"] = round(rng.uniform(*_BOUNDS["dropout"]), 3)
    else:  # k / bs -> pick an integer on the allowed grid
        lo, hi = _BOUNDS[key]
        out[key] = int(rng.choice([lo, (lo + hi) // 2, hi]))
    return out


def random_config(rng: random.Random | None = None) -> dict:
    """Uniform sample over the action space (no-LLM fallback proposer)."""
    rng = rng or random.Random()
    return normalize({
        "model": rng.choice(_MODELS),
        "loss": rng.choice(_LOSSES),
        "k": int(rng.choice([8, 16, 32, 64])),
        "lr": float(math.exp(rng.uniform(math.log(1e-4), math.log(3e-3)))),
        "dropout": round(rng.uniform(0.0, 0.3), 2),
        "dnn_hidden": rng.choice(_DNN_HIDDEN_CHOICES),
        "seed": rng.randint(0, 4),
        "use_videoside": rng.random() < 0.5,
        "use_userside": rng.random() < 0.5,
    })
