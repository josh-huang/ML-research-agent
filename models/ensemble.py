"""Seed-averaged + cross-model ensembling, with a repeated-(user,video) dedup diagnostic.

The three Phase-3 models (FM / DeepFM / DIN) carry different inductive biases (linear /
MLP tower / history attention) and partially decorrelated errors, so rank-averaging their
scores reliably lifts primary over any single model. Per model we also average over SEEDS
to kill single-seed noise — the official baseline itself is reported as a 5-seed mean
(std 0.0008), so a one-shot number is not a credible "we beat baseline" claim.

Combine method: **rank-average** (scale-free), never a plain score mean — raw logit scales
differ across models, and GAUC / nDCG@5 only care about within-user ordering.

Invoke from the project root::

    python -m models.ensemble
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_KIT = os.path.join(_ROOT, "kuairand-starter-kit", "kuairand-starter-kit")
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)

from data import load  # noqa: E402
from submit import write_submission  # noqa: E402

from models.data_loader import load_extended  # noqa: E402
from models.harness import evaluate  # noqa: E402
from models.train import run_experiment  # noqa: E402

DATA_DIR = os.path.join(_KIT, "KuaiRand-Pure", "data")
OUT = os.path.join(_ROOT, "submission", "final.csv")

# Verified-strong members (Phase 3d). DIN(k=32) is the tuned best single model.
MEMBERS = [
    ("fm",     dict(model="fm", loss="bce")),
    ("deepfm", dict(model="deepfm", loss="bce")),
    ("din",    dict(model="din", loss="bce", k=32, lr=3e-4, dropout=0.2)),
]
SEEDS = [0, 1, 2, 3, 4]


def rank_avg(scores: np.ndarray) -> np.ndarray:
    """Tie-averaged rank percentiles in [0, 1] (scale-free, order-preserving)."""
    a = np.asarray(scores, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    n = a.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) * 0.5  # mean 1-indexed rank over ties
        i = j
    return ranks / n


def dedup(scores: np.ndarray, rows, mode: str = "mean") -> np.ndarray:
    """Unify scores of repeated (user_id, video_id) test pairs (3.06% of rows).

    ``mode`` in {mean, max, min}. Repeated impressions are the *same* item shown to the
    same user, so their scores should agree; this collapses model noise across repeats.
    """
    out = np.asarray(scores, dtype=np.float64).copy()
    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault((r[1], r[2]), []).append(i)
    for idxs in groups.values():
        if len(idxs) > 1:
            vals = out[idxs]
            v = {"mean": float(vals.mean()), "max": float(vals.max()),
                 "min": float(vals.min())}[mode]
            out[idxs] = v
    return out


def _fmt(r: dict) -> str:
    return (f"GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | "
            f"primary {r['primary']:.4f}")


def main() -> None:
    """Build seed-averaged members + their rank-average ensemble; submit the valid-best.

    Selection discipline: every *choice* (which variant to submit, which dedup mode looks
    good) is made on the VALID split. Test is scored exactly once, for the chosen variant,
    and only reported — never compared across candidates.
    """
    print(f"loading data from {DATA_DIR} ...", flush=True)
    data = load_extended(DATA_DIR)
    official = load(DATA_DIR)
    va_rows, te_rows = official["valid"], official["test"]  # (date,user_id,video_id,...) tuples
    te_u, te_y = data["test"]["users"], data["test"]["y"]
    va_u, va_y = data["valid"]["users"], data["valid"]["y"]

    members = []  # (name, seed-avg valid scores, seed-avg test scores, valid mean, valid std)
    for name, cfg in MEMBERS:
        va_scores, te_scores, v_prim = [], [], []
        for s in SEEDS:
            res = run_experiment(data, dict(cfg, seed=s), verbose=False)
            va_scores.append(res["valid_scores"])
            te_scores.append(res["test_scores"])
            v_prim.append(res["valid"]["primary"])
            print(f"  {name:6s} seed={s}  valid {res['valid']['primary']:.4f}", flush=True)
        members.append((name, np.mean(va_scores, axis=0), np.mean(te_scores, axis=0),
                        float(np.mean(v_prim)), float(np.std(v_prim))))

    print(f"\n=== per-model ({len(SEEDS)} seeds), valid ===")
    for name, va, _, vp, vsd in members:
        print(f"  {name:6s}  valid primary per-seed {vp:.4f} ± {vsd:.4f} | "
              f"seed-avg scores {_fmt(evaluate(va_u, va_y, va))}")

    # Candidates, each carried as (valid scores, test scores) so selection stays on valid.
    by_name = {name: (va, te) for name, va, te, *_ in members}
    ens_va = np.stack([rank_avg(va) for _, va, *_ in members]).mean(axis=0)
    ens_te = np.stack([rank_avg(te) for _, _, te, *_ in members]).mean(axis=0)
    candidates = {"din_seed_avg": by_name["din"], "ensemble_rank_avg": (ens_va, ens_te)}

    print("\n=== candidates, valid ===")
    for label, (va, _) in candidates.items():
        print(f"  {label:18s} {_fmt(evaluate(va_u, va_y, va))}")
    # Repeated-(user,video) dedup diagnostic, also judged on valid.
    for mode in ("mean", "max", "min"):
        m = evaluate(va_u, va_y, dedup(ens_va, va_rows, mode))
        print(f"  ensemble+dedup({mode:4s}) {_fmt(m)}")

    label = max(candidates,
                key=lambda k: evaluate(va_u, va_y, candidates[k][0])["primary"])
    test_scores = candidates[label][1]
    write_submission(OUT, te_rows, test_scores)
    print(f"\nselected on valid: {label}")
    print(f"wrote {OUT} ({len(te_rows):,d} rows)")
    print(f"test (reported once, not used for selection): "
          f"{_fmt(evaluate(te_u, te_y, test_scores))}")


if __name__ == "__main__":
    main()
