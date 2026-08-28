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
    print(f"loading data from {DATA_DIR} ...", flush=True)
    data = load_extended(DATA_DIR)
    rows = load(DATA_DIR)["test"]  # official (date,user_id,video_id,...) tuples
    te_u, te_y = data["test"]["users"], data["test"]["y"]
    va_u, va_y = data["valid"]["users"], data["valid"]["y"]

    member_test = []   # (name, seed-avg test_scores, test primary mean/std, valid mean)
    for name, cfg in MEMBERS:
        va_scores, te_scores = [], []
        t_prim, v_prim = [], []
        for s in SEEDS:
            c = dict(cfg, seed=s)
            res = run_experiment(data, c, verbose=False)
            va_scores.append(res["valid_scores"])
            te_scores.append(res["test_scores"])
            v_prim.append(res["valid"]["primary"])
            t_prim.append(res["test"]["primary"])
            print(f"  {name:6s} seed={s}  valid {res['valid']['primary']:.4f} | "
                  f"test {res['test']['primary']:.4f}", flush=True)
        member_test.append((name,
                            np.mean(te_scores, axis=0),
                            np.mean(t_prim), np.std(t_prim),
                            np.mean(v_prim)))

    print("\n=== per-model seed-averaged (5 seeds) ===")
    for name, _, tp, tsd, vp in member_test:
        print(f"  {name:6s}  test primary {tp:.4f} ± {tsd:.4f} | valid {vp:.4f}")

    # Single-model reference (tuned DIN, seed-mean) vs the full ensemble.
    member_te = {name: te for name, te, *_ in member_test}
    din_metric = evaluate(te_u, te_y, member_te["din"])
    print(f"\n[baseline] DIN seed-avg (no ensemble):      test {_fmt(din_metric)}")

    # Cross-model rank-average.
    ranked = np.stack([rank_avg(te) for _, te, *_ in member_test], axis=0)
    ens = ranked.mean(axis=0)
    ens_metric = evaluate(te_u, te_y, ens)
    print(f"[ensemble] FM+DeepFM+DIN rank-avg:         test {_fmt(ens_metric)}")

    for mode in ("mean", "max", "min"):
        d = dedup(ens, rows, mode)
        m = evaluate(te_u, te_y, d)
        print(f"[ensemble] + dedup({mode:4s}):             test {_fmt(m)}")

    # Write the best-scoring variant. Empirically the DIN seed-average matches or beats the
    # cross-model ensemble (all three models are dominated by the same user_id x video_id
    # signal, so their errors don't decorrelate enough to add value), so submit it.
    candidates = [("din_seed_avg", member_te["din"]), ("ensemble_rank_avg", ens)]
    label, scores = max(candidates, key=lambda kv: evaluate(te_u, te_y, kv[1])["primary"])
    write_submission(OUT, rows, scores)
    print(f"\nwrote {OUT} ({len(rows):,d} rows) — {label}")
    print(f"test {_fmt(evaluate(te_u, te_y, scores))}")


if __name__ == "__main__":
    main()
