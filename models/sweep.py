"""Phase 3d sweep: grid-search model x loss to find the >=0.63 config.

Loads the extended data ONCE, then runs each (model, loss) config sequentially on the
single GPU (no parallelism — training must not oversubscribe the card), appending one
JSON line per config to ``run_logs/sweep_phase3d.jsonl``.

Baseline reference (valid primary): FM pointwise BCE ~= 0.6016.
"""
from __future__ import annotations

import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.data_loader import load_extended  # noqa: E402
from models.train import run_experiment  # noqa: E402

DATA_DIR = "kuairand-starter-kit/kuairand-starter-kit/KuaiRand-Pure/data"
OUT = os.path.join(_ROOT, "run_logs", "sweep_phase3d.jsonl")

MODELS = ["fm", "deepfm", "din"]
LOSSES = ["bce", "bpr", "listwise"]
BASE = dict(k=16, lr=1e-3, epochs=40, patience=8, bs=8192, seed=0, device="cuda")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("loading data...", flush=True)
    data = load_extended(DATA_DIR)
    results = []
    for model in MODELS:
        for loss in LOSSES:
            cfg = dict(BASE, model=model, loss=loss)
            t0 = time.time()
            print(f"\n=== {model} x {loss} ===", flush=True)
            res = run_experiment(data, cfg)
            row = {
                "model": model, "loss": loss,
                "valid": res["valid"], "test": res["test"],
                "best_valid": res["best_valid"], "epochs_run": res["epochs_run"],
                "wall_sec": round(time.time() - t0, 1),
            }
            results.append(row)
            with open(OUT, "a") as f:
                f.write(json.dumps(row, default=float) + "\n")
            print(f"  valid primary {res['valid']['primary']:.4f} | "
                  f"test primary {res['test']['primary']:.4f} | {row['wall_sec']}s", flush=True)

    print("\n=== SUMMARY (sorted by best valid primary) ===", flush=True)
    for row in sorted(results, key=lambda r: -r["best_valid"]):
        v = row["valid"]
        print(f"  {row['model']:7s} x {row['loss']:8s} | valid {v['primary']:.4f} "
              f"(GAUC {v['GAUC']:.4f} nDCG {v['nDCG@5']:.4f}) | test {row['test']['primary']:.4f} "
              f"| ep {row['epochs_run']}", flush=True)


if __name__ == "__main__":
    main()
