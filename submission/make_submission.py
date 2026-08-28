"""Generate the final submission.csv from the best verified config.

Trains the DIN anchor (k=32, lr=3e-4, dropout=0.2, pointwise BCE) on the train split,
early-stopped on valid, and writes per-row test scores in the official submission format
(row_id, user_id, video_id, score).

Row-order safety: ``models.data_loader.load_extended`` asserts its row order matches the
official ``data.load`` order, so ``test_scores`` line up index-for-index with
``data.load()["test"]``.

Usage (from project root)::

    python submission/make_submission.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_KIT = os.path.join(_ROOT, "kuairand-starter-kit", "kuairand-starter-kit")
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)

from data import load  # noqa: E402
from submit import write_submission  # noqa: E402

from models.data_loader import load_extended  # noqa: E402
from models.train import run_experiment  # noqa: E402

DATA_DIR = os.path.join(_KIT, "KuaiRand-Pure", "data")
OUT = os.path.join(_ROOT, "submission", "final.csv")

# Best verified config (Phase 3d hyperparameter sweep): valid 0.6047 / test 0.5978.
BEST_CONFIG = dict(model="din", loss="bce", k=32, lr=3e-4, dropout=0.2,
                   dnn_hidden=(64, 32), seed=0)


def main() -> None:
    data = load_extended(DATA_DIR)
    res = run_experiment(data, BEST_CONFIG, verbose=True)
    test_rows = load(DATA_DIR)["test"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    write_submission(OUT, test_rows, res["test_scores"])
    t = res["test"]
    print(f"\nwrote {OUT}: {len(test_rows):,d} rows")
    print(f"test GAUC {t['GAUC']:.4f} | nDCG@5 {t['nDCG@5']:.4f} | primary {t['primary']:.4f}")


if __name__ == "__main__":
    main()
