"""Generate the final submission.csv from the agent's confirmed-best config.

The config is read from ``run_logs/state.json``: the config group with the highest
seed-mean valid primary among those run with >= ``MIN_CONFIRM_SEEDS`` seeds
(``AgentState.confirmed_best``). Selection is therefore on valid only — test is scored
exactly once here, for the submitted model. ``--config '<json>'`` overrides it.

The model is trained on the train split, early-stopped on valid, and per-row test scores
are written in the official submission format (row_id, user_id, video_id, score).

Row-order safety: ``models.data_loader.load_extended`` asserts its row order matches the
official ``data.load`` order, so ``test_scores`` line up index-for-index with
``data.load()["test"]``.

Usage (from project root)::

    python submission/make_submission.py
    python submission/make_submission.py --config '{"model": "din", "k": 32}'
"""
from __future__ import annotations

import argparse
import json
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

from agent import search  # noqa: E402
from agent.state import MIN_CONFIRM_SEEDS, AgentState  # noqa: E402
from models.data_loader import load_extended  # noqa: E402
from models.train import run_experiment  # noqa: E402

DATA_DIR = os.path.join(_KIT, "KuaiRand-Pure", "data")
OUT = os.path.join(_ROOT, "submission", "final.csv")

def select_config(override: str | None) -> tuple[dict, str]:
    """(config, provenance): the --config override, else the confirmed best in state.json."""
    if override:
        return search.normalize(json.loads(override)), "--config override"
    best = AgentState.load().confirmed_best()
    if best is None:
        raise SystemExit(f"no confirmed best in run_logs/state.json (needs a config run with "
                         f">= {MIN_CONFIRM_SEEDS} seeds); run more seeds or pass --config.")
    return best["config"], (f"confirmed best from state.json: valid primary "
                            f"{best['primary']:.4f} ± {best['std']:.4f} over "
                            f"{best['n_seeds']} seeds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="JSON config overriding state.json")
    config, provenance = select_config(ap.parse_args().config)
    print(f"{provenance}\nconfig: {json.dumps(config, sort_keys=True)}", flush=True)
    data = load_extended(DATA_DIR)
    res = run_experiment(data, config, verbose=True)
    test_rows = load(DATA_DIR)["test"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    write_submission(OUT, test_rows, res["test_scores"])
    t = res["test"]
    print(f"\nwrote {OUT}: {len(test_rows):,d} rows")
    print(f"test GAUC {t['GAUC']:.4f} | nDCG@5 {t['nDCG@5']:.4f} | primary {t['primary']:.4f}")


if __name__ == "__main__":
    main()
