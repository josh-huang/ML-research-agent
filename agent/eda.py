"""Compact data-driven EDA summary for the researcher LLM.

Computes the evidence the agent's researcher needs to rank the organizer's headroom
map against the actual data, and returns it as a small dict / markdown block (token
frugality matters: this is injected into every LLM turn, so it must stay tight).

Signals examined (label = long_view, 0/1):
  * binary feedback (is_click/is_like/is_follow/is_comment/is_forward/is_hate/is_profile_enter)
    -> positive rate + phi correlation with long_view (ranked; only dense/useful ones matter)
  * watch time (play_time_ms vs duration_ms) -> censored-fraction + point-biserial corr
  * per-user impression counts + all-negative user share (bounds nDCG@5)
  * is_rand prevalence (dead in standard logs -> log_random is the unbiased-valid path)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.data_loader import _load_full_df  # noqa: E402
from models.data_loader import SPLITS, LOG_FILES  # noqa: E402

_BINARY = ["is_click", "is_like", "is_follow", "is_comment",
           "is_forward", "is_hate", "is_profile_enter"]
_CONT = ["play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time"]


def _phi(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    n11 = int((a & b).sum())
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n00 = int((~a & ~b).sum())
    denom = ((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01)) ** 0.5
    return float((n11 * n00 - n10 * n01) / denom) if denom > 0 else 0.0


def _point_biserial(a: np.ndarray, b: np.ndarray) -> float:
    a0, a1 = a[b == 0], a[b == 1]
    if len(a0) == 0 or len(a1) == 0:
        return 0.0
    n, n1 = len(a), len(a1)
    s = a.std(ddof=0)
    if s == 0:
        return 0.0
    return float((a1.mean() - a0.mean()) * np.sqrt(n1 * (n - n1)) / (s * n))


def compute_summary(data_dir: str) -> dict:
    df = _load_full_df(data_dir)
    y = df["long_view"].to_numpy()
    date = df["date"].to_numpy()

    # split labels: train / valid / test by date, matching the official loader.
    label = np.where(date <= SPLITS["train"][1], "train",
                     np.where(date <= SPLITS["valid"][1], "valid", "test"))

    # 1. label rate per split
    splits = {}
    for sp in ("train", "valid", "test"):
        m = label == sp
        splits[sp] = {
            "n": int(m.sum()),
            "long_view_rate": float(y[m].mean()),
        }

    # 2. binary feedback: positive rate + phi vs long_view
    binary = []
    for c in _BINARY:
        v = df[c].to_numpy()
        binary.append({
            "name": c,
            "pos_rate": float(v.mean()),
            "phi_long_view": round(_phi(v, y), 4),
        })
    binary.sort(key=lambda d: -d["phi_long_view"])

    # 3. watch time: censored fraction + point-biserial
    pt = df["play_time_ms"].to_numpy().astype(np.float64)
    dur = df["duration_ms"].to_numpy().astype(np.float64)
    watch = {
        "play_time_median_ms": float(np.median(pt)),
        "censored_frac_full_watch": float((pt >= dur).mean()),  # watched to the end
        "point_biserial_play_time": round(_point_biserial(pt, y), 4),
    }

    # 4. per-user impression counts + all-negative share (bounds nDCG@5)
    uid = df["user_id"].to_numpy()
    uid_sorted = np.sort(uid)
    change = np.where(uid_sorted[1:] != uid_sorted[:-1])[0] + 1
    counts = np.diff(np.concatenate([[0], change, [len(uid_sorted)]]))

    def _user_neg_frac(sp):
        m = label == sp
        # fraction of users (in this split) with zero positives
        d = pd.DataFrame({"u": uid[m], "y": y[m]})
        npos = d.groupby("u")["y"].sum()
        return float((npos == 0).mean())

    users = {
        "n_users_total": int(len(counts)),
        "impressions_per_user": {
            "mean": round(float(counts.mean()), 1),
            "median": int(np.median(counts)),
            "p90": int(np.percentile(counts, 90)),
            "max": int(counts.max()),
        },
        "all_negative_user_frac": {sp: round(_user_neg_frac(sp), 4) for sp in ("train", "valid", "test")},
    }

    # 5. is_rand prevalence (dead in standard logs)
    rand = {
        "is_rand_rate": float(df["is_rand"].mean()),
        "is_rand_rate_valid": float(df["is_rand"].to_numpy()[label == "valid"].mean()),
    }

    return {
        "splits": splits,
        "binary_feedback_ranked": binary,
        "watch_time": watch,
        "users": users,
        "is_rand": rand,
        "label": "long_view (0/1), within-user ranking",
        "metrics": "GAUC + nDCG@5, primary = mean",
    }


def render_markdown(s: dict) -> str:
    lines = [
        f"- label: long_view; metrics GAUC+nDCG@5",
        f"- long_view rate: " + " | ".join(
            f"{sp} {v['long_view_rate']:.3f} (n={v['n']})" for sp, v in s["splits"].items()),
        f"- feedback phi vs long_view (ranked): " + " | ".join(
            f"{d['name']}={d['phi_long_view']}({d['pos_rate']:.3f})" for d in s["binary_feedback_ranked"]),
        f"- watch: median play {s['watch_time']['play_time_median_ms']:.0f}ms, "
        f"full-watch {s['watch_time']['censored_frac_full_watch']:.3f}, "
        f"corr(play_time,long_view)={s['watch_time']['point_biserial_play_time']}",
        f"- users: {s['users']['n_users_total']}, imp/user med {s['users']['impressions_per_user']['median']} "
        f"p90 {s['users']['impressions_per_user']['p90']} max {s['users']['impressions_per_user']['max']}",
        f"- all-negative user frac: " + " | ".join(
            f"{sp} {v:.3f}" for sp, v in s["users"]["all_negative_user_frac"].items()),
        f"- is_rand: {s['is_rand']['is_rand_rate']:.4f} (valid {s['is_rand']['is_rand_rate_valid']:.4f}) -> dead",
    ]
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="kuairand-starter-kit/kuairand-starter-kit/KuaiRand-Pure/data")
    ap.add_argument("--json", action="store_true", help="dump JSON instead of markdown")
    a = ap.parse_args()
    s = compute_summary(a.data_dir)
    if a.json:
        print(json.dumps(s, indent=2))
    else:
        print(render_markdown(s))


if __name__ == "__main__":
    main()
