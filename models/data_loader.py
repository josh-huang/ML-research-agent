"""Extended data loading for Phase 3 models.

Extends the official 5-field encoding (reused verbatim via ``data.load``/``encode``)
with the signals the organizer's headroom map points at but the official loader drops:

  * ``click`` / ``like`` / ``profile`` — auxiliary feedback for multi-task learning.
    From EDA: ``is_click`` is ~46% positive (the only dense auxiliary signal),
    ``is_like`` ~1.9%, ``is_profile_enter`` ~2.4%. The remaining feedback signals
    (is_follow / is_comment / is_forward / is_hate) are <0.3% positive and useless
    as aux targets, so they are deliberately excluded.
  * ``play_time`` / ``duration`` — raw watch time for the censored watch-time (CWM)
    direction.
  * ``hist`` / ``hist_mask`` — per-row, the user's past K video ids, ordered by the
    absolute ``time_ms`` epoch timestamp (NOT ``date``/``hourmin``), the input to
    DIN's behavior-sequence attention.

Row order matches the official loader exactly (file1 then file2, split by date), so the
extended arrays line up index-for-index with the official ``X``/``y``/``users``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_KIT = os.path.join(_ROOT, "kuairand-starter-kit", "kuairand-starter-kit")
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)

from data import load, encode, SPLITS, FIELDS  # noqa: E402

LOG_FILES = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")
_AUX_COL = {
    "click": "is_click",
    "like": "is_like",
    "profile": "is_profile_enter",
    "play_time": "play_time_ms",
    "duration": "duration_ms",
}
AUX_NAMES = tuple(_AUX_COL)


def _load_full_df(data_dir: str) -> pd.DataFrame:
    """Both standard logs (original row order), author_id attached, all columns kept.

    author_id is attached via ``.map`` (order-preserving) rather than ``merge``, because
    pandas' hash-based ``merge`` does not guarantee the left frame's row order — which
    would break the row-order alignment this module relies on.
    """
    vid2author = pd.read_csv(os.path.join(data_dir, "video_features_basic_pure.csv"))
    vid2author = vid2author.set_index("video_id")["author_id"]
    df = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in LOG_FILES],
                   ignore_index=True)
    df["author_id"] = df["video_id"].map(vid2author).fillna("UNK")
    return df


def _build_history(df: pd.DataFrame, gvid: np.ndarray, K: int):
    """Per-row history of the user's past K global video-id indices, time-ordered.

    ``gvid`` is the global video_id index (== official ``X[:, 1]``), aligned to ``df``
    rows. Returns ``(hist, mask)`` shaped ``(N, K)`` in ``df``'s original row order.
    """
    n = len(df)
    hist = np.zeros((n, K), dtype=np.int64)
    mask = np.zeros((n, K), dtype=np.float32)
    # lexsort: primary key user_id, secondary key time_ms -> chronological per user.
    order = np.lexsort((df["time_ms"].values, df["user_id"].values))
    uids = df["user_id"].values[order]
    vids = gvid[order]
    rows = np.arange(n, dtype=np.int64)[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and uids[end] == uids[start]:
            end += 1
        for j in range(start, end):
            h = vids[max(start, j - K):j]
            row = rows[j]
            hist[row, :len(h)] = h
            mask[row, :len(h)] = 1.0
        start = end
    return hist, mask


def load_extended(data_dir: str, K: int = 50) -> dict:
    """Load extended ranking data for all models.

    Returns a dict with ``train``/``valid``/``test`` keys, each mapping to
    ``{X, y, users, aux, hist, hist_mask}`` where ``aux`` is a dict of ``{name: (N,)
    float32}`` for every name in :data:`AUX_NAMES`; plus top-level ``dim`` (total
    embedding dim), ``K`` and ``n_fields``.
    """
    df = _load_full_df(data_dir)

    splits = load(data_dir)
    enc, dim = encode(splits)
    n_train, n_valid, n_test = (len(splits[n]) for n in ("train", "valid", "test"))

    # Reorder df into [train, valid, test] to match the official per-split order,
    # preserving raw row order within each split (stable sort on the split label).
    # (The raw file2 interleaves valid/test dates; the official loader groups them.)
    date = df["date"].values
    split_label = np.where(date <= SPLITS["train"][1], 0,
                           np.where(date <= SPLITS["valid"][1], 1, 2))
    df_all = df.iloc[np.argsort(split_label, kind="stable")].reset_index(drop=True)

    users_all = enc["train"][2] + enc["valid"][2] + enc["test"][2]
    assert len(df_all) == len(users_all), "full df length != official loader length"
    assert df_all["user_id"].astype(str).tolist() == users_all, "row-order mismatch vs official loader"

    gvid_all = np.concatenate([enc[n][0][:, 1] for n in ("train", "valid", "test")]).astype(np.int64)
    hist_all, mask_all = _build_history(df_all, gvid_all, K)

    data = {"dim": dim, "K": K, "n_fields": len(FIELDS)}
    bounds = {"train": (0, n_train),
              "valid": (n_train, n_train + n_valid),
              "test": (n_train + n_valid, n_train + n_valid + n_test)}
    for name in ("train", "valid", "test"):
        s, e = bounds[name]
        X, y, users = enc[name]
        aux = {k: df_all[_AUX_COL[k]].to_numpy(dtype=np.float32)[s:e] for k in AUX_NAMES}
        data[name] = {
            "X": X, "y": y, "users": users, "aux": aux,
            "hist": hist_all[s:e], "hist_mask": mask_all[s:e],
        }
    return data
