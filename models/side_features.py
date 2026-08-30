"""Side-information feature engineering (item-side + user-side levers).

The official 5-field encoding is id-only; this module builds the organizer's video-side
and user-side signals into tensors a model can consume:

  * categorical side (``X_vside`` / ``X_uside``): sparse fields, offset-encoded (per-field
    vocab + UNK slot) so they merge into the shared embedding.
  * continuous side (``cont``): 6 video item-quality features, z-scored on TRAIN only.

Strict leakage口径: excludes ``complete_play_cnt`` / ``long_time_play_cnt`` /
``valid_play_cnt`` / ``play_cnt`` / raw ``show_cnt`` (near-label aggregates).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_VIDEO_FEATURES = ("video_type", "music_type")
CAT_USER_FEATURES = ("user_active_degree", "follow_user_num_range",
                     "fans_user_num_range", "friend_user_num_range", "register_days_range")
CONT_FEATURES = ("play_progress", "like_rate", "comment_rate",
                 "follow_rate", "share_rate", "log_duration")


def build_video_side_table(basic_df: pd.DataFrame, stat_df: pd.DataFrame) -> pd.DataFrame:
    """video_id-indexed table of categorical + continuous video side features (pure).

    ``basic_df``: columns include video_id, video_type, music_type, video_duration.
    ``stat_df``:  columns include video_id, show_cnt, play_progress, like_cnt,
                  comment_cnt, follow_cnt, share_cnt.
    """
    basic = basic_df.set_index("video_id")
    stat = stat_df.set_index("video_id")
    show = stat["show_cnt"].astype(float).clip(lower=1.0)
    return pd.DataFrame({
        "video_type": basic["video_type"],
        "music_type": basic["music_type"],
        "play_progress": stat["play_progress"].astype(float),
        "like_rate": (stat["like_cnt"].astype(float) / show).clip(0.0, 1.0),
        "comment_rate": (stat["comment_cnt"].astype(float) / show).clip(0.0, 1.0),
        "follow_rate": (stat["follow_cnt"].astype(float) / show).clip(0.0, 1.0),
        "share_rate": (stat["share_cnt"].astype(float) / show).clip(0.0, 1.0),
        "log_duration": np.log1p(basic["video_duration"].astype(float)),
    })


def build_user_side_table(user_df: pd.DataFrame) -> pd.DataFrame:
    """user_id-indexed categorical user-side table (pure)."""
    return user_df.set_index("user_id")[list(CAT_USER_FEATURES)]


def encode_cat_fields(table: pd.DataFrame, cols, keys, n_train: int):
    """Encode categorical columns ``cols`` of a key-indexed ``table`` for ``keys``.

    Vocab is built from the leading ``n_train`` rows only; unseen keys (or NaN) map to
    the field's UNK slot. Returns ``(X_cat, side_dim, n_fields)``: ``X_cat`` is
    (N, len(cols)) int32 with global offsets (starting at 0 — the caller adds the
    official ``dim`` when concatenating with ``X``).
    """
    cat = table[list(cols)].reindex(keys)
    X = np.empty((len(keys), len(cols)), dtype=np.int32)
    offset = 0
    dims = []
    for j, col in enumerate(cols):
        train_vals = cat[col].values[:n_train]
        vocab = {v: i for i, v in enumerate(pd.unique(train_vals))}
        unk = len(vocab)
        codes = cat[col].map(vocab).fillna(unk).astype(np.int32).values
        X[:, j] = codes + offset
        offset += len(vocab) + 1
        dims.append(len(vocab) + 1)
    return X, int(sum(dims)), len(cols)


def encode_cont_fields(table: pd.DataFrame, cols, keys, n_train: int):
    """Z-score continuous columns ``cols`` using TRAIN-only mean/std.

    Returns ``(cont, cont_dim)``; ``cont`` is (N, len(cols)) float32. Unseen keys are
    filled with 0 in the raw value (so they normalize to ``-mean/std``); constant
    columns get std=1 (no division by zero).
    """
    raw = table[list(cols)].reindex(keys).to_numpy(dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0)
    mean = raw[:n_train].mean(axis=0)
    std = raw[:n_train].std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    cont = ((raw - mean) / std).astype(np.float32)
    return cont, len(cols)
