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
