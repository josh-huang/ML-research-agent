"""Unit tests for side-feature engineering (models.side_features)."""
import numpy as np
import pandas as pd
import pytest

from models.side_features import (
    CAT_USER_FEATURES, CAT_VIDEO_FEATURES, CONT_FEATURES,
    build_user_side_table, build_video_side_table)


def _basic_df():
    return pd.DataFrame({
        "video_id": ["v1", "v2", "v3"],
        "video_type": ["music", "short", "news"],
        "music_type": ["pop", "none", "jazz"],
        "video_duration": [0.0, 60.0, 120.0],
    })


def _stat_df():
    return pd.DataFrame({
        "video_id": ["v1", "v2", "v3"],
        "show_cnt": [100, 50, 0],          # v3 zero shows -> denominator clips to 1
        "play_progress": [0.5, 0.9, 0.0],
        "like_cnt": [50, 60, 0],           # v2 like_cnt > show_cnt -> clip to 1.0
        "comment_cnt": [10, 0, 0],
        "follow_cnt": [5, 1, 0],
        "share_cnt": [2, 0, 0],
    })


def _user_df():
    return pd.DataFrame({
        "user_id": ["u1", "u2"],
        "user_active_degree": ["high", "low"],
        "follow_user_num_range": ["a", "b"],
        "fans_user_num_range": ["a", "b"],
        "friend_user_num_range": ["a", "b"],
        "register_days_range": ["a", "b"],
        "register_days": [10, 20],         # unrelated col, must not leak in
    })


def test_build_video_side_table_columns_and_rates():
    t = build_video_side_table(_basic_df(), _stat_df())
    assert list(t.index) == ["v1", "v2", "v3"]
    assert list(t.columns) == list(CAT_VIDEO_FEATURES) + list(CONT_FEATURES)
    assert t.loc["v1", "video_type"] == "music"
    assert t.loc["v3", "music_type"] == "jazz"
    assert t.loc["v1", "like_rate"] == pytest.approx(0.5)
    assert t.loc["v2", "like_rate"] == pytest.approx(1.0)   # 60/50 -> clip
    assert t.loc["v3", "like_rate"] == pytest.approx(0.0)   # 0 / max(0,1) -> 0
    assert t.loc["v2", "log_duration"] == pytest.approx(np.log1p(60.0))


def test_build_user_side_table_columns():
    t = build_user_side_table(_user_df())
    assert list(t.index) == ["u1", "u2"]
    assert list(t.columns) == list(CAT_USER_FEATURES)   # register_days excluded
    assert t.loc["u1", "user_active_degree"] == "high"
