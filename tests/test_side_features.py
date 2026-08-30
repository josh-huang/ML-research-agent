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


from models.side_features import encode_cat_fields, encode_cont_fields


def test_encode_cat_fields_offsets_and_unk():
    table = pd.DataFrame({
        "video_type": ["music", "short"],
        "music_type": ["pop", "none"],
    }, index=["v1", "v2"])
    keys = np.array(["v1", "v2", "v1", "v_new"])   # v1 repeats; v_new unseen
    X, side_dim, n_fields = encode_cat_fields(table, ["video_type", "music_type"], keys, n_train=2)

    assert X.shape == (4, 2)
    assert n_fields == 2
    # video_type vocab from train = {music:0, short:1}; UNK = 2
    assert X[0, 0] == 0          # v1 music
    assert X[1, 0] == 1          # v2 short
    assert X[3, 0] == 2          # v_new -> UNK
    # music_type offset starts after video_type dim (3)
    assert X[0, 1] == 3          # v1 pop -> 0 + 3
    assert X[1, 1] == 4          # v2 none -> 1 + 3
    assert side_dim == 6         # video_type(3) + music_type(3)


def test_encode_cont_fields_train_only():
    table = pd.DataFrame({"play_progress": [0.5, 0.9]}, index=["v1", "v2"])
    keys = np.array(["v1", "v2", "v1"])
    cont, cont_dim = encode_cont_fields(table, ["play_progress"], keys, n_train=2)

    assert cont_dim == 1
    # train mean 0.7, std 0.2 -> (0.5 - 0.7)/0.2 = -1.0
    assert cont[0, 0] == pytest.approx(-1.0)
    assert cont[0, 0] == pytest.approx(cont[2, 0])   # same video -> same value
