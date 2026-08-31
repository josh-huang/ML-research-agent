"""Probe-tool tests: leakage gate, catalog, rank-relevance (no data IO)."""
import numpy as np

from agent import inspect


def test_is_leaky_rejects_aggregates():
    for f in ("show_cnt", "play_cnt", "complete_play_cnt", "valid_play_user_num",
              "long_time_play_cnt", "short_time_play_user_num", "counts", "play_time_ms"):
        assert inspect._is_leaky(f), f


def test_is_leaky_allows_open_fields():
    for f in ("video_type", "server_width", "tag", "register_days", "is_click",
              "duration_ms", "hourmin", "is_lowactive_period", "play_progress"):
        assert not inspect._is_leaky(f), f


def test_list_features_highlights_unencoded_and_leakage():
    text = inspect.list_features()
    assert "upload_type" in text           # an un-encoded video field
    assert "register_days" in text         # an un-encoded user field
    assert "leakage" in text.lower() or "Leakage" in text
    assert "onehot_feat" in text


def test_probe_rejects_leakage_without_reading_df():
    ctx = inspect.InspectContext("/nonexistent")
    assert inspect.probe_feature(ctx, "complete_play_cnt").startswith("BLOCKED")


def test_probe_rejects_unknown_without_reading_df():
    ctx = inspect.InspectContext("/nonexistent")
    assert "unknown field" in inspect.probe_feature(ctx, "not_a_real_field")


def test_propose_rejects_unknown_op_without_reading_df():
    ctx = inspect.InspectContext("/nonexistent")
    assert "unknown feature op" in inspect.propose_feature(ctx, "nope")


def test_rank_relevance_per_user_constant_is_zero():
    uids = np.array(["a", "a", "a", "b", "b", "b"])
    v = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])  # constant within each user
    assert inspect._rank_relevance(v, uids) == 0.0


def test_rank_relevance_item_varying_is_high():
    uids = np.array(["a", "a", "b", "b"])
    v = np.array([1.0, 2.0, 1.0, 2.0])  # varies across items within each user
    assert inspect._rank_relevance(v, uids) > 0.9
