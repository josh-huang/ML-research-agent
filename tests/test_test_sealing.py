"""Test-split sealing: no agent-visible string may carry test metrics or test labels.

Test metrics are computed (for the run-log / human drift reporting) but must never reach
the LLM: the run_experiment tool result, the episode history line, read_run_log, the
episode preamble, the EDA system-prompt block, and the feature probes are all checked.
"""
import os

import numpy as np
import pandas as pd
import pytest

from agent import eda, inspect, memory, tools
from agent.state import AgentState
from models.data_loader import LOG_FILES

# Distinctive test-only numbers: if any shows up in an agent-facing string, test leaked.
TEST_METRICS = {"GAUC": 0.4321, "nDCG@5": 0.3579, "primary": 0.3950}
VALID_METRICS = {"GAUC": 0.6650, "nDCG@5": 0.5450, "primary": 0.6050}
LEAK_MARKERS = ("0.4321", "0.3579", "0.3950", "test GAUC", "test primary", "test  GAUC")


def _assert_sealed(text: str):
    for m in LEAK_MARKERS:
        assert m not in text, f"test metric leaked into agent-visible text: {m!r}"


def _fake_execute(data, config, timeout_s=600):
    return {"metrics": {"valid": dict(VALID_METRICS), "test": dict(TEST_METRICS)},
            "error": None, "wall_s": 1.0, "gpu_h": 1 / 3600, "trace": ""}


def _record(iteration=1):
    return {"iteration": iteration, "hypothesis": "h", "action": {"model": "din"},
            "metrics": {"valid": dict(VALID_METRICS), "test": dict(TEST_METRICS)},
            "verdict": "new_best", "errors": [], "lesson": "l"}


def test_run_experiment_result_has_valid_only(monkeypatch):
    monkeypatch.setattr(tools, "execute", _fake_execute)
    st = AgentState()
    text, run_record = tools.run_experiment(st, {}, {"model": "din", "loss": "bce"}, "h")
    assert "valid GAUC 0.6650" in text
    _assert_sealed(text)
    # The run record still carries test, for the run-log.
    assert run_record["metrics"]["test"]["primary"] == TEST_METRICS["primary"]


def test_run_experiment_reports_seed_group(monkeypatch):
    monkeypatch.setattr(tools, "execute", _fake_execute)
    st = AgentState()
    st.record(tools.search.normalize({"model": "din", "seed": 0}), 0.6040, 0, 0.0)
    text, _ = tools.run_experiment(st, {}, {"model": "din", "seed": 1}, "h")
    assert "same config across 2 seeds" in text
    assert "mean 0.6045" in text
    _assert_sealed(text)


def test_episode_line_read_run_log_and_preamble_are_sealed(monkeypatch):
    line = memory.episode_line("h", {"model": "din"}, 0.6050, "new_best", "l")
    _assert_sealed(line)
    monkeypatch.setattr(tools, "tail", lambda n: [_record(1), _record(2)])
    _assert_sealed(tools._read_run_log(5))
    st = AgentState()
    st.record({"model": "din", "seed": 0}, 0.6050, 0, 0.0)
    _assert_sealed(memory.build_preamble(st, [_record(1)], "- l", "- p"))


# --- data-level sealing (tiny synthetic CSVs) --------------------------------

TRAIN_DATE, VALID_DATE, TEST_DATE = 20220410, 20220425, 20220501


def _write_logs(d):
    """Two users; test rows carry an extreme label pattern so any leak is detectable."""
    rows = []
    t = 0
    for date, lv in ((TRAIN_DATE, [1, 0, 1, 0]), (VALID_DATE, [0, 1, 0, 1]),
                     (TEST_DATE, [1, 1, 1, 1])):
        for u in (1, 2):
            for i, y in enumerate(lv):
                t += 1000
                rows.append({
                    "user_id": u, "video_id": 10 * u + i, "date": date, "hourmin": 1200,
                    "time_ms": t, "is_click": y, "is_like": 0, "is_follow": 0,
                    "is_comment": 0, "is_forward": 0, "is_hate": 0, "long_view": y,
                    "play_time_ms": 9000 if y else 1000, "duration_ms": 10000,
                    "profile_stay_time": 0, "comment_stay_time": 0, "is_profile_enter": 0,
                    "is_rand": 0, "tab": 1})
    df = pd.DataFrame(rows)
    half = len(df) // 3
    df.iloc[:half].to_csv(os.path.join(d, LOG_FILES[0]), index=False)
    df.iloc[half:].to_csv(os.path.join(d, LOG_FILES[1]), index=False)
    pd.DataFrame({"video_id": sorted(df["video_id"].unique()), "author_id": 1}).to_csv(
        os.path.join(d, "video_features_basic_pure.csv"), index=False)


@pytest.fixture
def data_dir(tmp_path):
    _write_logs(str(tmp_path))
    return str(tmp_path)


def test_probe_dataframe_is_train_only(data_dir):
    ctx = inspect.build_inspect_context(data_dir)
    dates = ctx.df_light["date"].unique()
    assert set(dates) == {TRAIN_DATE}
    assert len(ctx.df_light) == 8


def test_eda_does_not_use_test_labels(data_dir):
    s = eda.compute_summary(data_dir)
    assert set(s["splits"]) == {"train", "valid"}
    assert s["n_test"] == 8
    assert set(s["users"]["all_negative_user_frac"]) == {"train", "valid"}
    # Train labels are 50% positive; including the all-positive test rows would shift it.
    click = next(b for b in s["binary_feedback_ranked"] if b["name"] == "is_click")
    assert click["pos_rate"] == pytest.approx(0.5)
    assert s["watch_time"]["censored_frac_full_watch"] == pytest.approx(0.0)
    md = eda.render_markdown(s)
    assert "labels sealed" in md
    assert np.isfinite(s["watch_time"]["point_biserial_play_time"])
