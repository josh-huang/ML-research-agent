"""AgentState tests: seed-mean best-tracking, verdicts, convergence, legacy load."""
import json

import pytest

from agent import state as state_mod
from agent.state import EPS, N_CONVERGE, AgentState

BASE = {"model": "din", "loss": "bce", "k": 32, "lr": 3e-4, "dnn_hidden": (64, 32)}


def cfg(seed=0, **kw):
    return dict(BASE, seed=seed, **kw)


def rec(st, config, primary, error=False):
    return st.record(config, primary, tokens=0, gpu_h=0.0, error=error)


def test_group_key_ignores_seed_full_key_does_not():
    st = AgentState()
    assert st.group_key(cfg(0)) == st.group_key(cfg(1))
    assert st.config_key(cfg(0)) != st.config_key(cfg(1))


def test_new_seed_is_not_a_duplicate():
    st = AgentState()
    rec(st, cfg(0), 0.60)
    assert st.seen(cfg(0))
    assert not st.seen(cfg(1))


def test_best_is_seed_mean_and_can_be_revised_down():
    st = AgentState()
    assert rec(st, cfg(0), 0.6050) == "new_best"
    assert st.best_primary == pytest.approx(0.6050)
    # A weaker second seed drags the group mean down: the best is revised, not kept.
    assert rec(st, cfg(1), 0.6040) == "reject"
    assert st.best_primary == pytest.approx(0.6045)
    assert st.best_n_seeds == 2
    assert st.best_std == pytest.approx(0.0005)
    assert st.best_iter == 2


def test_lucky_single_seed_loses_to_better_mean_elsewhere():
    st = AgentState()
    a, b = cfg(0), cfg(0, dropout=0.2)
    rec(st, a, 0.6050)                  # lucky single seed of config A
    rec(st, b, 0.6048)                  # config B, seed 0
    rec(st, dict(b, seed=1), 0.6048)    # config B, seed 1 -> mean 0.6048
    rec(st, dict(a, seed=1), 0.6040)    # A's second seed -> A mean 0.6045 < B
    assert st.best_primary == pytest.approx(0.6048)
    assert st.best_config["dropout"] == 0.2
    assert st.best_config["seed"] == 0
    assert st.best_iter == 3


def test_verdicts_and_convergence_counting():
    st = AgentState()
    assert rec(st, cfg(0), 0.60) == "new_best"
    assert st.stagnant == 0
    assert rec(st, cfg(0, k=16), 0.60 + EPS / 2) == "accept"   # positive but < eps
    assert rec(st, cfg(0, k=8), 0.50) == "reject"
    assert not st.converged
    assert rec(st, cfg(0, k=4), 0.0, error=True) == "error"
    assert st.stagnant == N_CONVERGE and st.converged
    assert rec(st, cfg(0, k=64), 0.60 + 2 * EPS) == "new_best"  # >= eps resets
    assert st.stagnant == 0


def test_errors_are_excluded_from_groups():
    st = AgentState()
    rec(st, cfg(0), 0.0, error=True)
    assert st.groups() == {}
    assert st.best_config is None
    rec(st, cfg(1), 0.60)
    g = st.group_of(cfg(5))
    assert g["n"] == 1 and g["seeds"] == {1: 0.60}


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(state_mod, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(state_mod, "LOG_DIR", str(tmp_path))
    st = AgentState()
    rec(st, cfg(0), 0.6050)
    rec(st, cfg(1), 0.6040)
    st.save()
    st2 = AgentState.load()
    assert st2.best_primary == pytest.approx(0.6045)
    assert st2.best_n_seeds == 2
    assert st2.best_iter == 2
    assert st2.iterations == 2


def test_legacy_state_is_rederived_to_seed_mean(tmp_path, monkeypatch):
    """Old state.json: single-seed max as best, no seen_iters."""
    path = tmp_path / "state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", str(path))
    key = lambda c: json.dumps(c, sort_keys=True, default=float)  # noqa: E731
    path.write_text(json.dumps({
        "best_primary": 0.6053, "best_iter": 2, "best_config": cfg(0), "iterations": 3,
        "seen_configs": {key(cfg(0)): 0.6053, key(cfg(1)): 0.6045, key(cfg(2)): 0.6049},
    }))
    st = AgentState.load()
    assert st.best_primary == pytest.approx((0.6053 + 0.6045 + 0.6049) / 3)
    assert st.best_n_seeds == 3
    assert st.best_iter == -1  # unknown for legacy files without seen_iters
