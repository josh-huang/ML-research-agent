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


def test_confirmed_best_requires_min_seeds():
    from agent.state import MIN_CONFIRM_SEEDS
    st = AgentState()
    a, b = cfg(0), cfg(0, dropout=0.2)
    rec(st, a, 0.6052)                      # 1-seed lead
    for s, p in enumerate([0.6049, 0.6046, 0.6052][:MIN_CONFIRM_SEEDS]):
        rec(st, dict(b, seed=s), p)
    assert st.best_primary == pytest.approx(0.6052)       # unconfirmed lead drives verdicts
    c = st.confirmed_best()
    assert c["n_seeds"] == MIN_CONFIRM_SEEDS
    assert c["config"]["dropout"] == 0.2 and c["config"]["seed"] == 0
    assert c["primary"] == pytest.approx(0.6049)
    assert st.to_dict()["confirmed_best"] == c


def test_confirmed_best_none_until_enough_seeds():
    st = AgentState()
    rec(st, cfg(0), 0.60)
    rec(st, cfg(1), 0.60)
    assert st.confirmed_best() is None


# --- persistence: atomic save, resume continuity, interventions (#6) ---------

@pytest.fixture
def state_path(tmp_path, monkeypatch):
    monkeypatch.setattr(state_mod, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(state_mod, "LOG_DIR", str(tmp_path))
    return tmp_path / "state.json"


def test_save_is_atomic_and_leaves_no_temp_files(state_path, monkeypatch):
    st = AgentState()
    rec(st, cfg(0), 0.60)
    st.save()
    good = state_path.read_text()
    # A crash mid-write must leave the previous file intact and no temp debris.
    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(state_mod.json, "dump", boom)
    rec(st, cfg(1), 0.61)
    with pytest.raises(RuntimeError):
        st.save()
    assert state_path.read_text() == good
    assert [p.name for p in state_path.parent.iterdir()] == ["state.json"]


def test_resume_continues_elapsed_and_interventions(state_path):
    st = AgentState()
    st.started_at -= 100.0          # pretend the run has been active for 100 s
    st.record_intervention("edited playbook")
    st.save()
    st2 = AgentState.load()
    assert st2.interventions == 1
    assert st2.intervention_log[0]["reason"] == "edited playbook"
    assert 100.0 <= st2.to_dict()["elapsed_s"] < 105.0   # continued, not reset to 0


def test_from_dict_roundtrips_to_dict(state_path):
    st = AgentState()
    rec(st, cfg(0), 0.6050)
    rec(st, cfg(1), 0.6040)
    st.record_intervention("x")
    d = st.to_dict()
    d2 = AgentState.from_dict(d).to_dict()
    for k in d:
        if k != "elapsed_s":
            assert d2[k] == d[k], k


def test_legacy_token_split_stays_unknown(state_path):
    state_path.write_text(json.dumps({"iterations": 7, "tokens_used": 383772}))
    st = AgentState.load()
    assert st.tokens_input is None and st.tokens_output is None
    rec(st, cfg(0), 0.60)           # resuming must not crash or invent a split
    assert st.tokens_input is None
    assert st.to_dict()["tokens_input"] is None


def test_start_state_counts_resume_and_declared_interventions(state_path, monkeypatch, tmp_path):
    from agent import main
    monkeypatch.setattr(main, "RUN_LOG_PATH", str(tmp_path / "run_log.jsonl"))
    fresh = main.start_state(True, 0, 0.0)
    assert fresh.interventions == 0                    # a fresh start is not an intervention
    rec(fresh, cfg(0), 0.60)
    fresh.save()
    resumed = main.start_state(False, 0, 0.0, ["raised max_iters"])
    assert resumed.interventions == 2
    assert "manual resume" in resumed.intervention_log[0]["reason"]
    assert AgentState.load().interventions == 2        # persisted immediately
