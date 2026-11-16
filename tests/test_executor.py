"""Executor: cooperative timeout, GPU-hour accounting, error classification."""
import numpy as np
import pytest

from agent import executor, tools
from agent.state import AgentState
from models import train


def tiny_data(n_users=6, per_user=8, seed=0):
    """Minimal data dict for FM/BCE: 5 id-like fields over a small vocabulary."""
    rng = np.random.default_rng(seed)

    def split():
        n = n_users * per_user
        X = rng.integers(0, 40, size=(n, 5)).astype(np.int64)
        X[:, 0] = np.repeat(np.arange(n_users), per_user)
        users = [str(u) for u in X[:, 0]]
        y = (rng.random(n) < 0.5).astype(np.float32)
        return {"X": X, "y": y, "users": users}

    return {"dim": 40, "n_fields": 5, "vside_dim": 0, "uside_dim": 0, "tag_dim": 0,
            "train": split(), "valid": split(), "test": split()}


CFG = {"model": "fm", "loss": "bce", "k": 4, "epochs": 6, "patience": 99, "bs": 16,
       "device": "cpu"}


def test_run_without_budget_runs_all_epochs():
    res = train.run_experiment(tiny_data(), CFG, verbose=False)
    assert res["epochs_run"] == 6 and res["timed_out"] is False


def test_time_budget_stops_after_one_epoch_and_still_scores(monkeypatch):
    clock = iter(np.arange(0.0, 1e6, 100.0))          # every time.time() call: +100 s
    monkeypatch.setattr(train.time, "time", lambda: float(next(clock)))
    res = train.run_experiment(tiny_data(), CFG, verbose=False, time_budget_s=50)
    assert res["timed_out"] is True
    assert res["epochs_run"] == 1                      # at least one epoch always completes
    assert np.isfinite(res["valid"]["primary"])


def test_execute_passes_timeout_and_flags_result(monkeypatch):
    seen = {}

    def fake(data, config, verbose=True, time_budget_s=None):
        seen["budget"] = time_budget_s
        m = {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55}
        return {"valid": m, "test": m, "timed_out": True, "device": "cpu"}

    monkeypatch.setattr(executor, "run_experiment", fake)
    out = executor.execute({}, {"model": "fm"}, timeout_s=123)
    assert seen["budget"] == 123
    assert out["timed_out"] is True and out["error"] is None
    assert out["gpu_h"] == 0.0                          # CPU run: no GPU-hours


def test_gpu_hours_only_on_cuda():
    assert executor._gpu_hours(3600, "cuda") == 1.0
    assert executor._gpu_hours(3600, "cuda:1") == 1.0
    assert executor._gpu_hours(3600, "cpu") == 0.0
    assert executor._gpu_hours(3600, None) == 0.0


def test_timeout_is_reported_to_agent_and_run_log(monkeypatch):
    m = {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55}
    monkeypatch.setattr(tools, "execute", lambda data, cfg, timeout_s=600: {
        "metrics": {"valid": m, "test": m}, "error": None, "wall_s": 1.0, "gpu_h": 0.0,
        "trace": "", "timed_out": True})
    text, rr = tools.run_experiment(AgentState(), {}, {"model": "fm"}, "h")
    assert "time limit" in text
    assert "timeout" in rr["recovery"]


@pytest.mark.parametrize("exc, expected", [
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "oom"),
    (MemoryError(), "oom"),
    (RuntimeError("mat1 and mat2 shapes cannot be multiplied (8x10 and 12x4)"), "shape"),
    (RuntimeError("The size of tensor a (3) must match the size of tensor b (4)"), "shape"),
    (RuntimeError("size mismatch for emb.weight"), "shape"),
    (ValueError("operands could not be broadcast together"), "shape"),
    (TypeError("forward() got an unexpected keyword argument 'cont'"), "syntax"),
    (ValueError("batch size must be positive"), "runtime"),   # "size" alone is not a shape error
    (KeyError("vside_dim"), "runtime"),
])
def test_classify_error(exc, expected):
    assert executor.classify_error(exc) == expected
