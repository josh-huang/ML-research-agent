"""LLM failure policy: retry -> tolerate -> degrade -> re-probe -> recover (no API calls)."""
import json
import random

import pytest

from agent import logger, main, memory, state as state_mod, tools
from agent.main import LLMHealth
from agent.react import EpisodeResult, ReActAgent
from agent.state import AgentState

VALID = {"GAUC": 0.66, "nDCG@5": 0.54, "primary": 0.60}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect every file the loop writes and stub out training."""
    monkeypatch.setattr(state_mod, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(state_mod, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "RUN_LOG_PATH", str(tmp_path / "run_log.jsonl"))
    monkeypatch.setattr(memory, "PLAYBOOK_PATH", str(tmp_path / "playbook.md"))
    monkeypatch.setattr(tools, "execute", lambda data, cfg, timeout_s=600: {
        "metrics": {"valid": dict(VALID), "test": dict(VALID)},
        "error": None, "wall_s": 0.0, "gpu_h": 0.0, "trace": ""})
    return tmp_path


def _log(tmp_path):
    return [json.loads(l) for l in (tmp_path / "run_log.jsonl").read_text().splitlines()]


class ScriptedReact:
    """Fake agent: each episode fails or succeeds per the script (True = fail)."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def run_episode(self, ctx):
        fail = self.script[self.calls] if self.calls < len(self.script) else False
        self.calls += 1
        if fail:
            return EpisodeResult(failed=True, llm_error="APIConnectionError: down")
        cfg = {"model": "din", "seed": 100 + self.calls}
        _, rr = tools.run_experiment(ctx.state, ctx.data, cfg, "llm hypothesis")
        tools.finalize_run(ctx.state, rr, tokens=10, lesson="")
        return EpisodeResult(ran=True)


def _run(react, max_iters):
    st = AgentState()
    ctx = tools.Ctx(state=st, data={})
    health = LLMHealth(available=True)
    main.run_loop(st, ctx, react, random.Random(0), max_iters, health)
    return st, health


# --- policy unit tests -------------------------------------------------------

def test_health_tolerates_isolated_failures():
    h = LLMHealth(available=True, fail_limit=3)
    h.record(True); h.record(True)
    assert h.should_try() and not h.degraded
    h.record(False)                       # a success resets the streak
    assert h.consecutive_failures == 0


def test_health_degrades_then_reprobes_and_recovers():
    h = LLMHealth(available=True, fail_limit=2, reprobe_every=3)
    h.record(True); h.record(True)
    assert h.degraded and not h.should_try()
    h.record_fallback(); h.record_fallback()
    assert not h.should_try()
    h.record_fallback()
    assert h.should_try()                 # periodic re-probe
    h.record(True)                        # probe fails -> wait another window
    assert not h.should_try()
    for _ in range(3):
        h.record_fallback()
    h.record(False)                       # probe succeeds
    assert not h.degraded and h.should_try()


def test_health_unavailable_never_tries():
    assert not LLMHealth(available=False).should_try()


# --- loop integration --------------------------------------------------------

def test_single_failure_does_not_disable_llm(sandbox):
    react = ScriptedReact([True, False, False, False])
    st, health = _run(react, max_iters=4)
    assert st.iterations == 4
    assert react.calls == 4               # LLM tried every iteration
    recs = _log(sandbox)
    assert recs[0]["hypothesis"].startswith("[no-llm]")       # failure covered by fallback
    assert recs[0]["llm_error"] == "APIConnectionError: down"  # ...and the cause is logged
    assert all(r["hypothesis"] == "llm hypothesis" and r["llm_error"] is None
               for r in recs[1:])


def test_outage_degrades_then_llm_takes_back_over(sandbox):
    # 3 failures -> degraded; 3 fallback iters; probe fails; 3 more; probe succeeds.
    react = ScriptedReact([True, True, True, True, False, False])
    st, health = _run(react, max_iters=10)
    assert st.iterations == 10
    recs = _log(sandbox)
    kinds = ["F" if r["hypothesis"].startswith("[no-llm]") else "L" for r in recs]
    #  iters 1-3: failed episodes (fallback) | 4-5: degraded | 6: failed probe (fallback)
    #  7-8: degraded | 9: probe succeeds | 10: LLM again
    assert kinds == ["F"] * 8 + ["L", "L"]
    assert react.calls == 6
    assert not health.degraded


def test_episode_exception_is_caught_and_logged(sandbox):
    class Exploding:
        calls = 0
        def run_episode(self, ctx):
            self.calls += 1
            raise RuntimeError("bug in agent")
    st, health = _run(Exploding(), max_iters=2)
    assert st.iterations == 2
    assert "RuntimeError: bug in agent" in _log(sandbox)[0]["llm_error"]


# --- ReActAgent surfaces the cause instead of swallowing it ------------------

class DownLLM:
    def complete(self, *a, **k):
        raise ConnectionError("proxy  refused\n connection")
    tool_uses = staticmethod(lambda content: [])
    tool_use = staticmethod(lambda content: {})


def test_react_reports_llm_error(sandbox):
    ep = ReActAgent(DownLLM(), []).run_episode(tools.Ctx(state=AgentState(), data={}))
    assert ep.failed and not ep.ran
    assert ep.llm_error == "ConnectionError: proxy refused connection"


def test_client_is_configured_with_retries(monkeypatch):
    from agent import llm
    monkeypatch.setattr(llm, "_ENV", {"ANTHROPIC_API_KEY": "sk-test-not-a-real-key"})
    client = llm.ClaudeClient().client
    assert client.max_retries == llm.MAX_RETRIES >= 4
    assert client.timeout == llm.REQUEST_TIMEOUT_S
