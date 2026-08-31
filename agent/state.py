"""Agent run state: best-so-far, convergence, budget, and config-dedup cache.

Persistence model: ``run_logs/state.json`` is a single small JSON file the dashboard
polls every tick and the agent rewrites after each iteration. All numeric metrics are
native Python floats so they JSON-serialize cleanly.
"""
from __future__ import annotations

import json
import os
import time

from agent.logger import LOG_DIR, git_sha

STATE_PATH = os.path.join(LOG_DIR, "state.json")

# convergence FLAG: set when N consecutive iterations improve valid primary by < eps.
# EPS/N are the locked competition口径 (ε=0.002, N=3) — do NOT change. The flag is for
# REPORTING only; it does not stop the loop (AgentState.done stops on budget only, so the
# agent keeps exploring past first-convergence up to max_iters).
EPS = 0.002
N_CONVERGE = 3


class AgentState:
    """In-memory state, mirrored to state.json on every mutation."""

    def __init__(self, best_primary: float = 0.0, budget_tokens: int = 0,
                 budget_gpu_h: float = 0.0):
        self.best_primary = best_primary
        self.best_iter = -1
        self.best_config = None
        self.stagnant = 0            # consecutive < eps improvements
        self.converged = False
        self.iterations = 0
        self.interventions = 0       # human interventions (autonomy score)
        self.tokens_used = 0
        self.tokens_input = 0        # input-side tokens (input + cache-write + cache-read)
        self.tokens_output = 0       # generated tokens
        self.git_sha = git_sha()     # run-start code commit (D3 traceability)
        self.gpu_hours = 0.0
        self.seen_configs = {}       # config-key -> best primary (dedup cache)
        self.budget_tokens = budget_tokens
        self.budget_gpu_h = budget_gpu_h
        self.started_at = time.time()
        self.errors = 0

    # -- config dedup -----------------------------------------------------
    def config_key(self, config: dict) -> str:
        return json.dumps(config, sort_keys=True, default=float)

    def seen(self, config: dict) -> bool:
        return self.config_key(config) in self.seen_configs

    def remember(self, config: dict, primary: float) -> None:
        self.seen_configs[self.config_key(config)] = float(primary)

    # -- update -----------------------------------------------------------
    def record(self, config: dict, valid_primary: float, test_primary: float,
               tokens: int, gpu_h: float, error: bool = False,
               tokens_input: int = 0, tokens_output: int = 0) -> str:
        """Record an iteration; returns 'error' | 'new_best' | 'accept' | 'reject'.

        Best-tracking is decoupled from convergence: *any* positive delta updates the
        tracked best, while only a delta >= EPS resets the stagnation counter. This way
        a genuine +0.0018 (DIN over baseline) is recorded as the best, but still counts
        toward the "3 consecutive sub-eps rounds -> converged" rule.
        """
        self.iterations += 1
        self.tokens_used += tokens
        self.tokens_input += tokens_input
        self.tokens_output += tokens_output
        self.gpu_hours += gpu_h
        if error:
            self.errors += 1
            self.stagnant += 1
            self.remember(config, -1.0)
            self.converged = self.stagnant >= N_CONVERGE
            return "error"

        delta = valid_primary - self.best_primary
        self.remember(config, valid_primary)
        if delta > 0:
            self.best_primary = float(valid_primary)
            self.best_iter = self.iterations
            self.best_config = config
        if delta >= EPS:
            self.stagnant = 0
            verdict = "new_best"
        else:
            self.stagnant += 1
            verdict = "accept" if delta > 0 else "reject"

        self.converged = self.stagnant >= N_CONVERGE
        return verdict

    # -- budget -----------------------------------------------------------
    def budget_exhausted(self) -> bool:
        if self.budget_tokens and self.tokens_used >= self.budget_tokens:
            return True
        if self.budget_gpu_h and self.gpu_hours >= self.budget_gpu_h:
            return True
        return False

    def done(self) -> bool:
        # Stop on budget only. `converged` is a REPORT flag (EPS/N locked above), not a
        # stop: the loop keeps exploring past first-convergence up to max_iters, which is
        # how the agent reaches the playbook's named-but-untried directions.
        return self.budget_exhausted()

    def summary(self) -> dict:
        """Compact state snapshot for the episode preamble / tool results."""
        return {
            "best_primary": self.best_primary,
            "best_iter": self.best_iter,
            "best_config": self.best_config,
            "stagnant": self.stagnant,
            "converged": self.converged,
            "iterations": self.iterations,
            "tokens_used": self.tokens_used,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "gpu_hours": round(self.gpu_hours, 4),
            "errors": self.errors,
            "n_configs_tried": len(self.seen_configs),
        }

    # -- persist ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "best_primary": self.best_primary,
            "best_iter": self.best_iter,
            "best_config": self.best_config,
            "stagnant": self.stagnant,
            "converged": self.converged,
            "iterations": self.iterations,
            "interventions": self.interventions,
            "git_sha": self.git_sha,
            "tokens_used": self.tokens_used,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "gpu_hours": round(self.gpu_hours, 4),
            "errors": self.errors,
            "n_configs_tried": len(self.seen_configs),
            "seen_configs": self.seen_configs,
            "elapsed_s": round(time.time() - self.started_at, 1),
        }

    def save(self) -> None:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=float)

    @classmethod
    def load(cls, **kwargs) -> "AgentState":
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH) as f:
                d = json.load(f)
            st = cls()
            st.best_primary = d.get("best_primary", 0.0)
            st.best_iter = d.get("best_iter", -1)
            st.best_config = d.get("best_config")
            st.stagnant = d.get("stagnant", 0)
            st.converged = d.get("converged", False)
            st.iterations = d.get("iterations", 0)
            st.tokens_used = d.get("tokens_used", 0)
            st.tokens_input = d.get("tokens_input", 0)
            st.tokens_output = d.get("tokens_output", 0)
            st.git_sha = d.get("git_sha", st.git_sha)
            st.gpu_hours = d.get("gpu_hours", 0.0)
            st.errors = d.get("errors", 0)
            st.seen_configs = d.get("seen_configs", {})
            for k, v in kwargs.items():
                setattr(st, k, v)
            return st
        return cls(**kwargs)
