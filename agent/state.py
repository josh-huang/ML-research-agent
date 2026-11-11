"""Agent run state: best-so-far, convergence, budget, and config-dedup cache.

Persistence model: ``run_logs/state.json`` is a single small JSON file the dashboard
polls every tick and the agent rewrites after each iteration. All numeric metrics are
native Python floats so they JSON-serialize cleanly.

Best-tracking is on the **seed-mean** of each config group (the config with ``seed``
removed), not on the single luckiest run: re-running a config with a new seed folds into
that config's mean, which can raise *or lower* the tracked best. This removes the
winner's-curse bias of "max over single-seed runs".
"""
from __future__ import annotations

import json
import os
import statistics
import time

from agent.logger import LOG_DIR, git_sha

STATE_PATH = os.path.join(LOG_DIR, "state.json")

# convergence FLAG: set when N consecutive iterations improve valid primary by < eps.
# EPS/N are the locked competition口径 (ε=0.002, N=3) — do NOT change. The flag is for
# REPORTING only; it does not stop the loop (AgentState.done stops on budget only, so the
# agent keeps exploring past first-convergence up to max_iters).
EPS = 0.002
N_CONVERGE = 3

# seen_configs value marking a run that errored (excluded from seed groups).
ERROR_SENTINEL = -1.0


class AgentState:
    """In-memory state, mirrored to state.json on every mutation."""

    def __init__(self, best_primary: float = 0.0, budget_tokens: int = 0,
                 budget_gpu_h: float = 0.0):
        self.best_primary = best_primary  # seed-mean valid primary of the best group
        self.best_iter = -1               # latest iteration that ran the best group
        self.best_config = None
        self.best_n_seeds = 0             # number of seeds behind best_primary
        self.best_std = 0.0               # population std over those seeds
        self.stagnant = 0            # consecutive < eps improvements
        self.converged = False
        self.iterations = 0
        self.interventions = 0       # human interventions (autonomy score)
        self.tokens_used = 0
        self.tokens_input = 0        # input-side tokens (input + cache-write + cache-read)
        self.tokens_output = 0       # generated tokens
        self.git_sha = git_sha()     # run-start code commit (D3 traceability)
        self.gpu_hours = 0.0
        self.seen_configs = {}       # full config-key (incl. seed) -> valid primary
        self.seen_iters = {}         # full config-key -> iteration it ran in
        self.budget_tokens = budget_tokens
        self.budget_gpu_h = budget_gpu_h
        self.started_at = time.time()
        self.errors = 0

    # -- config dedup -----------------------------------------------------
    # Two keys: the *full* key (incl. seed) refuses an exact re-run; the *group* key
    # (seed removed) aggregates the seeds of one config for seed-mean best-tracking.
    def config_key(self, config: dict) -> str:
        return json.dumps(config, sort_keys=True, default=float)

    @staticmethod
    def group_key(config: dict) -> str:
        return json.dumps({k: v for k, v in config.items() if k != "seed"},
                          sort_keys=True, default=float)

    def seen(self, config: dict) -> bool:
        return self.config_key(config) in self.seen_configs

    def remember(self, config: dict, primary: float) -> None:
        key = self.config_key(config)
        self.seen_configs[key] = float(primary)
        self.seen_iters[key] = self.iterations

    def groups(self) -> dict[str, dict]:
        """Successful runs aggregated by config-without-seed.

        Returns ``{group_key: {config, seeds: {seed: primary}, n, mean, std, last_iter}}``
        where ``config`` has no ``seed`` key. Errored runs are excluded.
        """
        out: dict[str, dict] = {}
        for key, primary in self.seen_configs.items():
            if primary <= ERROR_SENTINEL:
                continue
            cfg = json.loads(key)
            seed = int(cfg.pop("seed", 0))
            g = out.setdefault(self.group_key(cfg),
                               {"config": cfg, "seeds": {}, "last_iter": -1})
            g["seeds"][seed] = float(primary)
            g["last_iter"] = max(g["last_iter"], int(self.seen_iters.get(key, -1)))
        for g in out.values():
            vals = list(g["seeds"].values())
            g["n"] = len(vals)
            g["mean"] = float(statistics.fmean(vals))
            g["std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        return out

    def group_of(self, config: dict) -> dict | None:
        """Seed-aggregate stats of ``config``'s group, or None if it never ran successfully."""
        return self.groups().get(self.group_key(config))

    def _refresh_best(self) -> None:
        """Best = the config group with the highest seed-mean valid primary."""
        groups = self.groups()
        if not groups:
            return
        g = max(groups.values(), key=lambda x: x["mean"])
        self.best_primary = g["mean"]
        self.best_iter = g["last_iter"]
        self.best_n_seeds = g["n"]
        self.best_std = g["std"]
        self.best_config = dict(g["config"], seed=min(g["seeds"]))

    # -- update -----------------------------------------------------------
    def record(self, config: dict, valid_primary: float, tokens: int, gpu_h: float,
               error: bool = False, tokens_input: int = 0,
               tokens_output: int = 0) -> str:
        """Record an iteration; returns 'error' | 'new_best' | 'accept' | 'reject'.

        The improvement of a round is the change of the tracked best (seed-mean). Any
        positive change is recorded as the best; only a change >= EPS resets the
        stagnation counter (the locked "3 consecutive sub-eps rounds -> converged" rule).
        A new seed that drags the best group's mean down is a 'reject' with a negative
        change — the best is revised downward instead of keeping a lucky single seed.
        """
        self.iterations += 1
        self.tokens_used += tokens
        self.tokens_input += tokens_input
        self.tokens_output += tokens_output
        self.gpu_hours += gpu_h
        if error:
            self.errors += 1
            self.stagnant += 1
            self.remember(config, ERROR_SENTINEL)
            self.converged = self.stagnant >= N_CONVERGE
            return "error"

        prev_best = self.best_primary
        self.remember(config, valid_primary)
        self._refresh_best()
        delta = self.best_primary - prev_best
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
            "best_n_seeds": self.best_n_seeds,
            "best_std": self.best_std,
            "stagnant": self.stagnant,
            "converged": self.converged,
            "iterations": self.iterations,
            "tokens_used": self.tokens_used,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "gpu_hours": round(self.gpu_hours, 4),
            "errors": self.errors,
            "n_configs_tried": len(self.groups()),
            "n_runs": len(self.seen_configs),
        }

    # -- persist ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "best_primary": self.best_primary,
            "best_iter": self.best_iter,
            "best_config": self.best_config,
            "best_n_seeds": self.best_n_seeds,
            "best_std": self.best_std,
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
            "n_configs_tried": len(self.groups()),
            "n_runs": len(self.seen_configs),
            "seen_configs": self.seen_configs,
            "seen_iters": self.seen_iters,
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
            st.seen_iters = d.get("seen_iters", {})
            # Re-derive best from the seed groups: a legacy state.json stored the
            # single-seed max, which the seed-mean rule replaces.
            st._refresh_best()
            for k, v in kwargs.items():
                setattr(st, k, v)
            return st
        return cls(**kwargs)
