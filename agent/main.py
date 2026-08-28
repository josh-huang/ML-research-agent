"""Autonomous research loop — the agent's orchestrator.

One iteration: researcher proposes -> executor runs -> state records -> reflector
extracts a lesson -> run-log appended. The loop runs until the convergence criterion
(N consecutive improvements < eps) or a global budget (token / GPU-hours) trips.

Data is loaded once; EDA and the system prompt are computed once and cached across
LLM turns (prompt caching in :class:`agent.llm.ClaudeClient`).

Invoke from the project root::

    python -m agent.main --max_iters 12 --budget_tokens 300000

``--no_llm`` swaps the researcher for a uniform/mutation proposer so the whole loop can
be smoke-tested (and kept running) without API spend — a robustness fallback, not just a
test hook.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent import prompts, search  # noqa: E402
from agent.eda import compute_summary, render_markdown  # noqa: E402
from agent.executor import execute  # noqa: E402
from agent.llm import ClaudeClient  # noqa: E402
from agent.logger import log_iteration, tail  # noqa: E402
from agent.state import AgentState  # noqa: E402

_DATA_DIR = os.path.join(_ROOT, "kuairand-starter-kit", "kuairand-starter-kit",
                         "KuaiRand-Pure", "data")
BASELINE_HYPOTHESIS = "Reproduce the FM baseline (pointwise BCE) as the trust anchor."


def _tokens(usage: dict) -> int:
    return int(usage.get("input", 0)) + int(usage.get("output", 0))


def _propose_llm(state: AgentState, llm: ClaudeClient, tried: list[dict]):
    """Ask the researcher for the next experiment; returns (hypothesis, config, tokens)."""
    best = {"primary": state.best_primary, "config": state.best_config}
    msg = prompts.researcher_message(best, tail(5), tried)
    content, usage = llm.complete(
        msg, tools=[prompts.PROPOSE_TOOL],
        tool_choice={"type": "tool", "name": "propose_experiment"})
    inp = llm.tool_use(content)
    cfg = inp.get("config") or {}
    hypothesis = str(inp.get("hypothesis") or "").strip()
    if not hypothesis:
        hypothesis = f"propose {cfg.get('model')}+{cfg.get('loss')}"
    return hypothesis, cfg, _tokens(usage)


def _propose_fallback(state: AgentState, tried: list[dict], rng: random.Random):
    """No-LLM proposer: exhaust the seed configs first, then mutate the best."""
    for seed in search.SEED_CONFIGS:
        cfg = search.normalize(seed)
        if not state.seen(cfg):
            return f"[no-llm] explore seed {cfg['model']}+{cfg['loss']}", cfg, 0
    if state.best_config:
        cfg = search.mutate(state.best_config, rng)
    else:
        cfg = search.random_config(rng)
    return f"[no-llm] explore {cfg.get('model')}+{cfg.get('loss')} " \
           f"k={cfg.get('k')} lr={cfg.get('lr')}", cfg, 0


def _reflect(llm: ClaudeClient, result: dict) -> tuple[str, int]:
    """Extract a short reusable lesson from one result. Cheap; failure is non-fatal."""
    try:
        content, usage = llm.complete(
            prompts.reflector_message(result), tools=[prompts.REFLECT_TOOL],
            tool_choice={"type": "tool", "name": "reflect"})
        inp = llm.tool_use(content)
        return str(inp.get("lesson") or ""), _tokens(usage)
    except Exception:  # noqa: BLE001 — reflection must never kill the run
        return "", 0


def _run_one(data: dict, state: AgentState, llm: ClaudeClient, cfg: dict,
             hypothesis: str, reflect: bool) -> None:
    """Execute one config end-to-end: run -> record -> reflect -> log -> save."""
    cfg = search.normalize(cfg)
    t0 = time.time()
    res = execute(data, cfg)
    dur = time.time() - t0

    metrics = res["metrics"]
    error = res["error"]
    valid_primary = metrics["valid"]["primary"] if metrics else 0.0
    test_primary = metrics["test"]["primary"] if metrics else 0.0

    lesson, r_tokens = "", 0
    if reflect and not error:
        lesson, r_tokens = _reflect(llm, {"hypothesis": hypothesis,
                                          "config": cfg, "metrics": metrics})

    verdict = state.record(cfg, valid_primary, test_primary, r_tokens,
                           res["gpu_h"], error=bool(error))
    log_iteration(iteration=state.iterations, hypothesis=hypothesis, action=cfg,
                  metrics=metrics, tokens=r_tokens, gpu_h=res["gpu_h"],
                  errors=[error] if error else [], verdict=verdict,
                  duration_s=dur)
    state.save()

    if error:
        print(f"[{state.iterations:02d}] {cfg['model']}+{cfg['loss']} ERROR "
              f"({error}) | {dur:.0f}s", flush=True)
    else:
        print(f"[{state.iterations:02d}] {cfg['model']}+{cfg['loss']} k={cfg.get('k')} "
              f"lr={cfg.get('lr')} -> valid {valid_primary:.4f} / test {test_primary:.4f} "
              f"| {verdict} | {dur:.0f}s | {r_tokens}t", flush=True)
    if lesson:
        print(f"       lesson: {lesson}", flush=True)


def run(data_dir: str, max_iters: int, budget_tokens: int, budget_gpu_h: float,
        use_llm: bool = True, reflect: bool = True, fresh: bool = False) -> AgentState:
    from models.data_loader import load_extended  # noqa: E402  (import late: torch)
    data = load_extended(data_dir)
    eda_md = render_markdown(compute_summary(data_dir))

    if fresh:
        # Truncate (not delete) the previous run-log and start a fresh state.
        from agent.logger import RUN_LOG_PATH
        open(RUN_LOG_PATH, "w").close()
        state = AgentState(budget_tokens=budget_tokens, budget_gpu_h=budget_gpu_h)
    else:
        state = AgentState.load(budget_tokens=budget_tokens, budget_gpu_h=budget_gpu_h)
    llm = ClaudeClient() if use_llm else None
    if llm is not None:
        llm.set_system(prompts.build_system_prompt(eda_md))
    rng = random.Random(0)
    tried: list[dict] = []

    # Iteration 0: reproduce the FM baseline (task requirement #1, trust anchor).
    if state.iterations == 0:
        _run_one(data, state, llm, search.normalize({"model": "fm", "loss": "bce", "seed": 0}),
                 BASELINE_HYPOTHESIS, reflect=False)
        tried.append(search.normalize({"model": "fm", "loss": "bce", "seed": 0}))

    while not state.done() and state.iterations < max_iters:
        for attempt in range(3):
            if llm is not None:
                hypothesis, cfg, p_tokens = _propose_llm(state, llm, tried)
            else:
                hypothesis, cfg, p_tokens = _propose_fallback(state, tried, rng)
            cfg = search.normalize(cfg)
            if not state.seen(cfg):
                break
            # duplicate -> nudge and retry (bounded), else give up this turn
            cfg = search.mutate(cfg, rng)
            if not state.seen(cfg):
                hypothesis += " [dedup-mutated]"
                break
        else:
            cfg = search.random_config(rng)
            hypothesis = "[no-llm] random config after 3 duplicate proposals"

        tried.append(cfg)
        _run_one(data, state, llm, cfg, hypothesis, reflect=reflect)

    state.save()
    print(f"\n=== converged={state.converged} | iterations={state.iterations} | "
          f"best valid {state.best_primary:.4f} (iter {state.best_iter}) | "
          f"tokens {state.tokens_used} | gpu {state.gpu_hours:.3f}h | "
          f"errors {state.errors} ===", flush=True)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=_DATA_DIR)
    ap.add_argument("--max_iters", type=int, default=12)
    ap.add_argument("--budget_tokens", type=int, default=300_000)
    ap.add_argument("--budget_gpu_h", type=float, default=0.0)
    ap.add_argument("--no_llm", action="store_true", help="uniform/mutation proposer, no API")
    ap.add_argument("--no_reflect", action="store_true", help="skip the reflector LLM call")
    ap.add_argument("--fresh", action="store_true", help="truncate run-log and start a fresh state")
    a = ap.parse_args()
    run(a.data_dir, a.max_iters, a.budget_tokens, a.budget_gpu_h,
        use_llm=not a.no_llm, reflect=not a.no_reflect, fresh=a.fresh)


if __name__ == "__main__":
    main()
