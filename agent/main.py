"""Autonomous research loop — the agent's orchestrator.

The brain is a single persistent ReAct agent (:mod:`agent.react`) that sees results and
proposes the next step across iterations. The orchestrator keeps only the deterministic
skeleton: data/EDA loaded once, the iteration-0 FM baseline, convergence/budget, resume,
and a no-LLM fallback proposer for robustness.

Invoke from the project root::

    python -m agent.main --max_iters 12 --budget_tokens 300000

``--no_llm`` swaps the ReAct agent for a uniform/mutation proposer so the whole loop can
be smoke-tested (and kept running) without API spend. If the LLM raises mid-run the loop
falls back to this path *loudly* (a warning is printed and recorded) rather than silently.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent import prompts, search, tools  # noqa: E402
from agent.eda import compute_summary, render_markdown  # noqa: E402
from agent.llm import ClaudeClient  # noqa: E402
from agent.logger import RUN_LOG_PATH  # noqa: E402
from agent.react import ReActAgent  # noqa: E402
from agent.state import AgentState  # noqa: E402

_DATA_DIR = os.path.join(_ROOT, "kuairand-starter-kit", "kuairand-starter-kit",
                         "KuaiRand-Pure", "data")
BASELINE_HYPOTHESIS = "Reproduce the FM baseline (pointwise BCE) as the trust anchor."

TOOL_SCHEMAS = [prompts.SEARCH_ARXIV_TOOL, prompts.FETCH_PAPER_TOOL,
                prompts.READ_RUN_LOG_TOOL, prompts.RUN_EXPERIMENT_TOOL,
                prompts.FINISH_EPISODE_TOOL]


def _propose_fallback(state: AgentState, rng: random.Random) -> tuple[str, dict]:
    """No-LLM proposer: exhaust the seed configs first, then mutate the best."""
    for seed in search.SEED_CONFIGS:
        cfg = search.normalize(seed)
        if not state.seen(cfg):
            return f"[no-llm] explore seed {cfg['model']}+{cfg['loss']}", cfg
    if state.best_config:
        cfg = search.mutate(state.best_config, rng)
    else:
        cfg = search.random_config(rng)
    return f"[no-llm] explore {cfg.get('model')}+{cfg.get('loss')} " \
           f"k={cfg.get('k')} lr={cfg.get('lr')}", cfg


def _fallback_run(state: AgentState, ctx: tools.Ctx, rng: random.Random) -> bool:
    """No-LLM path: propose via fallback, run, finalize. Returns whether a run happened."""
    hypothesis, cfg = _propose_fallback(state, rng)
    _, run_record = tools.run_experiment(state, ctx.data, cfg, hypothesis)
    if run_record is not None:
        tools.finalize_run(state, run_record, tokens=0, lesson="")
    return run_record is not None


def run(data_dir: str, max_iters: int, budget_tokens: int, budget_gpu_h: float,
        use_llm: bool = True, fresh: bool = False) -> AgentState:
    from models.data_loader import load_extended  # noqa: E402  (import late: torch)
    data = load_extended(data_dir)
    eda_md = render_markdown(compute_summary(data_dir))

    if fresh:
        # Truncate (not delete) the previous run-log and start a fresh state.
        open(RUN_LOG_PATH, "w").close()
        state = AgentState(budget_tokens=budget_tokens, budget_gpu_h=budget_gpu_h)
    else:
        state = AgentState.load(budget_tokens=budget_tokens, budget_gpu_h=budget_gpu_h)

    llm, react = None, None
    if use_llm:
        try:
            llm = ClaudeClient()
            llm.set_system(prompts.build_system_prompt(eda_md))
            react = ReActAgent(llm, TOOL_SCHEMAS)
        except Exception as e:  # noqa: BLE001 — surface, then degrade to no-LLM
            print(f"[warn] LLM unavailable ({e}); falling back to no-LLM proposer.", flush=True)
            use_llm = False

    rng = random.Random(0)
    ctx = tools.Ctx(state=state, data=data)

    # Iteration 0: reproduce the FM baseline (task requirement #1, trust anchor). Runs
    # directly through the tool handler — zero tokens, deterministic, not in the conversation.
    if state.iterations == 0:
        baseline_cfg = search.normalize({"model": "fm", "loss": "bce", "seed": 0})
        _, run_record = tools.run_experiment(state, data, baseline_cfg, BASELINE_HYPOTHESIS)
        if run_record is not None:
            tools.finalize_run(state, run_record, tokens=0, lesson="")

    while not state.done() and state.iterations < max_iters:
        ran = False
        if use_llm and react is not None:
            try:
                ep = react.run_episode(ctx)
                if ep.failed:
                    print("[warn] LLM episode failed; degrading to no-LLM proposer.",
                          flush=True)
                    use_llm = False
                ran = ep.ran
            except Exception as e:  # noqa: BLE001 — never let an LLM bug kill the run
                print(f"[warn] LLM episode raised ({e}); degrading to no-LLM proposer.",
                      flush=True)
                use_llm = False
                ran = False
        if not ran:
            _fallback_run(state, ctx, rng)

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
    ap.add_argument("--fresh", action="store_true", help="truncate run-log and start a fresh state")
    a = ap.parse_args()
    run(a.data_dir, a.max_iters, a.budget_tokens, a.budget_gpu_h,
        use_llm=not a.no_llm, fresh=a.fresh)


if __name__ == "__main__":
    main()
