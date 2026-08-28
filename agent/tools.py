"""Tool handlers for the ReAct agent's action space.

Five tools, all dispatched locally (no MCP — single machine, single agent, and native
Anthropic function-calling already routes tool calls back into this process):

- search_arxiv / fetch_paper : literature grounding (``agent.research``)
- read_run_log              : episodic-memory recall (``agent.logger.tail``)
- run_experiment            : the act — train + eval + graceful-degrade (no record/log)
- finish_episode            : reflect — capture a reusable lesson

``dispatch`` routes one tool call and returns a string; side effects (a pending run, the
episode lesson) are stashed on the shared :class:`Ctx` and folded in by the orchestrator.
Recording/logging is deferred to :func:`finalize_run` so the episode's *total* token spend
(proposal + reflect) is known — which is what keeps convergence counting honest.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from agent import research, search
from agent.executor import degrade_config, execute
from agent.logger import log_iteration, tail

MAX_RUNS_PER_EPISODE = 1


@dataclass
class Ctx:
    """Per-run context threaded through the tool handlers."""
    state: "AgentState"
    data: dict
    episode_runs: int = 0
    run_record: dict | None = None
    lesson: str = ""
    finish_seen: bool = False


def run_experiment(state, data, config, hypothesis) -> tuple[str, dict | None]:
    """Execute one experiment (train + eval + one degradation retry); never records/logs.

    Returns ``(result_text, run_record)``. ``run_record`` is ``None`` when the config is a
    duplicate (refused without spending GPU). Recording/logging is deferred to
    :func:`finalize_run` so the episode's total token spend is known at record time.
    """
    cfg = search.normalize(config)
    if state.seen(cfg):
        prev = state.seen_configs[state.config_key(cfg)]
        return (f"Refusing: this config was already tried (valid primary {prev:.4f}). "
                f"Vary a dimension (model/loss/k/lr/dropout/aux/seed) and try again. "
                f"Normalized cfg: {json.dumps(cfg, sort_keys=True, default=float)}"), None

    t0 = time.time()
    res = execute(data, cfg)
    gpu_h = res["gpu_h"]
    recovery = ""

    if res["error"]:
        degraded = degrade_config(cfg, res["error"])
        tag = degraded.pop("_degraded", "")
        retry_cfg = search.normalize(degraded)
        retry_res = execute(data, retry_cfg)
        gpu_h += retry_res["gpu_h"]
        if not retry_res["error"]:
            res, cfg, recovery = retry_res, retry_cfg, tag
        else:
            recovery = f"{tag} FAILED"

    dur = time.time() - t0
    metrics = res["metrics"]
    error = res["error"]
    run_record = {
        "cfg": cfg, "hypothesis": hypothesis, "metrics": metrics,
        "error": error, "gpu_h": gpu_h, "recovery": recovery, "dur": dur,
    }

    lines = [f"Experiment executed: model={cfg['model']}, loss={cfg['loss']}, "
             f"k={cfg.get('k')}, lr={cfg.get('lr')}."]
    if error:
        lines.append(f"ERROR {error}{' | ' + recovery if recovery else ''} "
                     f"(wall {dur:.0f}s, gpu {gpu_h:.4f}h).")
    else:
        v, t = metrics["valid"], metrics["test"]
        lines.append(f"valid GAUC {v['GAUC']:.4f} nDCG@5 {v['nDCG@5']:.4f} "
                     f"primary {v['primary']:.4f}")
        lines.append(f"test  GAUC {t['GAUC']:.4f} nDCG@5 {t['nDCG@5']:.4f} "
                     f"primary {t['primary']:.4f}")
    lines.append(f"best valid so far {state.best_primary:.4f} (iter {state.best_iter}); "
                 f"iterations {state.iterations} done; stagnant {state.stagnant}/3.")
    lines.append("Now call finish_episode with a reusable lesson grounded in primary "
                 "(mean of GAUC and nDCG@5, ~0.60 — not GAUC ~0.66).")
    return "\n".join(lines), run_record


def finalize_run(state, run_record: dict, tokens: int, lesson: str):
    """Record + log + persist one executed experiment (exactly once per run).

    Returns ``(verdict, valid_primary, test_primary)``. This is where convergence counting
    and the token/GPU budgets update — preserving the original 1:1 iteration<->experiment
    semantics (``state.record`` is called exactly once per executed run).
    """
    error = run_record["error"]
    metrics = run_record["metrics"]
    valid_primary = metrics["valid"]["primary"] if metrics else 0.0
    test_primary = metrics["test"]["primary"] if metrics else 0.0
    verdict = state.record(run_record["cfg"], valid_primary, test_primary, tokens,
                           run_record["gpu_h"], error=bool(error))
    log_iteration(iteration=state.iterations, hypothesis=run_record["hypothesis"],
                  action=run_record["cfg"], metrics=metrics, tokens=tokens,
                  gpu_h=run_record["gpu_h"], errors=[error] if error else [],
                  verdict=verdict, duration_s=run_record["dur"],
                  recovery=run_record["recovery"], lesson=lesson)
    state.save()

    cfg = run_record["cfg"]
    if error:
        print(f"[{state.iterations:02d}] {cfg['model']}+{cfg['loss']} ERROR "
              f"({error}{' | ' + run_record['recovery'] if run_record['recovery'] else ''}) "
              f"| {run_record['dur']:.0f}s", flush=True)
    else:
        print(f"[{state.iterations:02d}] {cfg['model']}+{cfg['loss']} k={cfg.get('k')} "
              f"lr={cfg.get('lr')} -> valid {valid_primary:.4f} / test {test_primary:.4f} "
              f"| {verdict} | {run_record['dur']:.0f}s | {tokens}t", flush=True)
    if lesson:
        print(f"       lesson: {lesson}", flush=True)
    return verdict, valid_primary, test_primary


def _read_run_log(n: int) -> str:
    """Compact view of the last n records (metrics stripped to the summary lines)."""
    records = tail(max(1, min(int(n), 50)))
    if not records:
        return "(run log is empty)"
    lines = [f"last {len(records)} iterations (newest last):"]
    for r in records:
        m = r.get("metrics") or {}
        vp = m.get("valid", {}).get("primary")
        lines.append(
            f"- [{r['iteration']}] {r['hypothesis']} -> valid primary "
            f"{vp if vp is not None else 'err'}"
            f"{' (ERROR)' if r.get('errors') else ''} | verdict {r.get('verdict', '')}"
            f"{' | lesson: ' + r['lesson'] if r.get('lesson') else ''}")
    return "\n".join(lines)


def _guarded_run(ctx: Ctx, inp: dict) -> str:
    if ctx.episode_runs >= MAX_RUNS_PER_EPISODE:
        return ("Refusing: already ran one experiment this episode. Call finish_episode "
                "with your lesson now.")
    hypothesis = str(inp.get("hypothesis") or "").strip()
    config = inp.get("config") or {}
    text, run_record = run_experiment(ctx.state, ctx.data, config, hypothesis)
    if run_record is not None:
        ctx.episode_runs += 1
        ctx.run_record = run_record
    return text


def dispatch(name: str, inp: dict, ctx: Ctx) -> str:
    """Route one tool call to its handler; returns a string result (never raises)."""
    try:
        if name == "search_arxiv":
            return research.search_arxiv(str(inp.get("query", "")),
                                         int(inp.get("max_results", 5)))
        if name == "fetch_paper":
            return research.fetch_paper(str(inp.get("arxiv_id", "")))
        if name == "read_run_log":
            return _read_run_log(int(inp.get("n", 5)))
        if name == "run_experiment":
            return _guarded_run(ctx, inp)
        if name == "finish_episode":
            ctx.lesson = str(inp.get("lesson") or "").strip()
            ctx.finish_seen = True
            return "Episode recorded."
        return f"unknown tool: {name}"
    except Exception as exc:  # noqa: BLE001 — a tool must never kill the episode
        return f"tool {name} failed: {exc}"
