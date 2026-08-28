"""Agent memory: working-memory preamble, procedural-lessons digest, episode summary.

Four layers map onto existing artifacts:
- working    -> the current episode's multi-turn conversation, plus the one-line history
- episodic   -> run_log.jsonl (+ lesson field), queryable via the read_run_log tool
- semantic   -> the cached, stable system prompt
- procedural -> lessons_digest, derived from the log's lesson fields (survives resume)
"""
from __future__ import annotations

import json


def build_preamble(state, recent_tail: list[dict], lessons: str) -> str:
    """Fresh per-episode context (current state + lessons + recent iterations)."""
    s = state.summary()
    lines = [
        "## Current state",
        f"- best valid primary {s['best_primary']:.4f} (iter {s['best_iter']}), "
        f"config: {json.dumps(s['best_config'], default=float) if s['best_config'] else 'none'}",
        f"- iterations {s['iterations']} | stagnant {s['stagnant']}/3 | "
        f"converged {s['converged']}",
        f"- tokens used {s['tokens_used']} | gpu {s['gpu_hours']:.4f}h | errors {s['errors']}",
        f"- configs tried {s['n_configs_tried']}",
    ]
    if lessons:
        lines += ["", "## Lessons learned (do not repeat a failed direction)", lessons]
    lines += ["", "## Recent iterations (newest last)"]
    if recent_tail:
        for r in recent_tail:
            m = r.get("metrics") or {}
            vp = m.get("valid", {}).get("primary")
            lines.append(f"- [{r['iteration']}] {r['hypothesis']} -> valid primary "
                         f"{vp if vp is not None else 'err'}"
                         f"{' (ERROR)' if r.get('errors') else ''}")
    else:
        lines.append("  (none)")
    lines += ["", "Run exactly ONE experiment this episode, then finish_episode."]
    return "\n".join(lines)


def lessons_digest(records: list[dict], k: int = 8) -> str:
    """Dedup'd latest lessons from the run-log (newest first), newest last in output."""
    seen, out = [], []
    for r in reversed(records):
        lesson = (r.get("lesson") or "").strip()
        if lesson and lesson not in seen:
            seen.append(lesson)
            out.append(f"- {lesson}")
        if len(out) >= k:
            break
    return "\n".join(out)


def episode_line(hypothesis: str, cfg: dict, valid_primary: float, test_primary: float,
                 verdict: str, lesson: str) -> str:
    """One-line summary of a finished episode (the compressed working memory)."""
    cfg_s = json.dumps(cfg, sort_keys=True, default=float)
    line = (f"[episode] {hypothesis} -> config {cfg_s} -> valid primary {valid_primary:.4f} "
            f"/ test primary {test_primary:.4f} | verdict {verdict}")
    if lesson:
        line += f" | lesson: {lesson}"
    return line
