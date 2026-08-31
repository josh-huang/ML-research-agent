"""Agent memory: working-memory preamble, procedural-lessons digest, episode summary.

Four layers map onto existing artifacts:
- working    -> the current episode's multi-turn conversation, plus the one-line history
- episodic   -> run_log.jsonl (+ lesson field), queryable via the read_run_log tool
- semantic   -> the cached, stable system prompt
- procedural -> lessons_digest (run-local) + agent/playbook.md (persistent, survives --fresh)

The playbook is the agent's "SKILLS.MD": hard-won lessons append to it via finish_episode
and are re-injected every episode, so a fresh run starts from the frontier instead of zero.
"""
from __future__ import annotations

import json
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PLAYBOOK_PATH = os.path.join(_ROOT, "agent", "playbook.md")
PLAYBOOK_MAX_LESSONS = 15


def load_playbook(k: int = PLAYBOOK_MAX_LESSONS) -> str:
    """Persistent procedural prior (survives ``--fresh``), bounded to the last ``k`` lessons."""
    if not os.path.exists(PLAYBOOK_PATH):
        return ""
    with open(PLAYBOOK_PATH, encoding="utf-8") as f:
        lessons = [ln.strip() for ln in f if ln.strip().startswith("- ")]
    return "\n".join(lessons[-k:])


def append_lesson(lesson: str) -> None:
    """Append one lesson to the playbook (idempotent on exact text)."""
    lesson = (lesson or "").strip()
    if not lesson:
        return
    existing = set()
    if os.path.exists(PLAYBOOK_PATH):
        with open(PLAYBOOK_PATH, encoding="utf-8") as f:
            existing = {ln.strip()[2:].strip() for ln in f if ln.strip().startswith("- ")}
    if lesson in existing:
        return
    with open(PLAYBOOK_PATH, "a", encoding="utf-8") as f:
        f.write(f"- {lesson}\n")


def build_preamble(state, recent_tail: list[dict], lessons: str, playbook: str = "") -> str:
    """Fresh per-episode context (current state + playbook + lessons + recent iterations)."""
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
    if playbook:
        lines += ["", "## Playbook (persistent prior across runs — do not re-derive)",
                  playbook]
    if lessons:
        lines += ["", "## Lessons learned (this run, do not repeat a failed direction)",
                  lessons]
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
    lines += ["", "Run exactly ONE experiment this episode, then finish_episode.",
              "The 'converged' flag is informational only — keep proposing DISTINCT "
              "high-value configs (see the playbook's untried directions) rather than "
              "micro-tweaking tried ones."]
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
