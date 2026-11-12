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
import re

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PLAYBOOK_PATH = os.path.join(_ROOT, "agent", "playbook.md")
PLAYBOOK_MAX_LESSONS = 15

# --- test-split sealing for free text ----------------------------------------
# Lessons/hypotheses written before the test split was sealed quote test metrics
# ("valid 0.6053 (test 0.5988)"). Those files are kept verbatim as the historical record
# (D3), and every free-text field is scrubbed here at injection time instead.
_NUM = r"~?\d?\.\d+"
_TEST_INLINE = (
    re.compile(rf"\s*\((?:also\s+)?test(?:\s+primary)?\s+{_NUM}[^)]*\)", re.I),  # (test 0.59..)
    re.compile(rf",\s*(?:also\s+)?test(?:\s+primary)?\s+{_NUM}[^,;)]*", re.I),    # , test 0.59..
    re.compile(rf"\s*/\s*{_NUM}\s+test\b", re.I),                                 # / 0.5989 test
)
_CLAUSE_SPLIT = re.compile(r"(,\s+|;\s+|\s+—\s+|(?<=[.!?])\s+)")
_TEST_WORD = re.compile(r"\btest\b", re.I)
_END = (".", "!", "?")


def scrub_test_refs(text: str) -> str:
    """Remove test-split references from agent-visible free text.

    Inline forms (``(test 0.5988)``, ``, test 0.5989``, ``/ 0.5989 test``) are cut out;
    any clause still mentioning "test" is dropped whole. Valid-split content survives.
    """
    text = text or ""
    for pat in _TEST_INLINE:
        text = pat.sub("", text)
    parts = _CLAUSE_SPLIT.split(text)  # [clause, delim, clause, delim, ...]
    out: list[str] = []
    for i in range(0, len(parts), 2):
        clause = parts[i]
        if _TEST_WORD.search(clause):
            # keep the sentence terminated if we dropped its final clause
            if clause.rstrip().endswith(_END) and out and not out[-1].rstrip().endswith(_END):
                out[-1] = out[-1].rstrip() + "."
            continue
        out.append((parts[i - 1] if out else "") + clause)
    return "".join(out).strip()


def load_playbook(k: int = PLAYBOOK_MAX_LESSONS) -> str:
    """Persistent procedural prior (survives ``--fresh``), bounded to the last ``k`` lessons."""
    if not os.path.exists(PLAYBOOK_PATH):
        return ""
    with open(PLAYBOOK_PATH, encoding="utf-8") as f:
        lessons = [ln.strip()[2:] for ln in f if ln.strip().startswith("- ")]
    scrubbed = [scrub_test_refs(ln) for ln in lessons[-k:]]
    return "\n".join(f"- {ln}" for ln in scrubbed if ln)


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


def _confirmed_line(s: dict) -> str:
    """Preamble line on the confirmed (>= MIN_CONFIRM_SEEDS seeds) best."""
    from agent.state import MIN_CONFIRM_SEEDS
    c = s.get("confirmed_best")
    line = (f"- confirmed best (>= {MIN_CONFIRM_SEEDS} seeds; this is what gets submitted): "
            + (f"{c['primary']:.4f} ± {c['std']:.4f} over {c['n_seeds']} seeds"
               if c else "none yet"))
    if s.get("best_n_seeds", 0) < MIN_CONFIRM_SEEDS and s.get("best_config"):
        line += (f". The current best has only {s['best_n_seeds']} seed(s): re-run that "
                 f"config with a new `seed` to confirm it before trusting it.")
    return line


def build_preamble(state, recent_tail: list[dict], lessons: str, playbook: str = "") -> str:
    """Fresh per-episode context (current state + playbook + lessons + recent iterations)."""
    s = state.summary()
    lines = [
        "## Current state",
        f"- best valid primary {s['best_primary']:.4f} (seed-mean over "
        f"{s['best_n_seeds']} seed(s), std {s['best_std']:.4f}; iter {s['best_iter']}), "
        f"config: {json.dumps(s['best_config'], default=float) if s['best_config'] else 'none'}",
        _confirmed_line(s),
        f"- iterations {s['iterations']} | stagnant {s['stagnant']}/3 | "
        f"converged {s['converged']}",
        f"- tokens used {s['tokens_used']} | gpu {s['gpu_hours']:.4f}h | errors {s['errors']}",
        f"- configs tried {s['n_configs_tried']} ({s['n_runs']} runs incl. extra seeds)",
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
            lines.append(f"- [{r['iteration']}] {scrub_test_refs(r['hypothesis'])} -> valid primary "
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
        lesson = scrub_test_refs(r.get("lesson") or "")
        if lesson and lesson not in seen:
            seen.append(lesson)
            out.append(f"- {lesson}")
        if len(out) >= k:
            break
    return "\n".join(out)


def episode_line(hypothesis: str, cfg: dict, valid_primary: float,
                 verdict: str, lesson: str) -> str:
    """One-line summary of a finished episode (the compressed working memory).

    Valid only: this line is re-injected into the LLM context, and the test split is
    sealed from the agent.
    """
    cfg_s = json.dumps(cfg, sort_keys=True, default=float)
    hypothesis, lesson = scrub_test_refs(hypothesis), scrub_test_refs(lesson)
    line = (f"[episode] {hypothesis} -> config {cfg_s} -> valid primary {valid_primary:.4f} "
            f"| verdict {verdict}")
    if lesson:
        line += f" | lesson: {lesson}"
    return line
