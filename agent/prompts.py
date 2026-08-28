"""Prompts for the agent's researcher / reflector, plus the cached system prompt.

The system prompt is *stable* across the whole run (metric scope + anchors + headroom map
+ EDA evidence + action space), so it is markdown-cache-friendly. The volatile per-turn
state (best-so-far, recent run-log, tried configs) rides in the user message.
"""
from __future__ import annotations

import json

# The model/loss action space the researcher can pick from (Phase 5 verified components).
ACTION_SPACE = {
    "models": ["fm", "deepfm", "din"],
    "losses": ["bce", "bpr", "listwise"],
}

_SYSTEM_TEMPLATE = """\
You are the researcher of an autonomous ML research agent competing in TikTok TechJam 2026.
You iterate on a recommendation model over the KuaiRand-Pure dataset, one experiment per turn.

## Task (fixed, authoritative)
- Within-user ranking over logged impressions. Label = `long_view` (0/1).
- Metrics: GAUC + nDCG@5; primary = mean(GAUC, nDCG@5).
- You maximize primary on the VALID split; test is held out and reported for drift.

## Score anchors
- FM baseline (pointwise BCE): valid 0.6016, test 0.5946.
- Oracle ceiling (test primary): 0.8645. Progress is judged against this gap, not 1.0.

## What the prior sweep already established (do NOT re-try blindly)
- loss alignment: BPR ~ +0.001 over BCE; listwise is *worse* (unstable, overfits).
- model swap: DeepFM ~ +0.001, DIN ~ +0.002. All cluster at valid 0.602~0.604.
- CWM soft-label (train on watch_fraction / log_play_time): *much worse* (valid ~0.56) —
  the continuous target misaligns with the binary `long_view` metric. Dead end.
- multi-task aux (is_click / play_time head): no gain — `long_view` already carries the
  ranking signal; a dense near-redundant aux adds no gradient. Dead end.
- So `user_id x video_id` dominates; the plateau is ~0.604, gains here are <= +0.002.

## Open directions (all reachable within the action space below)
- hyperparameter tuning of the DIN anchor (k, lr, dropout, dnn_hidden) — the one
  direction that pushed valid past 0.604 in the prior sweep (din k=32 lr=3e-4 drop=0.2).
- model x loss combinations not yet on the frontier (din+bpr, deepfm+bpr, ...).

## EDA evidence
{eda}

## Action space
You propose ONE config per turn via the `propose_experiment` tool. Config fields:
- model: {models}
- loss: {losses}
- k (int, embedding dim, default 16), lr (float, default 1e-3),
- dnn_hidden (comma string, e.g. "64,32"), dropout (float, default 0.0), seed (int).

Rules:
- The action space is model/loss/hyperparameters ONLY. Your `config` must be the model
  you actually want to run — never describe a multi-task/CWM/ensemble idea in the
  hypothesis while proposing a plain config. Keep hypothesis and config consistent.
- Cite specific EDA evidence in your hypothesis. Never "just try a model".
- Prefer a config that is materially different from what has been tried.
- One experiment, one hypothesis. Keep the hypothesis concrete and falsifiable."""


def build_system_prompt(eda_md: str) -> str:
    return _SYSTEM_TEMPLATE.format(
        eda=eda_md, models=", ".join(ACTION_SPACE["models"]),
        losses=", ".join(ACTION_SPACE["losses"]),
    )


PROPOSE_TOOL = {
    "name": "propose_experiment",
    "description": "Propose the next training experiment with a hypothesis and a concrete config.",
    "input_schema": {
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string",
                           "description": "Falsifiable hypothesis with EDA evidence cited"},
            "config": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "enum": ACTION_SPACE["models"]},
                    "loss": {"type": "string", "enum": ACTION_SPACE["losses"]},
                    "k": {"type": "integer"},
                    "lr": {"type": "number"},
                    "dnn_hidden": {"type": "string"},
                    "dropout": {"type": "number"},
                    "seed": {"type": "integer"},
                },
                "required": ["model", "loss"],
            },
            "reason": {"type": "string", "description": "Why this config tests the hypothesis"},
        },
        "required": ["hypothesis", "config"],
    },
}


def researcher_message(best: dict, recent: list[dict], tried_configs: list[dict]) -> str:
    """Volatile per-turn context for the researcher."""
    lines = [
        "Current best so far:",
        json.dumps(best, indent=2, default=float) if best else "  (none yet — reproduce the FM baseline first)",
        "",
        f"Configs already tried ({len(tried_configs)}):",
        json.dumps(tried_configs, default=float),
        "",
        "Recent iterations (newest last):",
    ]
    if recent:
        for r in recent:
            m = r.get("metrics") or {}
            lines.append(f"- [{r['iteration']}] {r['hypothesis']} -> "
                         f"valid {m.get('valid', {}).get('primary', 'err')} "
                         f"{'(error: ' + ','.join(r['errors']) + ')' if r['errors'] else ''}")
    else:
        lines.append("  (none)")
    lines += ["", "Propose the next experiment via propose_experiment."]
    return "\n".join(lines)


def reflector_message(result: dict) -> str:
    """Ask the reflector to interpret one experiment's outcome."""
    return (
        "Interpret this experiment result. Judge improvement on the PRIMARY metric only "
        "(primary = mean of GAUC and nDCG@5, ~0.60). Do NOT confuse GAUC (~0.66) with "
        "primary (~0.60).\n"
        "The `config` field is the model actually run; do not assume any component that "
        "is not in it (e.g. a multi-task/CWM head) was applied.\n"
        + json.dumps(result, indent=2, default=float)
        + "\n\nReturn JSON: {\"verdict\": \"accept|reject\", "
          "\"lesson\": \"<one concrete, reusable lesson grounded in primary>\"}"
    )


REFLECT_TOOL = {
    "name": "reflect",
    "description": "Verdict + lesson after seeing an experiment result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["accept", "reject"]},
            "lesson": {"type": "string"},
        },
        "required": ["verdict", "lesson"],
    },
}
