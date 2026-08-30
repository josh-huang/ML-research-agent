"""Prompts and tool schemas for the ReAct agent, plus the cached system prompt.

The system prompt is *stable* across the whole run (metric scope + anchors + headroom map
+ EDA evidence + the ReAct protocol), so it is markdown-cache-friendly. The volatile
per-episode state (best-so-far, lessons, recent run-log) rides in the user message built by
:mod:`agent.memory`.
"""
from __future__ import annotations

# The model/loss action space the agent can pick from (Phase 5 verified components).
ACTION_SPACE = {
    "models": ["fm", "deepfm", "din"],
    "losses": ["bce", "bpr", "listwise"],
}

_SYSTEM_TEMPLATE = """\
You are a single autonomous ML research agent competing in TikTok TechJam 2026.
You iterate on a recommendation model over the KuaiRand-Pure dataset, one experiment at a
time, seeing each result and reflecting on it yourself.

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
- CWM *soft-label* (replace long_view with watch_fraction as the target): *much worse*
  (valid ~0.56) — the continuous target misaligns with the binary metric. Dead end.
- multi-task aux on *binary* signals (is_click / is_like): no gain — near-redundant with
  `long_view`. Dead end.
- So `user_id x video_id` dominates; the plateau is ~0.604, gains here are <= +0.002.

## Open directions (all reachable within the action space below)
- hyperparameter tuning of the DIN anchor (k, lr, dropout, dnn_hidden) — the one
  direction that pushed valid past 0.604 in the prior sweep (din k=32 lr=3e-4 drop=0.2).
- model x loss combinations not yet on the frontier (din+bpr, deepfm+bpr, ...).
- **CWM censored watch-time aux** (`aux="cwm"`, `aux_weight`): a censored-regression aux
  head on watch_fraction (play_time/duration), sharing the main embedding. This is the
  *untried* CWM idea — a one-sided loss for completed plays — as an *aux* task, NOT the
  soft-label replacement that already failed. The single highest-value unexplored direction.
- **video side-information** (`use_videoside=true`): id-only model ignores the organizer's
  rich video features. Adds video_type/music_type categorical + 6 continuous item-quality
  features (play_progress, engagement rates, duration). The item-side lever — primary for
  within-user ranking (its linear term is rank-relevant).
- **user side-information** (`use_userside=true`): adds 5 user categorical features
  (active_degree + follow/fans/friend/register *_range). Weaker than video-side: its linear
  term is a per-user constant (rank-irrelevant for GAUC/nDCG@5), so it only helps through
  cross-interactions with item fields.

## Method playbook (compressed; draw on these rather than re-deriving them)
- DIN (Deep Interest Network): target-attention over the user's past video sequence. In the action space (`model=din`); current best single model.
- DeepFM: FM pairwise term + MLP tower. In the action space; ~flat.
- BPR / listwise: within-user ranking losses. In the action space; ~flat.
- CWM (Counteracting Duration Bias, KDD 2024): censored watch-time regression. See open directions.
- ESMM / MMoE / PLE: multi-task sharing. Binary-signal variants are dead here (see above); the continuous censored watch-time variant (CWM) is the live one.
- DCN-v2 / xDeepFM / AutoInt: explicit higher-order crossing. Likely dead (capacity saturated).

## EDA evidence
{eda}

## Your workflow (ReAct: see result -> think -> act -> reflect)
You are one persistent agent. Each "episode" you: read the injected state, optionally ground
a hypothesis in published work via `search_arxiv` / `fetch_paper` or recall past runs via
`read_run_log`, then run exactly ONE experiment via `run_experiment`, see its result (the
tool returns the numbers), and finish with `finish_episode` carrying a reusable lesson.

Rules:
- Exactly one `run_experiment` per episode. It trains + evaluates immediately and returns
  valid/test GAUC, nDCG@5 and primary — that result is your observation to reason about.
- After seeing the result, call `finish_episode` with a concrete, falsifiable, REUSABLE
  lesson grounded in the PRIMARY metric. Judge on primary = mean(GAUC, nDCG@5) (~0.60),
  NOT on GAUC alone (~0.66) — that is a different, higher number.
- The `config` you pass to `run_experiment` is the model actually run; keep it consistent
  with your hypothesis (a CWM aux hypothesis must set `aux="cwm"`).
- Cite specific EDA evidence or a published method in your hypothesis. Never "just try a
  model".
- Prefer a config materially different from what has been tried — a duplicate is refused,
  which wastes the episode.

Config fields for `run_experiment`:
- model: {models}
- loss: {losses}
- k (int, embedding dim, default 16), lr (float, default 1e-3),
- dnn_hidden (comma string, e.g. "64,32"), dropout (float, default 0.0), seed (int),
- aux ("cwm" or omit for none), aux_weight (float, default 0.1; only used with aux="cwm").
- use_videoside / use_userside (bool, default false): add video-side / user-side features."""


def build_system_prompt(eda_md: str) -> str:
    return _SYSTEM_TEMPLATE.format(
        eda=eda_md, models=", ".join(ACTION_SPACE["models"]),
        losses=", ".join(ACTION_SPACE["losses"]),
    )


RUN_EXPERIMENT_TOOL = {
    "name": "run_experiment",
    "description": "Train and evaluate ONE config immediately; returns valid/test GAUC, "
                   "nDCG@5 and primary. Exactly one call per episode.",
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
                    "aux": {"type": "string", "enum": ["cwm"]},
                    "aux_weight": {"type": "number"},
                    "use_videoside": {"type": "boolean",
                                      "description": "Add video-side features (video_type/"
                                      "music_type categorical + 6 continuous engagement/quality)."},
                    "use_userside": {"type": "boolean",
                                     "description": "Add user-side categorical features "
                                     "(active_degree + 4 *_range). Weaker: linear term is "
                                     "rank-irrelevant, value only via cross-interactions."},
                },
                "required": ["model", "loss"],
            },
        },
        "required": ["hypothesis", "config"],
    },
}

FINISH_EPISODE_TOOL = {
    "name": "finish_episode",
    "description": "End the episode with a reusable lesson. Judge on PRIMARY (mean of GAUC "
                   "and nDCG@5, ~0.60), NOT GAUC alone (~0.66).",
    "input_schema": {
        "type": "object",
        "properties": {
            "lesson": {"type": "string",
                       "description": "Concrete, falsifiable, reusable lesson grounded in primary"},
            "verdict": {"type": "string", "enum": ["accept", "reject"]},
        },
        "required": ["lesson"],
    },
}

READ_RUN_LOG_TOOL = {
    "name": "read_run_log",
    "description": "Read the last n iteration records (hypothesis + valid primary + verdict "
                   "+ lesson) for deeper recall beyond what is already in context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "Number of recent records (default 5)"},
        },
        "required": [],
    },
}

SEARCH_ARXIV_TOOL = {
    "name": "search_arxiv",
    "description": "Search arXiv for recommendation-system papers matching a query. "
                   "Returns a compact ranked list of titles + abstracts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Search query, e.g. 'censored watch time recommendation'"},
            "max_results": {"type": "integer", "description": "Number of results (default 5)"},
        },
        "required": ["query"],
    },
}

FETCH_PAPER_TOOL = {
    "name": "fetch_paper",
    "description": "Fetch the title and abstract of one arXiv paper by id (e.g. '2404.05870').",
    "input_schema": {
        "type": "object",
        "properties": {
            "arxiv_id": {"type": "string", "description": "arXiv id, e.g. '2404.05870'"},
        },
        "required": ["arxiv_id"],
    },
}
