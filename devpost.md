# Autonomous ML Research Agent — TikTok TechJam 2026 (Devpost)

## What it does

This is an **autonomous machine-learning research agent**. Given the KuaiRand-Pure
recommendation dataset and a fixed metric (within-user ranking of `long_view`, scored by
`mean(GAUC, nDCG@5)`), the agent runs the entire research loop **by itself** — reproduce the
baseline, run data-driven EDA, propose a hypothesis, train, evaluate, reflect, and iterate —
until it converges, under an explicit token / GPU-hour / human-intervention budget.

In its autonomous run it reproduced the FM baseline (`primary 0.5946`), explored the
headroom, and converged at `primary 0.5978` (+0.0032 over baseline) **with zero human
interventions, 3,711 tokens, and 0.029 GPU-hours**, leaving a complete JSONL run-log and an
HTML report as evidence.

## How we built it

- **LLM brain (Claude Sonnet 5)** drives two roles: a *researcher* that proposes one
  config per turn with a falsifiable hypothesis citing EDA evidence, and a *reflector* that
  distills each result into a reusable lesson. Tool-use enforces structured JSON output, and
  prompt-caching + disabled extended-thinking keep the token bill tiny.
- **Models (PyTorch)** form the agent's action space: FM (baseline anchor), DeepFM, and
  **DIN** (user-history attention), each paired with pointwise BCE / pairwise BPR / listwise
  losses — all written on top of the official `data.py` encoding without touching
  `evaluate.py`.
- **A hand-built "floor" model** (DIN, k=32) is verified first so the score never depends on
  the agent getting lucky; the agent then autonomously confirms and explores around it.

## Challenges we ran into

The hardest problem was *not* engineering — it was discovering that the dataset's ranking
signal is already almost fully captured by `user_id × video_id` and the binary `long_view`
label. Every "richer" direction we (and the agent) tried — pairwise/listwise losses, deeper
models, multi-task with dense auxiliary feedback (`is_click`, phi 0.758), and
censored-watch-time soft-labels (play-time corr 0.634) — either plateaued at ~0.604 or
**actively hurt** because the continuous targets misalign with the binary metric. Honest
agents have to *rule out* dead-ends cheaply; that's the real win here.

## What's next

A censored-watch-time *ranking* loss (rather than soft-label regression), positional/temporal
features over the logged-impression sequence, and seed-ensembling the ~0.604 models.

## Tools, API, libraries, datasets

- **API:** Anthropic Messages API (`claude-sonnet-5`, tool-use + prompt caching)
- **Libraries:** PyTorch, NumPy, pandas, `anthropic` SDK, Streamlit (live dashboard)
- **Dataset:** KuaiRand-Pure (Zenodo), via the official TikTok starter kit
- **Stack:** Python, Claude Code (orchestration), single-GPU RTX 5070 Ti
