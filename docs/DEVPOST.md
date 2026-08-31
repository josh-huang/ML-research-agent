# Devpost — Autonomous ML Research Agent

**Title:** Autonomous ML Research Agent — a self-driving researcher for KuaiRand-Pure

**Subtitle:** An agent that reproduces the FM baseline, then iterates on its own — running EDA,
proposing falsifiable hypotheses, training, evaluating, and reflecting — to maximize within-user
ranking under a token / GPU-hour / intervention budget.

---

## What it does

It's a **fully autonomous ML research agent**. Given the KuaiRand-Pure dataset and the official
FM baseline, it:

1. **Reproduces the baseline** (pointwise BCE FM → valid primary 0.6020).
2. **Runs its own exploratory data analysis**, turning raw features into ranked, token-frugal
   hypotheses.
3. **Proposes one falsifiable experiment per turn** — each grounded in EDA evidence or published
   work (via arXiv retrieval).
4. **Trains and evaluates** against the *official, unmodified* `evaluate.py`.
5. **Reflects** — extracting a reusable lesson — and **converges** on its own (ε = 0.002 over 3
   rounds).

The delivered run went **baseline → plateau in 7 iterations, zero human interventions, zero
errors**, landing at **test primary 0.5988 (+0.0042 over the official 0.5946 baseline)**, and
did something a naive grid-search would not: it ran a **3-seed noise calibration** that proved its
"global max" was real signal, not a lucky seed — then *stopped*, correctly reporting the plateau
instead of burning budget on sub-noise permutations.

## How we built it

- **Model scaffold (PyTorch).** FM → DeepFM → DIN, with pointwise/pairwise/listwise losses,
  side-feature towers (video-side / user-side / tag-side categoricals), and a censored-watch-time
  (CWM) auxiliary loss. Built and verified by hand first, so the score never depends on agent luck.
- **Agent (ReAct).** A single persistent **Claude Sonnet 5** drives a loop over 8 tools
  (`run_experiment`, `propose_experiment`, EDA summary, `search_arxiv` / `fetch_paper`, state,
  logs). A hand-built "floor" is injected as *prior knowledge*, but every turn must still cite
  evidence for a falsifiable hypothesis — the agent re-derives, never blindly copies.
- **Robustness.** Config dedup, error classification (syntax/shape/OOM/NaN) with graceful
  degradation, a no-LLM fallback proposer, and prompt caching + disabled thinking to keep tokens
  low.

## Challenges we ran into

- **The metric misleads you.** Continuous signals (play-time corr 0.634, `is_click` φ 0.758)
  correlate with the label but *hurt* as targets, because they misalign with the binary metric.
  Every "rich" direction plateaus or regresses — the real discovery was *why*.
- **Seed noise is bigger than most "effects."** Single-seed valid-primary deltas below ~0.001 are
  indistinguishable from variance. Quantifying this (3-seed, ±0.0008) was the run's most valuable
  finding, and it re-read many earlier "definitive" flat verdicts as noise-floor-consistent.
- **Routing the agent's LLM calls correctly.** The dev environment injects its own gateway
  credentials; pinning the SDK to `api.anthropic.com` + the identity-linked key's workspace header
  was required for the agent to actually reach Claude.

## Accomplishments we're proud of

- A **real +0.0042 test-primary gain over the official baseline**, variance-bounded (3-seed test
  mean 0.5989 ± 0.0001), not a one-shot.
- **Full autonomy end-to-end:** baseline → EDA → propose → train → evaluate → reflect → converge,
  with 0 interventions, 0 errors, 7 iterations, 0.0535 GPU-hours.
- An agent that discovered and **codified a reusable research discipline** (the seed-noise floor)
  rather than just turning knobs.

## What we learned

- The `user_id × video_id` interaction dominates; the binary `long_view` label already captures
  the ranking signal, so achievable gain is genuinely small (+0.0042). The agent's value is
  **ruling out dead ends cheaply and reproducibly** — the actual job of an autonomous researcher.
- Test is the stable generalization signal (std ≈ 0.0001); valid carries real ±0.0008 seed jitter.
- Prompt caching + disabled thinking + token-frugal EDA keep a 7-iteration autonomous run to
  ~384k tokens.

## What's next

Two *unwired* signal sources the agent's EDA already ranked and named, both outside the current
config-only action space: (1) **true sequential user-history density** (`hist_len` /
`time_since_last`, rank-relevance up to 0.89), and (2) **duration-aware negative-sampling
reweighting**. Both need harness-level code — which is exactly why the agent correctly stopped
and reported the plateau.

## Built with

- **Language:** Python
- **Frameworks/libraries:** PyTorch, pandas, numpy, Streamlit, anthropic (SDK), python-dotenv
- **API:** Anthropic Claude (Messages API)
- **Tools:** PyCharm, VSCode
- **Dataset:** KuaiRand-Pure
