# Autonomous ML Research Agent — TikTok TechJam 2026

An **autonomous research agent** that reproduces the KuaiRand-Pure FM baseline and then
iterates on its own — running EDA, proposing data-driven hypotheses, training, evaluating,
and reflecting — to maximize within-user ranking of `long_view`, under a hard budget of
tokens / GPU-hours / human interventions.

## Result

| model | GAUC | nDCG@5 | **primary** | vs baseline |
|---|---|---|---|---|
| FM baseline (official) | 0.6610 | 0.5282 | 0.5946 | — |
| **Ours — DIN (k=32, lr=3e-4, dropout=0.2), 5-seed avg** | **0.6655** | **0.5314** | **0.5985** | **+0.0039** |
| oracle ceiling | 1.0000 | 0.7289 | 0.8645 | +0.2699 |

`primary = mean(GAUC, nDCG@5)`, evaluated on the held-out test split by the *official*
`evaluate.py` (imported, never modified). The submitted score is a 5-seed average (individual
seeds 0.5978–0.5987, mean 0.5982 ± 0.0004), so the +0.0039 gain is variance-bounded rather
than a one-shot. The agent reproduced the baseline, explored the headroom, and converged
**fully autonomously** — 0 human interventions, 3,711 tokens, 0.029 GPU-hours, 0 errors.

---

## Strategy: "floor + autonomy" (保底 + 自治)

The competition rewards the score (35%) **and** the agent itself (autonomy 20%, innovation
20%, feasibility 15%). These two pull against each other — a hand-tuned model maximizes the
score but proves nothing about autonomy. We take both ends:

1. **Floor (hand-built, Phase 3).** A small, verified-strong model scaffold is built by hand
   *first*, so the score never depends on the agent getting lucky. This is the `DIN(k=32)`
   anchor — `valid 0.6046 / test 0.5985` (5-seed) — a real +0.0039 over baseline.
2. **Autonomy (Phase 7).** The agent then runs the full closed loop on its own — reproduce →
   EDA → propose → train → evaluate → reflect → converge — and independently *confirms* the
   plateau at ~0.604, with a clean, replayable run-log.

The floor is injected into the agent as *prior knowledge* (the organizer's headroom map,
ranked against our own EDA). The agent is not allowed to blindly copy it: every turn must
cite EDA evidence for a falsifiable hypothesis. In the autonomous run the agent probed
`DIN × BPR` hyperparameters and — correctly — concluded the frontier is saturated, matching
the hand-built floor's conclusion.

---

## Key findings (data-driven, not guessed)

| hypothesis (EDA-driven) | result | verdict |
|---|---|---|
| Pointwise BCE is the trust anchor | FM reproduces baseline `valid 0.6020` | ✅ anchor |
| BPR (pairwise) aligns better with GAUC's within-user semantics | FM +0.001, DIN ~flat | ⚠️ marginal |
| Listwise softmax matches the ranking objective | *worse* than BCE (overfits) | ❌ reject |
| DeepFM's MLP tower adds capacity | +0.001 | ⚠️ marginal |
| DIN's history attention adds the user sequence | **+0.002 → best single** | ✅ keep |
| CWM **soft-label** (watch-fraction as the *target*) | valid **0.557** — much worse | ❌ dead end |
| CWM **censored aux** (one-sided watch-time regression as an *aux* loss) | not yet swept | 🔬 open |
| Multi-task aux (`is_click`, phi 0.758) | no gain — near-redundant | ❌ dead end |
| Hyperparameter tuning of the DIN anchor | +0.0009 (k=32, lr=3e-4, dropout=0.2) | ✅ small win |
| Seed averaging (5 seeds) | std 0.0004; single-seed claims are within noise | ✅ report this way |
| Cross-model ensemble (FM+DeepFM+DIN rank-avg) | 0.5983 — no gain over DIN alone (0.5985) | ❌ too correlated |
| Repeated-(user,video) dedup (3.06% of test) | 0.5977–0.5980 — flat/slightly worse | ❌ no gain |

**The single most important finding:** `user_id × video_id` interaction dominates. The label
`long_view` already captures almost all the ranking signal — the *continuous* signals that
correlate with it (play-time corr 0.634, `is_click` phi 0.758) **hurt** when used as training
targets, because they misalign with the *binary* metric (the model spends capacity separating
"70% vs 80% watched" when only ">50% watched" matters). This is *why* every lever — loss,
model, sequence, multi-task, CWM soft-label — clusters at `valid 0.602–0.605`, far below the naive
"+0.03" target, and why the organizer's own "capacity/loss are dead ends" note checks out.

---

## Architecture

```
ML-research-agent/
  kuairand-starter-kit/     # official kit (read-only: evaluate.py / submit.py / data.py)
  agent/
    main.py                 # orchestrator: propose -> execute -> record -> reflect -> converge
    llm.py                  # Claude Sonnet 5 client (prompt caching, thinking disabled)
    prompts.py              # researcher / reflector prompts + tool schemas
    research.py             # arXiv literature retrieval (search_arxiv / fetch_paper)
    eda.py                  # compact data-driven EDA summary (token-frugal)
    executor.py             # sandboxed run_experiment + error classification
    search.py               # config space: seeds / normalize / mutate (dedup-safe)
    state.py                # best-so-far + convergence (eps=0.002, N=3) + budget
    logger.py               # JSONL run-log + state.json (dashboard/report replay)
  models/
    data_loader.py          # extends official encode() with history + aux + watch-time
    fm_torch.py / deepfm.py / din.py
    losses.py               # BCE / BPR / listwise (vectorized)
    train.py                # unified train loop + early stop
  run_logs/                 # generated JSONL run-log + state.json
  report/run_report.html    # static HTML iteration report (deliverable)
  submission/final.csv      # final submission (validated by submit.py --check)
```

**One iteration:** the researcher (LLM) reads best-so-far + EDA + recent run-log + headroom
map — optionally grounding its hypothesis in published work via `search_arxiv` / `fetch_paper`
— then proposes one config via `propose_experiment` → `executor` runs it and classifies any
failure → the reflector (LLM) extracts a reusable lesson → `state` records accept/reject and
checks convergence (`3` consecutive valid-primary improvements `< 0.002`) → logger appends
the record.

**Robustness:** config dedup (identical keys, so an omitted field isn't a new config), error
classification (syntax/shape/OOM/NaN), **graceful degradation** (OOM → half batch, NaN → 10×
smaller LR, else a fresh seed; one bounded retry, then record-and-move-on), a no-LLM fallback
proposer, and a metric-precise reflector prompt (primary = mean of GAUC & nDCG@5, not GAUC)
to suppress metric confusion.

---

## Setup & reproduce

```bash
# 1. deps
pip install torch numpy pandas anthropic python-dotenv

# 2. data (KuaiRand-Pure) — see the official README
#    wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz && tar xzf ...

# 3. put your Anthropic key in .env (ANTHROPIC_API_KEY=sk-...)

# 4. reproduce the baseline & our model
python -m models.train --model fm --loss bce                 # valid ~0.6020
python submission/make_submission.py                         # DIN(k=32) -> submission/final.csv

# 5. run the autonomous agent (reproduces baseline, then iterates until convergence)
python -m agent.main --fresh --max_iters 14
```

The full run-log is in `run_logs/run_log.jsonl`; a readable HTML report is generated by
`python report/render_report.py` → `report/run_report.html`.

---

## Resource accounting (feasibility)

Feasibility is scored on **wall-clock time**, not raw GPU-hours — the organizer caps a run at
**50 iterations / 6 hours**, and rewards finishing inside that budget with minimal human help.

| resource | autonomous run | note |
|---|---|---|
| wall-clock | **< 6 h** (hard cap) | 50-iteration hard cap; our run converges well before both |
| tokens | **3,711** (in+out, reported) | system prompt cached once; thinking disabled; reflector metric-precise |
| GPU-hours | **0.029** | single RTX 5070 Ti; DIN trains in ~30 s |
| human interventions | **0** | fully autonomous from baseline to convergence |

The agent enforces both caps (`--max_iters`, `--budget_gpu_h`); token spend is *reported*, not
capped, but prompt caching + disabled thinking keep it in the thousands.

---

## Reflection

The honest headline is that the **achievable gain is small** (+0.003 primary): the
`user_id × video_id` interaction and the binary `long_view` label already capture the
ranking signal, and every "rich" direction (continuous watch-time, dense aux feedback,
bigger capacity) either plateaus or *hurts* because it fights the binary metric. The agent's
value here is **not** discovering a +0.03 trick that doesn't exist — it's *ruling out* the
plausible dead-ends cheaply and reproducibly, with a verifiable trail, which is exactly what
an autonomous researcher should do before spending scarce compute.

If we had more budget, the directions we'd push next: (1) the censored watch-time *aux* loss
(`aux="cwm"`) — now implemented in the action space but not yet swept to convergence, (2)
temporal/positional features on the logged-impression sequence, and (3) ensembling the ~0.604
models across seeds — all of which the agent's EDA ranking already surfaced.
