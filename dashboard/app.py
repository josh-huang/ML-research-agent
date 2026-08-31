"""Streamlit dashboard — interactive replay of the autonomous research run.

Reads ``run_logs/run_log.jsonl`` + ``run_logs/state.json`` and renders the
iteration trail the agent produced: header metrics (best primary + delta vs the
official baseline), the primary trajectory (valid & test vs iteration, with the
baseline reference), per-iteration detail, and the agent's procedural lessons.

Run from the project root::

    streamlit run dashboard/app.py

No state is written here — this is a read-only consumer of the log artifacts.
The rendering body is a 5s auto-refreshing fragment, so the dashboard tracks the
agent as it runs without any manual cache-bust / browser rerun.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOG = os.path.join(_ROOT, "run_logs", "run_log.jsonl")
STATE = os.path.join(_ROOT, "run_logs", "state.json")

BASELINE_VALID = 0.6016
BASELINE_TEST = 0.5946
ORACLE = 0.8645

st.set_page_config(page_title="Autonomous ML Research Agent — Dashboard",
                   page_icon="🧪", layout="wide")


def _load():
    # No @st.cache_data: this is a live monitor — each fragment tick re-reads the
    # (small) JSONL so freshly-written iterations show up with no cache clear.
    iters = []
    if os.path.exists(LOG):
        with open(LOG) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    iters.append(json.loads(ln))
    state = {}
    if os.path.exists(STATE):
        with open(STATE) as f:
            state = json.load(f)
    return iters, state


def _primary(m: dict | None, split: str) -> float | None:
    if not m:
        return None
    v = m.get(split, {}) or {}
    return v.get("primary")


def _frame(iters):
    rows = []
    for it in iters:
        m = it.get("metrics") or {}
        cfg = it.get("action") or {}
        rows.append({
            "iteration": it.get("iteration"),
            "model": cfg.get("model", "?"),
            "loss": cfg.get("loss", "?"),
            "k": cfg.get("k"),
            "lr": cfg.get("lr"),
            "dropout": cfg.get("dropout"),
            "aux": cfg.get("aux"),
            "use_videoside": bool(cfg.get("use_videoside")),
            "use_userside": bool(cfg.get("use_userside")),
            "use_tagside": bool(cfg.get("use_tagside")),
            "valid_primary": _primary(m, "valid"),
            "test_primary": _primary(m, "test"),
            "verdict": it.get("verdict", ""),
            "tokens": it.get("tokens", 0),
            "gpu_h": it.get("gpu_h", 0.0),
            "duration_s": it.get("duration_s", 0.0),
            "hypothesis": it.get("hypothesis", ""),
            "lesson": it.get("lesson", ""),
            "errors": "; ".join(it.get("errors") or []),
            "recovery": it.get("recovery") or "",
        })
    return pd.DataFrame(rows)


def _lessons(iters):
    seen, out = [], []
    for it in reversed(iters):
        lesson = (it.get("lesson") or "").strip()
        if lesson and lesson not in seen:
            seen.append(lesson)
            out.append({"lesson": lesson, "iteration": it.get("iteration")})
    return out


@st.fragment(run_every=5)
def _render():
    iters, state = _load()
    df = _frame(iters)
    lessons = _lessons(iters)

    st.title("🧪 Autonomous ML Research Agent")
    st.caption("KuaiRand-Pure · within-user ranking · label `long_view` · "
               "primary = mean(GAUC, nDCG@5) · progress measured vs oracle ceiling 0.8645")

    if not iters:
        st.warning("No run-log yet. Run `python -m agent.main --fresh --max_iters N` first.")
        return

    best_valid = state.get("best_primary", 0.0)
    best_test = None
    if state.get("best_config"):
        # locate the best iteration's test primary from the log
        for it in iters:
            if it.get("iteration") == state.get("best_iter"):
                best_test = _primary(it.get("metrics"), "test")
                break

    # --- header cards ---------------------------------------------------------
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Best valid primary", f"{best_valid:.4f}",
              f"+{best_valid - BASELINE_VALID:.4f} vs baseline", delta_color="normal")
    c2.metric("Best test primary", f"{best_test:.4f}" if best_test else "—",
              f"+{best_test - BASELINE_TEST:.4f} vs baseline" if best_test else None,
              delta_color="normal")
    c3.metric("Iterations", state.get("iterations", 0))
    c4.metric("Tokens", f"{state.get('tokens_used', 0):,}")
    c5.metric("GPU-hours", f"{state.get('gpu_hours', 0):.3f}")
    c6.metric("Converged", "yes" if state.get("converged") else "no",
              f"{state.get('stagnant', 0)}/3 stagnant")

    # oracle progress
    if best_test:
        headroom = ORACLE - BASELINE_TEST
        progress = (best_test - BASELINE_TEST) / headroom if headroom else 0.0
        st.progress(min(max(progress, 0.0), 1.0),
                    text=f"test primary progress vs oracle ceiling: {progress:.1%} "
                         f"({BASELINE_TEST:.4f} → {best_test:.4f} → {ORACLE})")

    # --- trajectory -----------------------------------------------------------
    st.subheader("Primary trajectory")
    chart = df.set_index("iteration")[["valid_primary", "test_primary"]].copy()
    chart["baseline valid"] = BASELINE_VALID
    chart["baseline test"] = BASELINE_TEST
    st.line_chart(chart)

    # --- per-iteration table --------------------------------------------------
    st.subheader("Iterations")
    show = df[["iteration", "model", "loss", "k", "lr", "aux", "use_videoside",
               "use_userside", "use_tagside", "valid_primary", "test_primary", "verdict",
               "tokens", "gpu_h", "errors"]]
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={
                     "valid_primary": st.column_config.NumberColumn(format="%.4f"),
                     "test_primary": st.column_config.NumberColumn(format="%.4f"),
                     "lr": st.column_config.NumberColumn(format="%.4g"),
                     "gpu_h": st.column_config.NumberColumn(format="%.4f"),
                     "use_videoside": st.column_config.CheckboxColumn(),
                     "use_userside": st.column_config.CheckboxColumn(),
                     "use_tagside": st.column_config.CheckboxColumn(),
                 })

    st.subheader("Hypothesis & lesson per iteration")
    for _, r in df.iterrows():
        with st.expander(f"[{int(r['iteration'])}] {r['model']}+{r['loss']} "
                         f"→ valid {r['valid_primary']:.4f}" if r["valid_primary"] is not None
                         else f"[{int(r['iteration'])}] {r['model']}+{r['loss']} → ERROR"):
            st.markdown(f"**Hypothesis** — {r['hypothesis']}")
            if r["lesson"]:
                st.markdown(f"**Lesson** — {r['lesson']}")
            if r["errors"]:
                st.error(f"errors: {r['errors']}" + (f" · recovery: {r['recovery']}"
                                                      if r["recovery"] else ""))

    # --- lessons (procedural memory) ------------------------------------------
    st.subheader("Lessons learned (procedural memory)")
    if lessons:
        for i, l in enumerate(lessons, 1):
            st.markdown(f"{i}. [iter {l['iteration']}] {l['lesson']}")
    else:
        st.caption("No lessons recorded.")

    st.caption(f"Best config: `{json.dumps(state.get('best_config'), sort_keys=True)}`")


_render()
