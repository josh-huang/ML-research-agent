"""Render the run-log into a self-contained static HTML report.

Reads ``run_logs/run_log.jsonl`` + ``run_logs/state.json`` and writes
``report/run_report.html`` (a single file, no external assets). This is the
"Run & Iteration Logs" deliverable: a readable trajectory chart + per-iteration
table (hypothesis / config / metrics / verdict / cost / delta vs baseline) +
the agent's procedural lessons.

Usage (from project root)::

    python report/render_report.py
"""
from __future__ import annotations

import html
import json
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOG = os.path.join(_ROOT, "run_logs", "run_log.jsonl")
STATE = os.path.join(_ROOT, "run_logs", "state.json")
OUT = os.path.join(_ROOT, "report", "run_report.html")

BASELINE_VALID = 0.6016
BASELINE_TEST = 0.5946
ORACLE = 0.8645


def _load():
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


def _esc(s) -> str:
    return html.escape(str(s))


def _chart(iters, w=900, h=260):
    """Inline SVG line chart of valid primary vs iteration."""
    if not iters:
        return "<p>No iterations yet.</p>"
    n = len(iters)
    vals = []
    for i, it in enumerate(iters):
        m = it.get("metrics") or {}
        v = (m.get("valid") or {}).get("primary")
        vals.append((i + 1, v if v is not None else None))

    pad_l, pad_r, pad_t, pad_b = 52, 16, 18, 34
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    lo = min((v for _, v in vals if v is not None), default=BASELINE_VALID) - 0.003
    hi = max((v for _, v in vals if v is not None), default=BASELINE_VALID) + 0.003
    if hi - lo < 0.004:
        hi = lo + 0.004
    xmax = max(n, 2)

    def X(i):
        return pad_l + (i - 1) / (xmax - 1) * iw if xmax > 1 else pad_l

    def Y(v):
        return pad_t + (1 - (v - lo) / (hi - lo)) * ih

    color = {"new_best": "#16a34a", "accept": "#2563eb", "reject": "#9ca3af", "error": "#dc2626", "": "#9ca3af"}
    parts = []
    # gridlines + y labels
    for g in range(4):
        v = lo + (hi - lo) * g / 3
        y = Y(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" '
                     f'stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
                     f'fill="#6b7280">{v:.3f}</text>')
    # baseline marker
    by = Y(BASELINE_VALID)
    parts.append(f'<line x1="{pad_l}" y1="{by:.1f}" x2="{w-pad_r}" y2="{by:.1f}" '
                 f'stroke="#f59e0b" stroke-dasharray="4,3" stroke-width="1.2"/>')
    parts.append(f'<text x="{w-pad_r}" y="{by-4:.1f}" text-anchor="end" font-size="9" '
                 f'fill="#d97706">baseline {BASELINE_VALID}</text>')
    # polyline
    pts = [(X(i), Y(v)) for i, v in vals if v is not None]
    if len(pts) > 1:
        d = " ".join(f"{'M' if j == 0 else 'L'}{px:.1f},{py:.1f}" for j, (px, py) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="#374151" stroke-width="1.6"/>')
    # points
    for i, v in vals:
        if v is None:
            continue
        c = color.get((iters[i - 1].get("verdict") or ""), "#9ca3af")
        parts.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{c}"/>')
    # x labels
    for i in range(1, n + 1, max(1, n // 12)):
        parts.append(f'<text x="{X(i):.1f}" y="{h-pad_b+16}" text-anchor="middle" font-size="9" '
                     f'fill="#6b7280">{i}</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">{"".join(parts)}</svg>'


def _rows(iters):
    out = []
    for it in iters:
        m = it.get("metrics") or {}
        v = (m.get("valid") or {}).get("primary")
        t = (m.get("test") or {}).get("primary")
        vv = f"{v:.4f}" if v is not None else "—"
        tt = f"{t:.4f}" if t is not None else "—"
        dv = f"{v - BASELINE_VALID:+.4f}" if v is not None else "—"
        cfg = it.get("action") or {}
        desc = f"{cfg.get('model','?')}+{cfg.get('loss','?')} "
        if cfg.get("k"):
            desc += f"k={cfg['k']} "
        if cfg.get("lr"):
            desc += f"lr={cfg['lr']:g}"
        if cfg.get("aux"):
            desc += f" aux={cfg['aux']}"
        if cfg.get("use_videoside"):
            desc += " +vside"
        if cfg.get("use_userside"):
            desc += " +uside"
        if cfg.get("use_tagside"):
            desc += " +tagside"
        verdict = it.get("verdict") or ""
        badge = {"new_best": "best", "accept": "ok", "reject": "—", "error": "ERR"}.get(verdict, verdict)
        errors = "; ".join(it.get("errors") or [])
        out.append(f"""<tr>
<td class="c">{it['iteration']}</td>
<td class="h">{_esc(it.get('hypothesis',''))}</td>
<td class="c mono">{_esc(desc)}</td>
<td class="c num">{vv}</td>
<td class="c num dlt">{dv}</td>
<td class="c num">{tt}</td>
<td class="c v v-{verdict}">{badge}</td>
<td class="c num">{it.get('tokens',0)}</td>
<td class="c num">{it.get('gpu_h',0):.3f}</td>
<td class="c err">{_esc(errors)}</td>
</tr>""")
    return "".join(out)


def _lessons(iters):
    """Dedup'd latest lessons, newest first (the agent's procedural memory)."""
    seen, out = [], []
    for it in reversed(iters):
        lesson = (it.get("lesson") or "").strip()
        if lesson and lesson not in seen:
            seen.append(lesson)
            out.append((it.get("iteration"), lesson))
    return out


def _best_test(iters, state):
    """test primary at the best iteration, else None."""
    bi = state.get("best_iter")
    for it in iters:
        if it.get("iteration") == bi:
            m = it.get("metrics") or {}
            return (m.get("test") or {}).get("primary")
    return None


def _html(iters, state):
    best = state.get("best_primary", 0.0)
    best_test = _best_test(iters, state)
    d_valid = best - BASELINE_VALID
    d_test = (best_test - BASELINE_TEST) if best_test is not None else None
    headroom = ORACLE - BASELINE_TEST
    progress = (d_test / headroom) if d_test is not None else None
    lessons = _lessons(iters)
    lessons_html = "".join(
        f"<li><span class='li'>iter {i}</span> {_esc(l)}</li>" for i, l in lessons
    ) if lessons else "<p>No lessons recorded.</p>"
    best_test_str = f"{best_test:.4f}" if best_test is not None else "—"
    d_test_str = f"{d_test:+.4f}" if d_test is not None else "—"
    progress_str = f"{progress:.1%}" if progress is not None else "—"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Report — Autonomous ML Research Agent</title>
<style>
:root {{ --fg:#111827; --muted:#6b7280; --bg:#ffffff; --card:#f9fafb; --line:#e5e7eb; --accent:#16a34a; }}
body {{ margin:0; font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--fg); background:var(--bg); }}
.wrap {{ max-width:1000px; margin:0 auto; padding:32px 20px 64px; }}
h1 {{ font-size:22px; margin:0 0 4px; }}
.sub {{ color:var(--muted); margin:0 0 24px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:28px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.card .k {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
.card .v {{ font-size:22px; font-weight:650; margin-top:2px; }}
.card .v.best {{ color:var(--accent); }}
.card .d {{ font-size:11px; color:var(--muted); margin-top:4px; }}
h2 {{ font-size:16px; margin:28px 0 10px; }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
th,td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
td.c {{ white-space:nowrap; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
td.dlt {{ color:var(--muted); }}
td.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.5px; }}
td.h {{ max-width:340px; }}
td.err {{ color:#dc2626; font-size:11px; }}
.v-new_best {{ color:#16a34a; font-weight:650; }}
.v-accept {{ color:#2563eb; }}
.v-reject {{ color:#9ca3af; }}
.v-error {{ color:#dc2626; font-weight:650; }}
ol.lessons {{ padding-left:20px; margin:8px 0; }}
ol.lessons li {{ margin:8px 0; font-size:13px; }}
.li {{ color:var(--muted); font-size:11px; }}
footer {{ margin-top:32px; color:var(--muted); font-size:11px; }}
</style></head><body><div class="wrap">
<h1>Autonomous ML Research Agent — Run Report</h1>
<p class="sub">KuaiRand-Pure · within-user ranking · label <code>long_view</code> · primary = mean(GAUC, nDCG@5)</p>
<div class="cards">
<div class="card"><div class="k">Best valid primary</div><div class="v best">{best:.4f}</div><div class="d">Δ {d_valid:+.4f} vs baseline {BASELINE_VALID}</div></div>
<div class="card"><div class="k">Best test primary</div><div class="v best">{best_test_str}</div><div class="d">Δ {d_test_str} vs baseline {BASELINE_TEST}</div></div>
<div class="card"><div class="k">Progress vs oracle</div><div class="v">{progress_str}</div><div class="d">ceiling {ORACLE}</div></div>
<div class="card"><div class="k">Iterations</div><div class="v">{state.get('iterations', 0)}</div></div>
<div class="card"><div class="k">Tokens</div><div class="v">{state.get('tokens_used', 0):,}</div></div>
<div class="card"><div class="k">GPU-hours</div><div class="v">{state.get('gpu_hours', 0):.3f}</div></div>
<div class="card"><div class="k">Converged</div><div class="v">{'yes' if state.get('converged') else 'no'}</div><div class="d">{state.get('stagnant', 0)}/3 stagnant</div></div>
<div class="card"><div class="k">Configs tried</div><div class="v">{state.get('n_configs_tried', 0)}</div></div>
</div>
<h2>Trajectory (valid primary vs iteration)</h2>
{_chart(iters)}
<h2>Iterations</h2>
<table><thead><tr><th>#</th><th>Hypothesis</th><th>Config</th><th>valid</th><th>Δ valid</th><th>test</th><th>verdict</th><th>tok</th><th>gpu-h</th><th>error</th></tr></thead>
<tbody>{_rows(iters)}</tbody></table>
<h2>Lessons learned</h2>
<ol class="lessons">{lessons_html}</ol>
<footer>Generated by report/render_report.py · baseline FM valid {BASELINE_VALID} / test {BASELINE_TEST} · oracle ceiling {ORACLE} · human interventions {state.get('interventions', 0)}</footer>
</div></body></html>"""


def main():
    iters, state = _load()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(_html(iters, state))
    best = state.get("best_primary", 0)
    print(f"wrote {OUT} ({len(iters)} iterations, best valid {best:.4f} "
          f"({best - BASELINE_VALID:+.4f} vs baseline))")


if __name__ == "__main__":
    main()
