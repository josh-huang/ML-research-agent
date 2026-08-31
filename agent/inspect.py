"""Read-only data/feature probes for the ReAct agent.

The agent's only "act" was ``run_experiment`` over a fixed config space, but the raw CSVs
hold many *un-encoded* signals (video ``upload_type``/``tag``/``server_width``, user
``is_lowactive_period``/``register_days``/``onehot_feat*``, log ``hourmin``/
``profile_stay_time``, ...) that the model never sees. These three tools let the agent see
them and quantify whether they are rank-relevant for *within-user* ranking — BEFORE burning
GPU to wire one in.

All three are strictly read-only: they import nothing from the model forward graph, never
touch ``evaluate.py``, and never mutate ``search``'s action space. Leakage fields (aggregated
play/engagement counts) are refused at the boundary.

Core diagnostic = **rank-relevance** = within-user variance / total variance (0..1). A field
whose value is constant per user (varies only across users, not across a user's items) has
within-user variance ~0 -> rank-relevance ~0 -> it cannot reorder a user's own list, so it is
rank-irrelevant for GAUC/nDCG@5. This turns the playbook's "user-side linear term is a
per-user constant" into something the agent can measure rather than memorise.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.eda import _phi, _point_biserial  # noqa: E402
from models.data_loader import LOG_FILES  # noqa: E402

# --- field registry ---------------------------------------------------------

# Open fields the agent may probe, grouped by source table. `encoded` means the field is
# already wired into the model (official 5-field encoding or side_features), so it is NOT a
# new-signal candidate. Leakage fields are NOT listed here — they live in BLOCKED and are
# refused outright by _is_leaky.
_LOG_COLS = ("is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
             "is_profile_enter", "profile_stay_time", "comment_stay_time", "duration_ms",
             "hourmin", "time_ms", "tab", "is_rand")
_VIDEO_BASIC_COLS = ("author_id", "video_type", "upload_dt", "upload_type", "visible_status",
                     "video_duration", "server_width", "server_height", "music_id",
                     "music_type", "tag")
_VIDEO_STAT_OPEN = ("play_progress", "comment_stay_duration")  # only non-aggregate, non-completion
_USER_COLS = ("user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
              "follow_user_num", "follow_user_num_range", "fans_user_num", "fans_user_num_range",
              "friend_user_num", "friend_user_num_range", "register_days", "register_days_range") \
             + tuple(f"onehot_feat{i}" for i in range(18))

# Leakage blacklist (strict口径): every *_cnt / *_user_num aggregate in video_stat is a
# near-label play/engagement count; plus play_duration (historical total watch), counts
# (opaque aggregate), and play_time_ms (the row's own watch outcome, unavailable at scoring).
_BLOCKED_EXACT = {"counts", "play_duration", "play_time_ms"}


def _is_leaky(field: str) -> bool:
    return field in _BLOCKED_EXACT or field.endswith("_cnt") or field.endswith("_user_num")


CATALOG = [
    # (name, source, dtype, encoded, note)
    ("video_type", "video_basic", "cat", True, "video category (encoded)"),
    ("music_type", "video_basic", "cat", True, "music category (encoded)"),
    ("author_id", "video_basic", "cat", True, "creator id (official field)"),
    ("video_duration", "video_basic", "cont", True, "duration; used as log_duration"),
    ("play_progress", "video_stat", "cont", True, "avg play progress (item-quality)"),
    ("upload_type", "video_basic", "cat", False, "upload channel type"),
    ("visible_status", "video_basic", "cat", False, "visibility flag"),
    ("server_width", "video_basic", "cont", False, "video width px"),
    ("server_height", "video_basic", "cont", False, "video height px"),
    ("upload_dt", "video_basic", "cont", False, "upload date (recency)"),
    ("music_id", "video_basic", "cat", False, "music track id"),
    ("tag", "video_basic", "cat", False, "free-text video tag"),
    ("comment_stay_duration", "video_stat", "cont", False, "avg comment stay (non-aggregate)"),
    ("user_active_degree", "user", "cat", True, "activity bucket (encoded)"),
    ("follow_user_num_range", "user", "cat", True, "follows bucket (encoded)"),
    ("fans_user_num_range", "user", "cat", True, "fans bucket (encoded)"),
    ("friend_user_num_range", "user", "cat", True, "friends bucket (encoded)"),
    ("register_days_range", "user", "cat", True, "age bucket (encoded)"),
    ("is_lowactive_period", "user", "binary", False, "low-activity window flag"),
    ("is_live_streamer", "user", "binary", False, "is live streamer"),
    ("is_video_author", "user", "binary", False, "is video author"),
    ("follow_user_num", "user", "cont", False, "raw follow count"),
    ("fans_user_num", "user", "cont", False, "raw fan count"),
    ("friend_user_num", "user", "cont", False, "raw friend count"),
    ("register_days", "user", "cont", False, "days since register"),
    ("onehot_feat*", "user", "cat", False, "anonymous one-hot features 0..17"),
    ("is_click", "log", "binary", True, "click feedback (aux target)"),
    ("is_like", "log", "binary", True, "like feedback (aux target)"),
    ("is_profile_enter", "log", "binary", True, "profile-enter (aux target)"),
    ("is_follow", "log", "binary", False, "follow feedback"),
    ("is_comment", "log", "binary", False, "comment feedback"),
    ("is_forward", "log", "binary", False, "forward feedback"),
    ("is_hate", "log", "binary", False, "hate feedback"),
    ("profile_stay_time", "log", "cont", False, "profile stay ms"),
    ("comment_stay_time", "log", "cont", False, "comment stay ms"),
    ("duration_ms", "log", "cont", True, "video duration (official dur_bucket)"),
    ("hourmin", "log", "cont", False, "impression HHMM"),
    ("time_ms", "log", "cont", False, "absolute epoch ms"),
    ("tab", "log", "cat", True, "tab (official field)"),
    ("is_rand", "log", "binary", False, "random-sample flag (~0 in standard logs)"),
]

_OPEN_FIELDS = tuple(c[0] for c in CATALOG)
_ALL_OPEN = set(_LOG_COLS) | set(_VIDEO_BASIC_COLS) | set(_VIDEO_STAT_OPEN) | set(_USER_COLS)


# --- lazy context -----------------------------------------------------------

_LIGHT_COLS = ("user_id", "video_id", "date", "time_ms", "hourmin", "tab", "is_rand",
               "long_view", "is_click", "is_like", "is_follow", "is_comment", "is_forward",
               "is_hate", "is_profile_enter", "duration_ms",
               "profile_stay_time", "comment_stay_time")


def _load_light_df(data_dir: str) -> pd.DataFrame:
    parts = [pd.read_csv(os.path.join(data_dir, f), usecols=_LIGHT_COLS) for f in LOG_FILES]
    return pd.concat(parts, ignore_index=True)


@dataclass
class InspectContext:
    """Lazily-loaded tables for the probes. Read once, cached for the whole run."""
    data_dir: str
    _video_basic: pd.DataFrame | None = None
    _video_stat: pd.DataFrame | None = None
    _user_feat: pd.DataFrame | None = None
    _df: pd.DataFrame | None = None
    _cache: dict = field(default_factory=dict)

    @property
    def video_basic(self) -> pd.DataFrame:
        if self._video_basic is None:
            self._video_basic = pd.read_csv(os.path.join(self.data_dir, "video_features_basic_pure.csv"))
        return self._video_basic

    @property
    def video_stat(self) -> pd.DataFrame:
        if self._video_stat is None:
            self._video_stat = pd.read_csv(os.path.join(self.data_dir, "video_features_statistic_pure.csv"))
        return self._video_stat

    @property
    def user_feat(self) -> pd.DataFrame:
        if self._user_feat is None:
            self._user_feat = pd.read_csv(os.path.join(self.data_dir, "user_features_pure.csv"))
        return self._user_feat

    @property
    def df_light(self) -> pd.DataFrame:
        if self._df is None:
            self._df = _load_light_df(self.data_dir)
        return self._df


def build_inspect_context(data_dir: str) -> InspectContext:
    return InspectContext(data_dir)


# --- feature ops (propose_feature whitelist; no eval) -----------------------

def _segment_apply(df: pd.DataFrame, cols, fn) -> np.ndarray:
    """Group rows by user (time-ordered), apply ``fn(seg_dict)`` per segment, map back.

    ``fn`` receives a dict of per-column float arrays for one user's segment and returns a
    same-length float array. Segments are found with a lexsort over (user_id, time_ms), so
    derived features like history length are computed over the user's *past* only.
    """
    order = np.lexsort((df["time_ms"].values, df["user_id"].values))
    uids = df["user_id"].values[order]
    n = len(df)
    vals = {c: df[c].values.astype(np.float64)[order] for c in cols}
    starts = np.r_[0, np.where(uids[1:] != uids[:-1])[0] + 1]
    ends = np.r_[starts[1:], n]
    out = np.empty(n, dtype=np.float64)
    for s, e in zip(starts, ends):
        seg = {c: v[s:e] for c, v in vals.items()}
        out[order[s:e]] = fn(seg)
    return out


def _hist_len(seg):
    return np.arange(len(seg["time_ms"]), dtype=np.float64)


def _hist_pos_rate(seg):
    y = seg["long_view"]
    r = np.arange(len(y), dtype=np.float64)
    return np.divide(np.cumsum(y) - y, r, out=np.zeros_like(r), where=r > 0)


def _hist_click_rate(seg):
    c = seg["is_click"]
    r = np.arange(len(c), dtype=np.float64)
    return np.divide(np.cumsum(c) - c, r, out=np.zeros_like(r), where=r > 0)


def _time_since_last(seg):
    t = seg["time_ms"]
    return np.concatenate([[np.nan], np.diff(t)])


def _hist_density(seg):
    t = seg["time_ms"]
    r = np.arange(len(t), dtype=np.float64)
    span = np.maximum(t - t[0], 1e-3) / 1000.0
    return np.divide(r, span, out=np.zeros_like(r), where=span > 0)


def _hour_frac(hourmin):
    return (hourmin // 100) + (hourmin % 100) / 60.0


FEATURE_OPS = {
    "hist_len": lambda df: _segment_apply(df, ["time_ms"], _hist_len),
    "hist_density": lambda df: _segment_apply(df, ["time_ms"], _hist_density),
    "hist_pos_rate": lambda df: _segment_apply(df, ["time_ms", "long_view"], _hist_pos_rate),
    "hist_click_rate": lambda df: _segment_apply(df, ["time_ms", "is_click"], _hist_click_rate),
    "time_since_last": lambda df: _segment_apply(df, ["time_ms"], _time_since_last),
    "hour_sin": lambda df: np.sin(2 * np.pi * _hour_frac(df["hourmin"].fillna(0).values.astype(np.float64)) / 24.0),
    "hour_cos": lambda df: np.cos(2 * np.pi * _hour_frac(df["hourmin"].fillna(0).values.astype(np.float64)) / 24.0),
    "log_duration": lambda df: np.log1p(df["duration_ms"].fillna(0).values.astype(np.float64)),
}


# --- diagnostics ------------------------------------------------------------

def _to_numeric(v: np.ndarray) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(pd.Series(v).dtype):
        return v.astype(np.float64)
    return pd.Series(v).factorize()[0].astype(np.float64)


def _rank_relevance(v_num: np.ndarray, uids: np.ndarray) -> float:
    """within-user variance / total variance (0..1); higher = more rank-relevant."""
    order = np.argsort(uids, kind="stable")
    suid = uids[order]
    sv = v_num[order]
    n = len(sv)
    starts = np.r_[0, np.where(suid[1:] != suid[:-1])[0] + 1]
    seg_lens = np.diff(np.r_[starts, n])
    seg_mean = np.repeat(np.add.reduceat(sv, starts) / seg_lens, seg_lens)
    within = float(np.mean((sv - seg_mean) ** 2))
    total = float(sv.var()) if n > 1 else 0.0
    if total == 0.0:
        return 0.0 if within == 0.0 else 1.0
    return within / total


def _rr_label(rr: float) -> str:
    return "HIGH" if rr > 0.3 else ("MID" if rr > 0.1 else "LOW(per-user constant)")


def _get_arrays(ctx: InspectContext, field: str):
    df = ctx.df_light
    y = df["long_view"].values.astype(np.float64)
    uids = df["user_id"].values
    if field in _LOG_COLS:
        return df[field].values, y, uids
    if field in _VIDEO_BASIC_COLS:
        v = ctx.video_basic.set_index("video_id")[field].reindex(df["video_id"].values).values
        return v, y, uids
    if field in _VIDEO_STAT_OPEN:
        v = ctx.video_stat.set_index("video_id")[field].reindex(df["video_id"].values).values
        return v, y, uids
    if field in _USER_COLS:
        v = ctx.user_feat.set_index("user_id")[field].reindex(df["user_id"].values).values
        return v, y, uids
    return None, None, None


def _diagnose(name: str, source: str, v: np.ndarray, y: np.ndarray, uids: np.ndarray) -> str:
    v = np.asarray(v)
    valid = ~pd.isna(v)
    n, n_total = int(valid.sum()), len(v)
    if n == 0:
        return f"{name} (source {source}): all missing."
    vv, yy, uu = v[valid], y[valid], uids[valid]
    cls = "binary" if pd.Series(vv).nunique(dropna=True) <= 2 else (
        "cat" if not pd.api.types.is_numeric_dtype(pd.Series(vv).dtype) else "cont")

    lines = [f"{name} (source {source}, {cls}, coverage {n / n_total:.1%})"]
    rr = _rank_relevance(_to_numeric(vv), uu)
    if cls == "binary":
        pos = float(vv.mean())
        lines.append(f"  pos_rate {pos:.3f} | phi(long_view) {_phi(vv, yy):+.4f} "
                     f"| rank-relevance {rr:.3f} [{_rr_label(rr)}]")
    elif cls == "cont":
        lines.append(f"  p0/p50/p100 {np.min(vv):.4g}/{np.median(vv):.4g}/{np.max(vv):.4g} "
                     f"| point_biserial(long_view) {_point_biserial(vv, yy):+.4f} "
                     f"| rank-relevance {rr:.3f} [{_rr_label(rr)}]")
    else:  # cat
        ser = pd.Series(vv)
        counts = ser.value_counts(dropna=True)
        top = counts.head(8)
        pos_by_cat = pd.Series(yy).groupby(ser).mean()
        spread = float(pos_by_cat.max() - pos_by_cat.min())
        lines.append(f"  unique {counts.size} | long_view rate spread {spread:.4f} "
                     f"| rank-relevance {rr:.3f} [{_rr_label(rr)}]")
        for c, k in top.items():
            lines.append(f"    {c!r}: n={k} ({k / n:.0%}), pos {pos_by_cat.get(c, np.nan):.3f}")
    return "\n".join(lines)


# --- tool entry points ------------------------------------------------------

def list_features() -> str:
    """Compact catalog, un-encoded (new-signal) fields first."""
    unencoded = [c for c in CATALOG if not c[3]]
    encoded = [c for c in CATALOG if c[3]]
    lines = ["Un-encoded signals (candidates for a NEW feature — probe them first):"]
    for name, src, dtype, _, note in unencoded:
        lines.append(f"- {name} [{src}, {dtype}] — {note}")
    lines.append("\nAlready encoded (in the model now): " +
                 ", ".join(sorted(set(name for name, *_ in encoded))))
    lines.append("\nLeakage-blacklisted (refused by probe_feature): all *_cnt / *_user_num "
                 "aggregates, plus play_duration / counts / play_time_ms.")
    return "\n".join(lines)


def probe_feature(ctx: InspectContext | None, field: str) -> str:
    field = str(field or "").strip()
    if not ctx:
        return "inspect context unavailable (data_dir missing)."
    if not field:
        return "specify a field name; call list_features for the open catalog."
    if _is_leaky(field):
        return (f"BLOCKED: '{field}' is leakage (aggregated play/engagement count or the row's "
                f"own watch outcome). Not probeable.")
    if field not in _ALL_OPEN:
        return f"unknown field '{field}'. call list_features to see open fields."
    v, y, uids = _get_arrays(ctx, field)
    return _diagnose(field, _source_of(field), v, y, uids)


def _source_of(field: str) -> str:
    if field in _LOG_COLS:
        return "log"
    if field in _VIDEO_BASIC_COLS:
        return "video_basic"
    if field in _VIDEO_STAT_OPEN:
        return "video_stat"
    return "user"


def propose_feature(ctx: InspectContext | None, name: str) -> str:
    name = str(name or "").strip()
    if not ctx:
        return "inspect context unavailable (data_dir missing)."
    if name not in FEATURE_OPS:
        return f"unknown feature op '{name}'. available: {', '.join(FEATURE_OPS)}."
    df = ctx.df_light
    v = FEATURE_OPS[name](df)
    y = df["long_view"].values.astype(np.float64)
    uids = df["user_id"].values
    body = _diagnose(name, "derived", v, y, uids)
    # Heuristic verdict: rank-relevance is the primary gate (within-user ranking), but a
    # strong point-biserial with at least mid rank-relevance is also worth wiring.
    m = ~pd.isna(v)
    vv, yy, uu = np.asarray(v)[m], y[m], uids[m]
    rr = _rank_relevance(_to_numeric(vv), uu)
    pb = _point_biserial(vv.astype(np.float64), yy)
    worth = rr > 0.3 or (rr > 0.1 and abs(pb) > 0.1)
    verdict = ("worth wiring in as a new side feature" if worth else
               "weak rank-relevance — low priority for wiring")
    return body + f"\n  verdict: {verdict}."
