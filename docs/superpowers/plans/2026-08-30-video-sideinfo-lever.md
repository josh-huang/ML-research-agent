# 视频侧 + 用户侧特征 lever 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 FM / DeepFM / DIN 加上视频侧 + 用户侧特征，作为两个独立可搜索的 toggle（`use_videoside` / `use_userside`），突破 valid primary ~0.604 平台期，并让 Agent 能独立消融两侧的边际贡献。

**Architecture:** 新建 `models/side_features.py` 负责纯特征工程（类别编码 + 连续归一化，train-only 统计），`data_loader.load_extended` 产出 `X_vside` / `X_uside` / `cont` 三个张量；三模型共享 embedding 扩维吃类别域，`cont` 拼进 deep tower（FM 走无偏置线性项）；两个 toggle 都关时与现状逐 bit 一致，保 baseline 不回归。

**Tech Stack:** Python 3 / numpy / pandas / PyTorch（CPU+单卡 CUDA）；无新依赖。

**Spec:** [docs/superpowers/specs/2026-08-30-video-sideinfo-lever-design.md](../specs/2026-08-30-video-sideinfo-lever-design.md)

## Global Constraints

- 代码/注释/命令/变量名英文；文档与交互中文。
- **不修改** starter-kit 内任何文件（`data.py` / `evaluate.py` / `ablation_features.py` 等只读；`evaluate.py` 只 import）。
- 评分口径唯一权威：label `long_view`，GAUC + nDCG@5，`primary = mean(GAUC, nDCG@5)`。
- 基线 valid 0.6016 / test 0.5946；oracle 0.8645；收敛 ε=0.002、N=3。
- 改必验：每个改动后跑 `python -m pytest`（现 10 用例全绿），关键改动跑 `python -m models.train ...` 或 `agent.main` 冒烟。
- 禁 `Any`、禁注释报错、禁 Bypass——必须解决根因。
- 凭证/密钥不入代码、不入 commit、不入日志。
- Git：commit 不加 `Co-Authored-By`，以用户本人（josh-huang）署名；只 commit 本计划涉及的文件。

---

### Task 1: `side_features` 表构建器（视频侧 + 用户侧）

**Files:**
- Create: `models/side_features.py`
- Test: `tests/test_side_features.py`

**Interfaces:**
- Produces:
  - `models.side_features.CAT_VIDEO_FEATURES = ("video_type","music_type")`
  - `models.side_features.CAT_USER_FEATURES = ("user_active_degree","follow_user_num_range","fans_user_num_range","friend_user_num_range","register_days_range")`
  - `models.side_features.CONT_FEATURES = ("play_progress","like_rate","comment_rate","follow_rate","share_rate","log_duration")`
  - `build_video_side_table(basic_df, stat_df) -> pd.DataFrame`（video_id 唯一索引）
  - `build_user_side_table(user_df) -> pd.DataFrame`（user_id 唯一索引）

- [ ] **Step 1: 写失败测试**

`tests/test_side_features.py`：

```python
"""Unit tests for side-feature engineering (models.side_features)."""
import numpy as np
import pandas as pd
import pytest

from models.side_features import (
    CAT_USER_FEATURES, CAT_VIDEO_FEATURES, CONT_FEATURES,
    build_user_side_table, build_video_side_table)


def _basic_df():
    return pd.DataFrame({
        "video_id": ["v1", "v2", "v3"],
        "video_type": ["music", "short", "news"],
        "music_type": ["pop", "none", "jazz"],
        "video_duration": [0.0, 60.0, 120.0],
    })


def _stat_df():
    return pd.DataFrame({
        "video_id": ["v1", "v2", "v3"],
        "show_cnt": [100, 50, 0],          # v3 zero shows -> denominator clips to 1
        "play_progress": [0.5, 0.9, 0.0],
        "like_cnt": [50, 60, 0],           # v2 like_cnt > show_cnt -> clip to 1.0
        "comment_cnt": [10, 0, 0],
        "follow_cnt": [5, 1, 0],
        "share_cnt": [2, 0, 0],
    })


def _user_df():
    return pd.DataFrame({
        "user_id": ["u1", "u2"],
        "user_active_degree": ["high", "low"],
        "follow_user_num_range": ["a", "b"],
        "fans_user_num_range": ["a", "b"],
        "friend_user_num_range": ["a", "b"],
        "register_days_range": ["a", "b"],
        "register_days": [10, 20],         # unrelated col, must not leak in
    })


def test_build_video_side_table_columns_and_rates():
    t = build_video_side_table(_basic_df(), _stat_df())
    assert list(t.index) == ["v1", "v2", "v3"]
    assert list(t.columns) == list(CAT_VIDEO_FEATURES) + list(CONT_FEATURES)
    assert t.loc["v1", "video_type"] == "music"
    assert t.loc["v3", "music_type"] == "jazz"
    assert t.loc["v1", "like_rate"] == pytest.approx(0.5)
    assert t.loc["v2", "like_rate"] == pytest.approx(1.0)   # 60/50 -> clip
    assert t.loc["v3", "like_rate"] == pytest.approx(0.0)   # 0 / max(0,1) -> 0
    assert t.loc["v2", "log_duration"] == pytest.approx(np.log1p(60.0))


def test_build_user_side_table_columns():
    t = build_user_side_table(_user_df())
    assert list(t.index) == ["u1", "u2"]
    assert list(t.columns) == list(CAT_USER_FEATURES)   # register_days excluded
    assert t.loc["u1", "user_active_degree"] == "high"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_side_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.side_features'`

- [ ] **Step 3: 写最小实现**

`models/side_features.py`：

```python
"""Side-information feature engineering (item-side + user-side levers).

The official 5-field encoding is id-only; this module builds the organizer's video-side
and user-side signals into tensors a model can consume:

  * categorical side (``X_vside`` / ``X_uside``): sparse fields, offset-encoded (per-field
    vocab + UNK slot) so they merge into the shared embedding.
  * continuous side (``cont``): 6 video item-quality features, z-scored on TRAIN only.

Strict leakage口径: excludes ``complete_play_cnt`` / ``long_time_play_cnt`` /
``valid_play_cnt`` / ``play_cnt`` / raw ``show_cnt`` (near-label aggregates).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_VIDEO_FEATURES = ("video_type", "music_type")
CAT_USER_FEATURES = ("user_active_degree", "follow_user_num_range",
                     "fans_user_num_range", "friend_user_num_range", "register_days_range")
CONT_FEATURES = ("play_progress", "like_rate", "comment_rate",
                 "follow_rate", "share_rate", "log_duration")


def build_video_side_table(basic_df: pd.DataFrame, stat_df: pd.DataFrame) -> pd.DataFrame:
    """video_id-indexed table of categorical + continuous video side features (pure).

    ``basic_df``: columns include video_id, video_type, music_type, video_duration.
    ``stat_df``:  columns include video_id, show_cnt, play_progress, like_cnt,
                  comment_cnt, follow_cnt, share_cnt.
    """
    basic = basic_df.set_index("video_id")
    stat = stat_df.set_index("video_id")
    show = stat["show_cnt"].astype(float).clip(lower=1.0)
    return pd.DataFrame({
        "video_type": basic["video_type"],
        "music_type": basic["music_type"],
        "play_progress": stat["play_progress"].astype(float),
        "like_rate": (stat["like_cnt"].astype(float) / show).clip(0.0, 1.0),
        "comment_rate": (stat["comment_cnt"].astype(float) / show).clip(0.0, 1.0),
        "follow_rate": (stat["follow_cnt"].astype(float) / show).clip(0.0, 1.0),
        "share_rate": (stat["share_cnt"].astype(float) / show).clip(0.0, 1.0),
        "log_duration": np.log1p(basic["video_duration"].astype(float)),
    })


def build_user_side_table(user_df: pd.DataFrame) -> pd.DataFrame:
    """user_id-indexed categorical user-side table (pure)."""
    return user_df.set_index("user_id")[list(CAT_USER_FEATURES)]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_side_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/side_features.py tests/test_side_features.py
git commit -m "feat: add video/user side-feature table builders"
```

---

### Task 2: `encode_cat_fields` + `encode_cont_fields`（train-only 编码/归一化）

**Files:**
- Modify: `models/side_features.py`（追加两个编码器）
- Test: `tests/test_side_features.py`（追加测试）

**Interfaces:**
- Produces:
  - `encode_cat_fields(table, cols, keys, n_train) -> (X_cat, side_dim, n_fields)`；`X_cat` (N, len(cols)) int32、`side_dim` int、`n_fields` int。
  - `encode_cont_fields(table, cols, keys, n_train) -> (cont, cont_dim)`；`cont` (N, len(cols)) float32。

- [ ] **Step 1: 写失败测试**

在 `tests/test_side_features.py` 追加：

```python
from models.side_features import encode_cat_fields, encode_cont_fields


def test_encode_cat_fields_offsets_and_unk():
    table = pd.DataFrame({
        "video_type": ["music", "short"],
        "music_type": ["pop", "none"],
    }, index=["v1", "v2"])
    keys = np.array(["v1", "v2", "v1", "v_new"])   # v1 repeats; v_new unseen
    X, side_dim, n_fields = encode_cat_fields(table, ["video_type", "music_type"], keys, n_train=2)

    assert X.shape == (4, 2)
    assert n_fields == 2
    # video_type vocab from train = {music:0, short:1}; UNK = 2
    assert X[0, 0] == 0          # v1 music
    assert X[1, 0] == 1          # v2 short
    assert X[3, 0] == 2          # v_new -> UNK
    # music_type offset starts after video_type dim (3)
    assert X[0, 1] == 3          # v1 pop -> 0 + 3
    assert X[1, 1] == 4          # v2 none -> 1 + 3
    assert side_dim == 6         # video_type(3) + music_type(3)


def test_encode_cont_fields_train_only():
    table = pd.DataFrame({"play_progress": [0.5, 0.9]}, index=["v1", "v2"])
    keys = np.array(["v1", "v2", "v1"])
    cont, cont_dim = encode_cont_fields(table, ["play_progress"], keys, n_train=2)

    assert cont_dim == 1
    # train mean 0.7, std 0.2 -> (0.5 - 0.7)/0.2 = -1.0
    assert cont[0, 0] == pytest.approx(-1.0)
    assert cont[0, 0] == pytest.approx(cont[2, 0])   # same video -> same value
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_side_features.py::test_encode_cat_fields_offsets_and_unk tests/test_side_features.py::test_encode_cont_fields_train_only -v`
Expected: FAIL — `ImportError: cannot import name 'encode_cat_fields'`

- [ ] **Step 3: 写最小实现**

在 `models/side_features.py` 追加：

```python
def encode_cat_fields(table: pd.DataFrame, cols, keys, n_train: int):
    """Encode categorical columns ``cols`` of a key-indexed ``table`` for ``keys``.

    Vocab is built from the leading ``n_train`` rows only; unseen keys (or NaN) map to
    the field's UNK slot. Returns ``(X_cat, side_dim, n_fields)``: ``X_cat`` is
    (N, len(cols)) int32 with global offsets (starting at 0 — the caller adds the
    official ``dim`` when concatenating with ``X``).
    """
    cat = table[list(cols)].reindex(keys)
    X = np.empty((len(keys), len(cols)), dtype=np.int32)
    offset = 0
    dims = []
    for j, col in enumerate(cols):
        train_vals = cat[col].values[:n_train]
        vocab = {v: i for i, v in enumerate(pd.unique(train_vals))}
        unk = len(vocab)
        codes = cat[col].map(vocab).fillna(unk).astype(np.int32).values
        X[:, j] = codes + offset
        offset += len(vocab) + 1
        dims.append(len(vocab) + 1)
    return X, int(sum(dims)), len(cols)


def encode_cont_fields(table: pd.DataFrame, cols, keys, n_train: int):
    """Z-score continuous columns ``cols`` using TRAIN-only mean/std.

    Returns ``(cont, cont_dim)``; ``cont`` is (N, len(cols)) float32. Unseen keys -> 0
    before normalization; constant columns get std=1 (no division by zero).
    """
    raw = table[list(cols)].reindex(keys).to_numpy(dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0)
    mean = raw[:n_train].mean(axis=0)
    std = raw[:n_train].std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    cont = ((raw - mean) / std).astype(np.float32)
    return cont, len(cols)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_side_features.py -v`
Expected: PASS（4 用例全绿）

- [ ] **Step 5: Commit**

```bash
git add models/side_features.py tests/test_side_features.py
git commit -m "feat: add train-only categorical/continuous side encoders"
```

---

### Task 3: 接进 `data_loader.load_extended`

**Files:**
- Modify: `models/data_loader.py`

**Interfaces:**
- Consumes: `build_video_side_table` / `build_user_side_table` / `encode_cat_fields` / `encode_cont_fields`（Task 1/2）。
- Produces: `load_extended` 返回值新增顶层 `vside_dim` / `vside_n_fields` / `uside_dim` / `uside_n_fields` / `cont_dim`，每个 split dict 新增 `X_vside`(N,2) int32、`X_uside`(N,5) int32、`cont`(N,6) float32，与 `X`/`y`/`users` 行序一致。原 `dim` / `n_fields` 语义不变。

- [ ] **Step 1: 改 `load_extended`**

`users_all` 断言之后、`data = {...}` 之前插入：

```python
    from models.side_features import (  # noqa: E402
        CAT_USER_FEATURES, CAT_VIDEO_FEATURES, CONT_FEATURES,
        build_user_side_table, build_video_side_table,
        encode_cat_fields, encode_cont_fields)
    basic = pd.read_csv(os.path.join(data_dir, "video_features_basic_pure.csv"))
    stat = pd.read_csv(os.path.join(data_dir, "video_features_statistic_pure.csv"))
    user = pd.read_csv(os.path.join(data_dir, "user_features_pure.csv"))
    vside = build_video_side_table(basic, stat)
    uside = build_user_side_table(user)
    X_vside, vside_dim, vside_n_fields = encode_cat_fields(
        vside, CAT_VIDEO_FEATURES, df_all["video_id"].values, n_train)
    cont, cont_dim = encode_cont_fields(
        vside, CONT_FEATURES, df_all["video_id"].values, n_train)
    X_uside, uside_dim, uside_n_fields = encode_cat_fields(
        uside, CAT_USER_FEATURES, df_all["user_id"].values, n_train)
```

`data = {"dim": dim, "K": K, "n_fields": len(FIELDS)}` 改为：

```python
    data = {"dim": dim, "K": K, "n_fields": len(FIELDS),
            "vside_dim": vside_dim, "vside_n_fields": vside_n_fields,
            "uside_dim": uside_dim, "uside_n_fields": uside_n_fields,
            "cont_dim": cont_dim}
```

split 循环里 `data[name] = {...}` 追加三个键：

```python
        data[name] = {
            "X": X, "y": y, "users": users, "aux": aux,
            "hist": hist_all[s:e], "hist_mask": mask_all[s:e],
            "X_vside": X_vside[s:e], "X_uside": X_uside[s:e], "cont": cont[s:e],
        }
```

- [ ] **Step 2: 形状自检（真实数据）**

Run: `python -c "from models.data_loader import load_extended; d=load_extended('kuairand-starter-kit/kuairand-starter-kit/KuaiRand-Pure/data'); print('dim', d['dim'], 'vside', d['vside_dim'], d['vside_n_fields'], 'uside', d['uside_dim'], d['uside_n_fields'], 'cont', d['cont_dim']); [print(k, d[k]['X'].shape, d[k]['X_vside'].shape, d[k]['X_uside'].shape, d[k]['cont'].shape) for k in ('train','valid','test')]"`
Expected: 打印 `vside_n_fields=2`、`uside_n_fields=5`、`cont_dim=6`；三个 split 的 `X`/`X_vside`/`X_uside`/`cont` 第 0 维（行数）完全一致。

- [ ] **Step 3: Commit**

```bash
git add models/data_loader.py
git commit -m "feat: wire video/user-side tensors into load_extended"
```

---

### Task 4: 模型连续特征接线（`deepfm.py` / `din.py`）

**Files:**
- Modify: `models/deepfm.py`（`FM` + `DeepFM`）
- Modify: `models/din.py`（`DIN`）
- Test: `tests/test_models_side.py`

**Interfaces:**
- Produces: `deepfm.FM(dim, k=16, aux_watch=False, cont_dim=0)`；`deepfm.DeepFM(dim, n_fields, k=16, dnn_hidden=(64,32), dropout=0.0, aux_watch=False, cont_dim=0)`；`din.DIN(dim, n_fields, k=16, hidden=(64,32), dropout=0.0, aux_watch=False, cont_dim=0)`。三者 `forward` 接受 `cont=None`（默认），`cont_dim=0` 时与现状逐 bit 一致。

- [ ] **Step 1: 写失败测试**

`tests/test_models_side.py`：

```python
"""Continuous-side-feature wiring for FM / DeepFM / DIN (cont param)."""
import torch

from models import deepfm, din


def test_fm_zero_cont_equals_no_cont():
    torch.manual_seed(0)
    m = deepfm.FM(dim=20, k=8, cont_dim=6)
    x = torch.randint(0, 20, (4, 5))
    z = torch.zeros(4, 6)
    assert torch.allclose(m(x), m(x, z), atol=1e-6)   # cont_lin bias=False -> zero contribution


def test_fm_cont_changes_output():
    torch.manual_seed(0)
    m = deepfm.FM(dim=20, k=8, cont_dim=6)
    x = torch.randint(0, 20, (4, 5))
    o = torch.ones(4, 6)
    assert not torch.allclose(m(x, torch.zeros(4, 6)), m(x, o), atol=1e-6)


def test_deepfm_cont_shape_and_backcompat():
    torch.manual_seed(0)
    m0 = deepfm.DeepFM(dim=20, n_fields=5, k=8, dnn_hidden=(16,), cont_dim=0)
    x = torch.randint(0, 20, (4, 5))
    assert m0(x).shape == (4,)                              # cont=None default still works
    m = deepfm.DeepFM(dim=20, n_fields=5, k=8, dnn_hidden=(16,), cont_dim=6)
    out = m(x, torch.randn(4, 6))
    assert out.shape == (4,) and torch.isfinite(out).all()


def test_din_cont_shape():
    torch.manual_seed(0)
    m = din.DIN(dim=20, n_fields=5, k=8, hidden=(16,), cont_dim=6)
    x = torch.randint(0, 20, (4, 5))
    hist = torch.randint(0, 20, (4, 3))
    mask = torch.ones(4, 3)
    out = m(x, hist, mask, torch.randn(4, 6))
    assert out.shape == (4,) and torch.isfinite(out).all()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_models_side.py -v`
Expected: FAIL — `TypeError: FM.__init__() got an unexpected keyword argument 'cont_dim'`

- [ ] **Step 3: 改 `deepfm.py`**

`FM.__init__` 加 `cont_dim: int = 0` 参数，末尾加：

```python
        self.cont_lin = nn.Linear(cont_dim, 1, bias=False) if cont_dim > 0 else None
```

`FM.forward` 改为：

```python
    def forward(self, x: torch.Tensor, cont: torch.Tensor | None = None) -> torch.Tensor:
        e = self.V(x)                       # (B, F, k)
        s = e.sum(dim=1)                    # (B, k)
        inter = 0.5 * (s.square().sum(dim=1) - e.square().sum(dim=(1, 2)))
        lin = self.W(x).sum(dim=1).squeeze(1)  # (B,)
        out = self.b + lin + inter
        if cont is not None and self.cont_lin is not None:
            out = out + self.cont_lin(cont).squeeze(1)
        return out
```

`DeepFM.__init__` 加 `cont_dim: int = 0`，MLP 输入维改 `in_dim = n_fields * k + cont_dim`；`DeepFM.forward` 改为：

```python
    def forward(self, x: torch.Tensor, cont: torch.Tensor | None = None) -> torch.Tensor:
        e = self.emb(x)                     # (B, F, k)
        s = e.sum(dim=1)
        inter = 0.5 * (s.square().sum(dim=1) - e.square().sum(dim=(1, 2)))
        lin = self.lin(x).sum(dim=1).squeeze(1)
        fm = self.b + lin + inter
        flat = e.reshape(x.size(0), -1)
        dnn_in = torch.cat([flat, cont], dim=1) if cont is not None else flat
        dnn = self.mlp(dnn_in).squeeze(1)
        return fm + dnn
```

- [ ] **Step 4: 改 `din.py`**

`DIN.__init__` 加 `cont_dim: int = 0`，deep scorer 输入维改 `in_dim = k + k + (n_fields - 1) * k + cont_dim`；`DIN.forward` 签名改为 `def forward(self, x, hist=None, hist_mask=None, cont=None):`，`deep` 输入拼接处改为：

```python
        other = e[:, self.other_idx].reshape(x.size(0), -1)    # (B, (F-1)*k)
        deep_in = torch.cat([q, pooled, other], dim=-1)
        if cont is not None:
            deep_in = torch.cat([deep_in, cont], dim=-1)
        deep = self.mlp(deep_in).squeeze(1)
        return wide + deep
```

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_models_side.py tests/test_losses.py tests/test_evaluate.py -v`
Expected: PASS（新 4 用例 + 原 10 用例全绿）

- [ ] **Step 6: Commit**

```bash
git add models/deepfm.py models/din.py tests/test_models_side.py
git commit -m "feat: add continuous-side-feature input to FM/DeepFM/DIN deep towers"
```

---

### Task 5: 训练接线 `train.py`（两 toggle 分支）

**Files:**
- Modify: `models/train.py`

**Interfaces:**
- Consumes: `data["vside_dim"]` / `vside_n_fields` / `uside_dim` / `uside_n_fields` / `cont_dim`、`split["X_vside"]` / `X_uside` / `cont`（Task 3）；模型 `cont` 参数（Task 4）。
- Produces: `make_model(config, dim, n_fields, cont_dim=0)`；`run_experiment(data, config)` 读 `use_videoside`/`use_userside`（默认 False）决定拼接；`main()` 加 `--use_videoside` / `--use_userside`。

- [ ] **Step 1: 改 `make_model`**

签名加 `cont_dim=0`，三处构造透传 `cont_dim=cont_dim`：

```python
def make_model(config, dim, n_fields, cont_dim=0):
    name = config["model"]
    k = config.get("k", 16)
    aux_watch = config.get("aux") == "cwm"
    if name == "fm":
        return deepfm.FM(dim, k=k, aux_watch=aux_watch, cont_dim=cont_dim)
    if name == "deepfm":
        return deepfm.DeepFM(dim, n_fields, k=k,
                             dnn_hidden=config.get("dnn_hidden", (64, 32)),
                             dropout=config.get("dropout", 0.0), aux_watch=aux_watch,
                             cont_dim=cont_dim)
    if name == "din":
        return din.DIN(dim, n_fields, k=k,
                       hidden=config.get("dnn_hidden", (64, 32)),
                       dropout=config.get("dropout", 0.0), aux_watch=aux_watch,
                       cont_dim=cont_dim)
    raise ValueError(f"unknown model {name!r}")
```

- [ ] **Step 2: 改 `_forward` + `_predict`，加 `_input_tensor`**

```python
def _input_tensor(split, idx, device, use_videoside, use_userside):
    xb = torch.from_numpy(split["X"][idx]).to(device)
    if use_videoside:
        xb = torch.cat([xb, torch.from_numpy(split["X_vside"][idx]).to(device)], dim=1)
    if use_userside:
        xb = torch.cat([xb, torch.from_numpy(split["X_uside"][idx]).to(device)], dim=1)
    cont = torch.from_numpy(split["cont"][idx]).to(device) if use_videoside else None
    return xb, cont


def _forward(model, xb, cont, split, idx, device):
    if isinstance(model, din.DIN):
        hist = torch.from_numpy(split["hist"][idx]).to(device)
        mask = torch.from_numpy(split["hist_mask"][idx]).to(device)
        return model(xb, hist, mask, cont)
    return model(xb, cont)


@torch.no_grad()
def _predict(model, split, device, use_videoside, use_userside, bs=200_000):
    model.eval()
    outs = []
    for i in range(0, len(split["X"]), bs):
        sl = slice(i, i + bs)
        xb, cont = _input_tensor(split, sl, device, use_videoside, use_userside)
        outs.append(_forward(model, xb, cont, split, sl, device).cpu().numpy())
    model.train()
    return np.concatenate(outs)
```

- [ ] **Step 3: 改 `run_experiment`**

`dim, n_fields = data["dim"], data["n_fields"]` 之后、`make_model` 之前插入：

```python
    use_videoside = bool(config.get("use_videoside", False))
    use_userside = bool(config.get("use_userside", False))
    if use_videoside:
        dim += data["vside_dim"]
        n_fields += data["vside_n_fields"]
        cont_dim = data["cont_dim"]
    else:
        cont_dim = 0
    if use_userside:
        dim += data["uside_dim"]
        n_fields += data["uside_n_fields"]
```

`make_model(config, dim, n_fields)` 改为 `make_model(config, dim, n_fields, cont_dim=cont_dim)`。

训练循环内 `xb = torch.from_numpy(Xtr[idx]).to(device)` 替换为：

```python
            xb, cont = _input_tensor(tr, idx, device, use_videoside, use_userside)
```

`logits = _forward(model, xb, tr, idx, device)` 改为 `logits = _forward(model, xb, cont, tr, idx, device)`。CWM 分支 `aux_logits = model.aux_forward(xb)` 不变（`xb` 已是拼接后输入）。

评估与最终打分：`_predict(model, va, va["X"], device)` → `_predict(model, va, device, use_videoside, use_userside)`，`_predict(model, te, te["X"], device)` → `_predict(model, te, device, use_videoside, use_userside)`；训练循环内的 `evaluate(va["users"], va["y"], _predict(...))` 同步改。

- [ ] **Step 4: `main()` 加两个 flag**

```python
    ap.add_argument("--use_videoside", action="store_true")
    ap.add_argument("--use_userside", action="store_true")
```

config dict 里加：

```python
    if a.use_videoside:
        config["use_videoside"] = True
    if a.use_userside:
        config["use_userside"] = True
```

- [ ] **Step 5: 验证 baseline 不回归**

Run: `python -m models.train --model fm --loss bce --seed 0`
Expected: valid primary **≈ 0.6015–0.6020**（两 toggle 默认关，零回归）。

- [ ] **Step 6: 验证两侧独立生效（量 lift）**

Run 三个命令，记录各自 valid primary：
- `python -m models.train --model fm --loss bce --seed 0 --use_videoside`
- `python -m models.train --model fm --loss bce --seed 0 --use_userside`
- `python -m models.train --model fm --loss bce --seed 0 --use_videoside --use_userside`

Expected: 三个都正常训练并打印 valid primary；与 Step 5 差值即两侧 lift。（形状不匹配会在此处崩溃——同时是 Task 3/4/5 的集成校验。）

- [ ] **Step 7: Commit**

```bash
git add models/train.py
git commit -m "feat: wire use_videoside/use_userside branches through the training loop"
```

---

### Task 6: 搜索空间 + 提示词（`search.py` / `prompts.py` / `tools.py`）

**Files:**
- Modify: `agent/search.py`
- Modify: `agent/prompts.py`
- Modify: `agent/tools.py`（拒绝提示文案，1 行）
- Test: `tests/test_search.py`

**Interfaces:**
- Produces: `search.normalize` 每个输出都带 `use_videoside: bool` / `use_userside: bool`（默认 False）；`mutate` 可翻转；`random_config` 以概率置 True。`RUN_EXPERIMENT_TOOL` schema 的 config 增加两个 boolean。

- [ ] **Step 1: 写失败测试**

`tests/test_search.py`：

```python
"""Config-space tests: side-feature toggles are normalize-complete and dedup-sound."""
import json

from agent import search


def test_normalize_carries_side_toggles():
    cfg = search.normalize({"model": "fm", "loss": "bce"})
    assert cfg["use_videoside"] is False and cfg["use_userside"] is False
    cfg2 = search.normalize({"model": "fm", "loss": "bce", "use_videoside": True})
    assert cfg2["use_videoside"] is True and cfg2["use_userside"] is False


def test_side_toggles_are_dedup_distinct():
    a = search.normalize({"model": "fm", "loss": "bce"})
    b = search.normalize({"model": "fm", "loss": "bce", "use_videoside": True})
    c = search.normalize({"model": "fm", "loss": "bce", "use_userside": True})
    ka = json.dumps(a, sort_keys=True)
    assert ka != json.dumps(b, sort_keys=True)
    assert ka != json.dumps(c, sort_keys=True)
    assert json.dumps(b, sort_keys=True) != json.dumps(c, sort_keys=True)


def test_mutate_can_toggle_side():
    from agent.search import _DEFAULT
    assert "use_videoside" in _DEFAULT and "use_userside" in _DEFAULT
    cfg = search.normalize({"model": "fm", "loss": "bce", "use_videoside": False})
    from agent.search import mutate
    # mutate flips one dimension; over enough draws both toggles are reachable
    keys = set()
    for _ in range(200):
        m = mutate(cfg, __import__("random").Random(1))
        if m["use_videoside"] != cfg["use_videoside"]:
            keys.add("videoside")
        if m["use_userside"] != cfg["use_userside"]:
            keys.add("userside")
    assert "videoside" in keys
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_search.py -v`
Expected: FAIL — `KeyError: 'use_videoside'`（`_DEFAULT` 尚未含该键）

- [ ] **Step 3: 改 `search.py`**

`_DEFAULT` 加两个键：

```python
_DEFAULT = dict(k=16, lr=1e-3, epochs=40, bs=8192, patience=8,
                dropout=0.0, dnn_hidden=(64, 32), seed=0, aux=None, aux_weight=0.1,
                use_videoside=False, use_userside=False)
```

`normalize` 末尾（`out["aux_weight"] = ...` 之后）加：

```python
    out["use_videoside"] = bool(config.get("use_videoside", False))
    out["use_userside"] = bool(config.get("use_userside", False))
```

`mutate` 的 key 列表加两个 toggle，并在 `elif` 链加分支：

```python
    key = rng.choice(["k", "lr", "dropout", "dnn_hidden", "bs", "aux",
                      "use_videoside", "use_userside"])
    ...
    elif key == "use_videoside":
        out["use_videoside"] = not bool(out.get("use_videoside"))
    elif key == "use_userside":
        out["use_userside"] = not bool(out.get("use_userside"))
```

`random_config` 的 normalize 输入 dict 加：

```python
        "use_videoside": rng.random() < 0.5,
        "use_userside": rng.random() < 0.5,
```

- [ ] **Step 4: 改 `prompts.py`**

`RUN_EXPERIMENT_TOOL` 的 `config.properties` 加：

```python
                    "use_videoside": {"type": "boolean",
                                      "description": "Add video-side features (video_type/"
                                      "music_type categorical + 6 continuous engagement/quality)."},
                    "use_userside": {"type": "boolean",
                                     "description": "Add user-side categorical features "
                                     "(active_degree + 4 *_range). Weaker: linear term is "
                                     "rank-irrelevant, value only via cross-interactions."},
```

`_SYSTEM_TEMPLATE` 的 "Open directions" 段，在 CWM 那条之后加两条：

```text
- **video side-information** (`use_videoside=true`): id-only model ignores the organizer's
  rich video features. Adds video_type/music_type categorical + 6 continuous item-quality
  features (play_progress, engagement rates, duration). The item-side lever — primary for
  within-user ranking (its linear term is rank-relevant).
- **user side-information** (`use_userside=true`): adds 5 user categorical features
  (active_degree + follow/fans/friend/register *_range). Weaker than video-side: its linear
  term is a per-user constant (rank-irrelevant for GAUC/nDCG@5), so it only helps through
  cross-interactions with item fields.
```

`_SYSTEM_TEMPLATE` 的 "Config fields" 段末尾加：

```text
- use_videoside / use_userside (bool, default false): add video-side / user-side features.
```

- [ ] **Step 5: 改 `tools.py` 拒绝文案**

`run_experiment` 里重复拒绝串的提示 `Vary a dimension (model/loss/k/lr/dropout/aux/seed)` 改为：

```python
                f"Vary a dimension (model/loss/k/lr/dropout/aux/seed/"
                f"use_videoside/use_userside) and try again. "
```

- [ ] **Step 6: 跑测试确认通过**

Run: `python -m pytest tests/test_search.py tests/test_side_features.py tests/test_models_side.py tests/test_losses.py tests/test_evaluate.py -v`
Expected: PASS（全绿）

- [ ] **Step 7: Commit**

```bash
git add agent/search.py agent/prompts.py agent/tools.py tests/test_search.py
git commit -m "feat: expose use_videoside/use_userside as searchable config dimensions"
```

---

### Task 7: 全量验证 + 冒烟

**Files:** 无新建（验证阶段）

- [ ] **Step 1: 全量单元测试**

Run: `python -m pytest -v`
Expected: 全部 PASS（原 10 用例 + 新 side_features/search/models_side 用例）。

- [ ] **Step 2: baseline 不回归**

Run: `python -m models.train --model fm --loss bce --seed 0`
Expected: valid primary ≈ 0.6015–0.6020。

- [ ] **Step 3: 独立消融（核心数字）**

Run:
- `python -m models.train --model fm --loss bce --seed 0 --use_videoside`
- `python -m models.train --model fm --loss bce --seed 0 --use_userside`
- `python -m models.train --model fm --loss bce --seed 0 --use_videoside --use_userside`

Expected: 记录各自 valid primary，算两侧独立 lift。

- [ ] **Step 4: no-LLM 冒烟**

Run: `python -m agent.main --no_llm --fresh --max_iters 5`
Expected: iteration-0 FM ~0.6020；后续 seed/mutate 实验能出现两个 toggle；0 error；`state.json` 的 `seen_configs` 键含 `use_videoside`/`use_userside`。

- [ ] **Step 5: 真实 ReAct 冒烟**

Run: `python -m agent.main --fresh --max_iters 3`
Expected: LLM 能在 `run_experiment` 的 config 里显式设两个 toggle 并读回结果；`finish_episode` 产出 lesson；lesson 回写日志。

- [ ] **Step 6: 提交计划文档本身**

```bash
git add docs/superpowers/plans/2026-08-30-video-sideinfo-lever.md docs/superpowers/specs/2026-08-30-video-sideinfo-lever-design.md
git commit -m "docs: implementation plan for video+user side-feature levers"
```
