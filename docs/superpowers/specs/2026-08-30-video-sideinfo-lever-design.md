# 视频侧特征 lever — 突破 valid primary ~0.604 平台期

- 日期：2026-08-30
- 状态：待评审（评审通过后进入 writing-plans）
- 分支：`feat/autonomous-agent`

## 1. 为什么改（Context）

当前 Agent 已在 valid primary **0.6046** 处平台化：3 轮迭代把 CWM aux_weight 从 0.1 调到 0.5，只涨了 +0.0002（0.6044 → 0.6046），lesson 自判「CWM 方向大致见顶，需要不同 lever」。抬高 max_iters 无济于事（收敛判据 ε=0.002/N=3 会提前停，且曲线已走平）。

根因：**当前所有模型都是 id-only**。官方 `FIELDS = ['user_id','video_id','author_id','tab','dur_bucket']` 只有 5 个稀疏 id 域，而数据里富余的视频侧信息（`video_features_basic_pure.csv` 的 `video_type/music_type/video_duration`、`video_features_statistic_pure.csv` 的 51 列统计）完全没用上。

**核心论据（为什么是「视频侧」而不是「用户侧」）**：任务口径是**用户内排序**（within-user ranking，GAUC + nDCG@5）。对同一用户，加一个「用户级常数」不改变该用户内部的排序，所以 GAUC/nDCG@5 对 per-user 常数平移不变——`user_id` embedding 已经吸收了用户级水平，用户侧特征（`user_active_degree` 等）只能通过 FM 交叉项间接起作用，是低优先级的次优解。**视频侧（item-side）特征直接区分同一用户内的不同视频**，正是这个口径下唯一有真实 headroom 的方向。官方 `data.py` 里也明确标注「想加特征就往这里加 —— 这是学生最该动的地方之一」。

## 2. 现有参考（不要重复造轮子）

`kuairand-starter-kit/kuairand-starter-kit/ablation_features.py` 已有一套「FM 上加视频侧类别特征」的参考实现（CWM 论文 13 域：`author_id/music_id/video_type/upload_type` + 5 用户侧），手写 per-field vocab + offset + UNK 槽的编码。**本设计是它的泛化**：

1. 不只 FM——接进共享 embedding 的 FM / DeepFM / DIN 三模型。
2. 不只类别——补上 `video_features_statistic_pure.csv` 的连续统计（ablation 脚本完全没用这 51 列）。
3. 做成可搜索的 config 维度 `use_sideinfo`，让 Agent 能自己跑 ablation 证明增益，而不是写死 always-on。

只读引用该脚本，**不改 starter-kit 内任何文件**。

## 3. 特征集（v1，严口径）

### 类别侧（F_cat = 2，新增稀疏 embedding）
| 字段 | 来源 | 说明 |
|---|---|---|
| `video_type` | basic CSV | 必选，强先验 |
| `music_type` | basic CSV | 粗粒度；避开 `music_id` 原始 id 的高基数/过拟合风险 |

**可选后续**（不进 v1）：`upload_type`（ablation 脚本用过）、`music_id`（原始 id，基数高）。

### 连续侧（C = 6，归一化后进 deep tower）
| 特征 | 构造 | 说明 |
|---|---|---|
| `play_progress` | statistic CSV 原值 | 平均播放进度，**最强先验** |
| `like_rate` | `like_cnt / max(show_cnt,1)` 截断 [0,1] | 参与率 |
| `comment_rate` | `comment_cnt / max(show_cnt,1)` 截断 [0,1] | 参与率 |
| `follow_rate` | `follow_cnt / max(show_cnt,1)` 截断 [0,1] | 参与率 |
| `share_rate` | `share_cnt / max(show_cnt,1)` 截断 [0,1] | 参与率 |
| `log_duration` | `log1p(video_duration)`（basic CSV） | 连续时长，比 `dur_bucket` 更细 |

**明确排除**（近泄漏）：`complete_play_cnt`、`long_time_play_cnt`、`valid_play_cnt`、`play_cnt`、`play_duration`、原始 `show_cnt`。这些是「这个视频有多少人 long-view」的聚合，直接喂进去近乎把 label 当特征，结果站不住。用 `play_progress` + 参与率这类「质量先验」既强又干净。

**诚实标注（时间泄漏权衡）**：`video_features_statistic_pure.csv` 是全周期快照，某个视频的 `play_progress` 可能含 test 期统计，属轻微时间泄漏。但这是组织者公开提供的同一份数据（对所有参赛者公平），且视频级流行度/质量特征在生产推荐里也是标准做法。更严格的口径是「只用 train 期日志自己聚合 item 统计」，本设计**不采用**（工程成本高、收益小），在此显式声明以便你否决。

### 归一化（train-only，无泄漏）
- 比率类（`play_progress` + 4 个 `*_rate`，天然 [0,1]）：直接 z-score。
- `log_duration`：先 `log1p` 再 z-score。
- `mean/std` 只在 **train** 上算，valid/test 复用；`std==0` 的常量特征置 `std=1`。

## 4. 数据流（`models/data_loader.py`）

`load_extended` **始终**构建侧特征（构建本身很便宜），新增返回：

- 顶层：`side_dim`（新增类别域 vocab 和）、`side_n_fields`（= F_cat）、`cont_dim`（= C）。
- 每 split：`X_side`(N, F_cat) int32（offset 从 `dim` 之后续接）、`cont`(N, C) float32（已归一化）。

`dim` / `n_fields` **保持原语义不变**（baseline 路径零改动）。关键对齐保证：侧特征按行对齐官方 `X/y/users`（沿用现有 `df_all` 的 `[train,valid,test]` 重排 + `video_id` `.map` 保持行序，不 `merge`）。

编码沿用 `ablation_features.py` 的 per-field vocab + offset + UNK 槽模式，但类别侧（`video_type`/`music_type`）与连续侧（`video_features_statistic_pure.csv`）合并成一张 `video_id → side` 表一次性构建。

## 5. 模型改动（`deepfm.py` / `din.py`；`fm_torch.py` 不动）

**共享 embedding 扩维**：`dim_total = dim + side_dim`。当 `use_sideinfo=True`，输入张量 = `torch.cat([X, X_side], dim=1)`，共享 `nn.Embedding(dim_total, k)` 自动吃到新类别域；FM 的 sum/交叉项、DeepFM 的 MLP、DIN 的 `other` 域（`other_idx` 随 `n_fields` 增长自动包含）全部天然覆盖。`video_id` 仍在下标 1（候选不变），DIN 的 attention 不受影响。

连续特征 `cont` 进入各模型的 deep 部分：

| 模型 | forward 签名 | 连续特征接入 |
|---|---|---|
| `deepfm.FM` | `forward(x, cont=None)` | 新增 `self.cont_lin = nn.Linear(cont_dim, 1)`，分数 `= b + lin + inter + cont_lin(cont)` |
| `deepfm.DeepFM` | `forward(x, cont=None)` | MLP 输入维 `n_fields*k + cont_dim`，`dnn_input = cat([e.reshape(B,-1), cont], 1)` |
| `din.DIN` | `forward(x, hist=None, hist_mask=None, cont=None)` | MLP 输入维 `k+k+(n_fields-1)*k + cont_dim`，`deep` 输入拼上 `cont` |

`cont_dim=0`（即 `use_sideinfo=False`）时三者**行为与现状逐 bit 一致**（`cont=None` 默认、`cont_lin` 不建、MLP 输入维不变）——这是 baseline 0.6020 不回归的保证。`aux_forward` 不改，但 `run_experiment` 里它拿到的 `xb` 也是拼接后的输入（`use_sideinfo` 时），共享 embedding 顺带把侧域信号喂给 CWM 头，无额外代码。

## 6. 训练接线（`models/train.py`）

- `make_model(config, dim, n_fields)` 加参数 `cont_dim=0`，透传给三个模型（`dim`/`n_fields` 由 `run_experiment` 传入有效值）。
- `run_experiment`：读 `use_sideinfo = bool(config.get("use_sideinfo", False))`；据此算 `dim`/`n_fields`/`cont_dim` 有效值，并在构建 batch 输入时：
  ```python
  def _inputs(split, idx, device):
      xb = torch.from_numpy(split["X"][idx]).to(device)
      if use_sideinfo:
          xb = torch.cat([xb, torch.from_numpy(split["X_side"][idx]).to(device)], dim=1)
          cont = torch.from_numpy(split["cont"][idx]).to(device)
      else:
          cont = None
      return xb, cont
  ```
- `_forward(model, xb, cont, split, idx, device)`：DIN 分支传 `cont`，其余 `model(xb, cont)`；`_predict` 同步改。

## 7. 搜索空间 / Agent（`agent/search.py` + `agent/prompts.py`）

- `search._DEFAULT` 加 `use_sideinfo=False`；`normalize` 强转 `bool(config.get("use_sideinfo", False))`——**让每个 normalized config 都带这个 key**，config 去重才 sound。
- `mutate` 把 `use_sideinfo` 加入扰动维度（toggle）；`random_config` 以一定概率置 `True`，no-LLM 兜底也能探索新 lever。
- `prompts.py`：`run_experiment` 的 tool schema 加 `use_sideinfo: boolean`；action-space 段加一句「视频侧信息：`video_type`/`music_type` 类别域 + 6 个连续参与率/质量特征，经 `use_sideinfo` 开关」。
- 现有 `SEED_CONFIGS` 的已验证种子不动；待本设计验证出 lift 后，再把 `use_sideinfo=True` 的胜出配置回填为种子。

## 8. 验证（改必验）

1. **单元**：`python -m pytest` 现有 10 用例全绿。
2. **baseline 不回归**：`python -m models.train --model fm --loss bce`（默认 `use_sideinfo=False`）→ valid primary **≈ 0.6020**（与 run_log iter 1 一致）。
3. **新 lever 有效**：同一 config 下 `use_sideinfo=False` vs `True`，量 FM / DeepFM / DIN 三模型的 valid lift——这是真正要报告的数字。
4. **no-LLM 冒烟**：`python -m agent.main --no_llm --fresh --max_iters 5`，确认 `use_sideinfo` 能出现在 proposed config、去重正确、0 error。
5. **真实 ReAct 冒烟**：`python -m agent.main --fresh --max_iters 3`，确认 LLM 能在 tool 里显式设 `use_sideinfo` 并读回结果。

## 9. 预期（诚实）

方向是唯一有真实 headroom 的，但**数字不打包票**：`play_progress` + 参与率是很强的 item 先验，理性预期 valid primary +0.02~+0.10；也可能被「`video_id` embedding 已学到部分视频流行度」吃掉一块。跑了才知道。

## 10. 不做什么（YAGNI）

- 不做用户侧特征（within-user ranking 下 rank-irrelevant，见 §1）。
- 不做上下文特征（`hourmin`/`is_rand`）、不做序列架构重写（DIN 之外的 DIEN/SIM 等）——那是后续独立 lever。
- 不碰 `fm_torch.py`（trust-anchor）与 starter-kit 任何文件。
- 不做「train-only 自聚合 item 统计」的严泄漏版（§3 已声明权衡）。
