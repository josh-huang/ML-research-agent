# 视频侧 + 用户侧特征 lever — 突破 valid primary ~0.604 平台期

- 日期：2026-08-30
- 状态：待评审（评审通过后进入 writing-plans）
- 分支：`feat/autonomous-agent`

## 1. 为什么改（Context）

当前 Agent 已在 valid primary **0.6046** 处平台化：3 轮迭代把 CWM aux_weight 从 0.1 调到 0.5，只涨了 +0.0002（0.6044 → 0.6046），lesson 自判「CWM 方向大致见顶，需要不同 lever」。抬高 max_iters 无济于事（收敛判据 ε=0.002/N=3 会提前停，且曲线已走平）。

根因：**当前所有模型都是 id-only**。官方 `FIELDS = ['user_id','video_id','author_id','tab','dur_bucket']` 只有 5 个稀疏 id 域，而数据里富余的视频侧信息（`video_features_basic_pure.csv` 的 `video_type/music_type/video_duration`、`video_features_statistic_pure.csv` 的 51 列统计）与用户侧信息（`user_features_pure.csv` 的 31 列）完全没用上。官方 `data.py` 也明确标注「想加特征就往这里加 —— 这是学生最该动的地方之一」。

**两侧价值的精确区别（这是设计核心）**：任务口径是**用户内排序**（within-user ranking，GAUC + nDCG@5），两者对 per-user 常数平移不变。

- **视频侧（item-side）**：线性项就 rank-relevant（同一用户内不同 item 的 `video_type`/`play_progress` 不同，直接改变排序）→ **主 lever**。
- **用户侧（user-side）**：线性项是 per-user 常数、对 within-user ranking **严格无关**；唯一价值路径是 FM/DeepFM 的**交叉项** `<V_user_feature, V_item_field>`（如 `user_active_degree × video_type`），以及作为低基数先验正则化稀疏的 `user_id` embedding → **次 lever，预期增益更小但非零**。组织者的 `ablation_features.py` 也测了用户侧（CWM 5 用户域）。

因此：**两侧都加，但用两个独立 toggle 拆开**，让 Agent 能分别量出「视频侧贡献多少、用户侧贡献多少」，消融不被混淆。

## 2. 现有参考（不要重复造轮子）

`kuairand-starter-kit/kuairand-starter-kit/ablation_features.py` 已有一套「FM 上加 CWM 13 域」的参考实现：视频侧 `author_id/music_id/video_type/upload_type` + 用户侧 `follow_user_num_range/register_days_range/fans_user_num_range/friend_user_num_range/user_active_degree`，手写 per-field vocab + offset + UNK 槽。**本设计是它的泛化**：

1. 不只 FM——接进共享 embedding 的 FM / DeepFM / DIN 三模型。
2. 不只类别——补上 `video_features_statistic_pure.csv` 的连续统计（ablation 脚本完全没用这 51 列）。
3. 做成两个可搜索的 config 维度 `use_videoside` / `use_userside`，让 Agent 能独立跑 ablation。

只读引用该脚本，**不改 starter-kit 内任何文件**。

## 3. 特征集（v1，严口径）

### 视频侧 — 类别（F_vcat = 2，新增稀疏 embedding）
| 字段 | 来源 | 说明 |
|---|---|---|
| `video_type` | basic CSV | 必选，强先验 |
| `music_type` | basic CSV | 粗粒度；避开 `music_id` 原始 id 的高基数/过拟合风险 |

**可选后续**（不进 v1）：`upload_type`、`music_id`。

### 视频侧 — 连续（C = 6，归一化后进 deep tower）
| 特征 | 构造 | 说明 |
|---|---|---|
| `play_progress` | statistic CSV 原值 | 平均播放进度，**最强先验** |
| `like_rate` | `like_cnt / max(show_cnt,1)` 截断 [0,1] | 参与率 |
| `comment_rate` | `comment_cnt / max(show_cnt,1)` 截断 [0,1] | 参与率 |
| `follow_rate` | `follow_cnt / max(show_cnt,1)` 截断 [0,1] | 参与率 |
| `share_rate` | `share_cnt / max(show_cnt,1)` 截断 [0,1] | 参与率 |
| `log_duration` | `log1p(video_duration)`（basic CSV） | 连续时长，比 `dur_bucket` 更细 |

**明确排除**（近泄漏）：`complete_play_cnt`、`long_time_play_cnt`、`valid_play_cnt`、`play_cnt`、`play_duration`、原始 `show_cnt`。

### 用户侧 — 类别（F_ucat = 5，新增稀疏 embedding，对齐 ablation_features.py 的 USER_FE）
| 字段 | 来源 | 说明 |
|---|---|---|
| `user_active_degree` | user CSV | CWM 用户活跃度 |
| `follow_user_num_range` | user CSV | 关注数分桶 |
| `fans_user_num_range` | user CSV | 粉丝数分桶 |
| `friend_user_num_range` | user CSV | 好友数分桶 |
| `register_days_range` | user CSV | 注册天数分桶 |

**用户侧连续**（原始 `follow_user_num`/`fans_user_num`/`register_days` 等）：**不做**（与 `*_range` 分桶冗余，YAGNI）。

**诚实标注（时间泄漏权衡）**：`video_features_statistic_pure.csv` 是全周期快照，某视频的 `play_progress` 可能含 test 期统计，属轻微时间泄漏。但这是组织者公开提供的同一份数据（对所有参赛者公平），且视频级流行度/质量特征是生产推荐标准做法。更严格口径是「只用 train 期日志自聚合 item 统计」，本设计**不采用**，在此显式声明以便你否决。用户侧特征无此问题（用户属性不随 test 期变化）。

### 归一化（train-only，无泄漏）
- 比率类（`play_progress` + 4 个 `*_rate`，天然 [0,1]）：直接 z-score。
- `log_duration`：先 `log1p` 再 z-score。
- `mean/std` 只在 **train** 上算，valid/test 复用；`std==0` 的常量特征置 `std=1`。
- 类别域 vocab 只在 **train** 上建，未见值落 UNK 槽。

## 4. 数据流（`models/data_loader.py` + 新 `models/side_features.py`）

新模块 `models/side_features.py` 负责纯特征工程（可单测）：`build_video_side_table` / `build_user_side_table` 产出两张 video_id/user_id 索引表，`encode_cat_fields` / `encode_cont_fields` 产出对齐张量。

`load_extended` **始终**构建侧特征，新增返回：

- 顶层：`vside_dim`/`vside_n_fields`（视频类别域）、`uside_dim`/`uside_n_fields`（用户类别域）、`cont_dim`（= C）。
- 每 split：`X_vside`(N, F_vcat) int32、`X_uside`(N, F_ucat) int32（offset 从各自前缀之后续接）、`cont`(N, C) float32（已归一化）。

`dim` / `n_fields` **保持原语义不变**（baseline 路径零改动）。关键对齐保证：侧特征按行对齐官方 `X/y/users`（沿用现有 `df_all` 的 `[train,valid,test]` 重排 + `video_id`/`user_id` `.map` 保持行序，不 `merge`）。

## 5. 模型改动（`deepfm.py` / `din.py`；`fm_torch.py` 不动）

**共享 embedding 扩维**：有效输入 = `torch.cat([X, X_vside?, X_uside?], dim=1)`（按启用的 toggle 拼接），`dim_total = dim + vside_dim? + uside_dim?`。FM 的 sum/交叉项、DeepFM 的 MLP、DIN 的 `other` 域自动覆盖新域；`video_id` 仍在下标 1，DIN 的 attention 不受影响。

连续特征 `cont`（仅视频侧，随 `use_videoside`）进入 deep 部分：

| 模型 | forward 签名 | 连续特征接入 |
|---|---|---|
| `deepfm.FM` | `forward(x, cont=None)` | `self.cont_lin = nn.Linear(cont_dim, 1, bias=False)`，`= b + lin + inter + cont_lin(cont)` |
| `deepfm.DeepFM` | `forward(x, cont=None)` | MLP 输入维 `n_fields*k + cont_dim` |
| `din.DIN` | `forward(x, hist=None, hist_mask=None, cont=None)` | MLP 输入维 `k+k+(n_fields-1)*k + cont_dim` |

`cont_dim=0`（两 toggle 都关）时三者**行为与现状逐 bit 一致**——baseline 0.6020 不回归的保证。`aux_forward` 不改，但 `run_experiment` 里它拿到的 `xb` 是拼接后输入，共享 embedding 顺带把侧域信号喂给 CWM 头。

## 6. 训练接线（`models/train.py`）

- `make_model(config, dim, n_fields, cont_dim=0)` 加 `cont_dim` 参数透传。
- `run_experiment` 读两个 toggle：`use_videoside = bool(config.get("use_videoside", False))`、`use_userside = bool(config.get("use_userside", False))`；据此算 `dim`/`n_fields`/`cont_dim` 有效值，`_input_tensor` 拼接对应张量，`cont` 仅当 `use_videoside`。

## 7. 搜索空间 / Agent（`agent/search.py` + `agent/prompts.py` + `agent/tools.py`）

- `search._DEFAULT` 加 `use_videoside=False, use_userside=False`；`normalize` 强转 `bool`——**让每个 normalized config 都带这两个 key**，config 去重才 sound。
- `mutate` 把两个 toggle 加入扰动维度（各自翻转）；`random_config` 各自以概率置 True。
- `prompts.py`：`run_experiment` tool schema 加 `use_videoside`/`use_userside` boolean；action-space 段加一句说明两侧特征与价值差异（视频侧主、用户侧经交叉项次）。
- `tools.py`：重复拒绝文案的「可调维度」提示加这两个 key。

## 8. 验证（改必验）

1. **单元**：`python -m pytest`（原 10 用例 + 新 video_side/search/models_side 用例全绿）。
2. **baseline 不回归**：`python -m models.train --model fm --loss bce --seed 0`（两 toggle 默认关）→ valid primary **≈ 0.6015–0.6020**。
3. **独立消融**：同一 config 下，量 (a) 仅 `use_videoside`、(b) 仅 `use_userside`、(c) 两者同开的 valid lift——这是本设计要报告的核心数字。
4. **no-LLM 冒烟**：`python -m agent.main --no_llm --fresh --max_iters 5`，确认两 toggle 能出现在 proposed config、去重正确、0 error。
5. **真实 ReAct 冒烟**：`python -m agent.main --fresh --max_iters 3`，确认 LLM 能在 tool 里显式设两 toggle 并读回结果。

## 9. 预期（诚实）

方向有真实 headroom，但**数字不打包票**：`play_progress` + 参与率是很强的 item 先验（理性预期 valid primary +0.02~+0.10）；用户侧经交叉项预期增益更小（可能 +0~+0.005）。都可能被「`video_id`/`user_id` embedding 已学到部分流行度」吃掉一块。跑了才知道。

## 10. 不做什么（YAGNI）

- 不做用户侧连续特征（与 `*_range` 分桶冗余）。
- 不做上下文特征（`hourmin`/`is_rand`）、不做序列架构重写（DIEN/SIM 等）——后续独立 lever。
- 不碰 `fm_torch.py`（trust-anchor）与 starter-kit 任何文件。
- 不做「train-only 自聚合 item 统计」的严泄漏版（§3 已声明权衡）。
