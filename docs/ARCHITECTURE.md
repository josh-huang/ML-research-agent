# 项目架构说明

本文档说明 ML-research-agent 的代码结构与数据流，便于快速回忆与上手。
评分口径、交付物等规范见根目录 `CLAUDE.md`；结果与策略见 `README.md`。

## 一句话

一个**自主 ML 研究 Agent**：在 KuaiRand-Pure 上先复现官方 FM baseline，然后由 Claude 自主
「提假设 → 跑实验 → 总结 lesson」迭代，最大化 `primary = mean(GAUC, nDCG@5)`。

整体分两层：

- **`models/`**：「怎么训一个模型」（Phase 3，人工搭建并验证的地板）
- **`agent/`**：「下一个该训什么」（Phase 7，Claude 驱动的自治循环）

Agent **只改 config，不改代码**——每轮的 "code diff" 即 config 增量，加上 run 起始的 `git_sha`。

---

## 目录总览

```
ML-research-agent/
├── kuairand-starter-kit/   官方 kit（只读）：data.py / evaluate.py / submit.py / baseline.py
├── models/                 模型 + 训练层
├── agent/                  自治 Agent 层
├── run_logs/               运行产物：run_log.jsonl（每轮一行）+ state.json + sweep_phase3d.jsonl
├── submission/             make_submission.py → final.csv
├── dashboard/app.py        Streamlit 仪表盘（只读 run_logs，5s 自动刷新）
├── tests/                  pytest 单测
├── conftest.py             把项目根加入 sys.path
└── docs/                   REPORT.pdf、本文档、设计文档（superpowers/）
```

---

## `models/` —— 训练层

| 文件 | 作用 |
|---|---|
| `data_loader.py` | `load_extended()`：复用官方 `data.load/encode` 的 5 字段编码，补充 aux 信号（click / like / profile / play_time / duration）、用户历史序列 `hist` + `hist_mask`（供 DIN）、side features；断言行顺序与官方 loader 逐行一致 |
| `side_features.py` | 视频侧 / 用户侧 / tag 特征表构建与编码（词表仅在 train 上建立） |
| `fm_torch.py` | 官方 FM 的 PyTorch 精确复刻，作为 **Phase 3 验证门**（valid primary ≈ 0.6016） |
| `deepfm.py` | `FM`、`DeepFM`（共享 embedding + DNN tower） |
| `din.py` | `DIN`：候选视频对用户历史序列做 attention |
| `losses.py` | BCE / BPR / listwise softmax + CWM（censored watch-time）辅助 loss |
| `train.py` | **核心入口** `run_experiment(data, config)`：config dict → 建模 → 训练 → 按 valid 早停 → 返回 valid/test 指标与逐行打分；`make_model()` 按 `config["model"]` 选模型 |
| `harness.py` | 包装官方 `evaluate`，作为全仓库统一评测入口 |
| `ensemble.py` | 多 seed / 跨模型 rank-average 集成实验 |
| `sweep.py` | Phase 3d 的 model × loss 网格搜索（一次性脚本） |

### config 字段（由 `agent/search.py` 统一规范化）

`model`（fm / deepfm / din）、`loss`（bce / bpr / listwise）、`k`、`lr`、`epochs`、`bs`、`patience`、
`dropout`、`dnn_hidden`、`seed`、`aux`（None / cwm）、`aux_weight`、
`use_videoside`、`use_userside`、`use_tagside`。

---

## `agent/` —— 自治层

| 文件 | 作用 |
|---|---|
| `main.py` | 编排器：加载数据与 EDA、第 0 轮 baseline、主循环、LLM 失败时降级为 no-LLM 提议器 |
| `react.py` | `ReActAgent`：单个持久 LLM，每个 episode 是有界的 tool loop（最多 6 turn） |
| `tools.py` | 工具分发 `dispatch()`、`run_experiment`（执行，不记录）、`finalize_run`（记录 + 落盘，每次实验恰好一次） |
| `llm.py` | Claude 客户端：system prompt 缓存、关闭 thinking、返回 input/output/cache token 分列 |
| `prompts.py` | system prompt 构建 + 全部 tool schema |
| `search.py` | 配置空间：`normalize`（补全键并截断到合法范围，保证去重可靠）、`mutate`、`random_config`、`SEED_CONFIGS` |
| `executor.py` | 训练调用的保护层：错误分类（oom / nan / shape / syntax / runtime）+ `degrade_config` 降级重试一次 |
| `state.py` | `AgentState`：best、收敛计数、预算、配置去重缓存，持久化为 `run_logs/state.json` |
| `memory.py` | 记忆管理：episode preamble、lessons 摘要、`playbook.md` 读写 |
| `logger.py` | 写 `run_log.jsonl`（每轮一行固定 schema）、记录 `git_sha` |
| `eda.py` | 紧凑的数据驱动 EDA 摘要（注入 system prompt） |
| `inspect.py` | 只读特征探针：`list_features` / `probe_feature` / `propose_feature` |
| `research.py` | arXiv 文献检索（`search_arxiv` / `fetch_paper`，仅用标准库） |
| `playbook.md` | 跨 run 持久的程序性记忆（`--fresh` 不清空） |

### 主循环（`main.py`）

```
启动
 ├─ load_extended()           数据只加载一次
 ├─ eda.py → system prompt     稳定内容，prompt caching 只付一次费
 ├─ 第 0 轮：FM + BCE baseline 直接执行（不调 LLM，0 token）
 └─ while 预算未耗尽 且 iterations < max_iters:
        ReActAgent.run_episode()    每个 episode 恰好执行 1 个实验
        LLM 异常 → 降级为 no-LLM 提议器（先跑 SEED_CONFIGS，再 mutate best）
```

### 单个 episode（`react.py` + `tools.py`）

```
preamble = 当前 state + playbook + 本轮 lessons + 最近 5 轮
   ↓
Claude 在 ≤ 6 turn 内调用工具：
   search_arxiv / fetch_paper       查文献          (research.py)
   read_run_log                     回看历史        (logger.py)
   list_features / probe_feature /
   propose_feature                  探查原始特征    (inspect.py)
   run_experiment                   ★ 训练 + 评测   (executor.py → models/train.py)
   finish_episode                   写 lesson
   ↓  （跑了实验却没 finish → 强制调用 finish_episode，保证 run 不丢失）
finalize_run：state.record() 给出 verdict → 写 run_log.jsonl → lesson 追加 playbook.md → 保存 state.json
   ↓
episode 压缩为一行进 history（上限 20 行，防止上下文二次增长）
```

### verdict 与收敛

- `state.record()` 返回 `error` / `new_best` / `accept` / `reject`。
- best 按 **seed 均值**计：同一配置（去掉 `seed` 后相同）的多个 seed 合为一组，取组内均值；
  补跑的新 seed 可能把 best 往上推，也可能往下拉。完全相同的配置（含 seed）会被拒绝重跑。
- 每轮的提升 = 跟踪中的 best 的变化量。任何正变化都会更新 best；只有变化 ≥ ε（0.002）才重置停滞计数。
- 连续 N=3 轮提升 < ε → `converged=True`。该标志**仅用于汇报**，不会停止循环；
  循环只在预算耗尽或达到 `--max_iters` 时停止。
- **已确认的 best**（`confirmed_best()`）：只在跑过 ≥ 3 个 seed（`MIN_CONFIRM_SEEDS`）的配置里取
  seed 均值最高者。最终提交和对外汇报用它；单 seed 领先无法胜出。

### test 封存

test 指标照常计算并写入 `run_log.jsonl`（供人看 drift），但**任何给 LLM 的内容都不含 test**：

- `run_experiment` 返回、episode 一行摘要、system prompt：只有 valid。
- EDA 摘要：标签相关统计只用 train，test 只给行数。
- 特征探针（`inspect.py`）：只读 train 日期段。
- 历史 lesson / hypothesis 里的 test 数字：注入前由 `memory.scrub_test_refs()` 过滤；
  `playbook.md` 与 `run_log.jsonl` 原文保留作历史记录。
- `models/ensemble.py`：按 valid 选方案，test 只对选中方案报告一次。

### 四层记忆（`memory.py`）

| 层 | 载体 |
|---|---|
| 工作记忆 | 当前 episode 的多轮对话 + 一行一条的 history |
| 情节记忆 | `run_logs/run_log.jsonl`（可通过 `read_run_log` 查询） |
| 语义记忆 | 缓存的 system prompt（口径 + EDA + 先验） |
| 程序记忆 | 本 run 的 lessons 摘要 + `agent/playbook.md`（跨 run 持久） |

---

## 数据流

```
KuaiRand CSV ──► data_loader ──► data dict
                                     │
   Claude ──提 config──► search.normalize ──► executor ──► train.run_experiment
     ▲                                                        │
     │                                         官方 evaluate.py（GAUC / nDCG@5）
     │                                                        │
     └── preamble ◄── memory ◄── state.json + run_log.jsonl ◄─┘
                                       │
                                       └──► dashboard（Streamlit 实时读取）

最终提交：submission/make_submission.py（state.json 的 confirmed_best）──► submission/final.csv
```

---

## 运行产物

| 文件 | 内容 |
|---|---|
| `run_logs/run_log.jsonl` | 每轮一行：`iteration`、`git_sha`、`hypothesis`、`action`（config）、`metrics`、`verdict`、`tokens*`、`gpu_h`、`errors`、`recovery`、`lesson` |
| `run_logs/state.json` | best（seed 均值 + `best_n_seeds` / `best_std`）、`confirmed_best`、收敛状态、迭代数、`interventions`、token / GPU 用量、`seen_configs` / `seen_iters`、`elapsed_s` |
| `run_logs/sweep_phase3d.jsonl` | Phase 3d 网格搜索结果 |
| `submission/final.csv` | starter-kit schema 的最终提交 |

---

## 常用命令（在项目根目录执行）

复现 baseline：

```bash
python -m models.train --model fm --loss bce
```

运行 Agent（清空旧 run-log，从头开始）：

```bash
python -m agent.main --fresh --max_iters 10
```

不调用 API 的冒烟测试：

```bash
python -m agent.main --no_llm --max_iters 5
```

实时仪表盘：

```bash
streamlit run dashboard/app.py
```

生成最终提交：

```bash
python submission/make_submission.py
```

单测：

```bash
python -m pytest -q
```
