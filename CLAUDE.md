# ML-research-agent — TikTok TechJam 2026 项目规范

## 目标（一句话）

做一个「自主 ML 研究 Agent」：在 KuaiRand-Pure 上复现官方 FM baseline 后自治迭代，
最大化 primary（GAUC 与 nDCG@5 的平均）对 baseline 的提升，并以最少的 token / GPU-hours / 人工干预收敛。

## 评分口径（唯一权威，一律照此，不可改）

- 任务：用户内排序（within-user ranking over logged impressions），不做全库检索。
- 相关性标签：`long_view`（0/1）。
- 指标：`GAUC` + `nDCG@5`；`primary = mean(GAUC, nDCG@5)`。
- 实现与唯一口径：`kuairand-starter-kit/kuairand-starter-kit/evaluate.py` —— 只 import，绝不修改。
- 数据划分：train 20220408–20220421 / valid 20220422–20220428 / test 20220429–20220508（按日期，写死）。
- 收敛判据：ε=0.002，N=3（连续 3 轮 valid primary 提升 < 0.002 判定收敛）。
- 官方 baseline（test）：GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946；valid primary 0.6016。
- oracle 上限（test primary）：0.8645。评估进展以 oracle 为分母，不以 1.0 为分母。

## 目录结构

```
ML-research-agent/
  kuairand-starter-kit/    # 官方 kit（只读参考；evaluate.py / submit.py 不改）
  agent/                   # 自治 Agent：main/llm/prompts/eda/executor/logger/search/state
  models/                  # PyTorch 模型脚手架：data_loader/fm_torch/deepfm/din/losses/train
  run_logs/                # 生成的逐轮 JSONL 日志 + state.json
  dashboard/               # Streamlit 实时监控仪表盘
  submission/              # 最终 submission.csv
```

## 约定

- 代码 / 注释 / 命令 / 变量名一律英文；文档与交互中文。
- Python 模块 snake_case；模型类 PascalCase；config 一律用 dict 传入 `train.py`。
- 不修改 starter kit 内任何文件；扩展功能一律写新模块，从 starter kit import 复用。
- 模型改动必须过 Phase 3 验证门（`fm_torch` 复现 baseline valid primary ≈ 0.6016）后才进入 Agent 搜索空间。

## 工程纪律

- 改必验：每个改动后跑对应验证命令（pytest / train.py / submit.py --check）。
- 寻根究底：禁止为过测试而注释报错、用 `Any`、加 Bypass。
- 凭证 / 密钥不入代码、不入 commit、不入日志。
- 红线（继承全局 CLAUDE.md）：删文件/目录、改 .env/密钥、迁移 DB、git push/rebase/reset --hard、装全局系统依赖、公开发布 —— 一律先显式确认。

## 提交交付物 (Deliverables)

竞赛 4 类交付物，全部必须产出并映射到仓库。核心汇报口径：**对官方 baseline 的绝对 delta**（进展以 oracle 0.8645 为分母，不以 1.0 为分母）。

### D1 书面项目描述（Devpost）
- 必答：方案如何解题 / 开发工具（PyCharm + VSCode + Python）/ API（**Anthropic Claude**，Messages API）/ 库与框架（PyTorch、pandas、numpy、Streamlit、anthropic、python-dotenv）/ 数据集（KuaiRand-Pure）。
- 产出：Devpost 文本稿（**新写**；从 README「Strategy / Key findings」提炼）。

### D2 公开代码 / GitHub 仓库
- README 必含：概览 / 安装 / 复现 / 局限与改进方向 / 团队分工。仓库 `README.md` 已覆盖前四者，**缺「团队分工」——暂移除，交付前补实际团队信息（非 solo）**。
- 代码注释齐全、结构清晰；凭证/密钥绝不入仓库（`.env` 不入 commit）。

### D3 Run & Iteration Logs（每轮日志）
- 每轮四要素，映射现状：
  - hypothesis → `run_log.jsonl` 的 `hypothesis` 字段 ✓
  - **code diff** → 本 Agent 每轮只改 config 不改代码，`action` 字段即 config 增量；为可复现性补一个 run 起始 `git_sha` 即可，无需逐轮 code diff（**待补**）。
  - 指标 → `metrics`（GAUC/nDCG@5/primary）✓
  - error/recovery → `errors` / `recovery` ✓
- 「人工干预次数」总结 → `state.json` 的 `interventions`（交付时显式写入结果表）✓

### D4 最终提交 + 结果汇总
- 最终 `submission/final.csv`（starter-kit schema，`submit.py --check` 校验）。
- 结果表：valid-best GAUC/nDCG@5 + 对官方 baseline 的绝对 delta → README「Result」表已有雏形，交付时刷新为最终数字。
- 资源用量（供 Feasibility 打分）：迭代数（**50 轮 cap**，我们 ε=0.002/N=3 会提前收敛）、墙钟（`state.elapsed_s`）、GPU-hours（`state.gpu_hours`）、token **input+output 分列**（`llm.complete()` 已返回 input/output/cache 分列，但 `state` 只存总量 `tokens_used`，**交付前补 `tokens_input`/`tokens_output` 聚合**）。

### 交付前待办（缺口清单）
1. README 补「团队分工」（暂移除，交付前填实际团队，非 solo）。
2. `run_log` 补 run 起始 `git_sha` 字段（D3「code diff」可追溯）。
3. `state` 补 token input/output 分列聚合（D4）。
4. 新写 D1 Devpost 描述稿。
5. 交付时刷新 README 结果表与资源用量为最终跑的数字。
