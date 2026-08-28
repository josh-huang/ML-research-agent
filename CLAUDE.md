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
  report/                  # 静态 HTML run-log 报告（deliverable）
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
