# Week14 — 自进化 Agent：习题 → 知识点 Skill 优化

参照 `study/week14 自进化agent` 的 Skill Nudge 机制，场景从电商客服改为
**初中数学习题知识点匹配**。评估数据为知识点抽样题干与判定标准。

## 目标

1. 用大模型维护可进化的匹配 Skills（create / patch）
2. 从失败样本驱动进化，量化准确率提升
3. 再压 Skill 体积，对比 **准确率 / Skill token / 评测 prompt token**

## 快速开始

```bash
cd self_evolving_agent
pip install -r requirements.txt

python scripts/build_eval_from_dataset.py

export LLM_PROVIDER=qwen
export DASHSCOPE_API_KEY=sk-xxx   # 或 DEEPSEEK_API_KEY + LLM_PROVIDER=deepseek

# 自进化主实验（基线 → 分块 Nudge → 最终）
python src/demo_runner.py

# 可选：压缩 Skill 体积后再评测
python scripts/compress_skills.py
```

结果：`outputs/evolution_log.json`、`outputs/compress_comparison.json`。

## 一次实测对比（Qwen Plus，64 题）

| 阶段 | 准确率 | Skill est_tokens | 说明 |
|------|--------|------------------|------|
| 基线（2 个初始 Skill） | **25.0%** | 468 | 仅覆盖最简整数比 / 韦达 |
| 进化后（+6 create，1 patch） | **78.1%** | 2477 | 覆盖 8 个知识点 |
| 压缩后 | **76.6%** | 1780 | token −28%，准确率仅 −1.5pp |

## 目录

```
self_evolving_agent/
├── src/                 # Agent / Reviewer / Evaluator / SkillManager
├── data/
│   ├── topic_bundle.json   # 从 dataset 抽样的 8 个知识点
│   ├── policies.md         # 仅 Reviewer 可读的判定标准
│   ├── eval_set.json
│   └── demo_script.json
├── skills/              # 活动 Skills
├── scripts/
│   ├── build_eval_from_dataset.py
│   └── compress_skills.py
└── outputs/
```

## 契约

| 角色 | 行为 |
|------|------|
| Agent | 只读 Skills；能匹配则只输出知识点名；否则只说「无法匹配知识点」 |
| Evaluator | 含「无法匹配」→ 推脱失败；否则检查 required / forbidden |
| Reviewer | 只看失败样本 + policies；最小改动 create/patch，并控制 token |

## 与客服 Demo 的对应关系

| 客服版 | 本项目 |
|--------|--------|
| 退款政策问答 | 习题 → 知识点名称 |
| policies.md 电商规则 | policies.md 知识点定义 + 代表题 |
| 「联系人工客服」 | 「无法匹配知识点」 |
| refund / vip Skills | ratio-simplify / vieta / … |
