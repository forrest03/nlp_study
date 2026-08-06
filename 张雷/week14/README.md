# Week14 · 超级全能周报生成器 Skill（V1/V2 + 四维度优化）

> 一个"周报生成器" Claude Skill 的完整生命周期：从 V1.0 设计、试跑实测，到按四个维度优化出 V2.0，再用真实 API 调用验证"省 Token 不丢功能"，最后产出带改动注释的 Prompt 文件与优化报告并入库。

## 任务闭环

```
① 创建 skill (V1.0 完整详尽版)
        ↓
② V1 试跑 ── deepseek-chat 实测 Token 用量与质量，发现"系统提示词占 2324 tok、推测数据未标注"等硬伤
        ↓
③ 四维度优化 → V2.0（效能优化版）
        提示词蒸馏 · Few-shot 动态裁剪 · 输出格式压缩 · CoT 剥离
        （约束：功能逻辑与 V1.0 完全一致，不偷工减料）
        ↓
④ 实测验证 ── V1/V2 同输入各跑 2 次，Token −56% / 加速 2.39× / 质量 0 衰减 / 成本省 58.9%
        ↓
⑤ 生成带注释的 Prompt 文件（V1 Baseline / V2 Optimized）+ 优化报告（数据看板 + 归因分析）
        ↓
⑥ 入库 ── commit b662d04（week14 分支），生成的周报样例按 .gitignore 排除
```

## 核心结论（数据看板）

实验：deepseek / deepseek-chat，temperature=0.7，同一 60 字符输入，V1/V2 各跑 2 次取均值。

| 对比项 | 公式 | V1(Before) | V2(After) | 实测 | 目标 | 达标 |
|---|---|---|---|---|---|---|
| Token 压缩率 | `(B−A)/B`（Total） | 4,412 tok | 1,939.5 tok | **56.0%** | ≥30% | ✅ |
| 成本节省率 | GPT-4o $5/M in、$15/M out | 4.29 美分/次 | 1.77 美分/次 | **省 2.53 美分/次 ≈ ¥0.182/次；58.9%** | 美分/元 | ✅ |
| 质量衰减率（关键） | `(B_Score−A_Score)/B_Score` | 7.5 | 7.5 | **0.0%** | <5% | ✅ 有效 |
| 响应加速比 | `B_Time/A_Time` | 24.18 s | 10.13 s | **2.39×** | ≥1.2× | ✅ |

**四项全部达标，质量零衰减，优化有效。** 规模化估算：5 万次/月约省 ¥9,100。

## 四维度优化策略

| 维度 | 操作 | 实测效果 |
|---|---|---|
| ① 提示词蒸馏 | 啰嗦自然语言 → 结构化关键词；删角色叙事/感叹强调/重复小节 | system_prompt 4021→1704 字符(−57.6%)；**最大单项贡献 ~870 tok(~35%)** |
| ② Few-shot 动态裁剪 | 5 示例固定内联 → 移到 references/，输入<80字载1例、≥80字 zero-shot | vs V1(5例) 省 260-415 tok/次 |
| ③ 输出格式压缩 | 去 Emoji/多余空行/`<output>`包裹，状态改 `[正常/风险/阻塞]` 文本 | completion −52%〜−62%（含④贡献） |
| ④ CoT 剥离 | `<thinking>` 输出推理 → 6步改内部处理、不输出 | 省 ~460 completion tok；最划算（零质量代价） |

## 目录结构

```
week14/
├── README.md                          # 本文件
├── .gitignore                         # 排除 sample_report*.md（生成周报不入库）
├── skills/
│   ├── weekly-report-generator/       # V1.0 skill（完整详尽版）
│   │   └── SKILL.md
│   └── weekly-report-generator-v2/    # V2.0 skill（效能优化版）
│       ├── SKILL.md
│       └── references/
│           └── few_shot.md            # 动态示例库（按需加载1个）
├── Skill_V1_Baseline.md               # 交付物：V1 Prompt + 逐段改动注释
├── Skill_V2_Optimized.md              # 交付物：V2 Prompt + 逐段改动注释
├── Optimization_Report.md             # 交付物：优化报告（看板+删改理由+归因+质量分析）
├── test_log.md                        # 实验记录（每次调用的 token/评分原始数据）
├── run_test.py                        # 通用测试脚本（v1/v2 + 动态Few-shot模拟 + 计时）
└── sample_report*.md                  # 生成的周报样例（gitignored，仅本地）
```

## 如何复现

环境：conda env `py312`，需 `DEEPSEEK_API_KEY`（备选 `DASHSCOPE_API_KEY`）在环境变量中。

```bash
# V1 试跑（同口径）
conda run -n py312 python run_test.py --version v1 \
  --input "本周主要在写新的用户登录模块……周五跟测试吵了一架，因为提的bug太多了。"

# V2 试跑（短输入自动加载1示例；--quiet 仅打印计时/Token摘要）
conda run -n py312 python run_test.py --version v2 \
  --input "本周主要在写新的用户登录模块……" --quiet

# V2 zero-shot（≥80字符清晰输入）
conda run -n py312 python run_test.py --version v2 \
  --input "本周完成了用户登录模块开发，含三种登录方式后端接口8个；重构后台页面统一12页设计规范……" --quiet
```

脚本逻辑：把 `SKILL.md` 全文作 system prompt，用户输入作 user message，调 OpenAI 兼容接口；v2 且输入<80 字符时自动追加 1 个 few-shot 示例（模拟动态加载），如实计入 prompt tokens，并计时。

## 关键发现

- **最有效的策略**：①提示词蒸馏（~870 tok，~35%）。V1 大量篇幅是渲染性自然语言，蒸馏为结构化关键词后语义零损失。
- **质量零衰减的代价**：6步 CoT 改为内部处理后，输出结构未降级（仍分类/价值升华/风险/计划），证明 CoT 剥离不伤功能。
- **V2 一处进步**：姓名正确用 `[待补充]`（V1 曾虚构"张明"）。
- **持续硬伤（如实标注）**：推测数据标注仍不一致（V1=0 处、V2#2=0 处、V2#3=1 处）。模型倾向把编造的数字当事实写，属模型合规问题，非 V2 功能逻辑缺失。
- **后续 V2.1 方向**：把"推测必标注"升级为输出前强校验、纳入日期占位约束，目标把 0 标注拉到 ≥3 处。

## 相关文件索引

- 两版 Prompt 逐行对比：`Skill_V1_Baseline.md` / `Skill_V2_Optimized.md`
- 完整分析与看板：`Optimization_Report.md`
- 原始实验数据：`test_log.md`
