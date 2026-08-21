# ARCHITECTURE.md — 渐进式 Skills Harness 技术方案

## 一、项目定位

本项目对应 week13 课件中的两条主线：

1. **Skills 范式**：把能力封装为 `SKILL.md`（行为 + 知识 + 工具调用），而不是把全部 schema 塞进 system prompt。
2. **Harness Engineering**：为模型提供「在什么机制里运行」——触发、加载、执行、释放、审计与边界。

核心问题：

> 能力越强，上下文越重，推理越低效 —— 如何用渐进式披露（Progressive Disclosure）只加载当前需要的 Skill？

---

## 二、渐进式披露三层

```
┌─────────────────────────────────────────────────────────────┐
│                     Context Window                           │
│                                                             │
│  L0 常驻层（Always）     SKILLS.md 索引，每 Skill 一行摘要   │
│                          通常 < 300 tokens                   │
│                                                             │
│  L1 触发层（On Demand）  activate_skill → 完整 SKILL.md      │
│                          约 500–3000 tokens                  │
│                                                             │
│  L2 执行层（In Context） read_skill_file → references 等     │
│                          按需追加，用完可 release             │
└─────────────────────────────────────────────────────────────┘
```

| 层级 | 何时进入 Context | 实现 |
|------|------------------|------|
| L0 | 每次请求 | `SkillRegistry.build_index_text()` → system prompt |
| L1 | 模型判定需要某 Skill | 工具 `activate_skill` → `ProgressiveLoader.activate` |
| L2 | Skill 流程要求读参考资料 | 工具 `read_skill_file` → `ProgressiveLoader.read_resource` |
| 释放 | 任务完成 | 工具 `release_skill`（教学演示 + 统计清零） |

与课件对照：L0 ≈ MEMORY.md 索引；L1 ≈ 触发后加载的 `pptx.md`；L2 ≈ Skill 执行期再读的 references。

---

## 三、Harness 执行流水线

```
用户消息
    │
    ▼
触发初筛（关键词 / triggers，零 LLM 成本）
    │
    ▼
组装 System：Harness 规则 + 初筛提示 + L0 索引
    │
    ▼
ReAct / Function Calling 循环
    ├── list_skills / get_load_stats
    ├── activate_skill (L1)
    ├── read_skill_file (L2)
    ├── write_file / read_file
    ├── run_skill_script
    └── release_skill
    │
    ▼
Final Answer + Token 对照（当前 vs 全量加载）
```

**设计取舍**：Skill **不是**直接注册成几十个 Function；Harness 只暴露少量「加载与执行原语」。Skill 正文进入 context 后，由模型按自然语言流程调度这些原语 —— 这正是课件「Skills 整合 FC / MCP / RAG」的工程落点。

---

## 四、模块职责

| 模块 | 职责 |
|------|------|
| `skill_registry.py` | 扫描 `skills/*/SKILL.md`，解析 frontmatter，生成 `SKILLS.md` |
| `progressive_loader.py` | L0/L1/L2 状态、token 粗估、加载事件、全量对比 |
| `tools.py` | Harness 工具 schema + 安全边界（路径限制、危险命令黑名单） |
| `harness.py` | Agent 循环、教学事件流 |
| `agent.py` | CLI |
| `serve.py` + `index.html` | SSE Web 演示：轨迹 + Context 仪表盘 |

---

## 五、内置 Skills（三种形态）

| Skill | 形态 | 说明 |
|-------|------|------|
| `flash-card` | 知识复合 + 代码工具 | 生成 JSON → 脚本渲染 HTML |
| `text-stats` | 纯代码工具 | Skill 描述流程，统计由 `analyze.py` 完成 |
| `baoyu-diagram` | 工作流 + 知识复合 | 大 SKILL.md + `references/*` 示范 L2 按需加载 |

---

## 六、Token 对照（教学指标）

`ProgressiveLoader.snapshot()` 输出：

- `l0_tokens` / `l1_tokens` / `l2_tokens` / `current_tokens`
- `full_load_tokens`：假设把所有 Skill 与 references 一次性注入
- `saved_tokens`：`full_load - current`

用于直观回答课件问题：「为什么 Prompt 越来越长是个问题？」

---

## 七、目录结构

```
progressive_skills_harness/
├── src/
│   ├── llm_config.py
│   ├── skill_registry.py
│   ├── progressive_loader.py
│   ├── tools.py
│   ├── harness.py
│   ├── agent.py
│   └── serve.py
├── skills/
│   ├── SKILLS.md              # 自动生成的 L0 索引
│   ├── flash-card/
│   ├── text-stats/
│   └── baoyu-diagram/
├── workspace/                 # 运行产物
├── index.html
├── requirements.txt
├── ARCHITECTURE.md
└── USAGE_GUIDE.md
```
