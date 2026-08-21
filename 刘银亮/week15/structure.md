# Week15：支持 Sub-Agent 分发的国产汽车博主 Agent

## 概述

实现了一个基于 ReAct 模式的 AI Agent，定位为 **国产汽车博主**，专注于中国品牌汽车（比亚迪、蔚来、小鹏、理想、吉利、长城、小米等）的评测与对比分析。

核心能力：当遇到多品牌/多车型/多维度对比问题时，主 Agent 自动拆分子问题，通过线程池并行派发子 Agent 进行独立调研，最后汇总结果输出博文风格的对比分析。

## 文件结构

```
week15/src/
├── llm_client.py        # LLM 客户端（阿里百炼 DashScope）
├── tavily_search.py     # Tavily 联网搜索工具
├── react_loop.py        # 通用 ReAct 循环引擎
└── agent.py             # 主 Agent 编排（子 Agent 分发逻辑）
```

### 依赖关系

```
agent.py
  ├── react_loop.py  → llm_client.py
  └── tavily_search.py
```

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────────────────┐
│                    agent.py                          │
│  汽车主编（主 Agent）                                 │
│  ├── 工具: web_search         (直接搜索)             │
│  └── 工具: dispatch_subagents (派发子 Agent)         │
│         │                                            │
│         │  ThreadPoolExecutor 并行                   │
│         ▼                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ 子研究员-1 │  │ 子研究员-2 │  │ 子研究员-3 │  ...     │
│  │ReActLoop  │  │ReActLoop  │  │ReActLoop  │          │
│  │web_search │  │web_search │  │web_search │          │
│  └──────────┘  └──────────┘  └──────────┘           │
│         │            │            │                   │
│         └────────────┴────────────┘                   │
│                      │                                │
│              汇总结果 → Observation                    │
│                      │                                │
│              主 Agent 综合 → Final Answer              │
└─────────────────────────────────────────────────────┘
│                    react_loop.py                      │
│  ReActLoop（通用引擎）                                │
│  Thought → Action → Action Input → Observation → ...  │
└─────────────────────────────────────────────────────┘
│                    llm_client.py                      │
│  llm_chat() → OpenAI SDK → 阿里百炼 DashScope         │
└─────────────────────────────────────────────────────┘
```

## 实现理念

### 1. 主 Agent 与子 Agent 共享同一引擎

核心洞察：**主 Agent 和子 Agent 都是 ReAct 循环，区别仅在于拥有哪些工具。**

- 主 Agent 工具集：`web_search` + `dispatch_subagents`
- 子 Agent 工具集：仅 `web_search`

`ReActLoop` 类通过 `tools` 参数实现工具集的可插拔，同一个类实例化出不同能力的 Agent，避免了代码重复。

### 2. 工具驱动而非代码驱动

子 Agent 的派发不是通过代码硬编码的"如果 A 则 B"逻辑，而是：

1. **System Prompt 引导**：主 Agent 的 `MAIN_SYSTEM` prompt 告诉 LLM 什么情况下用 `dispatch_subagents`、如何用 `||` 分隔子问题
2. **LLM 自主决策**：LLM 分析问题，自行判断是否需要拆分、拆成几个子问题
3. **工具函数执行**：`dispatch_subagents` 函数接收 LLM 生成的子问题列表，执行派发

这种设计让 Agent 的行为由 prompt 和工具塑造，而非硬编码的分支逻辑，更灵活、更易扩展。

### 3. 线程池并行派发

子问题之间通常相互独立（如"比亚迪汉性能"和"蔚来ET7价格"），采用 `ThreadPoolExecutor` 并行执行，显著缩短总耗时：

```
串行：子Agent1(3s) + 子Agent2(3s) + 子Agent3(3s) = 9s
并行：max(子Agent1, 子Agent2, 子Agent3) ≈ 3s
```

每个子 Agent 有 120s 超时保护，防止单个任务卡死阻塞整体。

### 4. shared_state 共享状态

通过 `shared_state` 字典贯穿整个调用链，实现：

- **Trace 收集**：子 Agent 的执行轨迹存入 `shared_state["subagent_traces"]`，供调试和可视化
- **无侵入传递**：`ReActLoop._exec_tool()` 自动将 `shared_state` 传给工具函数，工具函数无需感知调用方

### 5. System Prompt 角色分离

三个 System Prompt 各司其职：

| Prompt | 位置 | 角色 | 特点 |
|--------|------|------|------|
| `REACT_SYSTEM` | `react_loop.py` | 默认汽车博主 | 通用 ReAct 格式，子 Agent 未指定 prompt 时使用 |
| `MAIN_SYSTEM` | `agent.py` | 汽车主编 | 引导拆分子问题、使用 `dispatch_subagents`、博文风格输出 |
| `SUB_SYSTEM` | `agent.py` | 汽车研究员 | 专注搜索与信息整理，不带派发逻辑 |

### 6. ReAct 经典实现技巧

使用 `stop=["Observation:"]` 让 LLM 在生成完 Action Input 后自动停止，Runner 执行工具后再补上 Observation 文本续写——这是 ReAct 论文中的经典实现方式，避免了 LLM 自行编造工具结果（幻觉）。

## 数据流

```
用户问题: "对比分析比亚迪汉2026款和小米SU7 2026款的性能、价格和续航"
  │
  ▼
main_agent.run(question, shared_state={})
  │
  ▼ [Step 0]
LLM 生成:
  Thought: 涉及两个品牌、三个维度，需要拆分子问题并行调研
  Action: dispatch_subagents
  Action Input: 比亚迪汉2026款性能参数 || 比亚迪汉2026款价格和续航 || 小米SU7 2026款性能参数 || 小米SU7 2026款价格和续航
  │
  ▼
dispatch_subagents() 执行:
  ├── 解析 "||" → 4 个子问题
  ├── 创建 4 个 ReActLoop 子 Agent
  ├── ThreadPoolExecutor 并行执行
  ├── 收集结果 → shared_state["subagent_traces"]
  └── 返回汇总文本
  │
  ▼ [Step 0 Observation]
[子研究员-1] 子问题: 比亚迪汉2026款性能参数
调研结果: 比亚迪汉2026款搭载...（耗时 3.2s）
---
[子研究员-2] 子问题: 比亚迪汉2026款价格和续航
...
  │
  ▼ [Step 1]
LLM 综合子调研结果:
  Thought: 我已收集到两款车在性能、价格、续航三个维度的信息
  Final Answer: （博主口吻的对比分析长文）
```

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 派发粒度 | 批量并行（线程池） | 子问题独立，并行可显著缩短总耗时 |
| 输入格式 | `\|\|` 分隔符 | 简单直观，LLM 容易生成，解析成本低 |
| 子 Agent 步数 | 3 步 | 子任务聚焦单一搜索，3 步足够；主 Agent 6 步，因需要拆解+综合 |
| 超时策略 | 120s 单子任务 | 防止个别子任务卡死阻塞整体 |
| 模型选择 | deepseek-v4-flash | 阿里百炼兼容接口，性价比高 |

## 扩展方向

- **增加子 Agent 工具类型**：除 `web_search` 外，可给子 Agent 增加 `calculator`（计算）、`image_search`（图片搜索）等
- **支持嵌套派发**：子 Agent 也可拥有 `dispatch_subagents` 工具，实现多级树状分解
- **流式输出**：通过 `ReActLoop.run()` 的 `on_step` 回调，实现 SSE 流式推送每步决策和执行结果
- **多模型支持**：主 Agent 用强模型（如 deepseek-chat），子 Agent 用快模型（如 deepseek-v4-flash），平衡质量与速度