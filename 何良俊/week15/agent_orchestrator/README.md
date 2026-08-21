# Agent Orchestrator · 可并行下发 subagent 的编排 Agent

一个能**拆解任务 → 并行下发 subagent → 收集结果 → 汇总答案**的编排型 Agent。
编排者与 subagent 都是完整 agent，共用同一套决策循环核心；编排者多了一个
`dispatch_subagents` 工具，用于把任务拆给多个**独立决策循环**的 subagent 并行执行。

```
用户问题
   │
   ▼
┌──────────────────────── 主 Agent（orchestrator）───────────────────┐
│ 系统提示词(角色规则) → LLM 推理 → 决策                              │
│   ├─ 能直接回答 → 直接答（可选 calculator / current_date 辅助）     │
│   └─ 可拆分为并行子任务 → 调用 dispatch_subagents 工具              │
│      │                                                              │
│      ▼ ThreadPoolExecutor（并行）                                    │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                       │
│  │ subagent-1 │  │ subagent-2 │  │ subagent-3 │                     │
│  │ 角色提示词  │  │ 角色提示词  │  │ 角色提示词  │                     │
│  │ 决策循环    │  │ 决策循环    │  │ 决策循环    │                     │
│  │ calculator │  │ calculator │  │ calculator │   ← 与主 agent 相同的 │
│  │ current_date│ │ current_date│ │ current_date│     通用能力，仅无     │
│  │            │  │            │  │            │     下发能力          │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                       │
│        └──────────────┼──────────────┘                             │
│                       ▼                                             │
│              收集全部结果 → 完整结果展示给用户                       │
│              → 压缩摘要作为 Observation 回传主 LLM → 最终汇总       │
└─────────────────────────────────────────────────────────────────────┘
```

## Agent 必须组件清单（对应实现）

| # | 组件 | 位置 | 说明 |
|---|------|------|------|
| 1 | 系统提示词 | `orchestrator.py` / `agent.py` | 主 agent 角色（何时直接答、何时下发）+ 每个 subagent 独立的子任务角色 |
| 2 | LLM 推理 | `llm.py` | DeepSeek Chat Completions 封装（SSE 流式 + function calling），线程安全 |
| 3 | 决策循环 | `agent.py` | 迭代：调 LLM（**流式**）→ 有 tool_calls 则执行并回传 → 无则产出最终答案；**步数上限兜底**（主 agent `--max-steps` 默认 12，subagent `--subagent-max-steps` 默认 6，防无限循环） |
| 3.5 | ReAct 过程展示 | `llm.py` + `display.py` | 主 agent 的思维链（`reasoning_content` 字段）流式显示为 🧠 Thought，配合步骤横幅、Action/Obs、最终汇总构成完整可见的 ReAct 轨迹 |
| 4 | 工具 / Function Calling | `tools.py` + `orchestrator.py` | 通用工具集 `calculator`/`current_date` 主 agent 与 subagent 都有；`dispatch_subagents` 只有主 agent 有——**是否下发由 LLM 自主决定**（`tool_choice="auto"`），拆不开就直接答 |
| 5 | 并行 subagent 下发 | `orchestrator.py` | `ThreadPoolExecutor` 并发执行，每个 subagent 独立 messages 与循环 |
| 6 | 结果回传 | `agent.py` | 观察截断回传 LLM 保护上下文；完整结果走事件给展示层 |
| 7 | 结果展示 | `display.py` | 线程安全；主循环轨迹 + subagent 分栏报告 + 最终汇总 |
| 8 | 最终汇总 | `orchestrator.py` | 主 LLM 基于全部子结果输出结构化答案 |

## 使用方式

```bash
# 内置 demo：三个子主题并行调研
python main.py --demo

# 自定义任意可拆分任务（编排者自己拆解）
python main.py --question "请从产品定位、技术路线、商业化三个角度分析三家 AI 公司，并横向对比"

# 调并行度 / 步数 / 保存完整报告
python main.py --question "……" --max-workers 5 --subagent-max-steps 8 --save-json report.json
```

配置：`DEEPSEEK_API_KEY` 必填；`DEEPSEEK_MODEL`（默认 `deepseek-v4-flash`）可选。
依赖：仅 `requests`。

## 教学要点

1. **主 agent 与 subagent 是同构的**：都是「角色提示词 + 决策循环 + 工具集」。
   subagent = 主 agent 的能力全集 - `dispatch_subagents`（不能再往下发），
   subagent 拿到的是孤立的自包含任务，彼此无通信——这是最简单的多 agent 协作形态。
2. **下发即工具调用**：`dispatch_subagents` 是普通 function call，是否调用完全由
   主 agent 的 LLM 决定（能直接答就直接答）；其返回值就是 Observation，
   主 LLM 据此汇总——多 agent 系统可以挂在任何 agent 循环上。
3. **工具纪律写在提示词里**：研究/分析类任务默认不调工具；calculator 只在确实
   需要数值计算时调用一次，绝不连环调用——避免模型"为了用工具而用工具"。
4. **主 agent 流式、subagent 整块输出**：主 agent 的最终汇总走 SSE 流式，
   边生成边打印；subagent 并行跑时结果统一在完成后以分栏报告块一次性展示，
   方便逐块查看（`agent.py` 的 `stream_final` 开关控制）。
5. **主 agent 的 ReAct 过程完整可见**：DeepSeek 的 `reasoning_content` 字段暴露
   模型的思维链（Thought），配合"步骤横幅 → 🧠 Thought → 🔧 Action → 👁 Obs →
   ✅ 最终汇总"的轨迹呈现，主 agent 每一步决策过程都实时可见。
3. **并行与上下文保护**：subagent 在线程池里并行跑（共享客户端线程安全）；
   完整结果展示给用户，回传 LLM 的只是截断摘要，避免上下文爆炸。
4. **展示即调试**：实时事件流（启动/完成/工具调用）与最终分栏报告并存，
   既能观察并行推进，也能事后细看每个 subagent 做了什么。

## 目录结构

```
agent_orchestrator/
├── llm.py            # DeepSeek 客户端（线程安全，function calling）
├── tools.py          # 通用小工具：calculator / current_date
├── agent.py          # 通用 Agent 决策循环核心（编排者与 subagent 共用）
├── orchestrator.py   # 编排 Agent：拆解 → 并行下发 → 收集 → 汇总
├── display.py        # 线程安全展示层
├── main.py           # CLI 入口
└── README.md
```
