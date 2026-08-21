# STOCK_AGENT_GUIDE.md — 股票分析 Agent 使用说明

## 1. 功能概述

`stock_agent.py` 是在 `agents.py`（市场调研 Agent）基础上新增的**股票分析 Agent**，输入一个股票代码或名称，自动并行派发 3 个子调研员完成以下工作：

| 子课题 | 调研内容 |
|--------|----------|
| 公司概况 | 主营业务、成立时间、总部、上市情况、市值 |
| 近期走势 | 近 3 个月股价变动、成交量趋势、技术指标 |
| 近期新闻 | 最近 1 个月重大公告、行业政策、事件动态 |

架构与 `agents.py` 完全一致：**Orchestrator-Workers 拓扑**，主 Agent 是 ReAct 循环（2 个工具），3 个子 Agent 用 `ThreadPoolExecutor` 并行执行，wall-clock ≈ max(子时长) 而非 sum。

---

## 2. 环境准备

### 2.1 依赖
与现有项目完全相同，无需额外安装：
```bash
cd market_research_subagents
pip install -r requirements.txt
```

### 2.2 API Key
```bash
export DEEPSEEK_API_KEY="sk-xxx"     # LLM 推理（主 + subagent）
export TAVILY_API_KEY="tvly-xxx"     # 联网搜索
```

### 2.3 验证环境
```bash
python -c "import sys; sys.path.insert(0, 'src'); from stock_agent import analyze_stock; print('stock_agent 导入成功')"
```

---

## 3. 快速开始

### 3.1 命令行直接运行
```bash
python src/stock_agent.py
# 提示输入：请输入股票代码或名称（如 600519 或 贵州茅台）：
```

内置自测会打印：主 Agent 动作序列、派发次数、subagent 数量、并行统计、最终报告。

### 3.2 作为模块调用
```python
import sys; sys.path.insert(0, "src")
from stock_agent import analyze_stock

# 方式一：股票名称
r = analyze_stock("贵州茅台")

# 方式二：股票代码
r = analyze_stock("600519")

# 查看结果
print(r["final_answer"])           # 最终分析报告
print(r["parallel_stats"])         # 并行加速统计
print(len(r["subagents"]))         # 派发的 subagent 数量
```

### 3.3 带回调（对接 SSE / 日志 / 可视化）
```python
import sys; sys.path.insert(0, "src")
from stock_agent import analyze_stock

def on_main_step(step):
    print(f"[主] {step['action']}: {step['action_input'][:40]}")

def on_subagent_step(sid, step):
    print(f"[{sid}] {step['action']}: {step['action_input'][:40]}")

def on_subagent_done(sid, duration, topic):
    print(f"[{sid}] 完成 ({duration}s) - {topic}")

def on_dispatch(info):
    print(f"派发 {len(info['subagent_ids'])} 个子agent: {info['subtopics']}")

r = analyze_stock("比亚迪",
                   on_main_step=on_main_step,
                   on_subagent_step=on_subagent_step,
                   on_subagent_done=on_subagent_done,
                   on_dispatch=on_dispatch)

print(f"\n最终报告:\n{r['final_answer']}")
```

---

## 4. 返回数据结构

`analyze_stock()` 返回字典，与 `run_research()` 格式完全一致：

```python
{
    "final_answer": "结构化股票分析报告（公司概况/走势/新闻/投资建议/风险提示）",
    "main_trace": [                                 # 主 Agent 的 ReAct 全过程
        {
            "idx": 0,                               # 步骤序号
            "agent": "stock_main",                  # agent 名称
            "thought": "这是股票分析...",            # 推理
            "action": "dispatch_subagents",          # 动作
            "action_input": "...",                   # 参数
            "observation": "...",                    # 工具返回
            "final": False                          # 是否最终步
        },
        ...
    ],
    "subagents": {                                  # 各 subagent 详情
        "sub_abc123": {
            "subtopic": "贵州茅台 公司概况...",
            "trace": [...],                         # 该 subagent 的 ReAct trace
            "duration": 12.3,                       # 执行时长（秒）
            "final_answer": "子调研结论"
        },
        ...
    },
    "parallel_stats": [                             # 并行加速统计
        {
            "n_subagents": 3,
            "wall_clock": 15.2,                     # 并行墙钟时间
            "serial_sum": 35.7,                     # 串行时间之和
            "speedup": 2.35                          # 加速倍数
        }
    ],
    "dispatches": [                                 # 派发记录
        {
            "subtopics": ["公司概况...", "近期走势...", "近期新闻..."],
            "subagent_ids": ["sub_abc123", "sub_def456", "sub_ghi789"]
        }
    ]
}
```

---

## 5. 架构详解

### 5.1 流水线
```
用户输入 "贵州茅台"
   ↓
主 Agent (stock_main) — ReAct 循环
  工具: web_search + dispatch_subagents
  系统提示: STOCK_SYSTEM（针对股票分析优化）
   │
   │  Thought: 股票分析有 3 个侧面，必须并行派发
   │  Action: dispatch_subagents
   │  Action Input:
   │    贵州茅台 公司概况：主营业务、成立时间、总部、上市情况、市值 |
   │    贵州茅台 股票近期走势：近3个月股价变动、成交量趋势、技术指标 |
   │    贵州茅台 公司近期新闻：最近1个月重大公告、行业政策、事件动态
   ↓
┌─────────────┬─────────────┬─────────────┐
│  subagent1  │  subagent2  │  subagent3  │  ThreadPoolExecutor 并行
│  公司概况    │  近期走势    │  近期新闻    │  wall-clock ≈ max(3个)
│  (ReAct×4)  │  (ReAct×4)  │  (ReAct×4)  │  而非 sum
└─────────────┴─────────────┴─────────────┘
   ↓ 汇总 Observation（含并行加速统计）
主 Agent 综合 3 份结果
  → 结构化股票分析报告
    （公司概况 / 走势分析 / 新闻事件 / 投资建议 / 风险提示）
```

### 5.2 与 agents.py 的对比

| 维度 | agents.py（市场调研） | stock_agent.py（股票分析） |
|------|----------------------|--------------------------|
| 系统提示 | `MAIN_SYSTEM`（通用调研） | `STOCK_SYSTEM`（股票分析专用） |
| 默认子课题数 | LLM 自主决定 | 3 个（公司概况 / 走势 / 新闻） |
| 入口函数 | `run_research(question)` | `analyze_stock(stock_input)` |
| 返回格式 | 完全一致 | 完全一致 |
| 并行机制 | ThreadPoolExecutor | ThreadPoolExecutor |
| ReAct 引擎 | 共用 `react_loop.py` | 共用 `react_loop.py` |

### 5.3 关键设计
- **主 Agent 自主路由**：`STOCK_SYSTEM` 引导 LLM 在 Thought 阶段决定派发 3 个子课题，而非硬编码。LLM 会根据输入（名称/代码）自动调整查询词。
- **并行加速**：3 个 subagent 用 `ThreadPoolExecutor(max_workers=3)` 同时执行。A/B 对比（`serial=True`）可量化加速比。
- **完整 trace**：每个 subagent 的 Thought/Action/Observation 全程记录，用于调试和可视化。
- **结果截短**：每个子结果截短到 500 字喂回主 Agent，避免撑爆 context（完整 trace 仍在 `shared_state` 中）。

---

## 6. 接入 HTTP 服务（可选）

`stock_agent.analyze_stock()` 的回调接口与 `agents.run_research()` 完全一致，可以直接在 `serve.py` 中新增一个 `/stock` 端点：

```python
# serve.py 中新增
from stock_agent import analyze_stock as run_stock_analysis

@app.post("/stock")
def stock_query(req: QueryRequest):
    """SSE 流式：股票分析"""
    def event_stream():
        q = queue.Queue()
        SENTINEL = object()

        def push(ev): q.put(ev)
        def on_main_step(step): push({"type": "main_step", **step})
        def on_dispatch(info): push({"type": "dispatch", **info})
        def on_subagent_step(sid, step):
            push({"type": "subagent_step", "subagent_id": sid, **step})
        def on_subagent_done(sid, duration, topic):
            push({"type": "subagent_done", "subagent_id": sid,
                  "duration": duration, "subtopic": topic})

        def run():
            try:
                r = run_stock_analysis(
                    req.question,
                    on_main_step=on_main_step,
                    on_dispatch=on_dispatch,
                    on_subagent_step=on_subagent_step,
                    on_subagent_done=on_subagent_done,
                )
                push({"type": "final", "answer": r["final_answer"],
                      "parallel_stats": r["parallel_stats"]})
            except Exception as e:
                push({"type": "error", "message": f"{type(e).__name__}: {str(e)[:200]}"})
            finally:
                push(SENTINEL)

        threading.Thread(target=run, daemon=True).start()
        yield "data: " + json.dumps({"type": "start", "question": req.question},
                                    ensure_ascii=False) + "\n\n"
        while True:
            ev = q.get()
            if ev is SENTINEL:
                yield "data: " + json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"
                break
            yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

---

## 7. 串行对比测试

使用 `serial=True` 让 3 个 subagent 串行执行，与并行对比量化加速：

```python
import sys, time; sys.path.insert(0, "src")
from stock_agent import analyze_stock

stock = "贵州茅台"

t0 = time.time()
r_parallel = analyze_stock(stock, serial=False)
parallel_wall = time.time() - t0

t0 = time.time()
r_serial = analyze_stock(stock, serial=True)
serial_wall = time.time() - t0

print(f"并行墙钟: {parallel_wall:.1f}s")
print(f"串行墙钟: {serial_wall:.1f}s")
print(f"加速比: {serial_wall/parallel_wall:.2f}x")
print(f"dispatch 加速: {r_parallel['parallel_stats'][-1]['speedup']}x")
```

---

## 8. 调试与常见问题

**Q: 主 Agent 不派发 subagent，自己串行搜索？**
A: `STOCK_SYSTEM` 已引导股票分析必须派发 3 个子课题。若仍偶发，确认 `system_prompt=STOCK_SYSTEM` 传给了 `ReActLoop`。

**Q: 主 Agent 只派发 1-2 个 subagent 而非 3 个？**
A: LLM 可能判断某些侧面信息重叠。在 `STOCK_SYSTEM` 中已明确要求 3 个标准子课题（公司概况/走势/新闻），LLM 会遵守。

**Q: 股票代码搜不到公司信息？**
A: Tavily 搜索用自然语言，代码（如 `600519`）可能匹配不佳。建议输入股票名称（如 `贵州茅台`），Agent 会自动补充"600519"等关键词。

**Q: 并行加速比小于预期？**
A: Amdahl 定律。总墙钟 = 主 Agent 串行段（规划 + 综合）+ dispatch 并行段。只有并行段加速，主 Agent 的两次 LLM 调用（规划 + 综合）是串行的，拉低了总加速比。

**Q: 想自定义子课题？**
A: 修改 `STOCK_SYSTEM` 中的「标准子课题拆分」部分，或在调用 `analyze_stock` 前自行构造 question。也可以直接用 `_dispatch_subagents` 函数：
```python
from stock_agent import _dispatch_subagents
result = _dispatch_subagents("贵州茅台 主营业务 | 贵州茅台 市值", shared_state={})
```

**Q: serial 模式怎么跑？**
A: `analyze_stock(stock, serial=True)`，subagent 改 for 循环顺序执行（eval 基线）。

---

## 9. 目录结构

```
market_research_subagents/
├── src/
│   ├── tavily_search.py     # Tavily 搜索（urllib 零依赖）
│   ├── react_loop.py        # 通用 ReAct 引擎
│   ├── llm_client.py        # 极简 DeepSeek 客户端
│   ├── agents.py            # 市场调研 Agent（参考模板）
│   ├── stock_agent.py       # ★ 股票分析 Agent（新增）
│   ├── serve.py             # FastAPI + SSE
│   └── eval_compare.py      # parallel vs serial A/B
├── static/
│   ├── index.html
│   └── viz/topology.js
├── outputs/
├── requirements.txt
├── ARCHITECTURE.md          # 架构说明（通用）
├── USAGE_GUIDE.md           # 使用指南（通用）
├── STOCK_AGENT_GUIDE.md     # ★ 本文档（股票分析专用）
└── RESUME_GUIDE.md          # 简历指导
```