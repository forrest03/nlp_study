# Week15 · 股票 Subagent 多空并行分析

基于 **Orchestrator-Workers** 拓扑的多 Agent 系统：主 Agent 拉取 A 股行情，并行派发「看多 / 看空」两个 Subagent 独立分析，最后综合给出多空判定。支持 SSE 流式可视化与并行/串行 A/B 对比。

---

## 1. 项目结构

```text
stock_subagents/
├── src/
│   ├── stock_data.py       # akshare 拉行情 + 近 5 日表格（防幻觉）
│   ├── react_loop.py       # 通用 ReAct 引擎（主/subagent 共用）
│   ├── agents.py           # 主 Agent + dispatch_subagents 并行派发
│   ├── llm_client.py       # DashScope Qwen 客户端
│   ├── serve.py            # FastAPI + SSE 流式服务
│   └── eval_compare.py     # parallel vs serial 对比实验
├── static/
│   ├── index.html          # 左拓扑 + 右 ReAct 过程流
│   └── viz/topology.js     # SVG 拓扑动画
├── .cache/                 # 股票代码映射缓存
├── outputs/                # eval_compare.json（运行后生成）
└── requirements.txt
```

---

## 2. 环境准备

```bash
cd stock_subagents
pip install -r requirements.txt

export DASHSCOPE_API_KEY="sk-xxx"
# 可选：export QWEN_MODEL="qwen-plus"
```

> 行情数据通过 `akshare` 获取，无需额外 API Key。

---

## 3. 启动服务

```bash
cd src
python -m uvicorn serve:app --host 0.0.0.0 --port 8003
```

浏览器打开：http://localhost:8003

- 输入公司名 + 日期（如 `比亚迪` / `2026-08-04`）
- 左侧：主 Agent → 看多/看空 Subagent 拓扑
- 右侧：各节点 ReAct 步骤（Thought / Action / Observation）实时滚动
- 底部：综合多空判定 + 并行加速统计

---

## 4. 核心流程

```text
用户问题
  ↓
主 Agent ReAct（工具: get_stock_data + dispatch_subagents）
  ├─ Step1: get_stock_data(公司|日期) → 拉行情，写入 shared_state
  ├─ Step2: dispatch_subagents(公司|日期)
  │           ├─ 看多 Subagent ReAct（read_stock_data）  ─┐
  │           └─ 看空 Subagent ReAct（read_stock_data）  ─┤ ThreadPool 并行
  └─ Step3: Final Answer → 综合多空观点 + 最终倾向
```

### 主 Agent 工具

| 工具 | 作用 |
|------|------|
| `get_stock_data` | 拉取当日 30 分钟 K 线 + 近 5 日日线表格 |
| `dispatch_subagents` | 派发看多/看空两个 Subagent 并行分析 |

### Subagent 工具

| 工具 | 作用 |
|------|------|
| `read_stock_data` | 读取主 Agent 已拉取的共享行情（不重复联网） |

---

## 5. CLI 运行

```bash
# 单次多空分析
cd src
python agents.py

# 并行 vs 串行对比（2 题）
python eval_compare.py
python eval_compare.py --limit 1   # 快速版
```

Python 调用示例：

```python
import sys; sys.path.insert(0, "src")
from agents import run_research

r = run_research("查询 比亚迪 在 2026-08-04 的股票，给出多空分析")
print(r["final_answer"])
print(r["parallel_stats"])
```

---

## 6. API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（LLM / 模型名） |
| POST | `/query` | SSE 流式分析，`{question}` 或 `{company, date}` |
| GET | `/` | Web 可视化页面 |

SSE 事件类型：`start` → `main_step` → `dispatch` → `subagent_step` → `subagent_done` → `final` → `done`

---

## 7. 并行 vs 串行实验

`eval_compare.py` 对同一 `(公司, 日期)` 分别跑：

- **并行**：`ThreadPoolExecutor` 同时跑看多/看空 Subagent
- **串行**：`serial=True`，for 循环顺序执行（基线）

对比指标：

| 指标 | 含义 |
|------|------|
| `wall_clock` | 并行墙钟时间（≈ max(单 agent)） |
| `serial_sum` | 各 Subagent 时长之和（串行基线） |
| `speedup` | `serial_sum / wall_clock` |

预期结论：2 个独立 Subagent 并行后，dispatch 阶段墙钟从 **sum 压到 ≈ max**，体现 Subagent 并行的核心价值。

---

## 8. 与 Week15 课程对应

| 概念 | 本项目体现 |
|------|------------|
| Orchestrator-Workers | 主 Agent 取数据 → 派发 Worker → 综合 |
| ReAct | 主/Subagent 均为 Thought → Action → Observation 循环 |
| 并行 Subagent | `ThreadPoolExecutor` 并行看多/看空分析 |
| 可视化 | 拓扑图 + 点节点看 ReAct trace |
| 防幻觉 | Subagent 必须引用近 5 日行情表格中的具体数值 |
