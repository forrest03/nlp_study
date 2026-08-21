# GraphRAG Subagent 并行查询系统 — 架构设计与代码说明

## 目录

- [1. 概述](#1-概述)
- [2. 架构设计](#2-架构设计)
- [3. 执行流程](#3-执行流程)
- [4. 目录结构](#4-目录结构)
- [5. 后端代码说明](#5-后端代码说明)
- [6. 前端代码说明](#6-前端代码说明)
- [7. SSE 事件协议](#7-sse-事件协议)
- [8. 关键技术点](#8-关键技术点)
- [9. 性能分析](#9-性能分析)
- [10. API 接口文档](#10-api-接口文档)

---

## 1. 概述

本系统在 GraphRAG 知识图谱问答的基础上，引入了 **Subagent（子代理）并行查询架构**。主代理（Main Agent）接收用户问题后，同时派发两个子代理分别执行 Local Search 和 Global Search 检索，两个子代理并行运行、各自独立调用 LLM 生成回答，最后由主代理合并两个子代理的结果并再次调用 LLM 生成综合最终回答。

### 核心特性

| 特性 | 说明 |
|------|------|
| 并行执行 | Local Search 和 Global Search 通过 `ThreadPoolExecutor` 真正并行运行 |
| 实时状态推送 | 基于 SSE (Server-Sent Events) 流式推送 agent 状态变更 |
| 树形状态可视化 | 前端通过 CSS 树形结构展示主代理和子代理的实时状态 |
| 分步计时 | 精确记录每个 agent 的执行时间（Local / Global / Merge / Total） |
| 容错处理 | 单个子代理失败不影响另一个，主代理可降级使用单边结果 |

---

## 2. 架构设计

### 2.1 Agent 角色定义

```
┌─────────────────────────────────────────────────────────────┐
│                    🤖 主代理 (Main Agent)                     │
│                                                             │
│  职责：接收问题 → 派发子代理 → 等待并行完成 → 合并结果       │
│        → 调用 LLM 生成最终综合回答                           │
└──────────┬──────────────────────────────────┬──────────────┘
           │                                  │
           │  并行派发                         │  并行派发
           ▼                                  ▼
┌─────────────────────┐          ┌─────────────────────────┐
│ 🔍 Local Search     │          │ 🌍 Global Search        │
│    子代理           │          │    子代理               │
│                     │          │                         │
│ 关键词提取          │          │ 问题向量化              │
│ → Cypher 1跳邻域    │          │ → 社区摘要向量匹配      │
│ → 组装 Context      │          │ → Top-K 选择            │
│ → LLM 生成回答      │          │ → Map: 逐社区中间回答   │
│                     │          │ → Reduce: 汇总回答      │
│ 输出：局部检索回答  │          │ 输出：全局检索回答      │
└─────────┬───────────┘          └───────────┬─────────────┘
          │                                  │
          └──────────┐    ┌──────────────────┘
                     ▼    ▼
          ┌──────────────────────────────────┐
          │     🔀 主代理合并阶段             │
          │                                  │
          │  合并 Local + Global 回答为      │
          │  Context → LLM 生成最终回答      │
          │                                  │
          │  输出：综合最终回答              │
          └──────────────────────────────────┘
```

### 2.2 数据流

```
用户问题
    │
    ▼
┌──────────┐     ┌────────────────────┐     ┌───────────────────────┐
│ 前端页面 │ ──► │ POST /api/subagent │ ──► │   主代理 (Flask SSE)  │
│ (浏览器) │     │ /parallel-query    │     │                       │
│          │ ◄── │                    │ ◄── │  SSE 事件流 (实时推送) │
└──────────┘     └────────────────────┘     └───────────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
           ThreadPoolExecutor            local_agent()               global_agent()
           (max_workers=2)               ├─ extract_keywords()       ├─ get_embedding()
                    │                    ├─ graphrag_local_search()  ├─ cosine_sim 匹配
                    │                    └─ call_llm()               ├─ Map: call_llm() ×K
                    │                                                ├─ Reduce: call_llm()
                    │                                                └─ 返回 final_answer
                    │                            │                            │
                    └────────────────────────────┼────────────────────────────┘
                                                 ▼
                                        主代理合并阶段
                                        ├─ 组装 merge_context
                                        ├─ call_llm(合并回答)
                                        └─ emit("merge_complete")
```

### 2.3 技术选型

| 组件 | 技术 | 选型理由 |
|------|------|----------|
| 后端框架 | Flask | 与现有 GraphRAG 系统一体化，无需额外服务 |
| 并发模型 | `concurrent.futures.ThreadPoolExecutor` | Python GIL 下 I/O 密集型任务（LLM API 调用）并行效率高 |
| 实时通信 | SSE (Server-Sent Events) | 单向服务器推送，比 WebSocket 轻量，无需额外协议升级 |
| 前端接收 | `fetch()` + `ReadableStream` | 支持 POST 请求体的 SSE 接收（标准 EventSource 仅支持 GET） |
| LLM 调用 | DashScope API (Qwen-plus) | 通过环境变量 `DASHSCOPE_API_KEY` 认证 |
| 向量嵌入 | DashScope TextEmbedding (text-embedding-v3) | 1024 维向量，用于社区摘要匹配 |

---

## 3. 执行流程

### 3.1 完整流程时序图

```
时间轴 ──────────────────────────────────────────────────────────►

主代理   ├── 派发任务 ├── (等待并行) ──────────────────├── 合并 ├── 完成
          │            │                                │         │
Local    │ ├── 运行 ├── 关键词 ├── Cypher ├── LLM ├── 完成
  子代理  │           │  提取    │  检索     │  生成 │
          │           │          │           │       │
Global   │ ├── 运行 ├── 向量化 ├── 社区匹配 ├── Map ├── Reduce ├── 完成
  子代理  │           │          │           │  (×5)  │
          │           │          │           │        │
          t0         t1         t2          t3       t4         t5         t6

时间节点说明：
  t0 = 主代理启动，派发两个子代理
  t1 = 子代理各自开始工作
  t2 = Local 关键词提取完成 / Global 问题向量化完成
  t3 = Local Cypher 检索完成 / Global 社区匹配完成
  t4 = Local LLM 回答完成 (Local 子代理结束)
  t5 = Global Map-Reduce 完成 (Global 子代理结束)
  t6 = 主代理合并 LLM 完成，整个流程结束

并行节省时间 = Local 子代理执行时间 (t4 - t1)
```

### 3.2 主代理流程

```
1. 接收用户问题 (POST /api/subagent/parallel-query)
2. 初始化 SSE 事件队列 + 完成标志
3. 启动 worker 线程：
   a. emit("agent_update", main, running, "派发并行查询任务")
   b. 创建 ThreadPoolExecutor(max_workers=2)
   c. 并行提交 local_agent() 和 global_agent()
   d. 等待两个子代理都完成 (fut.result())
   e. emit("agent_update", main, merging, "合并子代理结果")
   f. 组装 merge_context = Local回答 + Global回答
   g. call_llm(question, merge_context) → 生成最终回答
   h. emit("merge_complete", { answer, timings })
   i. emit("agent_update", main, completed)
4. SSE 生成器持续从队列读取事件并 yield 给前端
5. 所有事件推送完毕，发送 "done" 事件
```

### 3.3 Local Search 子代理流程

```
1. emit("agent_update", local, running, "关键词提取 + 实体邻域检索")
2. extract_keywords(question)
   → 匹配知识图谱中的实体名称 + 提取通用中文关键词
   → emit("agent_step", local, "关键词提取完成")
3. graphrag_local_search(keywords)
   → 构造 Cypher 查询 (CONTAINS 匹配)
   → Neo4j 检索实体 + 1跳出度/入度关系
   → 组装 Context 文本
   → emit("agent_step", local, "Cypher 1跳邻域检索完成")
4. call_llm(question, context)
   → 调用 Qwen-plus 生成回答
5. emit("agent_update", local, completed, elapsed, result)
```

### 3.4 Global Search 子代理流程

```
1. emit("agent_update", global, running, "社区摘要向量匹配 + Map-Reduce")
2. graphrag_global_search(question, top_k=5)
   a. get_embedding(question) → 问题向量化 (1024维)
   b. 遍历 181 个社区摘要向量，计算余弦相似度
   c. 按相似度降序排序，选取 Top-5 社区
   d. Map 阶段：对每个选中社区，用其摘要调用 LLM 生成中间回答
   e. Reduce 阶段：汇总 5 个中间回答，调用 LLM 生成全局回答
3. emit("agent_step", global, "Map-Reduce 完成")
4. emit("agent_update", global, completed, elapsed, result)
```

### 3.5 合并阶段流程

```
1. emit("agent_update", main, merging)
2. 组装 merge_context:
     "=== Local Search 子代理检索结果 ===\n{local_answer}\n\n
      === Global Search 子代理检索结果 ===\n{global_answer}"
3. call_llm(question, merge_context, system_prompt=合并提示词)
   → 合并提示词要求：去除重复信息，整合互补信息，引用实体名称
4. emit("merge_complete", {
     answer: 最终回答,
     local_answer, global_answer,
     local_elapsed, global_elapsed,
     parallel_elapsed, merge_elapsed, total_elapsed
   })
5. emit("agent_update", main, completed, total_elapsed)
```

---

## 4. 目录结构

```
graphrag/app/
├── app.py                          # Flask 主应用（含 Subagent SSE 端点）
├── data/
│   └── communities.json            # 社区摘要 + 向量数据（Global Search 依赖）
├── templates/
│   └── index.html                  # 原始 GraphRAG 页面（Local/Global 单选）
└── subagent/                       # ★ Subagent 并行查询系统
    ├── README.md                   # 本说明文档
    └── templates/
        └── subagent.html           # Subagent 并行查询可视化页面
```

### 文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `app.py` (Subagent 部分) | ~280 行 | SSE 后端端点，第 974-1250 行 |
| `subagent/templates/subagent.html` | ~530 行 | 前端页面（含 CSS + JS） |
| `subagent/README.md` | 本文件 | 架构设计与代码说明 |

---

## 5. 后端代码说明

### 5.1 模块导入

```python
# app.py 顶部新增导入
import time                                              # 计时
import queue                                             # 线程安全队列（SSE 事件传递）
import threading                                         # 线程管理
from concurrent.futures import ThreadPoolExecutor        # 并行执行子代理
from flask import Response, stream_with_context, send_from_directory  # SSE 响应
```

### 5.2 页面路由

```python
SUBAGENT_TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "subagent", "templates"
)

@app.route("/subagent")
def subagent_page():
    """Serve the subagent parallel query HTML page."""
    return send_from_directory(SUBAGENT_TEMPLATE_DIR, "subagent.html")
```

通过 `send_from_directory` 直接从 `subagent/templates/` 目录返回 HTML 文件，无需注册 Flask 蓝图或修改全局模板路径。

### 5.3 时间戳工具函数

```python
def _ts_dict():
    """Return timestamp dict with epoch and human-readable string."""
    t = time.time()
    return {
        "epoch": round(t, 3),
        "str": time.strftime("%H:%M:%S", time.localtime(t))
        + f".{int(t * 1000) % 1000:03d}",
    }
```

每个 SSE 事件都附带精确到毫秒的时间戳，用于前端日志展示和计时计算。

### 5.4 SSE 端点核心结构

```python
@app.route("/api/subagent/parallel-query", methods=["POST"])
def api_subagent_parallel_query():
    data = request.get_json()
    question = data.get("question", "").strip()
    top_k = int(data.get("top_k", 5))

    def generate():
        event_queue = queue.Queue()      # 线程安全事件队列
        done_flag = threading.Event()    # 完成标志

        def emit(event_type, payload):
            # worker 线程向队列推送事件
            event_queue.put({"event": event_type, "data": payload, "ts": _ts_dict()})

        def worker():
            results = {}
            # ... local_agent() 和 global_agent() 定义 ...

            # 并行执行两个子代理
            with ThreadPoolExecutor(max_workers=2) as executor:
                fut_local = executor.submit(local_agent)
                fut_global = executor.submit(global_agent)
                fut_local.result()    # 阻塞等待 Local 完成
                fut_global.result()   # 阻塞等待 Global 完成

            # 合并阶段
            # ... call_llm 合并 ...
            done_flag.set()

        # 启动 worker 线程
        threading.Thread(target=worker, daemon=True).start()

        # SSE 生成器：持续从队列读取并 yield
        while not done_flag.is_set() or not event_queue.empty():
            try:
                evt = event_queue.get(timeout=0.5)
                yield f"data: {_json.dumps(evt, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"    # SSE 心跳

        yield f"data: {_json.dumps({'event': 'done', ...})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

### 5.5 关键设计：双线程模型

系统使用 **双线程模型** 解决 SSE 流式输出与后台并行计算的协调问题：

```
┌─ Flask 请求线程 ──────────────────────────────────────────┐
│                                                          │
│  generate() 生成器                                       │
│  ├── 启动 worker 线程 (daemon=True)                      │
│  ├── while 循环：                                        │
│  │   ├── 从 event_queue 读取事件                         │
│  │   ├── yield "data: ...\n\n"  → 推送给前端             │
│  │   └── 队列为空时 yield ": keepalive\n\n" (心跳)       │
│  └── worker 完成后发送 "done" 事件                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
          │ event_queue (线程安全)
          ▼
┌─ worker 线程 ────────────────────────────────────────────┐
│                                                          │
│  ├── emit("main running")                                │
│  ├── ThreadPoolExecutor 并行:                            │
│  │   ├── local_agent 线程                                │
│  │   │   ├── emit("local running")                       │
│  │   │   ├── extract_keywords() → emit("step")           │
│  │   │   ├── graphrag_local_search() → emit("step")      │
│  │   │   ├── call_llm()                                  │
│  │   │   └── emit("local completed")                     │
│  │   └── global_agent 线程                               │
│  │       ├── emit("global running")                      │
│  │       ├── graphrag_global_search()                    │
│  │       │   ├── get_embedding()                         │
│  │       │   ├── cosine_sim() × 181                      │
│  │       │   ├── Map: call_llm() × 5                     │
│  │       │   └── Reduce: call_llm()                      │
│  │       └── emit("global completed")                    │
│  ├── emit("main merging")                                │
│  ├── call_llm(merge_context)  ← 合并 LLM 调用            │
│  ├── emit("merge_complete")                              │
│  ├── emit("main completed")                              │
│  └── done_flag.set()  → 通知生成器结束                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 5.6 合并逻辑

```python
# 合并阶段代码
parallel_elapsed = round(time.time() - main_start, 2)
local_elapsed = results.get("local", {}).get("elapsed", 0)
global_elapsed = results.get("global", {}).get("elapsed", 0)

merge_start = time.time()

if "local" in results and "global" in results:
    merge_context = (
        "=== Local Search 子代理检索结果 ===\n"
        f"{results['local']['answer']}\n\n"
        "=== Global Search 子代理检索结果 ===\n"
        f"{results['global']['answer']}"
    )
    final_answer = call_llm(
        question,
        merge_context,
        system_prompt=(
            "你是一个节能降碳政策知识助手。以下是两个子代理"
            "（Local Search 基于实体邻域检索、Global Search 基于社区摘要向量检索）"
            "分别基于知识图谱检索得到的回答。"
            "请综合两个子代理的回答，去除重复信息，整合互补信息，"
            "生成一个完整、准确、有逻辑的最终回答。"
            "回答时请引用具体的实体名称和关系。"
        ),
    )
elif "local" in results:
    # 降级：仅 Local 有结果
    final_answer = results["local"]["answer"]
elif "global" in results:
    # 降级：仅 Global 有结果
    final_answer = results["global"]["answer"]
else:
    final_answer = "两个子代理均未返回有效结果。"

merge_elapsed = round(time.time() - merge_start, 2)
total_elapsed = round(time.time() - main_start, 2)
```

### 5.7 计时体系

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `local_elapsed` | `time.time() - local_start` | Local 子代理从启动到完成的耗时 |
| `global_elapsed` | `time.time() - global_start` | Global 子代理从启动到完成的耗时 |
| `parallel_elapsed` | `time.time() - main_start`（合并前） | 并行阶段耗时 ≈ max(local, global) |
| `merge_elapsed` | `time.time() - merge_start` | 合并阶段 LLM 调用耗时 |
| `total_elapsed` | `time.time() - main_start`（合并后） | 整个流程总耗时 = parallel + merge |

---

## 6. 前端代码说明

### 6.1 页面结构

```html
<div class="container">
    <!-- 1. 问题输入区 -->
    <div class="query-card">
        <textarea id="question"></textarea>
        <button onclick="runParallelQuery()">🚀 并行查询</button>
    </div>

    <!-- 2. Agent 树形状态 -->
    <div class="tree-card">
        <div class="agent-tree">
            <!-- 主代理节点 -->
            <div class="agent-node main-agent" id="node-main">
                <span class="agent-icon">🤖</span>
                <span class="agent-name">主代理 (Main Agent)</span>
                <span class="agent-step" id="step-main"></span>
                <span class="agent-status status-pending" id="status-main">等待中</span>
                <span class="agent-time" id="time-main"></span>
            </div>
            <!-- 子代理节点（带树形连接线） -->
            <div class="agent-children">
                <div class="agent-node" id="node-local">🔍 Local Search 子代理</div>
                <div class="agent-node" id="node-global">🌍 Global Search 子代理</div>
            </div>
        </div>
        <!-- 计时芯片 -->
        <div class="timing-bar">
            <div class="timing-chip local">🔍 Local: <span id="time-local-chip"></span></div>
            <div class="timing-chip global">🌍 Global: <span id="time-global-chip"></span></div>
            <div class="timing-chip">🔀 Merge: <span id="time-merge-chip"></span></div>
            <div class="timing-chip total">⏱ 总耗时: <span id="time-total-chip"></span></div>
        </div>
    </div>

    <!-- 3. 子代理结果（左右分栏） -->
    <div class="results-grid">
        <div class="result-card local">🔍 Local Search 子代理结果</div>
        <div class="result-card global">🌍 Global Search 子代理结果</div>
    </div>

    <!-- 4. 合并结果 -->
    <div class="merged-card">🔀 主代理合并结果 (LLM 综合回答)</div>

    <!-- 5. 执行日志 -->
    <div class="log-card">📋 执行日志</div>
</div>
```

### 6.2 树形结构 CSS 实现

```css
/* 子代理容器：左侧虚线作为树干 */
.agent-children {
    margin-left: 28px;
    padding-left: 20px;
    border-left: 2px dashed #b0bec5;
}

/* 每个子代理节点：左侧横线作为树枝 */
.agent-children .agent-node::before {
    content: '';
    position: absolute;
    left: -22px;
    top: 50%;
    width: 20px;
    height: 2px;
    background: #b0bec5;
}
```

### 6.3 状态徽章样式

```css
.status-pending   { background: #eceff1; color: #78909c; }  /* 灰色 - 等待 */
.status-running   { background: #e3f2fd; color: #1565c0;    /* 蓝色 - 运行中 */
                    animation: pulse 1.5s infinite; }        /* 脉冲动画 */
.status-merging   { background: #fff3e0; color: #e65100;    /* 橙色 - 合并中 */
                    animation: pulse 1.5s infinite; }
.status-completed { background: #e8f5e9; color: #2e7d32; }  /* 绿色 - 完成 */
.status-error     { background: #ffebee; color: #c62828; }  /* 红色 - 错误 */

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }   /* 闪烁效果表示活跃状态 */
}
```

### 6.4 SSE 接收逻辑（fetch + ReadableStream）

由于标准 `EventSource` API 仅支持 GET 请求，而本系统需要 POST 请求体发送问题文本，因此采用 `fetch()` + `ReadableStream` 手动解析 SSE 数据流：

```javascript
async function runParallelQuery() {
    const response = await fetch('/api/subagent/parallel-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question, top_k: 5 }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE 事件以 \n\n 分隔
        const parts = buffer.split('\n\n');
        buffer = parts.pop();  // 保留可能不完整的最后一段

        for (const part of parts) {
            const lines = part.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const evt = JSON.parse(line.slice(6));
                    handleEvent(evt);  // 分发事件到对应处理函数
                }
                // 以 ": " 开头的行是 SSE 注释（keepalive），忽略
            }
        }
    }
}
```

### 6.5 事件处理函数

```javascript
function handleEvent(evt) {
    const { event, data, ts } = evt;

    switch (event) {
        case 'agent_update':
            // 更新树形结构中的 agent 状态
            handleAgentUpdate(data, ts);
            break;

        case 'agent_step':
            // 更新 agent 的当前步骤描述
            handleAgentStep(data, ts);
            break;

        case 'merge_complete':
            // 展示合并结果 + 计时芯片
            handleMergeComplete(data, ts);
            break;

        case 'done':
            // 流程结束
            addLog(ts, 'DONE', '查询流程结束');
            break;
    }
}

function handleAgentUpdate(data, tsStr) {
    const { agent_id, status, step, elapsed, result, error } = data;

    // 1. 更新状态徽章
    updateAgent(agent_id, status, step, elapsed);

    // 2. 如果子代理完成且有结果，填充结果卡片
    if (status === 'completed' && result && (agent_id === 'local' || agent_id === 'global')) {
        document.getElementById(`result-${agent_id}`).textContent = result;
    }

    // 3. 如果出错，显示错误信息
    if (status === 'error') {
        document.getElementById(`result-${agent_id}`).innerHTML = `❌ ${error}`;
    }

    // 4. 记录日志
    addLog(tsStr, agent_name, `${status} ${step}`);
}
```

### 6.6 前端状态映射

```javascript
const STATUS_MAP = {
    'pending':   { text: '等待中', cls: 'status-pending' },
    'running':   { text: '运行中', cls: 'status-running' },
    'merging':   { text: '合并中', cls: 'status-merging' },
    'completed': { text: '已完成', cls: 'status-completed' },
    'error':     { text: '错误',   cls: 'status-error' },
};
```

---

## 7. SSE 事件协议

### 7.1 事件格式

每个 SSE 事件为一个 JSON 对象，通过 `data:` 前缀发送：

```
data: {"event": "agent_update", "data": {...}, "ts": {"epoch": 1786346517.75, "str": "15:21:57.749"}}

```

### 7.2 事件类型

| 事件类型 | 触发时机 | data 字段 |
|----------|----------|-----------|
| `agent_update` | Agent 状态变更 | `agent_id`, `agent_name`, `status`, `step`, `elapsed`, `result`, `error` |
| `agent_step` | Agent 内部步骤完成 | `agent_id`, `step`, `detail` |
| `merge_complete` | 主代理合并完成 | `answer`, `local_answer`, `global_answer`, `local_elapsed`, `global_elapsed`, `parallel_elapsed`, `merge_elapsed`, `total_elapsed` |
| `done` | 整个流程结束 | `{}` |

### 7.3 事件时序示例

```
[15:24:13.145] agent_update  main    → running   "派发并行查询任务给子代理"
[15:24:13.146] agent_update  local   → running   "关键词提取 + 实体邻域检索"
[15:24:13.149] agent_update  global  → running   "社区摘要向量匹配 + Map-Reduce"
[15:24:13.848] agent_step    local   "关键词提取完成" {matched: 0, general: 17}
[15:24:13.855] agent_step    local   "Cypher 1跳邻域检索完成" {entity_count: 27}
[15:24:19.360] agent_update  local   → completed (6.94s)  result="..."
[15:24:29.175] agent_step    global  "Map-Reduce完成" {total_communities: 181, top_k: 5}
[15:24:29.175] agent_update  global  → completed (16.04s) result="..."
[15:24:29.176] agent_update  main    → merging   "合并子代理结果 + 调用 LLM 生成最终回答"
[15:24:38.087] merge_complete  answer="..." local=6.94s global=16.04s merge=9.73s total=25.77s
[15:24:38.087] agent_update  main    → completed (25.77s)
[15:24:38.088] done
```

### 7.4 SSE 心跳机制

当 worker 线程正在执行耗时操作（如 LLM 调用）而事件队列为空时，SSE 生成器会发送心跳注释行防止连接超时：

```
: keepalive

```

前端解析时会自动忽略以 `:` 开头的 SSE 注释行。

---

## 8. 关键技术点

### 8.1 线程安全的 SSE 事件传递

**问题**：Flask 的 SSE 响应是通过 Python 生成器函数实现的，生成器运行在 Flask 请求线程中。而后台并行计算运行在 `ThreadPoolExecutor` 的子线程中。子线程不能直接 `yield` 数据到 Flask 的生成器。

**解决方案**：使用 `queue.Queue` 作为线程间通信桥梁：

```python
# 子线程通过 emit() 向队列推送事件
def emit(event_type, payload):
    event_queue.put({"event": event_type, "data": payload, "ts": _ts_dict()})

# Flask 生成器从队列读取事件并 yield
while not done_flag.is_set() or not event_queue.empty():
    try:
        evt = event_queue.get(timeout=0.5)
        yield f"data: {_json.dumps(evt, ensure_ascii=False)}\n\n"
    except queue.Empty:
        yield ": keepalive\n\n"
```

`queue.Queue` 是 Python 标准库中的线程安全队列，`put()` 和 `get()` 操作内部已加锁，可安全地在多线程环境下使用。

### 8.2 ThreadPoolExecutor 并行执行

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    fut_local = executor.submit(local_agent)
    fut_global = executor.submit(global_agent)
    fut_local.result()    # 阻塞直到 Local 完成
    fut_global.result()   # 阻塞直到 Global 完成
```

- `max_workers=2`：恰好为两个子代理各分配一个工作线程
- `submit()`：立即返回 `Future` 对象，不阻塞主线程
- `result()`：阻塞调用线程直到对应任务完成
- 两个子代理在各自的工作线程中**真正并行**执行（由于 LLM 调用是 I/O 密集型，Python GIL 不构成瓶颈）
- `with` 语句确保所有线程在退出时被正确清理

### 8.3 容错与降级策略

```python
if "local" in results and "global" in results:
    # 两个子代理都成功 → 正常合并
    final_answer = call_llm(question, merge_context, ...)
elif "local" in results:
    # 仅 Local 成功 → 降级使用 Local 结果
    final_answer = results["local"]["answer"]
elif "global" in results:
    # 仅 Global 成功 → 降级使用 Global 结果
    final_answer = results["global"]["answer"]
else:
    # 两个都失败 → 返回错误信息
    final_answer = "两个子代理均未返回有效结果。"
```

每个子代理内部使用 `try/except` 包裹，异常不会中断另一个子代理的执行：

```python
def local_agent():
    try:
        # ... 正常流程 ...
    except Exception as e:
        emit("agent_update", {"agent_id": "local", "status": "error", "error": str(e)})
        # 不抛出异常，结果字典中不会有 "local" 键
```

### 8.4 SSE 心跳保活

LLM 调用可能耗时 5-15 秒，期间事件队列为空。如果 SSE 连接长时间无数据，可能被代理服务器或浏览器超时断开。通过发送 SSE 注释行（以 `:` 开头）作为心跳：

```python
except queue.Empty:
    yield ": keepalive\n\n"    # SSE 注释，浏览器忽略但连接保持
```

### 8.5 前端 POST + SSE 接收

标准 `EventSource` API 仅支持 GET 请求，无法发送 POST 请求体。本系统使用 `fetch()` + `ReadableStream` 替代：

```javascript
const response = await fetch('/api/subagent/parallel-query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: question }),
});

const reader = response.body.getReader();
// 逐块读取并解析 SSE 数据
```

这使得可以在 POST 请求体中发送用户问题，同时接收 SSE 流式响应。

---

## 9. 性能分析

### 9.1 并行 vs 顺序执行对比

以问题 "钢铁行业有哪些节能降碳目标？" 为例：

| 模式 | Local 耗时 | Global 耗时 | Merge 耗时 | 总耗时 |
|------|-----------|------------|-----------|--------|
| 顺序执行 | 6.94s | 16.04s | 9.73s | **32.71s** |
| 并行执行 | 6.94s | 16.04s | 9.73s | **25.77s** |
| **节省** | - | - | - | **6.94s (21%)** |

并行执行的总耗时 = max(Local, Global) + Merge = 16.04 + 9.73 ≈ 25.77s
顺序执行的总耗时 = Local + Global + Merge = 6.94 + 16.04 + 9.73 = 32.71s
节省时间 = Local 子代理的执行时间 = 6.94s

### 9.2 各阶段耗时分解

```
                    t0 ──────────── t1 ──────────── t2 ──── t3
                    │               │               │       │
Local 子代理:       ├── 关键词提取(0.2s) ├── Cypher(0.1s) ├── LLM(6.6s) ──┤
                    │               │               │       │
Global 子代理:      ├── 向量化(0.3s) ├──────── Map-Reduce (15.7s) ────────┤
                    │               │               │       │
Merge:              │               │               │       ├── LLM(9.7s) ──┤
                    t0              t1              t2      t3              t4

时间节点:
  t0 = 0.00s   主代理派发
  t1 = 0.30s   Local 关键词+Cypher完成 / Global 向量化完成
  t2 = 6.94s   Local LLM 完成 (Local 子代理结束)
  t3 = 16.04s  Global Map-Reduce 完成 (并行阶段结束)
  t4 = 25.77s  Merge LLM 完成 (流程结束)
```

### 9.3 瓶颈分析

- **主要瓶颈**：Global Search 的 Map-Reduce 阶段需要多次 LLM 调用（1 次 Reduce + 5 次 Map = 6 次），是并行阶段的耗时主导因素
- **优化方向**：
  - 减少 `top_k` 参数（从 5 降到 3）可减少 Map 阶段 LLM 调用次数
  - 使用更快的 LLM 模型（如 qwen-turbo）用于 Map 阶段的中间回答
  - Map 阶段的多个 LLM 调用本身也可以并行化

---

## 10. API 接口文档

### 10.1 页面接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/subagent` | 返回 Subagent 并行查询 HTML 页面 |

### 10.2 查询接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/subagent/parallel-query` | SSE 流式返回并行查询过程 |

#### 请求参数

```json
{
    "question": "钢铁行业有哪些节能降碳目标？",
    "top_k": 5
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `question` | string | 是 | - | 用户问题 |
| `top_k` | int | 否 | 5 | Global Search 选取的社区数量 |

#### 响应格式

`Content-Type: text/event-stream`

响应为 SSE 事件流，每个事件格式为：

```
data: {"event": "<事件类型>", "data": {...}, "ts": {"epoch": 1234567890.123, "str": "HH:MM:SS.mmm"}}

```

#### 事件类型详情

**1. agent_update — Agent 状态变更**

```json
{
    "event": "agent_update",
    "data": {
        "agent_id": "local",           // "main" | "local" | "global"
        "agent_name": "Local Search 子代理",
        "status": "completed",          // "running" | "merging" | "completed" | "error"
        "step": "关键词提取 + 实体邻域检索",
        "elapsed": 6.94,               // 耗时（秒），仅 completed 时存在
        "result": "根据知识图谱..."     // 回答文本，仅子代理 completed 时存在
    },
    "ts": {"epoch": 1786346517.75, "str": "15:21:57.749"}
}
```

**2. agent_step — Agent 内部步骤**

```json
{
    "event": "agent_step",
    "data": {
        "agent_id": "local",
        "step": "Cypher 1跳邻域检索完成",
        "detail": {"entity_count": 27}
    },
    "ts": {"epoch": 1786346518.032, "str": "15:21:58.031"}
}
```

**3. merge_complete — 合并完成**

```json
{
    "event": "merge_complete",
    "data": {
        "answer": "根据知识图谱中整合的...",
        "local_answer": "根据提供的知识图谱...",
        "global_answer": "根据多源知识图谱...",
        "local_elapsed": 6.94,
        "global_elapsed": 16.04,
        "parallel_elapsed": 16.04,
        "merge_elapsed": 9.73,
        "total_elapsed": 25.77
    },
    "ts": {"epoch": 1786346528.087, "str": "15:22:08.070"}
}
```

**4. done — 流程结束**

```json
{
    "event": "done",
    "data": {},
    "ts": {"epoch": 1786346528.088, "str": "15:22:08.088"}
}
```

### 10.3 依赖接口

Subagent 系统复用了以下已有接口和函数：

| 函数/接口 | 位置 | 用途 |
|-----------|------|------|
| `extract_keywords(question)` | app.py | Local Search 关键词提取 |
| `graphrag_local_search(keywords)` | app.py | Local Search 实体邻域检索 |
| `graphrag_global_search(question, top_k)` | app.py | Global Search 社区向量匹配 + Map-Reduce |
| `call_llm(question, context)` | app.py | 调用 DashScope Qwen-plus 生成回答 |
| `load_communities()` | app.py | 加载社区摘要数据 |
| `/api/graphrag/build-communities` | app.py | 构建社区摘要（Global Search 前置依赖） |

### 10.4 环境变量

| 变量名 | 用途 | 必填 |
|--------|------|------|
| `DASHSCOPE_API_KEY` | DashScope API 认证（LLM + 向量嵌入） | 是 |
| `NEO4J_URI` | Neo4j 连接地址（默认 `bolt://localhost:7687`） | 否 |
| `NEO4J_USER` | Neo4j 用户名（默认 `neo4j`） | 否 |
| `NEO4J_PASSWORD` | Neo4j 密码（默认 `aaa111...`） | 否 |

### 10.5 前置条件

1. Neo4j 数据库已运行并导入知识图谱数据
2. `DASHSCOPE_API_KEY` 环境变量已设置
3. 社区摘要已构建（访问 `/api/graphrag/build-communities` 或在主页 GraphRAG Tab 中点击"构建社区摘要"）
4. Flask 服务已启动（`python app.py`）

### 10.6 访问地址

- Subagent 并行查询页面：`http://127.0.0.1:5000/subagent`
- 原始 GraphRAG 页面：`http://127.0.0.1:5000/`
