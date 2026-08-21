# ARCHITECTURE.md — 蛋糕商品采集 + 营销设计 Subagent

## 1. 项目定位

参照 `market_research_subagents` 的动态 Orchestrator-Workers 范式，落地垂直场景：

- **商品采集**：蛋糕类 SKU 的文字介绍、价格规格、**图片 URL**、来源链接
- **营销设计**：定位、人群、视觉方向、卖点文案、活动玩法

主 agent 自主决定是否 `dispatch_subagents`；多侧面任务并行，墙钟 ≈ max(子任务)，而非 sum。

## 2. 流水线

```
用户问题（蛋糕采集 / 营销设计）
   ↓
主 agent ReAct（web_search + dispatch_subagents）
   ├─ 单事实 → web_search → Final Answer
   └─ 多侧面 → dispatch_subagents("详情|竞品|营销")
                  ↓ ThreadPool 并行
         ┌─ sub: 商品图文详情 ─┐
         ├─ sub: 竞品与场景   ─┤
         └─ sub: 营销设计方案 ─┘
                  ↓
         主 agent 综合四段报告 → Final Answer
```

## 3. 相对参考项目的改动

| 点 | 改动 |
|----|------|
| 搜索 | **不用 Tavily**；`browser_search.py` 模拟浏览器 POST DuckDuckGo HTML，解析结果并可选打开详情页抽 `og:image` |
| Prompt | 主/子 agent 专为蛋糕采集 + 营销结构 |
| 汇总截断 | 子结果 800 字（图文更长） |
| LLM | **固定 qwen-plus**（`DASHSCOPE_API_KEY`） |
| 端口 | 默认 `8015`，避免与参考项目 8002 冲突 |

## 4. 目录

```
cake_marketing_subagents/
├── src/
│   ├── browser_search.py  # 模拟浏览器 Web 搜索（含图片）
│   ├── demo_catalog.py    # 搜索失败降级样例
│   ├── react_loop.py      # 通用 ReAct
│   ├── agents.py          # 主 agent + 并行派发
│   ├── serve.py           # FastAPI + SSE
│   ├── llm_client.py      # qwen-plus
│   └── eval_compare.py
├── static/
│   ├── index.html
│   └── viz/topology.js
├── outputs/
├── requirements.txt
├── ARCHITECTURE.md
└── USAGE_GUIDE.md
```

## 5. 范式对应

- 拓扑：动态 Orchestrator-Workers（运行时生长子节点）
- 并行收益：`wall_clock` vs `serial_sum`（见 `eval_compare.py`）
