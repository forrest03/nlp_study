# USAGE_GUIDE.md — 蛋糕采集与营销 Subagent

## 1. 环境

```bash
cd 徐斌/week15/cake_marketing_subagents
pip install -r requirements.txt

export DASHSCOPE_API_KEY="sk-xxx"   # 模型固定 qwen-plus
# 可选：export AGENT_MODEL=qwen-plus
```

搜索使用**模拟浏览器**访问 DuckDuckGo HTML（标准库 `urllib` + Chrome UA），**不依赖 Tavily**。若外网搜索失败，自动降级 `demo_catalog`。

## 2. CLI 跑一次

```bash
python src/agents.py
```

或：

```python
import sys; sys.path.insert(0, "src")
from agents import run_cake_research

r = run_cake_research(
    "采集生日蛋糕类商品详情（图片+文字介绍），并给出营销设计方案"
)
print(r["final_answer"])
print(r["parallel_stats"])
```

单独测搜索：

```bash
python src/browser_search.py
```

## 3. Web 可视化（SSE）

```bash
cd 徐斌/week15/cake_marketing_subagents
uvicorn src.serve:app --host 0.0.0.0 --port 8015
# 浏览器 http://localhost:8015
```

- `GET /health` → `{search: browser_ddg, llm, provider: qwen, model: qwen-plus}`
- `POST /query` → SSE：`start` → `main_step` → `dispatch` → `subagent_step` → `final` → `done`

## 4. 并行 vs 串行对比

```bash
python src/eval_compare.py --limit 1
```

结果写入 `outputs/eval_compare.json`。

## 5. 推荐提问

- 采集生日蛋糕类商品详情（图片URL + 文字介绍 + 价格规格），并基于竞品给出营销设计方案
- 芝士蛋糕热销商品采集（图文+价格），并输出下午茶场景营销设计
- 好利来黑森林蛋糕现在卖多少钱（单事实，通常不派发）
