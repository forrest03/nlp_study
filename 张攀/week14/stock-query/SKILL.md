---
name: "stock-query"
description: "Queries a company's past-week stock prices (7 trading days) via free Tencent APIs, no API key needed. Invoke when user says '查询xx公司股票' / '查一下xx的股价' or asks for a company's recent stock prices."
triggers: 股票, 股价, 行情, stock
entry: script/query_stock.py
argtype: text
---

# 股票查询 (Stock Query)

输入公司名称（中文/英文均可），返回该公司最近 7 个交易日的收盘价及周涨跌情况。

## 何时调用

- 用户输入"查询xx公司股票"，如"查询腾讯股票"、"查询 Apple 股票"
- 用户询问某家公司的股价 / 近一周行情 / 最近股价走势
- 用户直接给股票代码查行情（如 AAPL、600519、0700）

## 资产清单

| 文件 | 用途 |
|---|---|
| `script/query_stock.py` | 唯一入口模块，导出 `run(query="") -> dict`：公司名称 → 股票代码 → 近 7 个交易日收盘价 |

> 接口规范：每个 skill 的 `script/` 下只导出一个 `run(**kwargs) -> dict`，由 harness 用 importlib 动态加载后调用 `module.run(**params)`。依赖 `requests`（项目 `requirements.txt` 已包含），数据源为腾讯免费接口（smartbox 搜索 + fqkline K 线），国内可访问，无需 API Key。

## 数据源说明

- 智能搜索（名称 → 代码）：`https://smartbox.gtimg.cn/s3/`，返回 `v_hint="..."` 形式文本，值内含 `\uXXXX` 转义需按 JSON 字符串解码
- 日 K 线：`https://web.ifzq.gtimg.cn/appstock/app/fqkline/get`，`param=代码,day,,,7,qfq`，每行 `[日期, 开盘, 收盘, 最高, 最低, 成交量]`
- 类型过滤：搜索结果第 5 段为 `GP`（股票）才采用，跳过指数（ZS）等
- 代码规则：A 股 `sh600519` / `sz000001`，港股 `hk00700`，美股 `us` + 带后缀代码（如 `usaapl.oq`）

## ReAct 流程

### Step 1 · 提取公司名称

- **Thought**：从用户输入中提取要查询的公司名称。
- **Action**：直接取"查询...股票"中的公司名；用户直接给代码（AAPL / 600519 / 0700）同样支持。
- **Observation**：得到目标公司名；若用户已在消息中说明，跳过询问。

### Step 2 · 执行查询

- **Thought**：查询已封装进 `query_stock.run(query=...)`，直接执行。
- **Action**：`RunCommand` 运行 harness：
  ```
  python harness.py "查询腾讯股票"
  ```
  harness 会自动全量加载本 SKILL.md，并用 importlib 调用 `script/query_stock.py` 的 `run(query="腾讯")`。
- **Observation**：返回 dict（公司名、代码、市场、币种、近 7 日收盘价）。

### Step 3 · 校验并呈现

- **Thought**：确认查询成功，整理结果给用户。
- **Action**：
  - 成功（含 `points` 列表）：展示公司名 / 代码 / 市场 / 币种，逐日列出日期与收盘价，并用首末收盘价计算周涨跌额与百分比（涨用 ▲，跌用 ▼，持平用 —）
  - 失败（含 `error`）：按错误提示引导——"未找到"则建议换股票代码；"行情失败"则提示稍后重试
- **Observation**：用户获得近一周行情。

### 可选 · 网页查看

若 Flask 服务在运行（`python app.py`，端口 5050），可提示用户访问 http://127.0.0.1:5050/stock 输入公司名查看折线图版本；也可用 `curl http://127.0.0.1:5050/api/stock?q=腾讯` 走本地 API。

## 示例输出

```json
{
  "name": "腾讯控股",
  "display": "00700",
  "exchange": "香港",
  "currency": "HKD",
  "points": [
    {"date": "2026-07-28", "price": 447.2}
  ]
}
```

## 测试用例

- `python harness.py "查询腾讯股票"` → hk00700，HKD
- `python harness.py "查询Apple的股价"` → AAPL，USD
- `python harness.py "查询不存在的公司xyz"` → 返回 `error` 提示
