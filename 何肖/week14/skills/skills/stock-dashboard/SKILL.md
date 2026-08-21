---
name: "stock-dashboard"
description: "暗黑科技风格的股票看板 Skill：输入公司名+日期，自动拉取30分钟K线，保存JSON并生成独立HTML页面。Invoke when user asks for stock info/query/dashboard or mentions 股票/K线/行情/看板."
---

# Stock Dashboard · 股票信息看板（新架构）

按 **`json_data / html_data / scripts`** 三层结构组织：

```
stock-dashboard/
├── SKILL.md                    # 本文件（Skill 介绍）
├── json_data/                  # 存放查询到的公司信息（JSON）
│   ├── 平安银行_2026-07-28.json
│   └── 贵州茅台_2026-07-24.json
├── html_data/                  # 存放通过 JSON 生成的独立 HTML 页面
│   ├── 平安银行_2026-07-28.html
│   └── 贵州茅台_2026-07-24.html
├── scripts/                    # 存放 Python 脚本
│   ├── fetch_stock.py          # 数据获取 + JSON/HTML 生成
│   ├── app.py                  # FastAPI 前端服务（读取 html_data/）
│   ├── dashboard_template.html # 暗黑科技风格模板
│   └── requirements.txt
└── static/                     # 共享静态资源
    └── echarts.min.js
```

---

## 功能

- 输入「公司名 + 日期」，使用 akshare 拉取当日每 30 分钟 K 线；
- 计算买入/卖出成交量，按 **看多/中性/看空** 三色规则判定；
- 原始数据保存为 `json_data/{公司}_{日期}.json`；
- 基于模板生成独立 HTML 页面到 `html_data/{公司}_{日期}.html`；
- HTML 页面既可通过 FastAPI 服务访问，也可直接双击打开（已使用相对路径 `../static/`）。

---

## 调用时机（When to Invoke）

当用户提出以下任一请求时，应调用本 Skill：
- 「查询 XXX 公司 YYYY-MM-DD 的股票信息」
- 「帮我做一个股票看板 / K 线图」
- 「获取某股票某天的行情并保存为 JSON」
- 「画一个暗黑科技风格的证券看板」

---

## 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| 公司名 | A 股公司名称，支持模糊匹配 | `平安银行`、`贵州茅台` |
| 日期 | 交易日期，格式 `YYYY-MM-DD` | `2026-07-28` |

---

## 判定规则（三色柱图）

- **看多（红）**：买入成交量 / 总成交量 ≥ **2/3 (66.67%)**
- **看空（绿）**：卖出成交量 / 总成交量 ≥ **2/3 (66.67%)**
- **中性（黄）**：买入 / 卖出 比值 ≈ **1.0 ~ 1.2**

> akshare 分钟接口无主动买盘/卖盘拆分，采用业界通用 K 线分类法近似：收盘>开盘计买方量、收盘<开盘计卖方量、平盘 5:5 拆分。

---

## 快速开始

```bash
cd stock-dashboard

# 1. 安装依赖
pip install -r scripts/requirements.txt

# 2. 获取数据并生成页面（联网拉取 → JSON + HTML）
python scripts/fetch_stock.py --company 平安银行 --date 2026-07-28

# 3. 直接双击打开（离线可用）
#    html_data/平安银行_2026-07-28.html

# 或启动 FastAPI 服务
cd scripts
uvicorn app:app --reload
# 浏览器访问：
#   列表页：   http://127.0.0.1:8000/
#   看板页：   http://127.0.0.1:8000/stock/平安银行/2026-07-28
#   原始 JSON：http://127.0.0.1:8000/api/data/平安银行/2026-07-28
```

### 附加选项

```bash
# 从已有 JSON 重新生成 HTML（不联网）
python scripts/fetch_stock.py --company 平安银行 --date 2026-07-28 --from-json

# 仅获取 JSON，跳过 HTML 生成
python scripts/fetch_stock.py --company 平安银行 --date 2026-07-28 --skip-html
```

---

## 工作流对比

| 步骤 | 旧架构 | 新架构 |
|------|--------|--------|
| 数据获取 | `fetch_stock.py` → `data/xxx.json` | `fetch_stock.py` → `json_data/xxx.json` |
| 页面生成 | 请求时动态注入模板 | **预生成** 独立 HTML 到 `html_data/` |
| 页面访问 | 依赖 FastAPI | 可直接双击 HTML，或通过 FastAPI 访问 |
| 静态资源 | `/static/`（需 FastAPI） | `../static/`（相对路径，离线可用） |

---

## 渐进式执行步骤（Agent Side）

1. **Step 1 · 环境**：在 Skill 根目录确认 `json_data/`、`html_data/`、`scripts/`、`static/` 存在；安装 `scripts/requirements.txt` 依赖。
2. **Step 2 · 数据获取**：运行 `python scripts/fetch_stock.py --company X --date Y`，JSON 自动写入 `json_data/`，HTML 自动写入 `html_data/`。
3. **Step 3 · 页面查看**：直接打开 `html_data/{公司}_{日期}.html` 验证，或启动 FastAPI 服务查看。
4. **Step 4 · 增量更新**：新增一次查询即生成一对 JSON+HTML；`--from-json` 可从已有 JSON 重新生成页面。
5. **Step 5 · 模板修改**：调整 `scripts/dashboard_template.html` 后，使用 `--from-json` 重新生成所有页面。

---

## 代码位置

- 数据获取：[scripts/fetch_stock.py](scripts/fetch_stock.py)
- 前端服务：[scripts/app.py](scripts/app.py)
- 看板模板：[scripts/dashboard_template.html](scripts/dashboard_template.html)
- 依赖清单：[scripts/requirements.txt](scripts/requirements.txt)
- 原始 JSON：`json_data/{公司}_{日期}.json`
- 生成页面：`html_data/{公司}_{日期}.html`

---

## 常见问题

**Q: HTML 页面能离线打开吗？**
A: 可以。`html_data/` 下的页面通过 `../static/echarts.min.js` 引用共享资源，双击即可在浏览器中渲染 K 线图。

**Q: 如何只生成 HTML 不重新拉数据？**
A: 使用 `python scripts/fetch_stock.py --company X --date Y --from-json`。

**Q: 为什么判定是「偏多/偏空」而不是「看多/看空」？**
A: 这表示占比未达到 2/3 强阈值，但有倾向性。规则见 `scripts/fetch_stock.py::compute_buy_sell`。
