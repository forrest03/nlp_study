# Week13 · Harness + Skills 作业

基于 FastAPI + DashScope Function Calling 的 Skill Agent：用自然语言识别意图，自动调用对应 Skill 工具，并生成可访问的 JSON / HTML 产物。

包含两个 Skill：

- **stock-dashboard**：输入公司名 + 日期，拉取 30 分钟 K 线，生成股票看板
- **flash-card**：输入英语单词，补全音标/释义/例句，生成单词闪卡

---

## 1. 项目结构

```text
week13/
└── harness_homework/
    ├── skills/
    │   ├── stock-dashboard/          # 股票看板 Skill
    │   │   ├── SKILL.md
    │   │   ├── json_data/            # 行情 JSON
    │   │   ├── html_data/            # 看板 HTML
    │   │   ├── scripts/              # fetch_stock.py 等
    │   │   └── static/echarts.min.js
    │   └── flash-card/               # 单词闪卡 Skill
    │       ├── SKILL.md
    │       ├── json_data/            # 单词 JSON
    │       ├── html_data/            # 闪卡 HTML
    │       └── scripts/make_flashcard.py
    ├── src/
    │   ├── main.py                   # FastAPI 入口
    │   ├── llm_client.py             # DashScope LLM 客户端
    │   ├── skill_loader.py           # 扫描 / 解析 SKILL.md
    │   ├── skill_executor.py         # 工具注册与执行
    │   ├── models.py
    │   ├── chat.html                 # 对话前端
    │   └── requirements.txt
    └── start.ps1                     # Windows 启动脚本
```

---

## 2. 环境准备

```bash
cd harness_homework/src
pip install -r requirements.txt
pip install akshare   # stock-dashboard 拉行情需要

export DASHSCOPE_API_KEY="sk-xxx"
```

> 股票 Skill 额外依赖见 `skills/stock-dashboard/scripts/requirements.txt`。

---

## 3. 启动服务

```bash
cd harness_homework/src
python -m uvicorn main:app --host 127.0.0.1 --port 9001
```

浏览器打开：

- 首页：http://127.0.0.1:9001/
- 对话页：http://127.0.0.1:9001/chat

Windows 也可在 `harness_homework/` 下运行 `.\start.ps1`（默认端口 9000）。

---

## 4. 使用示例

在对话页直接用自然语言提问：

| 意图 | 示例 |
|------|------|
| 股票看板 | `查询平安银行2026-07-28股票信息` |
| 单词闪卡 | `做一个crazy的闪卡` |

Agent 会自动：

1. 识别 Skill
2. 调用对应工具（Function Calling）
3. 生成 JSON / HTML
4. 在回复中给出可点击链接

生成文件可通过服务访问：

```text
# 股票看板
/files/skills/stock-dashboard/json_data/平安银行_2026-07-28.json
/files/skills/stock-dashboard/html_data/平安银行_2026-07-28.html

# 单词闪卡
/files/skills/flash-card/json_data/crazy.json
/files/skills/flash-card/html_data/crazy.html
```

---

## 5. 核心流程

```text
用户自然语言
    ↓
DashScope LLM（Function Calling）
    ↓
skill_executor 执行工具
    ├── stock-dashboard → fetch_stock
    └── flash-card → save_flashcard_data → generate_flashcard
    ↓
写入 skills/<name>/json_data/ 与 html_data/
    ↓
前端通过 /files/skills/... 访问产物
```

### stock-dashboard

1. 解析公司名 + 日期  
2. `fetch_stock.py` 用 akshare 拉 30 分钟 K 线  
3. 保存 JSON，并基于模板生成暗黑风格 HTML 看板  

### flash-card

1. LLM 补全单词信息（音标、词性、释义、3 条例句、近义词）  
2. `save_flashcard_data` 写入 `json_data/<word>.json`  
3. `generate_flashcard` 调用 `make_flashcard.py` 生成 `html_data/<word>.html`  

---

## 6. 主要 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/skills` | 列出已加载 Skill |
| GET | `/api/skills/{name}` | Skill 详情 |
| POST | `/api/chat` | 自然语言对话 + 工具调用 |
| GET | `/files/skills/{name}/{path}` | 访问 Skill 生成的静态文件 |
| GET | `/api/health` | 健康检查 |

---

## 7. 已验证产物

本次提交中已包含示例结果：

- `skills/stock-dashboard/html_data/平安银行_2026-07-28.html`
- `skills/flash-card/html_data/crazy.html`

可直接启动服务后通过 `/files/...` 打开查看。
