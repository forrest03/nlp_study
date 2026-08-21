# NEURAL AGENT 办公助手 — 工程说明

## 项目概览

基于 Python 的智能办公助手，支持多轮对话、技能按需加载、记忆持久化和赛博朋克风格 Web 交互界面。采用阿里百炼 API（qwen-max）的 Function Calling 能力实现动态工具编排。

## 目录结构

```
week13/
├── index.html              # 赛博朋克风格聊天页面（会话列表/切换/新建/删除）
├── structure.md            # 本文件
├── memory/
│   └── memory.md           # 用户记忆持久化文件（LLM 语义提取写入）
├── skills/
│   ├── calculator/         # 计算器技能（script 执行器）
│   │   ├── skill.md
│   │   └── calculator.py
│   ├── time_query/         # 时间查询技能（llm 执行器，纯描述型）
│   │   └── skill.md
│   └── text_summary/       # 文本摘要技能（script 执行器）
│       ├── skill.md
│       └── text_summary.py
└── src/
    ├── server.py           # FastAPI Web 服务，SSE 流式对话 API
    ├── agent.py            # Agent 核心，LLM 编排与技能调度
    ├── skill.py            # 声明式技能加载器，按需加载与执行
    ├── memory.py           # 记忆管理器，LLM 语义提取写入 memory.md
    └── session_manager.py  # 会话管理与上下文清理策略
```

## 模块职责与交互

### src/server.py — FastAPI 服务

提供 REST API 和 SSE 流式对话接口。启动时自动预加载 skills，托管 `index.html` 作为首页。

**关键端点：**
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | SSE 流式对话，异步执行 agent.chat() |
| POST | `/session/create` | 创建新会话 |
| GET | `/sessions` | 列出所有会话 |
| GET | `/session/{id}/messages` | 获取会话消息历史 |
| DELETE | `/session/{id}` | 删除会话 |
| GET | `/skills` | 列出所有技能 |
| POST | `/skills/reload` | 热重载技能 |
| GET | `/memory` | 查看记忆内容 |
| POST | `/memory` | 手动更新记忆 |
| GET | `/health` | 健康检查 |

### src/agent.py — Agent 核心

接收用户输入，执行 Function Calling 循环，编排技能按需加载与执行。

**核心流程：**
1. 创建/恢复会话 → 加载历史消息并压缩
2. 构建 system prompt（技能目录 + 记忆内容）
3. FC 循环：默认仅暴露 `load_skill` 工具
   - LLM 调用 `load_skill` → 注入 `execute_skill` 工具
   - LLM 调用 `execute_skill` → 恢复为仅 `load_skill`
   - 执行完成后清理 `load_skill` 中间步骤
4. 保存消息到会话，更新记忆

### src/skill.py — 技能加载器

扫描 `skills/{name}/skill.md` 目录，解析 YAML frontmatter 声明式技能定义。

**技能定义格式（skill.md）：**
- `name` / `description` / `executor`（`script` 或 `llm`）
- `parameters` 参数定义（JSON Schema 格式）
- 内联或外联 Python 脚本

**关键函数：**
- `scan_skills()` / `reload()` — 扫描/热重载
- `get_skill_catalog()` — 返回轻量目录（仅名称和描述）
- `load_skill_detail()` — 返回完整技能定义（含参数）
- `execute_skill()` — 按 executor 类型执行
- `get_load_skill_schema()` / `get_execute_skill_schema()` — FC JSON Schema

### src/memory.py — 记忆管理器

将对话中用户的关键偏好、信息通过 LLM 语义提取，写入 `memory/memory.md`。

**关键方法：**
- `load_memory()` — 读取记忆文件
- `update_memory()` — 调用 LLM 提取最新对话中的关键信息，合并到现有记忆
- `save_memory()` — 直接写入

### src/session_manager.py — 会话管理

内存级多轮会话管理，线程安全。

**SessionManager：**
- 会话 CRUD（create / get / save / delete / list）
- 过期清理（默认 3600s）
- Token 估算

**ContextManager：**
- `smart_compact` 策略：移除 `load_skill` 的中间调用与结果，保留 `execute_skill` 的调用与结果，减少 token 消耗

### index.html — 前端页面

赛博朋克风格聊天界面，CSS 变量驱动主题（霓虹青、品红、黄绿），含扫描线动画、发光效果。

**功能：**
- 侧边栏会话列表（显示标题、轮次、消息数）
- 新建/切换/删除会话
- SSE 流式消息展示（LOAD / EXEC / ERROR 中间步骤 + Final Answer 高亮）
- Enter 发送 / Shift+Enter 换行

## 技能目录

| 技能 | 执行器 | 说明 |
|------|--------|------|
| calculator | script | 数学表达式计算（四则运算、幂、math 函数） |
| time_query | llm | 时间查询（纯描述型，由 LLM 直接回答） |
| text_summary | script | 文本摘要，支持 max_words 参数 |

## 启动方式

```bash
cd week13/src
python3 -m uvicorn server:app --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000` 打开聊天页面。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| ALIYUN_API_KEY | 阿里百炼 API 密钥 | 必填 |
| AGENT_MODEL | LLM 模型名 | qwen-max |
| SESSION_TIMEOUT | 会话超时时间（秒） | 3600 |
| SESSION_MAX_TOKENS | 上下文最大 token | 8192 |