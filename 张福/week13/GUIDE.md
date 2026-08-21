# Harness Memory Chat — 使用指南

## 启动服务

```bash
# 确保已设置 API Key
export DASHSCOPE_API_KEY=sk-xxxx

# 方式一：直接运行
python harness/serve.py

# 方式二：使用 uvicorn
uvicorn harness.serve:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问 http://localhost:8000

## 环境变量

| 变量 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `DASHSCOPE_API_KEY` | 是 | DashScope 通义千问 API Key | - |
| `HARNESS_MODEL` | 否 | LLM 模型名 | `qwen-plus` |
| `KMP_DUPLICATE_LIB_OK` | 否 | 避免 faiss 库冲突 | `TRUE`（代码自动设置） |

## 页面功能

### 1. 对话（Chat）
- 在底部输入框提问，按 Enter 发送
- Shift+Enter 换行
- 支持流式（streaming）回复，实时显示
- 侧边栏列出所有历史会话，点击可切换查看历史对话

### 2. 会话管理
- **+ 新会话**：创建全新会话，历史会话保留在侧边栏
- 点击侧边栏的会话可查看该会话的完整对话历史

### 3. 记忆管理
切换到「记忆管理」标签页，可查看和编辑以下记忆类型：

| 类型 | 说明 |
|------|------|
| `user_profile` | 用户特征/偏好记忆 |
| `long_term_raw` | 长期记忆原始缓冲区（对话积累的原始记录） |
| `compressed` | LLM 压缩后的长期记忆摘要 |
| `daily` | 按自然日记录的事件与对话摘要 |
| `memory_meta` | RAG 检索使用的结构化元数据（JSON 格式） |

- 选择类型后自动加载内容到文本编辑器
- 修改后点击「保存」即可写入
- 点击「刷新」重新读取

### 4. 工具按钮
- **压缩**：调用 LLM 将 `long_term_raw` 压缩为结构化记忆，更新检索索引
- **重建索引**：从 `memory_meta.json` 重新生成 FAISS 向量索引

## 工具调用模式 (Function Calling)

LLM 可通过 function call 自动创建目录、读写文件、执行 shell 命令。

### 安全机制

| 保护层 | 说明 |
|--------|------|
| ⛔ 危险命令自动拦截 | `rm -rf /`、`dd`、`mkfs`、`sudo`、`shutdown`、`reboot`、`curl \| bash` 等关键系统操作直接拦截，无需确认 |
| ⛔ 系统路径保护 | 写入 `/etc`、`/bin`、`/usr`、`/dev` 等系统目录自动拒绝 |
| ❓ 手动确认弹窗 | 所有未被拦截的 `execute_command` 在执行前弹出确认框，由用户手动点击「确认执行」或「拒绝」 |
| ⏱ 超时机制 | 确认弹窗 120 秒无操作自动拒绝 |

### 可用工具

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `create_directory` | 创建目录（自动创建父目录） | `path` (必填) |
| `write_file` | 写入文件内容 | `path` (必填), `content` (必填) |
| `read_file` | 读取文件内容 | `path` (必填) |
| `execute_command` | 执行 shell 命令 | `command` (必填), `timeout` (可选, 默认 30s) |

### Web 页面启用

在对话输入框下方勾选 **「工具模式」**，然后正常提问。
- LLM 执行工具时会实时显示调用日志（绿色消息气泡）
- 最终回答显示在对话气泡中

### CLI 启用

```bash
# 交互式对话（启用工具调用）
python harness/agent.py --allow-tools

# 单次提问（启用工具调用）
python harness/agent.py --allow-tools --question "创建一个 Python 项目，包含 main.py 和 README.md"

# 交互中动态切换
# 输入 tools 命令可开启/关闭工具模式
```

CLI 模式下，`execute_command` 执行前会弹出确认提示。

### 使用示例

用户提问：*"创建一个 Python 项目，包含 main.py 和 README.md"*

LLM 将依次调用：
1. `create_directory({"path": "my_project"})`
2. `write_file({"path": "my_project/main.py", "content": "..."})`
3. `write_file({"path": "my_project/README.md", "content": "..."})`
4. 最终回复创建结果

用户提问：*"运行 pip list 查看已安装的包"*

LLM 调用：`execute_command({"command": "pip list"})` 并返回输出结果。

## 命令行模式

```bash
# 交互式对话
python harness/agent.py

# 交互式对话（启用工具调用）
python harness/agent.py --allow-tools

# 单次提问
python harness/agent.py --question "你好"

# 单次提问（启用工具）
python harness/agent.py --allow-tools --question "创建 Python 项目"

# 指定会话继续
python harness/agent.py --session_id abc123 --question "继续上次话题"

# 手动压缩长期记忆
python harness/agent.py --compress

# 重建向量索引
python harness/agent.py --init-index
```

### 技能工具：PPT 拆解为 HTML

当用户要求将 PPT 课件拆解为逐页图片和 HTML 浏览页面时，LLM 会自动调用 `ppt_to_html` 工具：

```bash
# 在对话中开启工具模式后提问：
# "帮我把 ~/文档/learning_ai_ppt 下的 PPT 拆解成 HTML"
# LLM 将自动调用 ppt_to_html 工具完成转换
```

该工具功能：
- 读取指定目录下所有 `.pptx` / `.ppt` 文件
- 每页渲染为 1920×1080 PNG 图片
- 生成左右结构 HTML 页面，支持折叠/展开、键盘 ↑↓ 切换
- 输出到 `~/ppts/` 目录

## 技能系统

Harness 支持通过 `skills/` 目录扩展 LLM 可调用的工具。

### 内置技能

| 技能 | 触发场景 | 说明 |
|------|----------|------|
| `ppt_to_html` | 用户要求拆解 PPT 为 HTML | 将 PPT 逐页转为 PNG 并生成 HTML 浏览页面 |

### 开发新技能

在 `skills/` 下创建子目录，包含 `__init__.py`，导出 `skill` 实例：

```python
from skills.base import Skill

class MySkill(Skill):
    name = "my_tool"
    description = "技能描述（注入系统提示词）"
    tools = [...]  # tool definitions

    def execute(self, tool_name, arguments):
        # 执行逻辑
        return {"success": True, ...}

skill = MySkill()
```

技能工具会自动合并到全局工具列表中，LLM 可在对话中按需调用。

## 项目结构

```
├── harness/              # Harness 服务
│   ├── agent.py          # CLI 入口
│   ├── chat_harness.py   # 对话逻辑（含工具调用循环）
│   ├── llm_client.py     # LLM API 客户端（含 function calling）
│   ├── serve.py          # Web 服务 (FastAPI)
│   ├── tool_executor.py  # 工具执行器（目录/文件/命令）
│   └── static/           # 前端静态文件
│       └── index.html    # 主页面
├── skills/               # 技能扩展（自动发现）
│   ├── base.py           # 技能基类
│   ├── ppt_to_html/      # PPT → HTML 技能
│   │   └── __init__.py
├── src/                  # 记忆系统核心
│   ├── memory_store.py   # 存储读写
│   ├── memory_retriever.py  # 检索（BM25 + 向量）
│   ├── memory_compressor.py # 压缩与索引重建
│   └── paths.py          # 路径常量
├── memory/               # 记忆文件（Markdown）
│   ├── short_term/       # 短期会话
│   ├── long_term/        # 长期原始
│   ├── compressed/       # 压缩摘要
│   ├── daily/            # 按天日记
│   └── user_profile/     # 用户特征
├── databases/            # 检索数据库
│   ├── memory_meta.json
│   └── memory_faiss_index.bin
├── GUIDE.md              # 本指南
└── requirements.txt
```
