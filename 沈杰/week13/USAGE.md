# Harness Engineer 使用说明

## 项目简介

Harness Engineer 是一个基于 `deepseek-chat` 模型的 CLI 式问答 harness。它能够：

- 通过自然语言进行日常问答对话
- 自动发现并加载 `skills/` 目录下的 skill 元信息
- 当用户请求匹配某个 skill 时，渐进式加载 skill 内容，通过工具调用自动执行 skill 流程
- 将 skill 产物统一输出到 `output/` 目录
- 会话记忆：自动保存/恢复对话历史，支持多会话管理

## 目录结构

```
week13/
├── harness.py               # 主程序入口
├── USAGE.md                 # 本文档
├── output/                  # skill 产物输出目录
│   └── .gitkeep
├── sessions/                # 会话历史存储目录
│   └── .gitkeep
└── skills/
    └── flash-card/          # 示例 skill：单词闪卡生成
        ├── SKILL.md         # skill 说明文件（含 YAML frontmatter）
        ├── data/            # skill 数据存放目录
        │   ├── crazy.json
        │   ├── resilient.json
        │   └── thrill.json
        └── scripts/         # skill 脚本存放目录
            └── make_flashcard.py
```

## 环境要求

- Python 3.10+
- 依赖包：`openai`

## 安装依赖

```bash
pip install openai
```

## 配置 API Key

支持两种方式配置 API Key，优先级从高到低：

### 方式一：环境变量（推荐）

设置环境变量 `DEEPSEEK_API_KEY`：

**Windows (PowerShell)**
```powershell
$env:DEEPSEEK_API_KEY = "sk-你的API密钥"
```

**Windows (CMD)**
```cmd
set DEEPSEEK_API_KEY=sk-你的API密钥
```

**Linux / macOS**
```bash
export DEEPSEEK_API_KEY="sk-你的API密钥"
```

### 方式二：命令行参数

```bash
python harness.py -k sk-你的API密钥
```

## 启动方式

```bash
# 基本启动（自动加载最近的会话）
python harness.py

# 指定 API Key
python harness.py -k sk-你的API密钥

# 指定模型
python harness.py --model deepseek-chat

# 启动新会话（不加载历史）
python harness.py --no-memory

# 指定加载某个会话
python harness.py --session 20260730_143022_a1b2c3

# 指定 skills / output / sessions 目录
python harness.py --skills-dir ./my-skills --output-dir ./out --sessions-dir ./my-sessions
```

## 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--api-key` | `-k` | deepseek API Key | 环境变量 `DEEPSEEK_API_KEY` |
| `--base-url` | — | API 基础 URL | `https://api.deepseek.com` |
| `--model` | — | 模型名称 | `deepseek-chat` |
| `--skills-dir` | — | skills 目录路径 | `./skills` |
| `--output-dir` | — | 产物输出目录 | `./output` |
| `--sessions-dir` | — | 会话存储目录 | `./sessions` |
| `--session` | — | 指定要加载的会话 ID | 自动加载最近会话 |
| `--no-memory` | — | 禁用会话记忆（不加载、不保存） | `false` |

## 会话管理

### 自动保存与恢复

- 每次对话后，harness 会自动将对话历史保存到 `sessions/` 目录
- 下次启动时自动加载最近的会话，实现上下文延续
- 会话以 JSON 格式存储，包含 `id`、`name`、`created_at`、`updated_at`、`messages` 等字段

### 会话命令

在 CLI 中以 `/` 开头的命令会被识别和处理：

| 命令 | 说明 |
|------|------|
| `/help` | 显示所有命令帮助 |
| `/status` | 显示当前会话状态（ID、名称、轮数、存储状态） |
| `/sessions` | 列出所有已保存的会话 |
| `/load <id>` | 加载指定会话 |
| `/save [name]` | 保存当前会话（可选指定名称） |
| `/new` | 创建全新会话（当前会话会自动保存） |
| `/forget` | 清空当前上下文（历史文件保留） |
| `/delete <id>` | 删除指定会话 |

### 使用示例

```
>>> /sessions

已保存会话：
  20260730_143022_a1b2c3  20260730_143022_a1b2c3  (15 turns)  2026-07-30T14:45:10
  20260729_091500_d4e5f6  20260729_091500_d4e5f6  (8 turns)   2026-07-29T09:20:33

使用 /load <id> 加载会话。

>>> /save my-study-session
[session] 已保存会话 'my-study-session' (id=20260730_143022_a1b2c3)

>>> /status

当前会话:
  ID:    20260730_143022_a1b2c3
  名称:  my-study-session
  模型:  deepseek-chat
  轮数:  15
  内存:  42 messages
  路径:  D:\AI\workspace\week13\sessions
  状态:  已同步

>>> /new
[session] 已创建新会话: 20260730_150000_x1y2z3  (id=20260730_150000_x1y2z3)
```

### 历史裁剪

当对话轮数超过 20 轮时，harness 会自动裁剪较早的消息，保留系统提示和最近 20 轮对话，同时保持工具调用链的完整性，以避免超出模型上下文窗口。

## 基础使用

启动后会看到欢迎界面：

```
============================================================
  Harness Engineer  |  deepseek-chat CLI
  Skill: 1 个  |  Output: D:\AI\workspace\week13\output
  会话: 20260730_143022_a1b2c3 (id=20260730_143022_a1b2c3, 15 turns)
  输入 /help 查看命令  |  quit 退出
============================================================

>>>
```

### 日常问答

直接输入问题即可进行对话：

```
>>> 你好，介绍一下自己

你好！我是 Harness Engineer，能够进行日常问答，并在你需要时
自动调用 skills 完成特定任务。当前已加载 flash-card skill，
可以帮你生成英语单词的学习闪卡。

>>> Python 中装饰器怎么写？

Python 装饰器本质上是一个函数，它接受一个函数作为参数并返回
一个新的函数。常见的写法是...
```

### 使用 Skill

当你的请求与某个 skill 相关时，harness 会自动调用 skill，所有产物输出到 `output/` 目录：

```
>>> 给我做一张 resilient 的闪卡

[tool call] load_skill({"skill_name": "flash-card"})
[tool result]
[已加载 skill: flash-card，根目录: ...]

--- SKILL.md 完整内容 ---
# Flash Card 单词闪卡生成
...

[tool call] save_json_file({
  "path": "skills/flash-card/data/resilient.json",
  "content": "{\"word\":\"resilient\",...}"})

[tool result]
已保存 JSON 到 skills\flash-card\data\resilient.json

[tool call] run_script({
  "command": "python skills/flash-card/scripts/make_flashcard.py skills/flash-card/data/resilient.json -o output/resilient.html"})

[tool result]
STDOUT:
已生成: output\resilient.html

[tool call] open_file({"path": "output/resilient.html"})

已为你生成 resilient 的学习闪卡，文件位于 output/resilient.html，
已用浏览器打开预览。

>>>
```

### 退出程序

输入以下任意命令即可退出（退出前会话会自动保存）：

```
>>> quit
>>> exit
>>> q
>>> 退出
```

也可以使用 `Ctrl+C` 或 `Ctrl+D` 退出。

## Skill 系统

### Skill 发现机制

程序启动时会扫描 `skills/` 目录下的所有子目录，查找每个子目录中的 `SKILL.md` 文件。每个 `SKILL.md` 必须包含 YAML frontmatter 来声明 skill 的元信息：

```markdown
---
name: my-skill
description: >-
  简要描述 skill 的功能和触发场景。
  Use when the user asks to ...
---

# Skill 详细说明

...
```

### Skill 执行流程

1. **启动时**：harness 读取所有 `SKILL.md` 的 YAML frontmatter，提取 `name` 和 `description`，构建 skill 目录列表注入 system prompt
2. **请求匹配**：模型根据用户输入判断是否需要某个 skill
3. **渐进加载**：模型调用 `load_skill` 工具，harness 返回完整的 `SKILL.md` 内容
4. **执行流程**：模型按照 `SKILL.md` 中的执行说明，依次调用 `save_json_file`、`run_script`、`open_file` 等工具完成任务
5. **结果反馈**：模型用自然语言向用户汇报结果

### 内置工具

| 工具名 | 用途 |
|--------|------|
| `load_skill` | 加载指定 skill 的完整 SKILL.md 内容 |
| `save_json_file` | 将 JSON 文本写入指定文件路径 |
| `run_script` | 在本地执行命令行脚本，返回 stdout/stderr |
| `open_file` | 使用系统默认程序打开文件（HTML 用浏览器打开） |

## 扩展 Skill

添加新 skill 只需在 `skills/` 目录下创建子目录，并包含 `SKILL.md` 文件。

### 示例：创建一个翻译 skill

**目录结构**
```
skills/
└── translate/
    ├── SKILL.md
    └── scripts/
        └── translate.py
```

**SKILL.md 示例**
```markdown
---
name: translate
description: >-
  翻译中英文文本。
  Use when the user asks to translate / 翻译一段文字。
---

# Translate Skill

## 触发场景
- "帮我翻译这句话"
- "把这段英文翻译成中文"

## 执行流程
1. 从用户输入中提取待翻译的文本和目标语言
2. 调用 scripts/translate.py 进行翻译
3. 向用户展示翻译结果
```

创建完成后，重启 harness 即可在启动日志中看到新加载的 skill。

## 注意事项

- API Key 仅存储在内存中，不会写入任何文件
- 会话历史包含完整对话内容，请注意敏感信息
- `run_script` 工具的命令执行超时时间为 60 秒
- `save_json_file` 会自动创建目标目录
- 所有 skill 产物（HTML、JSON）统一输出到 `output/` 目录
- 如需使用其他模型，可通过 `--model` 参数指定，确保该模型支持 function calling
- 使用 `--no-memory` 可完全禁用会话记忆，适合敏感话题讨论

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| `未检测到 API Key` | 设置 `DEEPSEEK_API_KEY` 环境变量或使用 `-k` 参数 |
| `缺少依赖 openai` | 执行 `pip install openai` |
| `API 错误` | 检查 API Key 是否正确、网络是否通畅、余额是否充足 |
| 命令执行超时 | 检查脚本是否卡死，或联系开发者调整超时时间 |
| 会话加载失败 | 检查 `sessions/` 目录下的 JSON 文件是否损坏 |