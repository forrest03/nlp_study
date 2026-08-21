# Progressive Skills Harness (DeepSeek-powered)

一个接入 **DeepSeek 大模型** 的**对话型 agent** + 渐进式 skill 加载执行框架。

定位：一个能正常对话的 agent —— skills 作为「工具」通过 **function calling** 暴露给
LLM，由 LLM 在完整对话上下文中自然决定是否调用某个 skill。新增 skill 只需在 `skills/`
目录下放一个 `SKILL.md`，无需改任何 harness 代码。

**强依赖 LLM** —— agent loop 与 executor 都调用 DeepSeek，未配置 `DEEPSEEK_API_KEY`
时 CLI 直接报错退出，没有规则模式兜底。

LLM 介入两个核心环节：
1. **对话决策（function calling）** — LLM 在多轮对话中自主判断是否调用 skill、调哪个、
   传什么参数；skill 执行结果回传给 LLM，由它生成最终自然语言回复。
2. **执行阶段（数据生成）** — 通用执行器按 SKILL.md 声明的契约，可调 LLM 生成 skill
   所需数据（如让 LLM 写出单词 JSON、生成 SVG 内容）。

---

## 特性

- **对话型 agent**：基于 DeepSeek function calling，LLM 在完整对话上下文中决定是否
  调用 skill，纯闲聊与 skill 调用统一在一条 agent loop 中完成。
- **三阶段渐进加载**：启动只读 frontmatter，命中才读完整 markdown，执行前才枚举脚本目录。
- **声明式 skill 契约**：skill 在 `SKILL.md` frontmatter 声明 `params`/`entry`/`data_instructions`
  等字段，harness 通用执行，新增 skill 不改 harness 代码。
- **四种执行模式**：通用执行器按 frontmatter 字段组合自动分派（script+data_file /
  script+args / script+stdin / 纯生成模式）。
- **持久化对话记忆**：JSON 文件存储多轮对话历史 + 交互记录 + 每个 skill 的状态，支持
  上下文接续（「再来一个」「刚才那个」等指代可被 LLM 正确理解）。
- **REPL + CLI 双模式**：交互探索或一次性执行皆可。
- **最小依赖**：仅依赖 `requests` + `pyyaml`。

---

## 目录结构

```
agent_skills_system/
├── harness/
│   ├── __init__.py        # 包导出
│   ├── __main__.py        # python -m harness 入口
│   ├── config.py          # 配置加载（环境变量 + .env）
│   ├── llm.py             # DeepSeek 客户端（chat / chat_json / chat_with_tools）
│   ├── loader.py          # 三阶段渐进加载 (Phase 1/2/3) + frontmatter 契约解析
│   ├── executor.py        # 通用执行器（按声明式契约分派四种执行模式）
│   ├── memory.py          # JSON 持久化记忆层（对话 + 交互 + skill 状态）
│   ├── cli.py             # Harness agent loop + REPL + argparse
│   └── skills/            # skills 目录（自动发现）
│       ├── baoyu-diagram/ # 生成暗色主题 SVG 图表（生成模式）
│       ├── commit-message/# 根据 diff 生成 Conventional Commits（script+stdin）
│       └── flash-card/    # 生成英语单词学习闪卡（script+data_file）
├── data/
│   └── memory.json        # 运行时记忆文件（自动创建）
├── diagram/               # 运行时由 baoyu-diagram 生成（自动创建）
├── run.py                 # 项目入口: python run.py [...]
├── requirements.txt
├── .env.example           # DeepSeek 配置模板
└── README.md
```

---

## 环境要求

- **Python ≥ 3.8**
- **`requests` ≥ 2.28**、**`pyyaml` ≥ 6.0**
- **DeepSeek API Key** — 获取地址：https://platform.deepseek.com/api_keys
  （需支持 OpenAI 兼容的 function calling / tools 接口）

### 配置 API Key

任选一种方式：

**方式 1：环境变量**
```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-your-key-here"

# Linux/macOS
export DEEPSEEK_API_KEY=sk-your-key-here
```

**方式 2：项目根目录 `.env` 文件**

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-...
```

环境变量优先级高于 `.env` 文件。**未配置时 harness 启动即报错退出。**

### 可选 skill 运行时依赖

| Skill | 执行模式 | 运行时 | 说明 |
|-------|----------|--------|------|
| `flash-card` | script + data_file | Python 标准库 | 无额外依赖 |
| `commit-message` | script + stdin | Python 标准库 | 无额外依赖 |
| `baoyu-diagram` | 生成模式（+可选 post_process） | bun.exe | 仅 SVG→PNG 转换需要；无 bun 时仍产出 SVG |

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env  # 然后编辑填入 key

# 3. 列出 skills
python run.py --list

# 4. 对话 —— 可闲聊，也可让 LLM 自动调用 skill
python run.py "给我讲讲什么是 function calling"        # 纯对话，LLM 直接回复
python run.py "给我做 meticulous 的单词卡"              # LLM 自动调 flash-card
python run.py "画一个用户登录流程图"                    # LLM 自动调 baoyu-diagram

# 5. 进入持续对话 REPL
python run.py

# 6. 查看记忆
python run.py --history
python run.py --summary
```

---

## CLI 用法

```
python run.py [options] [request...]

选项:
  --skills-dir PATH   指定 skills 目录 (默认: harness/skills)
  --memory PATH       指定 memory JSON 路径 (默认: data/memory.json)
  --model NAME        覆盖 DEEPSEEK_MODEL (默认: deepseek-v4-flash)
  --quiet             减少 loader / agent loop 日志
  --list              仅列出 skills 后退出
  --history           打印执行历史后退出
  --summary           打印记忆汇总后退出
  --usage             打印 LLM 调用统计后退出
  request             单次请求文本（不进入 REPL），输出自然语言回复

退出码:
  0  成功
  2  配置错误（如未配置 DEEPSEEK_API_KEY）
```

### REPL 内置命令

| 命令 | 作用 |
|------|------|
| `/list` | 列出所有已注册 skills |
| `/history` | 查看最近 20 条执行历史 |
| `/state [name]` | 查看 skill 级记忆状态 |
| `/show <name>` | 打印某 skill 的完整 SKILL.md 正文 |
| `/reload` | 重新扫描 skills 目录 |
| `/summary` | 打印记忆汇总 |
| `/usage` | 打印 LLM 调用统计（calls/tokens/latency） |
| `/help` | 显示帮助 |
| `/quit` | 退出 |

其余输入进入对话：可正常聊天，也可请求执行 skill。

---

## 核心设计

### 对话型 agent loop（function calling 驱动）

`Harness.respond()` 是统一的对话入口，工作流：

1. 把当前用户输入写入 `memory.conversations`
2. 组装 `messages` = system prompt + 最近 N 轮对话历史（含 skill 调用后的总结）
3. 把所有 skills 的 frontmatter 契约转成 OpenAI function schema，作为 `tools` 传入
4. 调 DeepSeek `chat_with_tools`：
   - LLM 直接返回文本 → 这就是给用户的回复
   - LLM 返回 `tool_calls` → harness 执行对应 skill → 把结果作为 `tool` message 回传
     → 再次调 LLM 生成最终回复（可多轮 tool 调用，上限 5 轮防死循环）
5. 把最终回复写入 `memory.conversations`

关键点：**没有独立的 matcher 路由层**。是否调用 skill、调哪个、传什么参数，全部由 LLM
在完整对话上下文中决定。skill 执行结果回传给 LLM，由它生成自然语言总结 —— 这使得下一
轮对话中 LLM 完全知道刚才做了什么。

### 三阶段渐进加载

| 阶段 | 触发时机 | 动作 | 开销 |
|------|---------|------|------|
| **Phase 1** | 启动 | 仅解析每个 `SKILL.md` 的 YAML frontmatter（name/description/version + 执行契约） | 极小，可全量加载 |
| **Phase 2** | skill 被 LLM 选中时 | 才读取该 skill 的完整 markdown 正文 | 按需 |
| **Phase 3** | 准备执行时 | 才枚举该 skill 目录下的 `scripts/`、`data/`、`references/` 子目录 | 按需 |

### 声明式执行契约（frontmatter）

skill 在 `SKILL.md` frontmatter 中声明自己的执行契约，harness 通用执行。契约字段：

| 字段 | 作用 |
|------|------|
| `params` | 参数声明（name/type/description/required），转成 function schema 供 LLM 抽参 |
| `entry` | 执行脚本相对路径（无则走「生成模式」） |
| `entry_input` | `data_file` / `stdin` / `args`，决定如何把数据喂给 entry |
| `data_instructions` | LLM 生成数据的指令模板（支持 `{param}` 与 `{avoid_hint}` 占位） |
| `output_ext` | 生成模式产物扩展名（如 `.svg`） |
| `output_subdir` | 生成模式产物子目录（默认 skill name） |
| `output_name` | 产物命名模板（默认用首个参数值） |
| `post_process` | 生成模式后处理脚本（如 SVG→PNG） |

### 四种执行模式（按契约字段组合自动分派）

| 模式 | 触发条件 | 流程 |
|------|---------|------|
| **A: script + data_file** | `entry` + `entry_input=data_file` + `data_instructions` | LLM 生成 JSON → 写 `data/<name>.json` → 执行 entry |
| **B: script + args** | `entry` + `entry_input=args` | 参数直接作 argv → 执行 entry |
| **C: script + stdin** | `entry` + `entry_input=stdin` + `data_instructions` | LLM 生成 JSON → 通过 stdin 传给 entry |
| **D: 生成模式** | 无 `entry` 但有 `output_ext` | LLM 按 SKILL.md body 生成产物 → 写 `output_subdir/<name>.<ext>` → 可选 post_process |

必需参数缺失时，执行器调 LLM 结合历史补全（如「再来一个」时让 LLM 挑一个不重复的值）。

### 记忆机制

`memory.py` 持久化到 `data/memory.json`，三层结构：

| 字段 | 作用 |
|------|------|
| `conversations[]` | 多轮对话历史（role/content/ts），agent loop 每轮读写，支持上下文接续 |
| `interactions[]` | 每次 skill 执行的完整记录（输入、命中 skill、phase、artifacts、error） |
| `skill_state{}` | 每个 skill 的自定义状态（`items_generated[]` + `last_item`），执行时拼进 LLM prompt 避免重复 |

这让 LLM 能理解「再来一个」「刚才那个单词是什么」等指代 —— 因为对话历史完整可见。

---

## 模块导览

| 文件 | 职责 |
|------|------|
| [harness/config.py](harness/config.py) | `HarnessConfig` — 从环境变量/.env 读取 DeepSeek 配置 |
| [harness/llm.py](harness/llm.py) | `DeepSeekClient` — chat / chat_json / chat_with_tools（function calling），累计 token 用量 |
| [harness/loader.py](harness/loader.py) | `ProgressiveSkillLoader` — 三阶段加载 + frontmatter 契约解析 |
| [harness/executor.py](harness/executor.py) | `GenericExecutor` — 按声明式契约分派四种执行模式，无 skill 专用代码 |
| [harness/memory.py](harness/memory.py) | `MemoryStore` — JSON 持久化（对话 + 交互 + skill 状态） |
| [harness/cli.py](harness/cli.py) | `Harness` agent loop + REPL + argparse |

---

## 编写新 skill

1. 在 `harness/skills/<my-skill>/` 下创建 `SKILL.md`：

   ```yaml
   ---
   name: my-skill
   description: 一句话描述何时触发本 skill（LLM 据此决定是否调用）
   version: 0.1.0

   # 参数声明 → 自动转成 function schema
   params:
     - name: topic
       type: string
       description: 要处理的主题
       required: true

   # 执行契约（任选一种模式，见上表）
   entry: scripts/run.py
   entry_input: stdin
   data_instructions: |
     请为「{topic}」生成数据，输出 JSON：{ ... }
   ---

   # 详细说明（markdown 正文，Phase 2 才加载，生成模式会作为 LLM 指令）
   ```

2. （可选）放置 `scripts/`、`data/`、`references/` 子目录。
3. `python run.py /reload` 生效 —— **无需改任何 harness 代码**。

LLM 会自动发现新 skill 并在合适时机调用。

---

## 验证示例

实测输出（DeepSeek-V4-Flash `deepseek-v4-flash`）：

```
$ python run.py "给我讲讲什么是 function calling"
[loader] Phase 1: scanning ... → 3 skill(s) registered
你好！很高兴为你讲解这个概念 😊
...（LLM 直接文字回复，未调用任何 skill）

$ python run.py "给我做张 curious 闪卡"
[loader] Phase 1: scanning ... → 3 skill(s) registered
[agent] tool_call: flash-card args={'word': 'curious'}
[agent] completed after 1 tool round(s)
闪卡已经生成成功啦！🎉 以下是 curious 这个单词的详细内容：
...（LLM 基于 skill 执行结果生成的自然语言总结）

$ python run.py --quiet "刚才那个单词是什么？再用它造个句子"
刚才那个单词是 curious 🎯
...（LLM 从对话历史理解指代，直接回答未误调工具）
```

---

## 已知限制与扩展点

- **强依赖 LLM**：未配置 `DEEPSEEK_API_KEY` 时 CLI 直接退出（exit code 2），不做降级。
- **function calling 兼容性**：依赖 DeepSeek 的 OpenAI 兼容 `tools` 接口，模型需支持
  function calling（`deepseek-v4-flash` / `deepseek-v4-pro` 均已支持）。
- **对话历史长度**：默认取最近 12 轮拼入 prompt 控制 token 成本，超长对话会丢失早期上下文。
- **工具调用轮数上限**：agent loop 最多 5 轮工具调用，异常情况会优雅终止。
- **PNG 转换**：Windows 上 `npx` 通常是 `.cmd`，subprocess 直接调用失败 → 自动跳过 PNG，
  仍产出 SVG。如需 PNG，请安装 `bun.exe`。
- **执行隔离**：skill 脚本以 harness 同一用户身份运行，未做沙箱。生产部署请自行加容器/权限隔离。
