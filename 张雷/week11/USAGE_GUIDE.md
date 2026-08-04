# USAGE_GUIDE.md - 代码调用与测试指南

## 1. 环境准备

### 1.1 依赖安装
```bash
pip install -r requirements.txt
```
依赖：`openai`、`httpx`、`mcp>=1.0.0`。无需 Node.js。

**把 fincli 装成真实命令（推荐，一次即可）**：
```bash
pip install -e .
```
这会把 `fincli` 注册到 PATH（`pyproject.toml` 的 `[project.scripts]`），之后在任意目录都能像 `git`/`ls` 一样直接敲 `fincli geocode`/`fincli weather ...`。方式三的 CLI 形态依赖它；不装也能用（自动退回 `python mode_cli/cli/main.py`，只是命令不"漂亮"）。

### 1.2 API Key 配置（系统环境变量）
本项目不使用 `.env` 文件，API Key 一律从系统环境变量读取。

Windows（PowerShell）：
```powershell
$env:DEEPSEEK_API_KEY  = "sk-xxx"   # 默认 LLM，驱动工具调用
$env:DASHSCOPE_API_KEY = "sk-xxx"   # 备选 LLM qwen-plus（--provider dashscope 时用）
```
Windows（CMD，当前进程）：
```cmd
set DEEPSEEK_API_KEY=sk-xxx
set DASHSCOPE_API_KEY=sk-xxx
```
Linux / macOS：
```bash
export DEEPSEEK_API_KEY=sk-xxx
export DASHSCOPE_API_KEY=sk-xxx
```
- DeepSeek 申请：https://platform.deepseek.com/
- DashScope 申请：https://dashscope.aliyun.com/
- 想永久生效：Windows 用 `setx DEEPSEEK_API_KEY "sk-xxx"`（重开终端生效）；Linux/macOS 写入 `~/.bashrc` 或 `~/.zshrc`。
- 只用默认 DeepSeek 时，`DASHSCOPE_API_KEY` 可不设。

---

## 2. 各方式运行

> 三种方式的命令行接口统一：`--question/-q` 单问题、`--demo` 内置示例、`--provider deepseek|dashscope`、`--json --quiet`（供 compare.py）。

### 2.1 方式一：Function Call

```bash
python mode_function_call/run_function_call.py -q "宁德现在天气如何？"
python mode_function_call/run_function_call.py --demo
python mode_function_call/run_function_call.py -q "..." --provider dashscope
```
**内部流程**：
1. 手写 `TOOLS_SCHEMA`（`geocode` + `get_weather_by_coords` 两个工具的 JSON Schema）
2. `chat.completions.create(tools=TOOLS_SCHEMA, tool_choice="auto")`
3. 若返回 `tool_calls`：查 `TOOL_DISPATCH` 表调后端函数 -> 结果以 `role=tool` 回填
4. 再次 `create`，模型基于结果继续（多轮）或生成最终回答

**预期输出**：先打印 `-> [tool] geocode({...})` 再 `get_weather_by_coords({...})` 与结果预览，再打印最终回答。

### 2.2 方式二：MCP

```bash
python mode_mcp/run_mcp.py -q "宁德现在天气如何？如果气温低于20度再查哈尔滨，否则查广州。"
python mode_mcp/run_mcp.py --demo
```
**内部流程**：
1. `AsyncExitStack` 启动 weather_server 子进程（stdio）
2. Server `initialize()` 握手 -> `list_tools()` 发现工具，打印 `✓ [weather] geocode, get_weather_by_coords`；同时把 MCP `inputSchema` 转 OpenAI `parameters`（一次走完连接+发现+转 schema）
3. LLM 多轮循环：`tool_call` -> 查 `tool_registry` 路由到对应 `ClientSession` -> `session.call_tool()` -> 结果回填
4. 模型生成最终回答

**预期输出**：stderr 打印连接日志 + `共 2 个工具就绪`；stdout 打印工具调用与最终回答。

> MCP Server 的 INFO 日志会打到 stderr（如 `Processing request of type CallToolRequest`），属正常协议日志。

### 2.3 方式三：CLI

CLI 方式的核心思想是"把能力做成普通命令行工具"--它本身跟大模型没有任何关系，可以像 `ls`、`git` 一样独立使用；然后再让大模型通过一个 `run_cli`/`run_bash` 工具去调用它。所以下面分两步看：先把它当普通 CLI 用，再把它接给模型。

#### 2.3.1 作为命令行工具单独使用（不经过 LLM）

`mode_cli/cli/main.py` 是统一入口，`pip install -e .` 后就是 PATH 上的 `fincli` 命令。任何人都能直接敲命令拿到结果，跟大模型没有任何关系：

```bash
# 查天气（两步：先 geocode 拿坐标，再 weather 取预报）
fincli geocode --city 宁德                 # -> JSON 坐标
fincli weather --lat 26.66 --lon 119.52    # -> 天气预报（--name 等可选，用于标题）
```

> 没装 fincli？用 `python mode_cli/cli/main.py geocode --city 宁德` 也能跑，效果一样。

`fincli` 就是把 `src/weather_backend.py` 包了一层 `argparse`，输出走 stdout 纯文本。这就是 CLI 作为"工具实现层"的全部--**一条能跑、能管道拼接的真实命令，无需任何协议**。可以配合 shell 管道用：

```bash
fincli weather --lat 26.66 --lon 119.52 | head -20
```

#### 2.3.2 结合大模型调用

`run_cli.py` 把上面的 `fincli` 包成 LLM 可调用的工具，多轮循环。有两种形态，用 `--mode` 切换：

**形态 A（具名 `run_cli`，白名单，默认，更安全）**：
```bash
python mode_cli/run_cli.py --mode named -q "宁德现在天气如何？"
```
LLM 分两步调：先 `run_cli(command='geocode', args={city: '宁德'})` 拿坐标，再 `run_cli(command='weather', args={lat, lon, ...})` 取预报；host 按 `NAMED_COMMANDS` 白名单拼出 `fincli geocode`/`fincli weather ...` 执行。`command` 是 enum，模型只能选预批准的命令--安全可控，但每加一个命令要改代码。

**形态 B（通用 `run_bash`，沙箱，更灵活）**：
```bash
python mode_cli/run_cli.py --mode bash -q "宁德现在天气如何？如果气温低于20度再查哈尔滨。"
```
LLM 自己拼完整 shell 命令字符串（如先 `fincli geocode --city 宁德` 再 `fincli weather --lat .. --lon ..`），host 经 `sandbox_check` 后 `subprocess.run(shell=True)` 执行。最灵活也最危险，靠沙箱兜底。

> 两种形态对比是本方式的教学重点：形态 A = 安全的"工具白名单"，形态 B = 灵活的"通用 shell"，差异只在沙箱设计。

#### 2.3.3 沙箱拦截验证（形态 B）

```bash
python -c "from mode_cli.run_cli import run_bash; print(run_bash('rm -rf /'))"
# -> [run_bash] 沙箱拦截：命中危险模式 '\\brm\\b'
```
沙箱 = 危险命令正则黑名单（rm/del/format/sudo/curl|sh/nc…）+ 命令头白名单（fincli/python/git/ls/cat/echo/type/dir）+ 15s 超时 + 工作目录锁定。

---

## 3. 三方式对比

```bash
python compare.py
python compare.py --questions "宁德现在天气如何？" "北京和上海哪个气温更高？"
python compare.py --provider dashscope
```
对每个问题依次跑 Function Call / MCP / CLI(named) / CLI(bash) 四方式，记录工具调用、耗时、答案摘要，输出对比表到 `output/compare_result.md`，同时在控制台打印简表。

**预期**：四方式对同一问题调用工具基本一致；Function Call 进程内直调最快，MCP/CLI 有子进程或 IPC 开销更高；天气条件题四方式都跨多轮完成。

---

## 4. 作为模块调用

```python
import sys; sys.path.insert(0, ".")
from src.weather_backend import geocode, get_weather_by_coords

# 直接调后端（两步）
loc = geocode("宁德")                       # 城市名 -> 经纬度
print(get_weather_by_coords(loc["lat"], loc["lon"], loc["name"], loc["country"], loc["admin1"]))
```

```python
# 方式一作为模块
from mode_function_call.run_function_call import build_client, run, TOOLS_SCHEMA
client, model = build_client("deepseek")
result = run(client, model, "宁德现在天气如何？", verbose=True)
print(result["answer"], result["tool_calls"])
```

---

## 5. 调试与常见问题

**Q1：`RecursionError: maximum recursion depth exceeded`（MCP 方式）**
MCP Server 的 tool 函数与导入的后端函数同名导致递归。本项目已用 `as` 别名（`geocode as _geocode`、`get_weather_by_coords as _get_weather_by_coords`）修复，若你新增 Server 工具，注意别让 tool 函数名与 import 的后端函数名相同。

**Q2：MCP / 后端报 `未设置 DEEPSEEK_API_KEY`**
LLM 调用需要 Key。确认系统环境变量 `DEEPSEEK_API_KEY` 已设置（`echo $DEEPSEEK_API_KEY` 能看到值）。MCP Server 子进程通过 `env={**os.environ}` 继承父进程环境变量。

**Q3：CLI(bash) 模型生成的命令被沙箱拦截**
看拦截信息：命中黑名单（危险命令）或命令头不在白名单（只允许 fincli/python/git/ls/cat/echo/type/dir）。模型应在 prompt 提示的命令集内生成。

**Q4：天气查到奇怪的地方（如"宁德"查到西藏）**
已修复：geocoding 自动取行政级别更高的候选，并对裸城市名追加"市"重查。若仍异常，显式传 `--city 宁德市`。

**Q5：`python -m py_compile` 提示文件名以数字开头无法 import**
本项目所有文件用 snake_case，无数字前缀，可直接 import。

**Q6：DeepSeek 偶尔不返回 tool_calls，直接回答了**
偶发现象。系统 prompt 已强调天气查询可多轮。若仍频繁出现，换 `--provider dashscope`（qwen-plus）。
