# ARCHITECTURE.md - Function Call / MCP / CLI 三方式对比教学项目

## 1. 项目定位

本项目以"天气查询"为业务场景，**教学重点是对比让大模型"动手"调用工具的三种主流方式**：

| 方式 | 层次 | 工具从哪来 | 调用怎么执行 |
|------|------|-----------|-------------|
| **Function Call** | 模型能力层（意图生成） | 开发者手写 JSON Schema | 宿主直接调后端函数 |
| **MCP** | 协议标准层（接入规范） | 连接 Server 自动发现 | 跨进程 `call_tool`（stdio JSON-RPC） |
| **CLI** | 工具实现层（执行手段） | 命令行子命令 | 子进程执行，stdout 回传 |

三者**不互斥而是分层协作**：CLI 可作为 Function Call 的工具；MCP Server 内部也可调 CLI。本项目通过"同一份业务后端 × 三种封装"让学生看清差异。

> 业务能力是天气查询（Open-Meteo 免费 API，无需注册）。
> 项目聚焦"三种工具调用方式对比"，并支持**多轮循环**：天气条件题迫使模型先查一个城市、看到结果再决定查哪个，跨多轮完成。

## 2. 整体流水线

```
                         ┌─────────────────────────────┐
                         │  src/  共享业务后端（纯逻辑）  │
                         │   weather_backend.py (HTTP)  │
                         └──────────────┬──────────────┘
                                        │ 复用同一份后端
             ┌──────────────┬───────────┴───────────┬──────────────┐
             ▼              ▼                       ▼              ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ 方式一           │ │ 方式二           │ │ 方式三 形态A     │ │ 方式三 形态B     │
   │ Function Call    │ │ MCP              │ │ CLI (named)     │ │ CLI (bash)      │
   │                  │ │                  │ │                 │ │                 │
   │ 手写 tools schema│ │ @mcp.tool 注册   │ │ argparse 子命令  │ │ 同左 + 通用 shell│
   │ ↓                │ │ ↓                │ │ ↓               │ │ ↓               │
   │ LLM tool_call    │ │ Host list_tools  │ │ LLM run_cli     │ │ LLM run_bash    │
   │ ↓                │ │ ↓ schema 转换    │ │ ↓ 白名单 enum   │ │ ↓ 沙箱检查       │
   │ 直接调后端函数    │ │ call_tool IPC    │ │ subprocess      │ │ subprocess shell│
   └────────┬─────────┘ └────────┬─────────┘ └────────┬────────┘ └────────┬────────┘
            └─────────────────────┴─────────────────────┴──────────────────┘
                                  │
                       多轮循环：只要还在调工具就继续
                          compare.py 同一问题 × 四方式
                                  ↓
                    output/compare_result.md（对比表）
```

三种方式的多轮循环代码**几乎相同**（都是 `create(tools=) -> tool_calls -> 执行 -> role=tool 回填 -> 再 create`，直到无 tool_call 或触顶），差异只在"工具定义从哪来"和"tool_call 怎么执行"--这是对比的教学核心。

## 3. 各环节技术选型

### 3.1 共享后端 `src/`（选型原因：公平对比的前提）
- `weather_backend.py`：Open-Meteo 天气查询，同步实现（CLI 子进程同步更直观）；含地名歧义处理（裸"宁德"重查"市"并取行政级别更高的）。
- **为何抽后端**：三种方式必须调同一份逻辑，对比才公平--否则差异里混进了业务逻辑差异，掩盖了"接入方式"本身的差异。

### 3.2 方式一 Function Call（选型原因：最低接入成本，最直观）
- 手写 `TOOLS_SCHEMA`（name/description/parameters），`tool_choice="auto"`。
- `TOOL_DISPATCH` 表把工具名映射到后端函数--新增工具 = 加一行 schema + 一行 dispatch。
- 接入成本最低，但 schema 各家模型 API 略有差异（OpenAI/Claude/DeepSeek 字段名不完全一样），跨模型复用性差。

### 3.3 方式二 MCP（选型原因：标准化协议，工具自动发现）
- `mode_mcp/servers/weather_server.py`：FastMCP Server，`@mcp.tool()` 装饰后端函数（用 `as` 别名导入避免同名递归，见踩坑表）。
- `run_mcp.py` Host：`AsyncExitStack` 管理 Server 连接，`connect_all_servers()` 一次走完 `initialize()`->`list_tools()` 发现工具并把 MCP `inputSchema` 转 OpenAI `parameters`。
- 工具与模型解耦：同一个 MCP Server 可被 Claude Desktop / IDE / 任意 Host 复用，"写一次到处接"。代价是要实现 Server、走 stdio JSON-RPC，接入成本中等。

### 3.4 方式三 CLI（选型原因：复用现成命令生态，零封装）
- `mode_cli/cli/main.py`：argparse 子命令，把后端包成统一入口；`pyproject.toml` 用 `[project.scripts]` 注册为 `fincli`，`pip install -e .` 后是 PATH 上的真实命令（和 `git`/`ls` 一样），而非 `python xxx.py`。
- **形态 A `run_cli`**：白名单 enum 限定可执行命令集，host 拼出 `fincli geocode`/`fincli weather ...` 执行，安全；模型只能调预批准命令。
- **形态 B `run_bash`**：模型自己拼 shell 命令字符串（如先 `fincli geocode --city 宁德` 再 `fincli weather --lat .. --lon ..`），最灵活最危险，靠沙箱（黑名单正则 + 命令头白名单含 fincli + 超时 + 工作目录锁定）兜底。
- 与模型完全无关（任意能生成文本的 LLM 都能用），跨模型复用性最高。
- **降级**：`run_cli.py` 启动时 `shutil.which("fincli")` 探测，未安装则自动退回 `python mode_cli/cli/main.py`，功能不变。

### 3.5 LLM 选型
- 驱动工具调用的 LLM：默认 DeepSeek `deepseek-chat`（= `deepseek-v4-flash`），支持 function calling 与并行调用；备选 DashScope `qwen-plus`。统一走 OpenAI 兼容接口。

## 4. 实验结果（`python compare.py` 实跑）

对每个天气问题依次跑 Function Call / MCP / CLI(named) / CLI(bash) 四方式，记录工具调用、耗时、答案摘要，输出到 `output/compare_result.md`。

> LLM 调用有随机性，每次实跑的具体耗时与工具调用次数会有波动；下表为定性结论，具体数值以实跑为准。

**结果解读**：
1. **能力一致**：四方式对同一问题调用的工具与参数基本一致--说明底层能力相同，差异在"接入方式"而非"能力"。
2. **延迟**：Function Call 最快（进程内直调）；MCP 多一层 stdio IPC + Server 启动开销；CLI 多一层 subprocess 启动开销。后两者普遍高于 Function Call，具体排序随问题规模而异。
3. **多轮循环**：查天气分两步（`geocode` 拿坐标 -> `get_weather_by_coords` 取天气），坐标数据依赖使单城市查询也跨 2 轮；天气条件题（如"若气温低于20度则查A城，否则查B城"）迫使四方式都跨多轮完成--模型须先查一个城市、看到结果再决定查哪个，无法并行一把梭。
4. **触顶保护**：极端情况下模型连续调工具 8 轮仍不给最终答案时，`MAX_ROUNDS` 用 `tool_choice="none"` 强制收尾，不静默截断。

## 5. 关键工程决策与踩坑

| 问题 | 根因 | 解法 |
|------|------|------|
| MCP tool 函数递归调用自己（RecursionError） | `from src.weather_backend import geocode` 后又 `def geocode`，同名 tool 函数遮蔽了导入的后端函数，函数体内 `return geocode(...)` 调到自己 | 用 `as` 别名导入：`from src.weather_backend import geocode as _geocode`，tool 函数体调 `_geocode` |
| MCP Server stdout 混入普通 print 破坏协议 | stdout 是 JSON-RPC 通道 | 所有 log 写 `file=sys.stderr` |
| ClientSession 生命周期管理 | async context manager 一旦 `__aexit__` 连接就关 | `AsyncExitStack` 在 main 顶层持有连接 |
| "宁德"天气查到西藏那曲的村而非福建宁德 | Open-Meteo geocoding 裸"宁德"命中西藏 PPL 点（feature_code=PPL），福建宁德是"宁德市"（PPLA2）| count=10 取候选；若全是低级行政点且用户没带"市/县/区"后缀，用 `city+"市"` 重查；候选里按行政级别（PPLA/ADM）+ 人口排序 |
| CLI(bash) 安全风险 | 模型可能生成危险命令 | 黑名单正则（rm/del/format/sudo/curl\|sh/nc…）+ 命令头白名单（fincli/python/git/ls/cat/echo）+ 超时 15s + cwd 锁项目根 |
| fincli 未装时 bash 形态失效 | 形态 B 的 LLM 生成 `fincli ...`，shell 找不到命令 | run_cli.py 用 `shutil.which("fincli")` 探测；形态 A 自动降级到 `python mode_cli/cli/main.py`；形态 B 需先 `pip install -e .` |
| pyproject `[tool.setuptools] packages` 报配置错 | 包名用斜杠 `mode_cli/cli` 非法 | 必须用点号包名 `mode_cli.cli`；src/mode_cli/mode_cli.cli 加 `__init__.py` 才能被 console_script 导入 |
| compare.py 跨方式收集结果 | 各方式 lifecycle 不同（MCP 是 async + 子进程 Server）| 不跨 import，各方式支持 `--json --quiet` 输出单对象，compare.py 以子进程跑并解析 stdout 末行 JSON |

## 6. 目录结构

```
week11/
├── src/                              # 共享业务后端（三方式都复用）
│   └── weather_backend.py            # Open-Meteo：geocode（地名歧义）+ get_weather_by_coords（预报）
├── mode_function_call/
│   └── run_function_call.py          # 方式一：手写 schema + 多轮循环
├── mode_mcp/
│   ├── servers/
│   │   └── weather_server.py         # FastMCP Server（@mcp.tool 复用 weather_backend）
│   └── run_mcp.py                    # 方式二：Host 连接 Server + 工具发现 + 多轮循环
├── mode_cli/
│   ├── cli/
│   │   └── main.py                   # fincli 统一入口：argparse 子命令 geocode / weather
│   └── run_cli.py                    # 方式三：形态A run_cli(白名单) + 形态B run_bash(沙箱)
├── output/                           # compare.py 输出
│   └── compare_result.md
├── compare.py                        # 四方式对比运行器（教学 centerpiece）
├── test_loop.py                      # 多轮循环 mock 测试（不联网、不发 API）
├── pyproject.toml                    # 注册 fincli 为 console_script（pip install -e .）
├── requirements.txt
├── ARCHITECTURE.md                   # 本文件
├── USAGE_GUIDE.md
└── RESUME_GUIDE.md
```
