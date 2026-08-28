# CLAUDE.md

本文件为 Claude Code 提供本项目（`张雷/week11`）的工作指引。全局规则见用户 `~/.claude/CLAUDE.md`。

## 项目概览
「工具调用」三种方式的教学对比：同一套业务（**天气查询**）用三种工具接入方式实现并横向对比。

- **方式一 Function Call** - `mode_function_call/run_function_call.py`：手写 JSON Schema（`TOOLS_SCHEMA`），通过 `TOOL_DISPATCH` 直接调 `src/` 后端函数。
- **方式二 MCP** - `mode_mcp/run_mcp.py`：连接 `mode_mcp/servers/weather_server.py` 这一个 stdio Server，`connect_all_servers` 走 建管道->握手->`list_tools` 发现->转 OpenAI schema；执行走 `session.call_tool` 跨进程调用。
- **方式三 CLI** - `mode_cli/run_cli.py`：`fincli`（`pip install -e .` 注册到 PATH）作为真实命令。`named` 模式白名单 enum（`run_cli(command, args)`），`bash` 模式沙箱执行（`run_bash(command)`，有黑名单+白名单头+超时+工作目录锁定）。
- `src/` - 共享业务逻辑：`weather_backend.py`（`geocode` 城市名->经纬度含地名歧义 + `get_weather_by_coords` 经纬度->天气预报）。
- `compare.py` - 在内置 `DEFAULT_QUESTIONS`（天气题）上跑**四种**执行并对比：Function Call / MCP / CLI(named) / CLI(bash)（CLI 有 named+bash 两形态，故「三方式」实跑 4 路）。以 `--json --quiet` 子进程调用各 `run_*.py`，解析 stdout **最后一行** JSON `{answer, tool_calls, elapsed, rounds}`，写 `output/compare_result.md`（生成物，已 gitignore）。

## 关键设计：多轮循环
三个 `run_*.py` 的 `run()` 是**同一骨架**的多轮循环：只要模型还在调用工具，就继续「执行->回填->再请求」，直到给出不带 tool_call 的最终答案；`MAX_ROUNDS=8` 触顶用 `tool_choice="none"` 强制收尾并打印警告。`run()` 返回 `{answer, tool_calls, elapsed, rounds}`。
- 天气条件题（如「若气温低于 20 度则查哈尔滨，否则查广州」）天然跨多轮：模型须先查一个城市，看到结果才能决定下一步查哪个--数据依赖强制多轮，不会被模型并行一把梭。
- 工具集两个：`geocode`（城市名->经纬度）+ `get_weather_by_coords`（经纬度->天气）；坐标数据依赖强制多轮（拿不到 geocode 的经纬度就无法调 get_weather_by_coords），即使查单城市也跨 2 轮，循环无需区分工具类型（早期版本有「天气门控」让 RAG 工具单轮、天气多轮；RAG 删除后门控变死代码已移除，循环简化为纯多轮）。

## 常用命令
```bash
# 依赖
pip install -r requirements.txt
pip install -e .                       # 注册 fincli（cli 模式需要）

# 环境变量
export DEEPSEEK_API_KEY=sk-xxx         # 默认 LLM（deepseek-chat）；--provider dashscope 切 qwen-plus
export DASHSCOPE_API_KEY=sk-xxx        # 备选 LLM（qwen-plus）

# 单独运行（默认问题=天气多轮循环演示题）
python mode_function_call/run_function_call.py --demo
python mode_mcp/run_mcp.py --demo
python mode_cli/run_cli.py --mode named --demo
python mode_cli/run_cli.py --mode bash  --demo

python compare.py                      # 三方式（4 路）横向对比
python test_loop.py                    # 循环逻辑 mock 测试（不联网、不发 API）
```

## 约定与陷阱
- **三个 `run_*.py` 故意镜像**：结构、`DEMO_QUESTIONS`、`SYSTEM_PROMPT` 风格保持一致；改一个通常要同步另两个，避免分叉。
- **箭头字符陷阱**：源码注释 / f-string 里的 `->` 是单个 Unicode **U+2192**，不是 ASCII `->`；`--` 也常是 em dash（U+2014）而非 ASCII `--`。Read 会把它们渲染成普通箭头/短横，导致 Edit 大段 old_string 匹配失败。改大段时用只含 ASCII 的锚点拆小块，或写脚本按 ASCII 锚点定位后替换（见 `weather_server.py` docstring 改法）。注意类型注解 `-> dict:` 是真 ASCII，别混淆。
- **新增工具**：Function Call = 在 `TOOLS_SCHEMA` 加 schema + `TOOL_DISPATCH` 加映射；MCP = 在 server 加 `@mcp.tool()`；CLI named = 在 `NAMED_COMMANDS` 加白名单条目（并同步 fincli 子命令）。
- **天气地名歧义**：`weather_backend` 对裸「宁德」等会重查「市」并取行政级别更高的（福建宁德而非西藏同名点）。
- **分层**：业务逻辑改动放 `src/`，协议层（工具从哪来、怎么执行）改动放各 `mode_*/run_*.py`，二者分离。
- **MCP Server stdout 是协议通道**：`mode_mcp/servers/*.py` 里所有日志必须写 `sys.stderr`，混进 stdout 会破坏 JSON-RPC 连接。后端函数用 `as` 别名导入（如 `geocode as _geocode`、`get_weather_by_coords as _get_weather_by_coords`），否则 `@mcp.tool()` 同名函数会遮蔽后端、递归调自己。
- **`test_loop.py` 只直测 function_call 的 `run()`**：mcp/cli 的循环是镜像（结构相同、未单独测）。改 mcp/cli 的循环后靠 `python3 -m py_compile` + 镜像对照验证，别以为 test_loop 能兜底。
- LLM 走 OpenAI 兼容接口（`PROVIDERS` 配置 deepseek / dashscope）。

## 更深文档
- `ARCHITECTURE.md` - 大图：三方式分层协作流水线、接入成本/安全/跨模型复用对比。
- `USAGE_GUIDE.md` - 环境准备、API Key 配置、各方式调用与测试步骤。
- `RESUME_GUIDE.md` - 可量化指标表（4 路延迟、沙箱 13 正则+7 白名单头）。
