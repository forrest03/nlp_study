# ARCHITECTURE.md — Function Call / MCP / CLI 三方式对比（天气查询）

## 1. 项目定位

以**天气查询**为业务场景，对比让大模型"动手"调用工具的三种主流方式：

| 方式 | 层次 | 工具从哪来 | 调用怎么执行 |
|------|------|-----------|-------------|
| **Function Call** | 模型能力层 | 开发者手写 JSON Schema | 宿主直接调后端函数 |
| **MCP** | 协议标准层 | 连接 Server 自动发现 | 跨进程 `call_tool`（stdio） |
| **CLI** | 工具实现层 | 命令行子命令 | 子进程执行，stdout 回传 |

天气能力拆成两步工具，强制多轮循环：`geocode_city`（地区→坐标）→ `get_weather_by_coords`（坐标→天气）。

## 2. 整体流水线

```
              src/weather_backend.py（纯业务逻辑）
                           │
     ┌─────────────────────┼─────────────────────┐
     ▼                     ▼                     ▼
 Function Call           MCP                   CLI
 手写 schema          weather_server         fincli 子命令
 直接调后端            call_tool IPC         subprocess
     └─────────────────────┴─────────────────────┘
                           │
                     compare.py 对比
```

## 3. 目录结构

```
function_call_mcp_cli/
├── src/weather_backend.py            # geocode_city / get_weather_by_coords / get_weather
├── mode_function_call/run_function_call.py
├── mode_mcp/
│   ├── servers/weather_server.py
│   └── run_mcp.py
├── mode_cli/
│   ├── cli/main.py                   # fincli：geocode / weather-by-coords / weather
│   └── run_cli.py                    # named 白名单 + bash 沙箱
├── compare.py
├── requirements.txt
└── pyproject.toml                    # 注册 fincli
```
