# 简历要点 — Function Call / MCP / CLI 天气工具对比

| 项 | 内容 |
|----|------|
| 业务场景 | 天气查询（Open-Meteo：地名→坐标→预报） |
| 工具数 | 2 个（geocode_city / get_weather_by_coords） |
| 调用形态 | 多轮循环（先坐标后天气） |
| 对比维度 | Function Call / MCP / CLI(named) / CLI(bash) |

## 一句话

> 以天气查询为统一业务后端，横向对比 Function Call、MCP、CLI 三种大模型工具接入方式；将天气拆成 geocode + forecast 两步，实现依赖前一步结果的多轮 tool_call 闭环，并用 compare.py 量化延迟与工具调用差异。

## 可写细节

- Function Call：手写 JSON Schema + dispatch 表 + 多轮回填
- MCP：FastMCP Server + Host `initialize` / `list_tools` / `call_tool` + schema 适配
- CLI：`fincli` console_script；named 白名单与 bash 沙箱双形态
- 共享 `src/weather_backend`，保证对比公平
