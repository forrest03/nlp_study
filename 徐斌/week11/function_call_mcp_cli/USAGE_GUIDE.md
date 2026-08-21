# 使用指南 — Function Call / MCP / CLI 三方式对比（天气查询）

## 环境准备

```bash
cd function_call_mcp_cli
pip install -r requirements.txt
# 或：pip install -e .
```

依赖：`openai`、`httpx`、`mcp>=1.0.0`。

环境变量：

```bash
export DEEPSEEK_API_KEY=sk-xxx          # 默认 LLM
# 可选：export DASHSCOPE_API_KEY=sk-xxx  # --provider dashscope
```

## 方式一：Function Call

```bash
python mode_function_call/run_function_call.py -q "宁德现在天气怎么样？"
python mode_function_call/run_function_call.py --demo
```

预期：多轮工具调用，先 `geocode_city`，再 `get_weather_by_coords`，最后生成回答。

## 方式二：MCP

```bash
python mode_mcp/run_mcp.py -q "宁德现在天气怎么样？"
```

Host 启动 `weather_server.py` 子进程，`list_tools` 发现 `geocode_city` / `get_weather_by_coords`，再走同样的多轮闭环。

## 方式三：CLI

```bash
# 直接命令
fincli geocode --city 宁德
fincli weather-by-coords --latitude 26.67 --longitude 119.52 --location-name 宁德
fincli weather --city 宁德

# LLM 驱动
python mode_cli/run_cli.py --mode named -q "宁德现在天气怎么样？"
python mode_cli/run_cli.py --mode bash -q "北京今天天气如何？"
```

## 对比跑分

```bash
python compare.py
```

结果写入 `output/compare_result.md`。

## 后端自测

```bash
python -c "from src.weather_backend import geocode_city, get_weather; print(geocode_city('宁德')); print(get_weather('宁德'))"
```
