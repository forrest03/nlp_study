# 三方式对比结果（Function Call / MCP / CLI）

- LLM provider：`dashscope`
- 生成时间：本表由 `python compare.py` 实跑生成
- 问题数：3，方式数：4
- 业务场景：天气查询（geocode → weather-by-coords 多轮循环）

## 对比表

| 问题 | 方式 | 工具调用 | 工具数 | LLM耗时 | 答案摘要 |
|------|------|---------|:------:|:-------:|---------|
| 宁德现在天气怎么样？ | Function Call | geocode_city, get_weather_by_coords | 2 | 8.3s | 宁德当前天气为局部多云，气温27.2°C，湿度较高（89%），风速较轻（6.0 km/h）。  未来三天预报如下： - 7月17日：中毛毛雨，气温36.6°C ... |
| 宁德现在天气怎么样？ | MCP | geocode_city, get_weather_by_coords | 2 | 7.6s | 宁德市当前天气为局部多云，气温27.2°C，湿度较高（89%），风速较轻（6.0 km/h）。  未来三天预报如下：   - **7月17日**：中毛毛雨，气温... |
| 宁德现在天气怎么样？ | CLI(named) | run_cli, run_cli | 2 | 9.0s | 宁德市当前天气为局部多云，气温27.2°C，相对湿度89%，风速6.0 km/h。  未来三天预报： - 7月17日：中毛毛雨，气温27.0°C～36.6°C，... |
| 宁德现在天气怎么样？ | CLI(bash) | run_bash, run_bash | 2 | 7.9s | 宁德市当前天气为局部多云，气温27.2°C，相对湿度89%，风速6.0 km/h。  未来三天预报如下： - 7月17日：中毛毛雨，气温36.6°C / 27.... |
| 北京今天天气如何？另外未来三天呢？ | Function Call | geocode_city, get_weather_by_coords | 2 | 8.5s | 【北京】天气报告   坐标：39.91°N, 116.40°E    ✅ 当前天气：阴天   - 温度：24.9°C   - 相对湿度：91%   - 风速：3... |
| 北京今天天气如何？另外未来三天呢？ | MCP | geocode_city, get_weather_by_coords | 2 | 9.7s | 【北京】天气报告   坐标：39.91°N, 116.40°E    **当前天气**：阴天   - 温度：24.9°C   - 相对湿度：91%   - 风速... |
| 北京今天天气如何？另外未来三天呢？ | CLI(named) | run_cli, run_cli | 2 | 9.1s | 北京今天天气为阴天，当前温度24.9°C，相对湿度91%，风速3.5 km/h。  未来三天天气预报如下： - 7月17日（周四）：阴天，最高温31.3°C，最... |
| 北京今天天气如何？另外未来三天呢？ | CLI(bash) | run_bash, run_bash | 2 | 8.2s | 北京今天的天气是阴天，当前温度为24.9°C，相对湿度为91%，风速为3.5 km/h。  未来三天的天气预报如下： - 2026-07-17：阴天，最高温度3... |
| 对比一下上海和深圳现在的天气。 | Function Call | geocode_city, geocode_city, get_weather_by_coords, get_weather_by_coords | 4 | 13.8s | 以下是上海和深圳当前天气的对比：  ### 🌤 当前天气对比： | 项目         | 上海                     | 深圳      ... |
| 对比一下上海和深圳现在的天气。 | MCP | geocode_city, geocode_city, get_weather_by_coords, get_weather_by_coords | 4 | 13.3s | 以下是上海和深圳当前天气的对比：  ### 当前天气对比： - **温度**：     - 上海：31.4°C（较热）     - 深圳：24.5°C（相对凉爽... |
| 对比一下上海和深圳现在的天气。 | CLI(named) | run_cli, run_cli, run_cli, run_cli | 4 | 15.1s | 以下是上海和深圳当前天气的对比：  **温度** - 上海：31.4°C（明显更热） - 深圳：24.5°C（相对凉爽）  **天气状况** - 上海：局部多云... |
| 对比一下上海和深圳现在的天气。 | CLI(bash) | run_bash, run_bash, run_bash, run_bash | 4 | 16.5s | 以下是上海和深圳当前天气的对比：  ### **上海** - **当前天气**：局部多云   - **温度**：31.4°C   - **相对湿度**：69% ... |

## 解读

- **工具调用一致性**：四种方式对同一问题通常都走 geocode → weather_by_coords 两步。
- **接入成本**：Function Call 要手写 schema；MCP 要写 Server 但工具自动发现可跨产品复用；CLI(named) 写白名单；CLI(bash) 几乎零封装但需沙箱。
- **安全**：Function Call / MCP / CLI(named) 都走白名单；CLI(bash) 依赖沙箱拦截。
- **跨模型复用**：MCP 工具可被任意支持 MCP 的 Host 复用；CLI 与模型完全无关。
