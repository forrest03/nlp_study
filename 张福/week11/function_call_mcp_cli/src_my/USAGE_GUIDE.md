# USAGE_GUIDE.md — 代码调用与测试指南
## run_cli.py
### 查询天气 bash 方式
```
python run_cli.py --mode bash --question 重庆的天气
[CLI/bash] provider=dashscope model=qwen-plus

============================================================
Q1：重庆的天气
============================================================
lines314  → [bash] run_bash({'command': 'fincli weather --city 重庆'})
    ↩ 【中国 重庆市 重庆】天气报告 坐标：29.56°N, 106.56°E  当前天气：晴天   温度：41.0°C   相对湿度：29%   风速：11.9 km/h  未来3天预报：   2026-07-14：阴天，41.5°C / 30...

  → [llm] 最终回答（7.5s）

最终回答：
重庆当前天气为晴天，温度41.0°C，相对湿度29%，风速11.9 km/h。未来三天均为阴天，高温在39.7°C–41.5°C之间，低温在29.7°C–30.7°C之间，无降水。

```

### 根据经纬度查天气 bash 方式
```
python run_cli.py --mode bash --question 获取重庆的经纬度，再根据经纬度查天气
[CLI/bash] provider=dashscope model=qwen-plus

============================================================
Q1：获取重庆的经纬度，再根据经纬度查天气
============================================================
工具个数: 2
lines314  → [bash] run_bash({'command': 'fincli latlon --city 重庆'})
    ↩ ['【中国 重庆市 重庆】', '坐标：29.56°N, 106.56°E'] 

lines314  → [bash] run_bash({'command': 'fincli weather_latlon --lat 29.56 --lon 106.29'})
    ↩ 天气报告 坐标：29.56°N, 106.29°E  当前天气：晴天   温度：38.8°C   相对湿度：39%   风速：7.9 km/h  未来3天预报：   2026-07-14：阴天，39.5°C / 30.2°C，降水 0.0 ...

  → [llm] 最终回答（16.0s）

最终回答：
重庆的经纬度为：29.56°N, 106.56°E（注意：天气查询使用的是近似坐标 29.56°N, 106.29°E）。

根据该经纬度查询的天气信息如下：

- 当前天气：晴天  
- 温度：38.8°C  
- 相对湿度：39%  
- 风速：7.9 km/h  

未来3天预报：  
- 2026-07-14：阴天，39.5°C / 30.2°C，降水 0.0 mm  
- 2026-07-15：阴天，36.6°C / 28.4°C，降水 0.0 mm  
- 2026-07-16：局部多云，38.3°C / 28.6°C，降水 0.0 mm
```

### 查询天气 named模式
```
python run_cli.py --question 重庆的天气
[CLI/named] provider=dashscope model=qwen-plus

============================================================
Q1：重庆的天气
============================================================
工具个数: 1
lines314  → [named] weather({'city': '重庆'})
    ↩ 【中国 重庆市 重庆】天气报告 坐标：29.56°N, 106.56°E  当前天气：晴天   温度：41.1°C   相对湿度：29%   风速：11.9 km/h  未来3天预报：   2026-07-14：阴天，41.5°C / 30...

  → [llm] 最终回答（7.6s）

最终回答：
重庆当前天气为晴天，气温41.1°C，相对湿度29%，风速11.9 km/h。未来三天（7月14日至16日）均为阴天，最高温在39.7–41.5°C之间，最低温在29.7–30.7°C之间，无降水。
```

### 查询根据经纬度查询天气 named模式
```
python run_cli.py --question 获取重庆的经纬度,在通过经纬度查询天气
[CLI/named] provider=dashscope model=qwen-plus

============================================================
Q1：获取重庆的经纬度,在通过经纬度查询天气
============================================================
工具个数: 2
lines314  → [named] latlon({'city': '重庆'})
    ↩ ['【中国 重庆市 重庆】', '坐标：29.56°N, 106.56°E'] 

lines314  → [named] weather_latlon({'lat': 29.56301, 'lon': 106.551557})
    ↩ 天气报告 坐标：29.56301°N, 106.551557°E  当前天气：晴天   温度：41.1°C   相对湿度：29%   风速：11.9 km/h  未来3天预报：   2026-07-14：阴天，41.5°C / 30.2°C...

  → [llm] 最终回答（10.3s）

最终回答：
重庆的经纬度为：29.56°N, 106.56°E（即 29.56301°N, 106.551557°E）。

当前天气为晴天，气温高达41.1°C，相对湿度29%，风速11.9 km/h。  
未来三天（7月14日–16日）均为阴天，高温在39.7°C–41.5°C之间，低温在29.7°C–30.7°C之间，无降水。
```
---
## run_function_call.py 通过function_call 调用
### python run_function_call.py  -q 重庆的天气
```
python run_function_call.py  -q 重庆的天气
[rag_backend] 就绪：10353 个向量，10353 条元数据
[Function Call] provider=dashscope model=qwen-plus

============================================================
Q1：重庆的天气
============================================================
  → [tool] get_weather({'city': '重庆'})
【中国 重庆市 重庆】天气报告
坐标：29.56°N, 106.56°E

当前天气：晴天
  温度：41.2°C
  相对湿度：29%
  风速：12.0 km/h

未来3天预报：
  2026-07-14：阴天，41.5°C / 30.2°C，降水 0.0 mm
  2026-07-15：阴天，39.7°C / 29.7°C，降水 0.0 mm
  2026-07-16：阴天，40.0°C / 30.7°C，降水 0.0 mm
    ↩ 【中国 重庆市 重庆】天气报告 坐标：29.56°N, 106.56°E  当前天气：晴天   温度：41.2°C   相对湿度：29%   风速：12.0 km/h  未来3天预报：   2026-07-14：阴天，41.5°C / 30...

  → [llm] 最终回答（9.0s）

最终回答：
重庆当前天气为晴天，气温高达41.2°C，相对湿度29%，风速12.0 km/h。

未来3天预报如下：
- 7月14日：阴天，最高温41.5°C，最低温30.2°C，无降水；
- 7月15日：阴天，最高温39.7°C，最低温29.7°C，无降水；
- 7月16日：阴天，最高温40.0°C，最低温30.7°C，无降水。

请注意防暑降温，避免长时间户外活动。
```
### python run_function_call.py  -q 重庆经纬度是多少
```
python run_function_call.py  -q 重庆经纬度是多少
[rag_backend] 就绪：10353 个向量，10353 条元数据
[Function Call] provider=dashscope model=qwen-plus

============================================================
Q1：重庆经纬度是多少
============================================================
  → [tool] get_latlon({'city': '重庆'})
['【中国 重庆市 重庆】', '坐标：29.56°N, 106.56°E'] <class 'list'>
    ↩ 【中国 重庆市 重庆】坐标：29.56°N, 106.56°E

  → [llm] 最终回答（4.8s）

最终回答：
重庆的经纬度是：**北纬29.56°，东经106.56°**。
```
### python run_function_call.py  -q 获取重庆的经纬度,再根据经纬度查询天气
```
python run_function_call.py  -q 获取重庆的经纬度,再根据经纬度查询天气
[rag_backend] 就绪：10353 个向量，10353 条元数据
[Function Call] provider=dashscope model=qwen-plus

============================================================
Q1：获取重庆的经纬度,再根据经纬度查询天气
============================================================
  → [tool] get_latlon({'city': '重庆'})
['【中国 重庆市 重庆】', '坐标：29.56°N, 106.56°E'] <class 'list'>
    ↩ 【中国 重庆市 重庆】坐标：29.56°N, 106.56°E

  → [tool] get_weather_latlon({'lat': '29.5639', 'lon': '106.5516'})
天气报告
坐标：29.5639°N, 106.5516°E

当前天气：晴天
  温度：41.2°C
  相对湿度：29%
  风速：12.0 km/h

未来3天预报：
  2026-07-14：阴天，41.5°C / 30.2°C，降水 0.0 mm
  2026-07-15：阴天，39.7°C / 29.7°C，降水 0.0 mm
  2026-07-16：阴天，40.0°C / 30.7°C，降水 0.0 mm
    ↩ 天气报告 坐标：29.5639°N, 106.5516°E  当前天气：晴天   温度：41.2°C   相对湿度：29%   风速：12.0 km/h  未来3天预报：   2026-07-14：阴天，41.5°C / 30.2°C，降水...

  → [llm] 最终回答（9.3s）

最终回答：
重庆当前天气为晴天，气温高达41.2°C，相对湿度29%，风速12.0 km/h；未来三天（7月14日–16日）以阴天为主，日间高温维持在39.7–41.5°C，夜间低温约29.7–30.7°C，无降水。
```

## run_mcp.py
### 查询经纬度
```
python run_mcp.py  --question 重庆的经纬度
[MCP] provider=dashscope model=qwen-plus

正在连接 MCP Servers...

[rag_backend] 就绪：10353 个向量，10353 条元数据
RAG MCP Server 启动中（stdio 模式）...
[07/14/26 15:26:37] INFO     Processing request of type ListToolsRequest                                          server.py:733
  ✓ [rag]  search_annual_report, list_companies
Weather MCP Server 启动中（stdio 模式）...
[07/14/26 15:26:38] INFO     Processing request of type ListToolsRequest                                          server.py:733
  ✓ [weather]  get_weather, get_latlon

共 4 个工具就绪

============================================================
Q1：重庆的经纬度
============================================================
  → [mcp] get_latlon({'city': '重庆'})
[07/14/26 15:26:39] INFO     Processing request of type CallToolRequest                                           server.py:733
[07/14/26 15:26:41] INFO     HTTP Request: GET                                                                  _client.py:1025
                             https://geocoding-api.open-meteo.com/v1/search?name=%E9%87%8D%E5%BA%86&count=10&la                
                             nguage=zh&format=json "HTTP/1.1 200 OK"                                                           
    ↩ [weather] 【中国 重庆市 重庆】坐标：29.56°N, 106.56°E

  → [llm] 最终回答（4.6s）

最终回答：
重庆的经纬度为：**29.56°N, 106.56°E**。

```
### 根据经纬度查询天气
```
python run_mcp.py  --question 获取重庆的经纬度,再通过经纬度获取天气信息
[MCP] provider=dashscope model=qwen-plus

正在连接 MCP Servers...

[rag_backend] 就绪：10353 个向量，10353 条元数据
RAG MCP Server 启动中（stdio 模式）...
[07/14/26 15:28:36] INFO     Processing request of type ListToolsRequest                                          server.py:733
  ✓ [rag]  search_annual_report, list_companies
Weather MCP Server 启动中（stdio 模式）...
[07/14/26 15:28:37] INFO     Processing request of type ListToolsRequest                                          server.py:733
  ✓ [weather]  get_weather, get_latlon, get_weather_latlon

共 5 个工具就绪

============================================================
Q1：获取重庆的经纬度,再通过经纬度获取天气信息
============================================================
  → [mcp] get_latlon({'city': '重庆'})
[07/14/26 15:28:39] INFO     Processing request of type CallToolRequest                                           server.py:733
[07/14/26 15:28:40] INFO     HTTP Request: GET                                                                  _client.py:1025
                             https://geocoding-api.open-meteo.com/v1/search?name=%E9%87%8D%E5%BA%86&count=10&la                
                             nguage=zh&format=json "HTTP/1.1 200 OK"                                                           
    ↩ [weather] 【中国 重庆市 重庆】坐标：29.56°N, 106.56°E

  → [mcp] get_weather_latlon({'lat': '29.56301', 'lon': '106.551557'})
                    INFO     Processing request of type CallToolRequest                                           server.py:733
[07/14/26 15:28:41] INFO     HTTP Request: GET                                                                  _client.py:1025
                             https://api.open-meteo.com/v1/forecast?latitude=29.56301&longitude=106.551557&curr                
                             ent=temperature_2m%2Crelative_humidity_2m%2Cwind_speed_10m%2Cweather_code&daily=te                
                             mperature_2m_max%2Ctemperature_2m_min%2Cprecipitation_sum%2Cweather_code&timezone=                
                             Asia%2FShanghai&forecast_days=3 "HTTP/1.1 200 OK"                                                 
    ↩ [weather] 天气报告 坐标：29.56301°N, 106.551557°E  当前天气：局部多云   温度：41.4°C   相对湿度：29%   风速：11.6 km/h  未来3天预报：   2026-07-14：局部多云，41.6°C / 30...

  → [llm] 最终回答（8.0s）

最终回答：
重庆当前天气为局部多云，气温高达41.4°C，湿度较低（29%），风速11.6 km/h。未来三天预报如下：

- **7月14日**：局部多云，最高温41.6°C，最低温30.2°C，无降水；
- **7月15日**：阴天，最高温39.4°C，最低温29.2°C，无降水；
- **7月16日**：小毛毛雨，最高温40.0°C，最低温30.5°C，降水0.1 mm。

请注意防暑降温，并关注16日可能出现的微量降雨。
```