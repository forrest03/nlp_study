"""
weather_backend.py — 天气查询后端（三种方式共享的业务逻辑）

教学重点：
  1. 同样是"纯业务逻辑"，与 rag_backend 平级，被三种方式复用
  2. 内部两次 HTTP 请求：Geocoding（城市名→经纬度）+ 天气查询
  3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费
  4. 拆分为细粒度函数，支持大模型分步调用

使用方式（作为模块）：
  from src.weather_backend import search_city, get_current_weather, get_forecast
  cities = search_city("宁德")
  weather = get_current_weather(cities[0]['latitude'], cities[0]['longitude'])

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def search_city(city: str) -> list:
    """
    搜索城市，返回匹配的城市列表（含经纬度、行政级别、人口等信息）。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        匹配的城市列表，每个元素包含：name(城市名), latitude(纬度), longitude(经度),
        feature_code(行政级别代码), population(人口), country(国家), admin1(省/州)
    """
    with httpx.Client(timeout=10.0) as client:
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _geocode(city)
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry

        if not results:
            return []

        filtered = []
        for r in results:
            filtered.append({
                "name": r.get("name", ""),
                "latitude": r.get("latitude", 0),
                "longitude": r.get("longitude", 0),
                "feature_code": r.get("feature_code", ""),
                "population": r.get("population", 0),
                "country": r.get("country", ""),
                "admin1": r.get("admin1", ""),
            })
        return filtered


def get_current_weather(latitude: float, longitude: float) -> dict:
    """
    根据经纬度获取当前天气。

    Args:
        latitude: 纬度
        longitude: 经度

    Returns:
        当前天气数据字典，包含：temperature(温度°C), humidity(湿度%), 
        wind_speed(风速km/h), weather_code(天气代码), weather_desc(天气描述)
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(WEATHER_URL, params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 1,
            })
            resp.raise_for_status()
            data = resp.json()
            cur = data["current"]
            return {
                "temperature": cur["temperature_2m"],
                "humidity": cur["relative_humidity_2m"],
                "wind_speed": cur["wind_speed_10m"],
                "weather_code": cur["weather_code"],
                "weather_desc": WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}"),
                "latitude": latitude,
                "longitude": longitude,
            }
    except httpx.RequestError as e:
        return {"error": f"天气数据获取失败：{e}"}


def get_forecast(latitude: float, longitude: float, days: int = 3) -> dict:
    """
    根据经纬度获取未来天气预报。

    Args:
        latitude: 纬度
        longitude: 经度
        days: 预报天数，默认3天，最多14天

    Returns:
        预报数据字典，包含：latitude, longitude, daily(每日预报列表)
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(WEATHER_URL, params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": min(days, 14),
            })
            resp.raise_for_status()
            data = resp.json()
            daily = data["daily"]
            forecast_list = []
            for i in range(len(daily["time"])):
                forecast_list.append({
                    "date": daily["time"][i],
                    "max_temp": daily["temperature_2m_max"][i],
                    "min_temp": daily["temperature_2m_min"][i],
                    "precipitation": daily["precipitation_sum"][i],
                    "weather_code": daily["weather_code"][i],
                    "weather_desc": WEATHER_CODE_MAP.get(daily["weather_code"][i], ""),
                })
            return {
                "latitude": latitude,
                "longitude": longitude,
                "daily": forecast_list,
            }
    except httpx.RequestError as e:
        return {"error": f"预报数据获取失败：{e}"}


def get_weather(city: str) -> str:
    """
    （聚合方法）查询指定城市的当前天气及未来3天预报。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    cities = search_city(city)
    if not cities:
        return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

    def _rank(r):
        fc = str(r.get("feature_code", ""))
        admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
        pop = r.get("population") or 0
        return (admin_priority, pop)

    loc = max(cities, key=_rank)
    lat, lon = loc["latitude"], loc["longitude"]

    current = get_current_weather(lat, lon)
    if "error" in current:
        return current["error"]

    forecast = get_forecast(lat, lon, 3)
    if "error" in forecast:
        return forecast["error"]

    location_str = f"{loc.get('country', '')} {loc.get('admin1', '')} {loc.get('name', city)}".strip()

    lines = [
        f"【{location_str}】天气报告",
        f"坐标：{lat:.2f}°N, {lon:.2f}°E",
        "",
        f"当前天气：{current['weather_desc']}",
        f"  温度：{current['temperature']}°C",
        f"  相对湿度：{current['humidity']}%",
        f"  风速：{current['wind_speed']} km/h",
        "",
        "未来3天预报：",
    ]
    for day in forecast["daily"]:
        lines.append(
            f"  {day['date']}：{day['weather_desc']}，"
            f"{day['max_temp']}°C / {day['min_temp']}°C，"
            f"降水 {day['precipitation']} mm"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    print(get_weather(args.city))