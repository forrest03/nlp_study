"""
weather_query.py — 天气查询后端（三种方式共享的业务逻辑）

教学重点：
  1. 同样是"纯业务逻辑"，与 rag_backend 平级，被三种方式复用
  2. 内部两次 HTTP 请求：Geocoding（城市名→经纬度）+ 天气查询
  3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
  from src.weather_query import get_weather, get_geo
  print(get_weather("宁德"))

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


def get_geo(city: str) -> dict | None:
    """
    根据城市名获取地理位置信息（经纬度、行政级别等）。

    Args:
        city: 城市名称，支持中文

    Returns:
        包含 latitude, longitude, name, country, admin1 的字典；未找到返回 None
    """
    with httpx.Client(timeout=10.0) as client:
        def _geocode(name: str):
            try:
                resp = client.get(GEOCODING_URL, params={
                    "name": name, "count": 10, "language": "zh", "format": "json",
                })
                resp.raise_for_status()
                return resp.json().get("results") or []
            except httpx.HTTPError:
                return []

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
            return None

        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        return {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "name": loc.get("name", city),
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""),
        }


def get_weather_data(lat: float, lon: float) -> dict | None:
    """
    根据经纬度获取天气数据。

    Args:
        lat: 纬度
        lon: 经度

    Returns:
        包含 current 和 daily 的天气数据字典；获取失败返回 None
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
            return weather_resp.json()
        except httpx.HTTPError:
            return None


def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    loc = get_geo(city)
    if not loc:
        return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

    data = get_weather_data(loc["latitude"], loc["longitude"])
    if not data:
        return "天气数据获取失败"

    cur = data["current"]
    daily = data["daily"]

    weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
    location_str = f"{loc['country']} {loc['admin1']} {loc['name']}".strip()

    lines = [
        f"【{location_str}】天气报告",
        f"坐标：{loc['latitude']:.2f}°N, {loc['longitude']:.2f}°E",
        "",
        f"当前天气：{weather_desc}",
        f"  温度：{cur['temperature_2m']}°C",
        f"  相对湿度：{cur['relative_humidity_2m']}%",
        f"  风速：{cur['wind_speed_10m']} km/h",
        "",
        "未来3天预报：",
    ]
    for i in range(3):
        day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
        lines.append(
            f"  {daily['time'][i]}：{day_desc}，"
            f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
            f"降水 {daily['precipitation_sum'][i]} mm"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    print(get_weather(args.city))
