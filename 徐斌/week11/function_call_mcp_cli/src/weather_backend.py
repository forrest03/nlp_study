"""
weather_backend.py — 天气查询后端（三种方式共享的业务逻辑）

教学重点：
  1. 纯业务逻辑，被 Function Call / MCP / CLI 三种方式复用
  2. 拆成两步工具：Geocoding（城市名→经纬度）+ 按坐标查天气
     ——宿主侧用多轮循环让模型先 geocode 再 get_weather_by_coords
  3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
  from src.weather_backend import geocode_city, get_weather_by_coords, get_weather
  print(geocode_city("宁德"))
  print(get_weather_by_coords(26.67, 119.52))
  print(get_weather("宁德"))  # 兼容一步到位（CLI 便捷入口）

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import json

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


def _pick_location(city: str, results: list[dict]) -> dict:
    """在候选里优先取行政级别更高的，其次取有人口数据的。"""
    def _rank(r):
        fc = str(r.get("feature_code", ""))
        admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
        pop = r.get("population") or 0
        return (admin_priority, pop)

    return max(results, key=_rank)


def geocode_city(city: str) -> str:
    """
    将城市名解析为经纬度坐标（供下一步 get_weather_by_coords 使用）。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        JSON 字符串，含 latitude / longitude / name / country / admin1；
        失败时返回可读错误信息。
    """
    with httpx.Client(timeout=10.0) as client:
        # 中国地名常有歧义：裸"宁德"会命中西藏那曲市的一个村（PPL），
        # 而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
        # 策略：先按用户输入查；若命中的只是低级行政点（feature_code 纯 PPL），
        # 且用户没带"市/县/区"后缀，就用 city+"市" 重查一次并优先采用。
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        try:
            results = _geocode(city)
        except httpx.RequestError as e:
            return f"地理编码失败：{e}"

        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            try:
                retry = _geocode(city + "市")
            except httpx.RequestError as e:
                return f"地理编码失败：{e}"
            if retry:
                results = retry

        if not results:
            return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

        loc = _pick_location(city, results)
        payload = {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "name": loc.get("name", city),
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""),
            "feature_code": loc.get("feature_code", ""),
        }
        return json.dumps(payload, ensure_ascii=False)


def get_weather_by_coords(
    latitude: float,
    longitude: float,
    location_name: str = "",
) -> str:
    """
    按经纬度查询当前天气及未来3天预报。

    Args:
        latitude: 纬度
        longitude: 经度
        location_name: 可选，展示用地点名（如 geocode 返回的 name）

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = weather_resp.json()
        cur = data["current"]
        daily = data["daily"]

        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        title = location_name.strip() or f"{latitude:.2f}°N, {longitude:.2f}°E"

        lines = [
            f"【{title}】天气报告",
            f"坐标：{latitude:.2f}°N, {longitude:.2f}°E",
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


def get_weather(city: str) -> str:
    """
    一步查询城市天气（内部串联 geocode + weather，供 CLI 便捷入口）。

    教学场景下，Function Call / MCP 应拆成两步工具由模型循环调用；
    本函数保留给 `fincli weather --city` 等直接使用。
    """
    geo_raw = geocode_city(city)
    try:
        geo = json.loads(geo_raw)
    except json.JSONDecodeError:
        return geo_raw  # 错误信息原样返回

    location_str = f"{geo.get('country', '')} {geo.get('admin1', '')} {geo.get('name', city)}".strip()
    return get_weather_by_coords(
        latitude=float(geo["latitude"]),
        longitude=float(geo["longitude"]),
        location_name=location_str,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_geo = sub.add_parser("geocode", help="城市名 → 坐标")
    p_geo.add_argument("--city", required=True)

    p_wx = sub.add_parser("weather-by-coords", help="按坐标查天气")
    p_wx.add_argument("--latitude", type=float, required=True)
    p_wx.add_argument("--longitude", type=float, required=True)
    p_wx.add_argument("--location-name", default="")

    p_one = sub.add_parser("weather", help="一步查城市天气")
    p_one.add_argument("--city", required=True)

    args = parser.parse_args()
    if args.cmd == "geocode":
        print(geocode_city(args.city))
    elif args.cmd == "weather-by-coords":
        print(get_weather_by_coords(args.latitude, args.longitude, args.location_name))
    else:
        print(get_weather(args.city))
