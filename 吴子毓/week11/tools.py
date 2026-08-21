import requests


def geocode(city_name: str) -> dict:
    """城市名 → 经纬度：将城市名称转换为地理坐标

    含同名小村庄消歧策略（移植自 src/weather_backend.py）：
      - 取 count=10 个候选，按行政级别（PPLA/ADM 优先）+ 人口排序选最优
      - 若候选全是低级行政点（纯 PPL，非 PPLA）且用户没带"市/县/区/镇"后缀，
        用 city_name+"市" 重查一次并优先采用
      - 典型案例：裸"宁德"会命中西藏那曲的一个村（PPL），而福建宁德是
        地级市"宁德市"（PPLA2），重查"宁德市"才能命中正确地点
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"

    def _fetch(name: str):
        """发起一次 Geocoding 请求，返回候选列表"""
        params = {
            "name": name,
            "count": 10,
            "language": "zh",
            "format": "json",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("results") or []

    # 第一次查询
    results = _fetch(city_name)

    # 消歧：候选全是低级行政点（纯 PPL，非 PPLA）且没带"市/县/区/镇"后缀 → 用 city+"市" 重查
    is_low_admin = all(
        str(r.get("feature_code", "")).startswith("PPL")
        and not str(r.get("feature_code", "")).startswith("PPLA")
        for r in results
    ) if results else True
    has_suffix = any(city_name.endswith(s) for s in ("市", "县", "区", "镇"))
    if is_low_admin and not has_suffix:
        retry = _fetch(city_name + "市")
        if retry:
            results = retry

    if not results:
        raise ValueError(f"未找到城市: {city_name}")

    # 按行政级别（PPLA/ADM 优先）+ 人口排序，选最优候选
    def _rank(r):
        fc = str(r.get("feature_code", ""))
        admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
        pop = r.get("population") or 0
        return (admin_priority, pop)

    result = max(results, key=_rank)
    return {
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "city": result.get("name", city_name),
        "country": result.get("country", ""),
        "admin1": result.get("admin1", "")
    }


def get_weather(latitude: float, longitude: float) -> dict:
    """经纬度 → 天气：根据地理坐标获取当前及每日天气信息"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,weather_code",
        "timezone": "auto"
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    current = data.get("current", {})
    daily = data.get("daily", {})

    daily_max = daily.get("temperature_2m_max", [None])
    daily_min = daily.get("temperature_2m_min", [None])
    daily_code = daily.get("weather_code", [None])

    return {
        "current": {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "weather_code": current.get("weather_code"),
            "wind_speed": current.get("wind_speed_10m")
        },
        "daily": {
            "max_temp": daily_max[0] if daily_max else None,
            "min_temp": daily_min[0] if daily_min else None,
            "weather_code": daily_code[0] if daily_code else None
        }
    }
