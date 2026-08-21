
"""
根据输入的城市名称查询经纬度
"""

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

def query_geo(city_name: str):
    """
    根据输入的城市名称查询经纬度
    Args:
        city_name: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"
    Returns:
        (latitude, longitude): 经纬度元组
    """
    with httpx.Client(timeout=10.0) as client:
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []
        results = _geocode(city_name)
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city_name.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city_name + "市")
            if retry:
                results = retry

        if not results:
            raise Exception(f"未找到城市 '{city_name}'，请尝试其他写法（如'宁德市'改'宁德'）")

        # 在候选里优先取行政级别更高的（feature_code 含 A = 某级政府驻地），
        # 其次取有人口数据的，避免落到同名小村庄
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        lat = loc["latitude"]
        lon = loc["longitude"]
        return lat, lon

if __name__ == '__main__':
    test_cities = ["北京"]
    for city in test_cities:
        try:
            lat, lon = query_geo(city)
            print(f"{city}的经纬度为：{lat:.4f}, {lon:.4f}")
        except Exception as e:
            print(f"查询{city}失败: {e}")
