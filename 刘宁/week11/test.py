import httpx
from typing import Tuple, Dict, Any, Optional

# 常量定义
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CODE_MAP = {
    0: "晴天",
    1: "大部晴朗", 2: "多云", 3: "阴天",
    45: "雾", 48: "雾凇",
    51: "毛毛雨", 53: "小雨", 55: "中雨", 56: "冻毛毛雨", 57: "冻雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "冻大雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "阵雪", 86: "强阵雪",
    95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴伴冰雹"
}


class WeatherService:
    def __init__(self):
        self.client = httpx.Client(timeout=10.0)
        self.city: str = ""
        self.lat: Optional[float] = None
        self.lon: Optional[float] = None
        self.location_info: Dict[str, Any] = {}
        self.weather_result: str = ""

    def _geocode(self, name: str) -> list:
        resp = self.client.get(GEOCODING_URL, params={
            "name": name, "count": 10, "language": "zh", "format": "json",
        })
        resp.raise_for_status()
        return resp.json().get("results") or []

    def geocode(self, city: str) -> "WeatherService":
        """
        链式方法：查询城市经纬度信息，保存结果
        """
        self.city = city
        results = self._geocode(city)

        # 低级行政区则重试加"市"后缀
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = self._geocode(city + "市")
            if retry:
                results = retry

        if not results:
            raise ValueError(f"未找到城市 '{city}'，请尝试其他写法（如加上市/县）")

        # 优先级排序选最佳地点
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        self.lat = loc["latitude"]
        self.lon = loc["longitude"]
        self.location_info = loc
        return self

    def get_lat_lon(self) -> Tuple[Optional[float], Optional[float]]:
        """单独获取经纬度"""
        return self.lat, self.lon

    def get_location_info(self) -> Dict[str, Any]:
        """单独获取完整地址信息"""
        return self.location_info

    def fetch_weather(self) -> "WeatherService":
        """
        链式方法：基于已获取的经纬度查询天气
        """
        if self.lat is None or self.lon is None:
            raise RuntimeError("请先调用 .geocode(城市名) 获取经纬度！")

        weather_resp = self.client.get(WEATHER_URL, params={
            "latitude": self.lat,
            "longitude": self.lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "Asia/Shanghai",
            "forecast_days": 3,
        })
        weather_resp.raise_for_status()
        data = weather_resp.json()
        cur = data["current"]
        daily = data["daily"]

        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        city_name = self.location_info.get("name", self.city)
        country = self.location_info.get("country", "")
        admin1 = self.location_info.get("admin1", "")
        location_str = f"{country} {admin1} {city_name}".strip()

        lines = [
            f"【{location_str}】天气报告",
            f"坐标：{self.lat:.2f}°N, {self.lon:.2f}°E",
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
        self.weather_result = "\n".join(lines)
        return self

    def get_weather_result(self) -> str:
        """获取最终天气文本"""
        return self.weather_result

    def close(self):
        """关闭HTTP连接"""
        self.client.close()


if __name__ == "__main__":
    service = WeatherService()

    try:
        #循环交互查询
        print("\n===== 循环天气查询（输入 q 退出）=====")
        while True:
            city_input = input("\n输入城市：").strip()
            if city_input.lower() in ("q", "quit", "exit", "退出"):
                print("结束查询")
                break
            if not city_input:
                continue
            res = (
                service.geocode(city_input)
                       .fetch_weather()
                       .get_weather_result()
            )
            print("\n" + res)

    except Exception as e:
        print("错误：", e)
    finally:
        service.close()