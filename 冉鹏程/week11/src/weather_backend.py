"""Open-Meteo 地理位置与天气查询后端。

本模块刻意将两个外部能力解耦：``get_location`` 只负责城市名到坐标的解析，
``get_weather`` 只负责根据经纬度查询天气。所有公开接口返回统一 JSON 对象：
``{"success": bool, "data": object | None, "errMsg": str}``。
"""

import logging
import math
from typing import Any, TypeAlias

import httpx


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT_SECONDS = 10.0
ApiResponse: TypeAlias = dict[str, Any]
LOGGER = logging.getLogger(__name__)

# Open-Meteo WMO 天气代码 → 中文说明，避免调用方重复维护映射。
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天", 45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨", 61: "小雨", 63: "中雨",
    65: "大雨", 71: "小雪", 73: "中雪", 75: "大雪", 80: "小阵雨", 81: "中阵雨",
    82: "大阵雨", 95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def _success(data: Any) -> ApiResponse:
    """构造成功响应。

    Args: data: 需要返回的 JSON 兼容数据。
    Returns: 统一成功响应。
    """
    return {"success": True, "data": data, "errMsg": ""}


def _failure(message: str) -> ApiResponse:
    """构造失败响应。

    Args: message: 可安全展示给调用方的错误原因。
    Returns: 统一失败响应，data 固定为 None。
    """
    return {"success": False, "data": None, "errMsg": message}


def _request_locations(client: httpx.Client, city: str) -> list[dict[str, Any]]:
    """请求地理编码服务。

    Args: client: 已配置超时的 HTTP 客户端；city: 已校验的城市名称。
    Returns: 候选地点列表。
    Raises: httpx.HTTPError: 网络异常或非成功 HTTP 状态码。
    """
    LOGGER.info("event=geocoding_request city_length=%d", len(city))
    response = client.get(GEOCODING_URL, params={
        "name": city, "count": 10, "language": "zh", "format": "json",
    })
    response.raise_for_status()
    return response.json().get("results") or []


def _select_location(results: list[dict[str, Any]]) -> dict[str, Any]:
    """选择行政等级最高、人口最多的同名地点。

    Args: results: 地理编码服务返回的非空候选列表。
    Returns: 最匹配的地点对象。
    Raises: ValueError: 候选列表为空。
    """
    def rank(item: dict[str, Any]) -> tuple[int, int]:
        feature_code = str(item.get("feature_code", ""))
        is_administrative_location = feature_code.startswith(("PPLA", "ADM"))
        return int(is_administrative_location), int(item.get("population") or 0)

    return max(results, key=rank)


def _normalise_location(city: str, results: list[dict[str, Any]]) -> ApiResponse:
    """筛选地点并提取下游天气查询需要的坐标字段。"""
    if not results:
        return _failure(f"未找到城市“{city}”，请尝试使用更完整的名称，例如“宁德市”")
    try:
        location = _select_location(results)
        return _success({
            "input": city, "name": location.get("name", city),
            "country": location.get("country", ""), "admin1": location.get("admin1", ""),
            "latitude": location["latitude"], "longitude": location["longitude"],
            "timezone": location.get("timezone", ""),
        })
    except (KeyError, TypeError, ValueError) as exc:
        LOGGER.warning("event=geocoding_invalid_payload error=%s", type(exc).__name__)
        return _failure(f"地理位置服务返回了无效数据：{exc}")


def get_location(city: str) -> ApiResponse:
    """将城市名称解析为标准地理位置。

    Args: city: 城市中文名称，例如“宁德”或“北京”。
    Returns: 成功时 data 含 name、latitude、longitude、timezone；失败时 data 为 None。
    Raises: 无。网络和数据异常均转换为统一失败响应。
    """
    if not isinstance(city, str):
        return _failure("城市名称必须是字符串")
    city = city.strip()
    if not city:
        return _failure("城市名称不能为空")
    if len(city) > 100 or any(character in city for character in "\r\n\t"):
        return _failure("城市名称格式不合法")

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
            results = _request_locations(client, city)
            # 为什么：裸地名可能先命中同名村庄；追加“市”能提高城市级行政区的命中率。
            # TODO: 如需支持海外行政区，可改为依据用户传入的国家代码进行二次筛选。
            is_only_low_level_places = bool(results) and all(
                str(item.get("feature_code", "")).startswith("PPL")
                and not str(item.get("feature_code", "")).startswith("PPLA") for item in results
            )
            if is_only_low_level_places and not city.endswith(("市", "县", "区", "镇")):
                city_results = _request_locations(client, f"{city}市")
                results = city_results or results
    except httpx.HTTPError as exc:
        LOGGER.warning("event=geocoding_failed error=%s", type(exc).__name__)
        return _failure(f"地理位置查询失败：{exc}")
    except (AttributeError, TypeError, ValueError) as exc:
        LOGGER.warning("event=geocoding_invalid_json error=%s", type(exc).__name__)
        return _failure(f"地理位置服务返回了无效数据：{exc}")
    result = _normalise_location(city, results)
    LOGGER.info("event=geocoding_completed success=%s", result["success"])
    return result


def _validate_coordinates(latitude: float, longitude: float) -> str | None:
    """校验经纬度的类型、有限性和地理范围，返回错误文本或 None。"""
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        return "经纬度必须是数字"
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return "经纬度必须是数字"
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return "经纬度必须是有限数字"
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return "纬度范围必须为 -90 到 90，经度范围必须为 -180 到 180"
    return None


def _build_weather_data(weather_data: dict[str, Any], latitude: float, longitude: float) -> dict[str, Any]:
    """将上游天气 JSON 转换为稳定的应用数据结构。"""
    current = weather_data["current"]
    daily = weather_data["daily"]
    forecast = []
    for index in range(min(3, len(daily["time"]))):
        weather_code = daily["weather_code"][index]
        forecast.append({
            "date": daily["time"][index], "weatherCode": weather_code,
            "weather": WEATHER_CODE_MAP.get(weather_code, f"代码{weather_code}"),
            "temperatureMax": daily["temperature_2m_max"][index],
            "temperatureMin": daily["temperature_2m_min"][index],
            "precipitation": daily["precipitation_sum"][index],
        })
    weather_code = current["weather_code"]
    return {
        "location": {"latitude": weather_data.get("latitude", latitude),
                     "longitude": weather_data.get("longitude", longitude),
                     "timezone": weather_data.get("timezone", "")},
        "current": {"time": current.get("time", ""), "weatherCode": weather_code,
                    "weather": WEATHER_CODE_MAP.get(weather_code, f"代码{weather_code}"),
                    "temperature": current["temperature_2m"],
                    "relativeHumidity": current["relative_humidity_2m"],
                    "windSpeed": current["wind_speed_10m"]},
        "forecast": forecast,
    }


def get_weather(latitude: float, longitude: float, timezone: str = "auto") -> ApiResponse:
    """按经纬度查询当前天气及未来三天预报，不接受城市名称。

    Args: latitude: 纬度，范围 -90 至 90；longitude: 经度，范围 -180 至 180；
        timezone: IANA 时区名，默认 auto。
    Returns: 成功时 data 含坐标、当前天气和 forecast；失败时 data 为 None。
    Raises: 无。参数、网络和上游数据异常均转换为统一失败响应。
    """
    validation_error = _validate_coordinates(latitude, longitude)
    if validation_error:
        return _failure(validation_error)
    if not isinstance(timezone, str) or not timezone.strip():
        return _failure("时区必须是非空字符串")
    try:
        LOGGER.info("event=weather_request latitude=%.4f longitude=%.4f", latitude, longitude)
        with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = client.get(WEATHER_URL, params={
                "latitude": latitude, "longitude": longitude, "timezone": timezone.strip(),
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "forecast_days": 3,
            })
            response.raise_for_status()
            result = _success(_build_weather_data(response.json(), latitude, longitude))
            LOGGER.info("event=weather_completed success=true")
            return result
    except httpx.HTTPError as exc:
        LOGGER.warning("event=weather_request_failed error=%s", type(exc).__name__)
        return _failure(f"天气数据获取失败：{exc}")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        LOGGER.warning("event=weather_invalid_payload error=%s", type(exc).__name__)
        return _failure(f"天气服务返回了无效数据：{exc}")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="按经纬度查询天气")
    parser.add_argument("--latitude", required=True, type=float)
    parser.add_argument("--longitude", required=True, type=float)
    parser.add_argument("--timezone", default="auto")
    args = parser.parse_args()
    print(json.dumps(get_weather(args.latitude, args.longitude, args.timezone), ensure_ascii=False, indent=2))
