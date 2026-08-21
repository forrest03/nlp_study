"""
main.py — fincli：天气查询命令行入口

把 src/ 后端能力封装成一条真实命令。通过 pyproject.toml 的 [project.scripts]
注册为 console_script，`pip install -e .` 后即可全局调用：

  fincli geocode --city 宁德
  fincli weather-by-coords --latitude 26.67 --longitude 119.52 --location-name 宁德
  fincli weather --city 宁德   # 一步到位便捷入口

不想安装也可直接跑：
  python -m mode_cli.cli.main geocode --city 宁德

依赖：
  pip install httpx
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.weather_backend import geocode_city, get_weather, get_weather_by_coords  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        prog="fincli",
        description="fincli — 天气查询命令行工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_geo = sub.add_parser("geocode", help="城市名 → 经纬度（查天气第一步）")
    p_geo.add_argument("--city", required=True, help="城市中文名，如 宁德")

    p_wx = sub.add_parser("weather-by-coords", help="按坐标查天气（查天气第二步）")
    p_wx.add_argument("--latitude", type=float, required=True, help="纬度")
    p_wx.add_argument("--longitude", type=float, required=True, help="经度")
    p_wx.add_argument("--location-name", default="", help="可选展示用地点名")

    p_weather = sub.add_parser("weather", help="一步查询城市天气（内部串联 geocode+预报）")
    p_weather.add_argument("--city", required=True, help="城市中文名，如 宁德")

    args = parser.parse_args()

    if args.cmd == "geocode":
        print(geocode_city(args.city))
    elif args.cmd == "weather-by-coords":
        print(get_weather_by_coords(args.latitude, args.longitude, args.location_name))
    elif args.cmd == "weather":
        print(get_weather(args.city))


if __name__ == "__main__":
    main()
