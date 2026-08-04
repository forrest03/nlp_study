"""
main.py - fincli：天气查询 统一命令行入口

把 src/ 后端能力封装成一条"看起来像 git/ls 那样"的真实命令，而不是
`python xxx.py ...`。通过 pyproject.toml 的 [project.scripts] 注册为
console_script，`pip install -e .` 后即可全局调用：

  fincli geocode --city 宁德           # 城市名 -> 经纬度（输出 JSON）
  fincli weather --lat 26.66 --lon 119.52 --name 宁德市 --country 中国 --admin1 福建省

不想安装也可直接跑：
  python mode_cli/cli/main.py geocode --city 宁德
  python mode_cli/cli/main.py weather --lat 26.66 --lon 119.52

教学点：
  1. CLI 作为"工具实现层"，本质就是一个能跑的脚本--跟协议无关
  2. 用 pyproject + console_script 把脚本变成 PATH 上的真实命令，是 Python CLI 工具的标准发布方式
  3. 一个 fincli 含子命令（geocode / weather），对应 git 的子命令设计

依赖：
  pip install httpx
"""

import argparse
import json
import sys
from pathlib import Path

# 让本脚本能 import 项目根的 src/（无论从哪个工作目录 / 是否安装）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.weather_backend import geocode, get_weather_by_coords  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        prog="fincli",
        description="fincli - 天气查询 命令行工具（两步：geocode 拿坐标，weather 查预报）",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # fincli geocode --city 宁德  -> 输出 JSON {lat, lon, name, country, admin1} 或 null
    p_geo = sub.add_parser("geocode", help="城市名 -> 经纬度（输出 JSON）")
    p_geo.add_argument("--city", required=True, help="城市中文名，如 宁德")

    # fincli weather --lat --lon [--name --country --admin1]  -> 输出天气预报
    p_w = sub.add_parser("weather", help="经纬度 -> 天气预报")
    p_w.add_argument("--lat", type=float, required=True, help="纬度")
    p_w.add_argument("--lon", type=float, required=True, help="经度")
    p_w.add_argument("--name", default="", help="城市名（可选，用于报告标题）")
    p_w.add_argument("--country", default="", help="国家（可选）")
    p_w.add_argument("--admin1", default="", help="省/州（可选）")

    args = parser.parse_args()

    if args.cmd == "geocode":
        loc = geocode(args.city)
        print(json.dumps(loc, ensure_ascii=False) if loc is not None else "null")
    elif args.cmd == "weather":
        print(get_weather_by_coords(args.lat, args.lon, args.name, args.country, args.admin1))


if __name__ == "__main__":
    main()
