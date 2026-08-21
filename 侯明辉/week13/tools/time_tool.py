"""
time_tool.py — 当前时间查询（本地 datetime 实现，无外部 API）

时区用 IANA 名（如 Asia/Shanghai、UTC、America/New_York），
依赖 Python 3.9+ stdlib zoneinfo，跨平台一致。
"""

from datetime import datetime

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # type: ignore


def get_time(zone: str) -> str:
    """
    查询指定时区的当前时间。

    Args:
        zone: IANA 时区名，如 'Asia/Shanghai'、'UTC'、'America/New_York'

    Returns:
        包含 ISO 时间、星期、Unix 时间戳的文字描述
    """
    try:
        tz = ZoneInfo(zone)
    except (ZoneInfoNotFoundError, KeyError):
        return f"错误：不支持的时区 '{zone}'，请用 IANA 名（如 'Asia/Shanghai'、'UTC'）"

    now = datetime.now(tz)
    iso = now.isoformat(timespec="seconds")
    weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
    unix_ts = int(now.timestamp())

    return (
        f"【{zone}】当前时间\n"
        f"  ISO 格式：{iso}\n"
        f"  星期：{weekday_cn}\n"
        f"  Unix 时间戳：{unix_ts}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="Asia/Shanghai")
    args = parser.parse_args()
    print(get_time(args.zone))
