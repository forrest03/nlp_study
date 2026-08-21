"""
weather_loop.py — 天气查询循环调用（基于 weather_backend）
"""

from weather_backend import get_weather

def main():
    print("天气查询助手启动（输入 '退出' 结束）\n")
    while True:
        city = input("请输入城市名（如：北京）：").strip()
        if city.lower() in ["退出", "exit", "quit", "q"]:
            print("再见！")
            break
        if not city:
            continue
        
        print("\n正在查询...\n")
        result = get_weather(city)   # 调用业务函数
        print(result)
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
