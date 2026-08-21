"""计算器脚本：从 stdin 读取 JSON 参数，计算表达式并输出结果"""
import sys
import json
import math


def main():
    # 从 stdin 读取参数
    raw = sys.stdin.read()
    try:
        params = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        print("错误：参数 JSON 格式无效")
        return

    expr = params.get("expr")
    if not expr:
        print("错误：缺少必填参数 expr")
        return

    # 安全的数学计算环境
    safe_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    safe_names.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})

    try:
        result = eval(expr, {"__builtins__": {}}, safe_names)
        print(round(float(result), 6))
    except Exception as e:
        print(f"计算出错: {e}，表达式: {expr}")


if __name__ == "__main__":
    main()
