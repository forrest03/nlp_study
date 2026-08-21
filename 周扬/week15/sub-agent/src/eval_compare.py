"""同一通用任务分别跑并行和串行，量化 sub-Agent 的并行收益。"""

import time

from src.agents import run_task


QUESTIONS = [
    "帮我制定杭州三天旅行计划，需要交通、住宿区域和景点安排",
    "帮我比较 Python、Java 和 Go，分别说明学习难度、典型用途和适合的人群",
]


def run_one(question, serial):
    start_time = time.time()
    result = run_task(question, serial=serial)
    stat = result["parallel_stats"][-1] if result["parallel_stats"] else {}
    return {
        "total_time": round(time.time() - start_time, 2),
        "subagent_count": len(result["subagents"]),
        **stat,
    }


def main():
    for question in QUESTIONS:
        print("\n" + "=" * 66)
        print("问题：" + question)
        parallel = run_one(question, serial=False)
        serial = run_one(question, serial=True)
        print("子 Agent 数量：" + str(parallel["subagent_count"]))
        print("并行总耗时：" + str(parallel["total_time"]) + " 秒")
        print("串行总耗时：" + str(serial["total_time"]) + " 秒")
        print("并行调度统计：" + str(parallel))


if __name__ == "__main__":
    main()
