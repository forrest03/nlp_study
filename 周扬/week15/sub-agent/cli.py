"""通用 Sub-Agent 的命令行交互入口。"""

from src.agents import run_task


def print_step(step):
    """将一个 Agent 的 ReAct 步骤实时输出到终端。"""
    print("\n[" + step["agent"] + " | 第 " + str(step["idx"] + 1) + " 步]")
    print("Thought: " + step["thought"])
    print("Action: " + step["action"])
    print("Action Input: " + step["action_input"])
    if step.get("observation"):
        print("Observation: " + step["observation"][:800])


def main():
    print("=" * 66)
    print("通用任务 Sub-Agent CLI")
    print("主 Agent 工具：web_search、dispatch_subagents")
    print("子 Agent 工具：web_search（不能继续分发）")
    print("输入任务开始，输入 exit、quit 或 退出结束。")
    print("=" * 66)

    while True:
        question = input("\n你：").strip()
        if question.lower() in ["exit", "quit", "退出"]:
            print("再见！")
            return
        if not question:
            print("请输入一个任务。")
            continue

        try:
            result = run_task(
                question,
                on_main_step=print_step,
                on_subagent_step=lambda _, step: print_step(step),
                on_dispatch=lambda info: print("\n[主 Agent 分发] " + " | ".join(info["subtopics"])),
                on_subagent_done=lambda sid, duration, topic: print(
                    "[" + sid + "] 完成：" + topic + "，用时 " + str(duration) + " 秒"
                ),
            )
        except Exception as error:
            print("\n运行失败：" + str(error))
            print("请检查 config.py 中的 DEEPSEEK_API_KEY 与 TAVILY_API_KEY。")
            continue

        print("\n" + "=" * 66)
        print("最终答案")
        print(result["final_answer"])
        if result["parallel_stats"]:
            print("\n并行统计：" + str(result["parallel_stats"][-1]))
        print("=" * 66)


if __name__ == "__main__":
    main()
