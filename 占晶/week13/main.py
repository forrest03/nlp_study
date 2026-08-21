from __future__ import annotations

import os
from pathlib import Path

from harness import AgentHarness
from skill_loader import SkillRegistry


PROJECT_ROOT = Path(__file__).resolve().parent


def print_help() -> None:
    print(
        """
命令：
  /help          显示帮助
  /skills        显示 Skill 元数据目录
  /active        显示当前激活的 Skill
  /reset         清空对话和当前 Skill
  /quit          退出

显式调用示例：
  $setup-ffmpeg 检查一下 FFmpeg
  $add-video-subtitles 给 D:\\videos\\demo.mp4 加中文字幕
""".strip()
    )


def main() -> int:
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("未设置 DEEPSEEK_API_KEY。")
        print('$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"')
        return 1

    registry = SkillRegistry.discover(PROJECT_ROOT / "skills")
    agent = AgentHarness(registry)
    print("本地视频 Agent 已启动。输入 /help 查看命令，/quit 退出。")
    print(f"模型：{agent.model}")

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0

        if not user_input:
            continue
        if user_input in {"/quit", "/exit", "quit", "exit"}:
            print("再见。")
            return 0
        if user_input == "/help":
            print_help()
            continue
        if user_input == "/skills":
            print(agent.list_skills())
            continue
        if user_input == "/active":
            active = agent.session.active_skill
            print(active.name if active else "当前没有激活 Skill")
            continue
        if user_input == "/reset":
            agent.reset()
            print("对话和 Skill 状态已清空。")
            continue

        try:
            print(f"\nAgent：{agent.run_turn(user_input)}")
        except Exception as exc:
            print(f"\n[错误] {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
