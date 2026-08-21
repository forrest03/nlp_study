from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import media_tools
from skill_loader import Skill, SkillRegistry
from tool_definitions import CORE_TOOLS, SKILL_TOOLS


BASE_INSTRUCTIONS = """
你是一个运行在命令行中的本地视频助手。

工作方式：
1. 先理解用户目标。
2. 如果尚未激活 Skill，并且请求匹配 Skill 目录，调用 activate_skill。
3. Skill 激活后严格遵循完整 Skill 指令，逐步调用工具。
4. 每次工具调用后读取工具结果，再决定下一步；不要假设工具成功。
5. 工具失败时根据结构化错误修正计划，必要时向用户提问。
6. 不要编造本机状态、模型状态、文件路径或工具结果。
7. 安装软件和下载模型由 Harness 在终端进行最终确认。
8. 使用中文简洁回复。

当工作流真正完成或用户取消时调用 complete_skill。询问中间问题时不要结束 Skill。
""".strip()


@dataclass
class Session:
    active_skill: Skill | None = None
    active_instructions: str | None = None


class AgentHarness:
    def __init__(
        self,
        registry: SkillRegistry,
        *,
        model: str | None = None,
        client: Any | None = None,
        max_steps: int = 12,
    ):
        self.registry = registry
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.max_steps = max_steps
        self.session = Session()
        self.messages: list[Any] = []

        if client is None:
            from openai import OpenAI

            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("未设置 DEEPSEEK_API_KEY")
            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv(
                    "DEEPSEEK_BASE_URL",
                    "https://api.deepseek.com",
                ),
            )
        self.client = client

    def list_skills(self) -> str:
        return self.registry.catalog_text()

    def reset(self) -> None:
        self.session = Session()
        self.messages.clear()

    def activate_explicit_skill(self, user_input: str) -> None:
        stripped = user_input.strip()
        if not stripped.startswith("$"):
            return
        name = stripped[1:].split(maxsplit=1)[0]
        skill = self.registry.get(name)
        if skill:
            self.session.active_skill = skill
            self.session.active_instructions = self.registry.load_body(skill)

    def _instructions(self) -> str:
        sections = [BASE_INSTRUCTIONS, self.registry.catalog_text()]
        if self.session.active_skill:
            sections.append(
                "\n".join(
                    [
                        f'<active_skill name="{self.session.active_skill.name}">',
                        self.session.active_instructions or "",
                        "</active_skill>",
                    ]
                )
            )
        else:
            sections.append(
                "当前没有激活 Skill。需要专业工作流时，先调用 activate_skill。"
            )
        return "\n\n".join(sections)

    def _available_tools(self) -> list[dict]:
        if not self.session.active_skill:
            return CORE_TOOLS
        return SKILL_TOOLS.get(self.session.active_skill.name, CORE_TOOLS)

    def _confirm(self, title: str, detail: str) -> bool:
        print(f"\n[需要确认] {title}")
        print(detail)
        answer = input("是否继续？[y/N] ").strip().lower()
        return answer in {"y", "yes"}

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        print(f"[工具] {name} {json.dumps(arguments, ensure_ascii=False)}")

        if name == "activate_skill":
            skill = self.registry.get(arguments["name"])
            if skill is None:
                return {
                    "ok": False,
                    "error": {
                        "code": "SKILL_NOT_FOUND",
                        "message": f"未知 Skill：{arguments['name']}",
                        "retryable": False,
                    },
                }
            self.session.active_skill = skill
            self.session.active_instructions = self.registry.load_body(skill)
            return {
                "ok": True,
                "skill": skill.name,
                "message": "完整 Skill 指令已加载；请依据新指令重新规划。",
            }

        if name == "complete_skill":
            previous = self.session.active_skill.name if self.session.active_skill else None
            self.session = Session()
            return {"ok": True, "completed_skill": previous}

        if name == "read_skill_reference":
            if not self.session.active_skill:
                return {
                    "ok": False,
                    "error": {
                        "code": "NO_ACTIVE_SKILL",
                        "message": "当前没有激活 Skill",
                        "retryable": False,
                    },
                }
            try:
                content = self.registry.read_reference(
                    self.session.active_skill,
                    arguments["relative_path"],
                )
                return {
                    "ok": True,
                    "path": arguments["relative_path"],
                    "content": content,
                }
            except (OSError, ValueError) as exc:
                return {
                    "ok": False,
                    "error": {
                        "code": "REFERENCE_READ_FAILED",
                        "message": str(exc),
                        "retryable": False,
                    },
                }

        if name == "check_ffmpeg":
            return media_tools.check_ffmpeg()

        if name == "install_ffmpeg":
            command = media_tools.ffmpeg_install_command()
            if command is None:
                return media_tools.install_ffmpeg()
            if not self._confirm(
                "安装 FFmpeg",
                "将执行固定命令：\n" + " ".join(command),
            ):
                return {
                    "ok": False,
                    "cancelled": True,
                    "error": {
                        "code": "USER_CANCELLED",
                        "message": "用户取消了 FFmpeg 安装",
                        "retryable": False,
                    },
                }
            return media_tools.install_ffmpeg()

        if name == "check_whisper_model":
            return media_tools.check_whisper_model(**arguments)

        if name == "download_whisper_model":
            model_name = arguments["model_name"]
            if not self._confirm(
                "下载 Whisper 模型",
                f"将从 Hugging Face 下载 faster-whisper {model_name} 到 models 目录。",
            ):
                return {
                    "ok": False,
                    "cancelled": True,
                    "error": {
                        "code": "USER_CANCELLED",
                        "message": "用户取消了模型下载",
                        "retryable": False,
                    },
                }
            return media_tools.download_whisper_model(**arguments)

        if name == "transcribe_video":
            return media_tools.transcribe_video(**arguments)

        if name == "burn_subtitles":
            return media_tools.burn_subtitles(**arguments)

        return {
            "ok": False,
            "error": {
                "code": "UNKNOWN_TOOL",
                "message": f"未知工具：{name}",
                "retryable": False,
            },
        }

    def run_turn(self, user_input: str) -> str:
        self.activate_explicit_skill(user_input)
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._instructions()},
                    *self.messages,
                ],
                tools=self._available_tools(),
                extra_body={"thinking": {"type": "disabled"}},
            )
            message = response.choices[0].message
            self.messages.append(message)
            tool_calls = message.tool_calls or []

            if not tool_calls:
                return message.content or "模型没有返回可显示的文本。"

            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    result = self._execute_tool(
                        tool_call.function.name,
                        arguments,
                    )
                except Exception as exc:
                    result = {
                        "ok": False,
                        "error": {
                            "code": "TOOL_EXCEPTION",
                            "message": str(exc),
                            "retryable": False,
                        },
                    }

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        raise RuntimeError(f"Agent 超过最大工具循环步数：{self.max_steps}")
