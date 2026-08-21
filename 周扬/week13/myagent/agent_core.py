"""
通用智能体核心控制层。

这里放核心工程能力：
1. 多轮会话状态
2. tool calling 循环
3. skill 渐进式加载
4. 记忆读取与写入策略
5. 用户输入该写入 soul.md / user.md / memery.md 的判断
"""

from __future__ import annotations

import json
import re
from typing import Any, Generator


class AgentCore:
    """通用智能体核心。"""

    def __init__(self, tool_manager, skill_manager, memory_manager, session_id: str = "default_session"):
        self.tool_manager = tool_manager
        self.skill_manager = skill_manager
        self.memory_manager = memory_manager
        self.session_id = session_id
        self.session_messages: list[dict[str, Any]] = self.memory_manager.load_session_messages(session_id)

    def reset_session(self):
        """重置会话状态。"""
        self.session_messages = []
        self.skill_manager.reset_loaded_skills()
        self.memory_manager.clear_session_messages(self.session_id)
        self.memory_manager.clear_session_summary(self.session_id)

    def get_skill_status(self) -> dict[str, Any]:
        return {
            "manifest_text": self.skill_manager.get_manifest_text(),
            "loaded_skills": self.skill_manager.loaded_skills.copy(),
        }

    def get_memory_status(self) -> dict[str, Any]:
        summary = self.memory_manager.get_memory_summary()
        summary["session_file"] = str(self.memory_manager.get_session_file(self.session_id))
        summary["session_summary_file"] = str(self.memory_manager.get_session_summary_file(self.session_id))
        summary["session_summary_exists"] = bool(self.memory_manager.load_session_summary(self.session_id))
        summary["session_message_count"] = len(self.session_messages)
        return summary

    def _format_tool_calls(self, tool_calls) -> list[dict[str, Any]]:
        formatted = []
        for tool_call in tool_calls:
            formatted.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )
        return formatted

    def _safe_get_memory_prompt(self, user_input: str) -> str:
        try:
            return self.memory_manager.build_memory_prompt(user_input, top_k=3)
        except Exception as e:
            return f"长期记忆检索暂不可用：{e}"

    def _build_system_prompt(self, user_input: str) -> str:
        soul_text = self.memory_manager.read_soul()
        user_profile_text = self.memory_manager.read_user_profile()
        long_term_memory_text = self.memory_manager.read_long_term_memory()
        retrieved_memory_text = self._safe_get_memory_prompt(user_input)
        manifest_text = self.skill_manager.get_manifest_text()

        return (
            f"{soul_text}\n\n"
            f"以下是用户画像：\n{user_profile_text}\n\n"
            f"以下是长期记忆摘要：\n{long_term_memory_text}\n\n"
            f"以下是补充记忆信息：\n{retrieved_memory_text}\n\n"
            f"以下是当前可用的 skill 摘要清单（这里只是摘要，不是完整内容）：\n{manifest_text}\n\n"
            "你要遵守以下规则：\n"
            "1. 你是一个通用智能体，不是只会工具调用\n"
            "2. 你要先判断当前问题是普通问答、工具问题、skill 问题，还是记忆问题\n"
            "3. 需要外部信息时优先调用工具，不要编造\n"
            "4. skill 不要一次全加载，只在命中时渐进式加载\n"
            "5. 工具可能需要多轮调用，要循环到拿到最终答案\n"
            "6. 对于用户长期偏好、智能体规则、项目长期上下文，要判断写入哪一类记忆\n"
            "7. 如果要更新记忆，只能调用 update_soul_memory / update_user_memory / update_long_term_memory\n"
            "8. 严禁使用 run_local_script 修改任何记忆文件\n"
            "9. run_local_script 只接受真实脚本文件路径，不接受 ls -la、python -c、/bin/bash 这类命令串或解释器路径\n"
            "10. 如果任务需要生成文件，优先用 write_text_file；如果需要查看目录，优先用 list_directory；如果需要读取本地参考文件，优先用 read_text_file；如果需要预览结果，优先用 open_file\n"
            "11. 如果用户要求你把图、网页、闪卡、svg、html 等内容真正做出来，不能只口头描述，必须实际调用工具生成文件，并在最后明确告知输出路径\n"
            "12. 如果用户上一轮已经讲过需求，这一轮只说“画出来”“做出来”“生成吧”之类，你要结合最近会话继续执行，不要要求用户重复描述\n"
            "13. 不要返回空内容；若信息不足就提出最小澄清，否则继续执行\n"
            "14. 如果 skill 已经提供了 skill_dir 或真实文件路径，必须使用这些路径，不要猜测 .autoagent、.cursor 等旧目录\n"
        )

    def _decide_memory_actions(self, user_input: str, answer: str) -> dict[str, list[str]]:
        """
        判断这轮对话该写入哪类长期记忆。

        规则：
        - soul.md：关于智能体人格、行为准则、回答风格、必须遵守的规则
        - user.md：关于用户自己的稳定偏好、习惯、身份、常用配置
        - memery.md：项目上下文、长期任务、跨会话有价值的事实
        """
        text = user_input.strip()
        actions = {
            "soul": [],
            "user": [],
            "memery": [],
        }

        explicit_memory_markers = ["记住", "以后", "默认", "长期", "下次", "之后都", "一直按"]
        soul_markers = [
            "你要", "你应该", "你必须", "你的风格", "回答风格", "行为准则", "人格", "规则",
            "以后回答", "回答要", "更简洁", "更详细", "语气", "称呼", "叫你", "名字", "你就叫",
        ]
        user_markers = [
            "我喜欢", "我习惯", "我常用", "我一般", "我是", "我做", "我的偏好",
            "中文回答", "英文回答", "用中文", "用英文", "偏好",
        ]
        project_markers = ["这个项目", "当前项目", "我们在做", "后续", "路径", "目录", "项目规则", "约束", "当前任务"]

        has_memory_intent = any(marker in text for marker in explicit_memory_markers)

        if (
            ("你" in text or "智能体" in text or "助手" in text)
            and any(marker in text for marker in soul_markers)
        ):
            actions["soul"].append(text)

        if "我" in text and any(marker in text for marker in user_markers):
            actions["user"].append(text)

        if any(marker in text for marker in project_markers):
            actions["memery"].append(f"{text}\nassistant: {answer[:200]}")

        if has_memory_intent and not any(actions.values()):
            actions["memery"].append(f"用户希望长期记住：{text}\nassistant: {answer[:200]}")

        # 只有当这轮输入本身像“值得长期记住的信息”时，才写 md；
        # 普通对话只进向量记忆，不污染长文档。
        return actions

    def _should_invoke_llm_memory_router(self, user_input: str) -> bool:
        markers = [
            "记住", "以后", "默认", "长期", "偏好", "习惯", "我是", "我喜欢",
            "你要", "你以后", "回答要", "规则", "项目", "目录", "路径", "约束",
        ]
        return any(marker in user_input for marker in markers)

    def _llm_decide_memory_actions(self, user_input: str, answer: str, config) -> tuple[dict[str, list[str]], str]:
        """
        用 LLM 辅助判断该写入哪类长期记忆。
        返回 (actions, raw_content)。
        """
        empty_actions = {"soul": [], "user": [], "memery": []}

        if not config or not config.is_configured():
            return empty_actions, ""

        prompt = (
            "你是一个记忆路由分类器。请判断下面这轮对话里的用户输入，是否应该写入长期记忆。\n"
            "分类规则：\n"
            "1. soul: 智能体人格、回答风格、行为准则、必须遵守的规则\n"
            "2. user: 用户自己的长期偏好、习惯、背景、常用设置\n"
            "3. memery: 项目上下文、跨会话长期事实、目录路径、约束信息\n"
            "4. 如果不该写入长期记忆，对应数组返回空列表\n\n"
            "你必须只返回 JSON，格式如下：\n"
            '{"soul": [], "user": [], "memery": [], "reason": "..."}\n\n'
            f"用户输入：{user_input}\n"
            f"助手回复：{answer}\n"
        )

        try:
            response = config.client.chat.completions.create(
                model=config.model_name,
                messages=[
                    {"role": "system", "content": "你只输出 JSON，不要输出多余文字。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=300,
            )
            content = response.choices[0].message.content or ""
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1:
                return empty_actions, content

            data = json.loads(content[start:end + 1])
            actions = {
                "soul": data.get("soul", []) or [],
                "user": data.get("user", []) or [],
                "memery": data.get("memery", []) or [],
            }
            return actions, content
        except Exception as e:
            return empty_actions, f"LLM 路由失败: {e}"

    def _merge_memory_actions(
        self,
        base_actions: dict[str, list[str]],
        llm_actions: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        merged = {"soul": [], "user": [], "memery": []}
        for key in merged:
            values = []
            for item in base_actions.get(key, []) + llm_actions.get(key, []):
                if item and item not in values:
                    values.append(item)
            merged[key] = values
        return merged

    def _apply_memory_actions(self, actions: dict[str, list[str]]) -> list[dict[str, Any]]:
        events = []
        tool_plan = [
            ("soul", "update_soul_memory", "soul.md"),
            ("user", "update_user_memory", "user.md"),
            ("memery", "update_long_term_memory", "memery.md"),
        ]
        for key, tool_name, target_name in tool_plan:
            for item in actions[key]:
                events.append({"type": "tool_call", "tool": tool_name, "args": {"content": item}})
                result = self.tool_manager.execute_tool(tool_name, {"content": item})
                success = not result.startswith("MemoryManager 未注入")
                events.append({"type": "tool_result", "result": result, "success": success})
                if success:
                    events.append({"type": "memory_write", "target": target_name, "content": item})
        return events

    def _build_messages(self, user_input: str) -> list[dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(user_input),
            }
        ]
        session_summary = self.memory_manager.load_session_summary(self.session_id)
        if session_summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "以下是当前 session 的历史摘要，请把它视为本轮对话上下文的一部分：\n"
                        f"{session_summary}"
                    ),
                }
            )
        messages.extend(self.session_messages[-8:])
        messages.append({"role": "user", "content": user_input})
        return messages

    def _build_skill_context(self, user_input: str) -> str:
        """
        为 skill 命中构造轻量上下文。

        只拼接最近几轮用户/助手最终对话，不引入工具返回值，
        这样既能理解“把图画出来”这类跟进指令，也不容易被目录/脚本噪声带偏。
        """
        recent_messages = self.session_messages[-4:]
        parts: list[str] = []
        for item in recent_messages:
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            parts.append(f"{role}: {content[:300]}")
        parts.append(f"user: {user_input}")
        return "\n".join(parts)

    def _extract_svg_content(self, text: str) -> str:
        """从模型回复里提取完整 svg 文本。"""
        if not text:
            return ""
        match = re.search(r"(<svg[\s\S]*?</svg>)", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _auto_handle_svg_artifact(self, svg_text: str) -> list[dict[str, Any]]:
        """
        当模型直接返回 svg 文本时，自动落盘并尽量生成 png 预览。
        """
        events: list[dict[str, Any]] = []
        svg_path = "diagram/generated-diagram.svg"
        png_path = "diagram/generated-diagram@2x.png"

        events.append({"type": "tool_call", "tool": "write_text_file", "args": {"path": svg_path, "content": svg_text}})
        write_result = self.tool_manager.execute_tool("write_text_file", {"path": svg_path, "content": svg_text})
        events.append({"type": "tool_result", "result": write_result, "success": not write_result.startswith("工具")})

        meta = self.skill_manager.get_skill_meta("baoyu-diagram") or {}
        skill_dir = meta.get("dir_path")
        if skill_dir:
            script_path = f"{skill_dir}/scripts/main.ts"
            script_args = [svg_path, "-o", png_path]
            events.append({"type": "tool_call", "tool": "run_local_script", "args": {"script_path": script_path, "args": script_args}})
            png_result = self.tool_manager.execute_tool(
                "run_local_script",
                {"script_path": script_path, "args": script_args},
            )
            png_success = "返回码：0" in png_result
            events.append({"type": "tool_result", "result": png_result, "success": png_success})

        events.append({"type": "tool_call", "tool": "open_file", "args": {"path": svg_path}})
        open_result = self.tool_manager.execute_tool("open_file", {"path": svg_path})
        open_success = open_result.startswith("已打开文件")
        events.append({"type": "tool_result", "result": open_result, "success": open_success})
        return events

    def _persist_session_messages(self):
        self.memory_manager.save_session_messages(self.session_id, self.session_messages[-20:])

    def _build_session_transcript(self, messages: list[dict[str, Any]]) -> str:
        lines = []
        for item in messages:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            if not content:
                continue
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _update_session_summary(self, config) -> str:
        """
        根据当前 session 历史自动更新摘要。

        策略：
        - 历史不长时不生成
        - 超过阈值后，用旧摘要 + 较早的对话历史生成新摘要
        - 摘要生成后只保留最近几轮 messages
        """
        if len(self.session_messages) < 8:
            return ""

        old_summary = self.memory_manager.load_session_summary(self.session_id)
        recent_keep = 6
        to_summarize = self.session_messages[:-recent_keep]
        if not to_summarize:
            return ""

        transcript = self._build_session_transcript(to_summarize)
        prompt = (
            "请把下面这个 session 的历史对话总结成一份后续可继续使用的上下文摘要。\n"
            "摘要要求：\n"
            "1. 保留用户目标、偏好、已确认事实、未完成事项\n"
            "2. 保留当前任务进展和关键路径\n"
            "3. 不要写废话，不要逐句复述\n"
            "4. 输出用中文\n\n"
            f"旧摘要：\n{old_summary or '暂无'}\n\n"
            f"新增历史：\n{transcript}\n"
        )

        try:
            response = config.client.chat.completions.create(
                model=config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个会话摘要助手，请输出简洁摘要。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            summary = (response.choices[0].message.content or "").strip()
            if summary:
                self.memory_manager.save_session_summary(self.session_id, summary)
                self.session_messages = self.session_messages[-recent_keep:]
                self._persist_session_messages()
            return summary
        except Exception:
            return old_summary

    def run(self, user_input: str, config, max_steps: int = 8) -> Generator[dict[str, Any], None, None]:
        """真实 agent 主循环。"""
        if not config.is_configured():
            yield {"type": "error", "content": "未配置模型，请先输入 /model 进行配置"}
            return

        yield {"type": "thinking", "content": f"使用模型 [{config.model_name}] 处理用户问题..."}

        messages = self._build_messages(user_input)
        skill_context_text = self._build_skill_context(user_input)
        empty_response_retries = 0
        session_summary_exists = bool(self.memory_manager.load_session_summary(self.session_id))
        yield {
            "type": "context_status",
            "session_message_count": len(self.session_messages),
            "session_summary_exists": session_summary_exists,
            "loaded_skill_count": len(self.skill_manager.loaded_skills),
        }

        try:
            for step in range(1, max_steps + 1):
                yield {"type": "step", "num": step, "desc": "分析上下文，尝试渐进式加载 skill"}

                load_result = self.skill_manager.progressive_load(skill_context_text)
                yield {
                    "type": "skill_status",
                    "matched_skills": load_result["matched_skills"],
                    "new_skills": load_result["new_skills"],
                    "loaded_skills": load_result["loaded_skills"],
                }
                if load_result["new_skills"]:
                    yield {
                        "type": "thinking",
                        "content": f"本轮新加载 skill: {', '.join(load_result['new_skills'])}",
                    }
                    messages.append(
                        {
                            "role": "system",
                            "content": f"下面是本轮新加载的 skill，请按这些规则工作：\n\n{load_result['loaded_prompt']}",
                        }
                    )

                yield {"type": "thinking", "content": "开始请求模型判断是否需要调用工具..."}
                request_messages = list(messages)
                if load_result["active_prompt"]:
                    request_messages.append(
                        {
                            "role": "system",
                            "content": f"下面是当前命中的相关 skill，请优先按这些规则工作：\n\n{load_result['active_prompt']}",
                        }
                    )
                response_max_tokens = 6000 if "baoyu-diagram" in load_result["matched_skills"] else 2048

                response = config.client.chat.completions.create(
                    model=config.model_name,
                    messages=request_messages,
                    tools=self.tool_manager.get_tools_schema(),
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=response_max_tokens,
                )

                assistant_message = response.choices[0].message

                if not assistant_message.tool_calls:
                    content = (assistant_message.content or "").strip()
                    if not content:
                        if empty_response_retries < 1:
                            empty_response_retries += 1
                            yield {
                                "type": "thinking",
                                "content": "模型这一步返回了空内容，我会追加执行约束后再尝试一次。",
                            }
                            messages.append(
                                {
                                    "role": "system",
                                    "content": (
                                        "你刚刚返回了空内容。不要返回空内容。"
                                        "如果任务要求生成本地文件，请立即调用合适的工具真正生成文件；"
                                        "如果已经完成，请明确回复结果与输出路径。"
                                    ),
                                }
                            )
                            continue
                        content = "模型返回了空内容"
                    yield {
                        "type": "model_decision",
                        "decision": "final_response",
                        "tool_calls": [],
                    }
                    svg_text = self._extract_svg_content(content)
                    if svg_text:
                        yield {"type": "thinking", "content": "检测到模型直接返回了 SVG，我会自动保存并打开预览。"}
                        for event in self._auto_handle_svg_artifact(svg_text):
                            yield event
                        content = (
                            "图已经生成完成，并已保存到 `diagram/generated-diagram.svg`。"
                            "如果 PNG 转换成功，也会输出到 `diagram/generated-diagram@2x.png`。"
                        )
                    yield {
                        "type": "thinking",
                        "content": f"模型已返回最终结果（{response.usage.total_tokens} tokens）",
                    }

                    memory_actions = self._decide_memory_actions(user_input, content)
                    llm_memory_actions = {"soul": [], "user": [], "memery": []}
                    llm_reason = ""
                    if self._should_invoke_llm_memory_router(user_input):
                        llm_memory_actions, llm_reason = self._llm_decide_memory_actions(user_input, content, config)
                        memory_actions = self._merge_memory_actions(memory_actions, llm_memory_actions)

                    if any(memory_actions.values()):
                        yield {
                            "type": "thinking",
                            "content": (
                                f"记忆路由判断结果: "
                                f"soul={len(memory_actions['soul'])}, "
                                f"user={len(memory_actions['user'])}, "
                                f"memery={len(memory_actions['memery'])}"
                            ),
                        }
                    if llm_reason:
                        yield {"type": "thinking", "content": f"LLM 记忆路由结果: {llm_reason}"}

                    for event in self._apply_memory_actions(memory_actions):
                        yield event

                    self.session_messages.append({"role": "user", "content": user_input})
                    self.session_messages.append({"role": "assistant", "content": content})
                    self.session_messages = self.session_messages[-12:]
                    self._persist_session_messages()
                    summary_text = self._update_session_summary(config)
                    if summary_text:
                        yield {"type": "thinking", "content": "当前 session 摘要已自动更新"}

                    yield {"type": "response", "content": content}
                    return

                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": self._format_tool_calls(assistant_message.tool_calls),
                    }
                )
                yield {
                    "type": "model_decision",
                    "decision": "tool_call",
                    "tool_calls": [tool_call.function.name for tool_call in assistant_message.tool_calls],
                }

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    post_tool_system_message = None
                    try:
                        tool_args = json.loads(tool_call.function.arguments or "{}")
                    except Exception:
                        tool_args = {}

                    yield {"type": "tool_call", "tool": tool_name, "args": tool_args}
                    tool_result = self.tool_manager.execute_tool(tool_name, tool_args)
                    success = not (
                        tool_result.startswith("未知工具")
                        or tool_result.startswith("工具参数错误")
                        or tool_result.startswith("工具执行失败")
                    )
                    yield {"type": "tool_result", "result": tool_result, "success": success}

                    if (
                        not success
                        and tool_name == "write_text_file"
                        and "missing 2 required positional arguments" in tool_result
                    ):
                        post_tool_system_message = {
                            "role": "system",
                            "content": (
                                "你调用 write_text_file 时必须同时提供 path 和 content。"
                                "如果 SVG 内容太长，不方便放进工具参数，"
                                "可以直接在下一条回复中只输出完整 SVG 文本（从 <svg> 到 </svg>），"
                                "系统会帮你保存成文件。"
                            ),
                        }

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )
                    if post_tool_system_message is not None:
                        messages.append(post_tool_system_message)
            final_text = "已达到最大推理轮数，先停在这里。"
            for event in self._apply_memory_actions({"soul": [], "user": [], "memery": []}):
                yield event
            self.session_messages.append({"role": "user", "content": user_input})
            self.session_messages.append({"role": "assistant", "content": final_text})
            self.session_messages = self.session_messages[-12:]
            self._persist_session_messages()
            summary_text = self._update_session_summary(config)
            if summary_text:
                yield {"type": "thinking", "content": "当前 session 摘要已自动更新"}
            yield {"type": "error", "content": final_text}

        except Exception as e:
            yield {"type": "error", "content": f"调用模型失败: {e}"}

    def simulate(self, user_input: str) -> Generator[dict[str, Any], None, None]:
        """未配置模型时的模拟流程。"""
        yield {"type": "thinking", "content": f"用户问了一个问题：{user_input}，我需要分析一下..."}
        yield {
            "type": "context_status",
            "session_message_count": len(self.session_messages),
            "session_summary_exists": bool(self.memory_manager.load_session_summary(self.session_id)),
            "loaded_skill_count": len(self.skill_manager.loaded_skills),
        }
        load_result = self.skill_manager.progressive_load(self._build_skill_context(user_input))
        yield {
            "type": "skill_status",
            "matched_skills": load_result["matched_skills"],
            "new_skills": load_result["new_skills"],
            "loaded_skills": load_result["loaded_skills"],
        }
        if load_result["new_skills"]:
            yield {
                "type": "thinking",
                "content": f"模拟模式命中了 skill: {', '.join(load_result['new_skills'])}",
            }

        memory_actions = self._decide_memory_actions(user_input, "模拟回答")
        if any(memory_actions.values()):
            yield {
                "type": "thinking",
                "content": (
                    f"这条输入像是长期记忆，可能会写入："
                    f"{', '.join([k for k, v in memory_actions.items() if v])}"
                ),
            }

        yield {"type": "step", "num": 1, "desc": "模拟工具调用流程"}

        if "天气" in user_input or "经纬度" in user_input:
            yield {"type": "model_decision", "decision": "tool_call", "tool_calls": ["query_geo"]}
            yield {"type": "tool_call", "tool": "query_geo", "args": {"city_name": "北京"}}
            yield {"type": "tool_result", "result": "北京 的经纬度为：39.9075, 116.3972", "success": True}
        elif "系统" in user_input or "环境" in user_input:
            yield {"type": "model_decision", "decision": "tool_call", "tool_calls": ["get_system_info"]}
            yield {"type": "tool_call", "tool": "get_system_info", "args": {}}
            yield {"type": "tool_result", "result": self.tool_manager.get_system_info(), "success": True}
        else:
            yield {"type": "model_decision", "decision": "final_response", "tool_calls": []}
            yield {"type": "thinking", "content": "当前是模拟模式，没有真实模型参与。"}

        yield {
            "type": "response",
            "content": f"这是关于「{user_input}」的模拟回答。（如需真实工具循环，请先输入 /model 配置模型）",
        }
