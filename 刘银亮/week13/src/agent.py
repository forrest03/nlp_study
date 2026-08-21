"""
Agent 核心：接收用户输入，编排 LLM 调用、Skill 按需加载与执行、记忆更新

设计要点：
  1. 动态 tools 切换：默认只有 load_skill，load_skill 返回后注入 execute_skill
  2. 上下文清理：execute_skill 完成后，移除 load_skill 的调用与结果
  3. 记忆整合：system prompt 包含 skill catalog + memory 内容
  4. 生成器模式：yield 每步结构化结果，供 SSE 推送

使用方式：
  python agent.py --question "帮我算一下 123 * 456"
  python agent.py --question "现在几点" --max_steps 8
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Generator, Optional, List, Dict, Any

from openai import OpenAI

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from skill import (
    get_skill_catalog, load_skill_detail, execute_skill,
    get_load_skill_schema, get_execute_skill_schema,
)
from memory import memory_manager
from session_manager import session_manager, context_manager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ── LLM 客户端 ────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")


# ── System Prompt 构建 ─────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """构建 system prompt：skill catalog + memory"""
    catalog = get_skill_catalog()
    memory_content = memory_manager.load_memory()

    skill_list = "\n".join(
        f"  {i+1}. {s['name']} - {s['description']}"
        for i, s in enumerate(catalog)
    ) if catalog else "  （暂无可用技能）"

    return f"""你是一个智能办公助手，可以帮助用户回答问题、查询信息、处理办公任务。

## 可用技能
以下是目前可用的技能列表（仅显示名称和描述）：
{skill_list}

## 工作流程
1. 根据用户问题判断是否需要使用技能
2. 如果需要技能，先调用 load_skill(skill_name) 获取该技能的详细参数定义
3. 了解参数格式后，调用 execute_skill(skill_name, parameters) 执行技能
4. 根据执行结果回答用户问题
5. 如果不需要技能，直接回答即可

## 规则
- 每次只能调用一个技能
- execute_skill 前必须先 load_skill 了解参数
- 如果技能执行失败，可以重试或换一种方式回答
- 回答要简洁清晰

## 用户记忆
{memory_content}
"""


# ── Agent 类 ──────────────────────────────────────────────────────────────────

class Agent:
    """办公助手 Agent，支持技能按需加载和多轮对话"""

    def __init__(self):
        self.client = client
        self.model = MODEL

    def chat(
        self,
        question: str,
        session_id: Optional[str] = None,
        max_steps: int = 10,
    ) -> Generator[Dict, None, None]:
        """
        执行对话，yield 每一步结构化结果

        Args:
            question: 用户问题
            session_id: 会话 ID（为空则创建新会话）
            max_steps: 最大步数

        Yields:
            每步结果 dict，type 可能是：
            - "start": 对话开始
            - "load_skill": 加载 skill 详情
            - "execute_skill": 执行 skill
            - "final": 最终答案
            - "error": 错误
            - "max_steps": 达到最大步数
        """
        # 1. 创建或恢复会话
        if not session_id or not session_manager.session_exists(session_id):
            session_id = session_manager.create_session()
        else:
            logger.info(f"恢复会话: {session_id}")

        yield {"type": "start", "session_id": session_id, "question": question}

        # 2. 获取历史消息并压缩
        history = session_manager.get_messages(session_id)
        if history:
            history = context_manager.apply_strategy(history, "smart_compact")

        # 3. 构建 messages
        system_prompt = _build_system_prompt()
        if history:
            # 历史消息中可能包含旧的 system prompt，替换为最新的
            messages = [m for m in history if m.get("role") != "system"]
            messages.insert(0, {"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

        # 4. FC 循环（动态 tools）
        final_messages = messages
        for step_data in self._fc_loop(messages, max_steps):
            if step_data.get("type") in ("final", "error", "max_steps"):
                final_messages = step_data.get("messages", messages)
            yield step_data

        # 5. 保存消息到会话
        if final_messages:
            session_manager.save_messages(session_id, final_messages)
            session_manager.increment_turn(session_id)
            logger.info(f"会话 {session_id} 已保存，轮次: {session_manager.get_session_info(session_id)['turn_count']}")

        # 6. 更新记忆（同步执行，简化实现）
        if final_messages:
            try:
                memory_manager.update_memory(final_messages)
            except Exception as e:
                logger.warning(f"记忆更新失败: {e}")

        yield {"type": "done", "session_id": session_id}

    def _fc_loop(self, messages: List[Dict], max_steps: int) -> Generator[Dict, None, None]:
        """Function Calling 循环，动态切换 tools"""
        tools = [get_load_skill_schema()]

        for step in range(1, max_steps + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0,
                )
            except Exception as e:
                yield {
                    "step": step, "type": "error",
                    "error": f"LLM 调用失败: {e}",
                    "messages": messages,
                }
                return

            msg = response.choices[0].message
            reason = response.choices[0].finish_reason

            # 模型决定直接回答
            if reason == "stop" or not msg.tool_calls:
                content = msg.content or "（模型返回空内容）"
                messages.append({"role": "assistant", "content": content})
                yield {
                    "step": step, "type": "final",
                    "answer": content,
                    "messages": messages,
                }
                return

            # 处理工具调用（转为 dict 便于后续操作）
            messages.append(self._copy_message(msg))
            load_skill_called = False
            execute_skill_called = False

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                if fn_name == "load_skill":
                    skill_name = fn_args.get("skill_name", "")
                    observation = load_skill_detail(skill_name)
                    # 注入 execute_skill
                    tools = [get_load_skill_schema(), get_execute_skill_schema()]
                    load_skill_called = True
                    step_type = "load_skill"

                    yield {
                        "step": step, "type": step_type,
                        "skill": skill_name,
                        "observation": observation[:500],
                    }

                elif fn_name == "execute_skill":
                    skill_name = fn_args.get("skill_name", "")
                    params = fn_args.get("parameters", {})
                    observation = execute_skill(skill_name, params)
                    # 恢复为只有 load_skill
                    tools = [get_load_skill_schema()]
                    execute_skill_called = True
                    step_type = "execute_skill"

                    yield {
                        "step": step, "type": step_type,
                        "skill": skill_name,
                        "action_input": params,
                        "observation": str(observation)[:1000],
                    }

                else:
                    observation = f"未知工具: {fn_name}"
                    step_type = "error"

                    yield {
                        "step": step, "type": step_type,
                        "error": observation,
                    }

                # 回填工具结果
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(observation),
                })

            # execute_skill 完成后，清理 load_skill 的调用与结果
            if execute_skill_called and load_skill_called:
                messages = self._cleanup_load_skill(messages)


        yield {
            "step": max_steps + 1, "type": "max_steps",
            "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
            "messages": messages,
        }

    def _cleanup_load_skill(self, messages: List[Dict]) -> List[Dict]:
        """
        从 messages 中移除 load_skill 的 assistant 调用和对应的 tool 结果

        保留 execute_skill 的调用和结果，只清理中间的 load_skill 步骤
        """
        # 收集 load_skill 产生的 tool_call_id
        load_skill_tool_call_ids = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    if tc.get("function", {}).get("name") == "load_skill":
                        load_skill_tool_call_ids.add(tc.get("id"))

        if not load_skill_tool_call_ids:
            return messages

        cleaned = []
        for msg in messages:
            role = msg.get("role")

            # 跳过 load_skill 对应的 tool 结果
            if role == "tool" and msg.get("tool_call_id") in load_skill_tool_call_ids:
                continue

            # 处理包含 load_skill 的 assistant 消息
            if role == "assistant" and msg.get("tool_calls"):
                has_load_skill = any(
                    tc.get("function", {}).get("name") == "load_skill"
                    for tc in msg.get("tool_calls", [])
                )
                if has_load_skill:
                    other_calls = [
                        tc for tc in msg.get("tool_calls", [])
                        if tc.get("function", {}).get("name") != "load_skill"
                    ]
                    if not other_calls and not msg.get("content"):
                        # 纯 load_skill 调用，整条跳过
                        continue
                    else:
                        # 保留消息但移除 load_skill 的 tool_call
                        msg_copy = self._copy_message(msg)
                        msg_copy["tool_calls"] = other_calls
                        cleaned.append(msg_copy)
                        continue

            cleaned.append(msg)

        return cleaned

    @staticmethod
    def _copy_message(msg) -> Dict:
        """将 OpenAI message 对象或 dict 转为可序列化的 dict"""
        if isinstance(msg, dict):
            return msg.copy()

        # OpenAI ChatCompletionMessage 对象
        result = {"role": msg.role}
        if msg.content:
            result["content"] = msg.content
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return result


# ── CLI 打印 ──────────────────────────────────────────────────────────────────

COLORS = {
    "load":    "\033[36m",   # cyan
    "execute": "\033[33m",   # yellow
    "final":   "\033[35m",   # magenta
    "error":   "\033[31m",   # red
    "reset":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10):
    agent = Agent()
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}")
    print('='*60)

    start = time.time()

    for step_data in agent.chat(question, max_steps=max_steps):
        stype = step_data.get("type")

        if stype == "start":
            print(f"会话: {step_data['session_id']}")

        elif stype == "load_skill":
            print(f"\n[Step {step_data['step']}]")
            print(_c("load", f"📥 加载技能: {step_data['skill']}"))
            print(_c("load", f"   详情: {step_data['observation'][:200]}..."))

        elif stype == "execute_skill":
            print(f"\n[Step {step_data['step']}]")
            print(_c("execute", f"🔧 执行技能: {step_data['skill']}"))
            print(_c("execute", f"   参数: {json.dumps(step_data.get('action_input', {}), ensure_ascii=False)}"))
            print(_c("execute", f"   结果: {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ 最终答案:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('error', ''))}"))

        elif stype == "done":
            print(_c("final", f"\n会话已保存: {step_data['session_id']}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="办公助手 Agent")
    parser.add_argument("--question", default="帮我算一下 123 * 456")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps)
